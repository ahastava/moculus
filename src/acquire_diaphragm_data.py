"""
Diaphragmatic POCUS Ingest Pipeline (Phase 3)
==============================================

Generic ingest for diaphragm/PLAPS-visible POCUS frames sourced from
public datasets (POCUS Atlas, GrepMed, NEJM supplements, Butterfly
teaching cases, LITFL, Rochester LUS, etc.).

This script is intentionally **source-agnostic**: it doesn't scrape
the internet. You manually download whatever you legally can from
each source into a local staging directory, then run this script to:

  1. Walk the staging directory recursively for media files
     (images: .png .jpg .jpeg .bmp .tif .tiff .webp ; videos: .mp4 .mov .avi .webm .mkv)
  2. For videos, extract frames via ffmpeg at a configurable fps
  3. Center-crop each image to remove vendor UI overlays
  4. Letterbox-resize to 256×256 (preserves aspect ratio)
  5. Deduplicate via difference-hash (dhash) with Hamming threshold
  6. Auto-filter via the existing `_detect_pleural_line` heuristic
     from anatomy_bank.py — keeps frames where the pleural line is in
     the upper 20% of the image (i.e. lower BLUE / PLAPS / Diaphragm
     views, which is exactly what we want for the diaphragm bank)
  7. Write surviving candidates to a `candidates/` directory and
     emit a one-line CSV manifest per candidate

After this runs, you open scripts/labeling_review.html in your
browser pointed at the candidates directory, click through the
frames, and assign (pathology_class, zone_region) labels. The
labels CSV produced by the labeling page is then merged into
data/real_pocus/processed/metadata.csv via
scripts/merge_diaphragm_labels.py.

Why this design
---------------
* Source-agnostic: licensing is your call. Some sources (POCUS Atlas)
  are CC-licensed and can be batch-downloaded with wget/curl. Others
  (NEJM, Butterfly) require institutional access and manual saving.
  This script doesn't care — it just processes whatever's in the
  staging dir.
* No new dependencies: uses only numpy + PIL + scipy (already in
  the env) plus ffmpeg (system tool). Perceptual hashing is
  implemented inline as a 64-bit difference hash (dhash).
* Idempotent: re-running on the same staging dir is safe — already-
  processed files are skipped via the dedup hash.

Usage
-----
    # 1. Download / save sourced media into a staging dir
    mkdir -p data/diaphragm_staging/pocus_atlas
    # ... save .png/.mp4 files there ...

    # 2. Run the ingest pipeline
    python3 -m src.acquire_diaphragm_data \\
        --staging data/diaphragm_staging \\
        --output data/diaphragm_candidates \\
        --fps 2

    # 3. Open scripts/labeling_review.html, label the candidates,
    #    save the labels CSV.

    # 4. Merge labels into the main metadata.csv:
    python3 scripts/merge_diaphragm_labels.py \\
        --labels data/diaphragm_candidates/labels.csv \\
        --candidates-dir data/diaphragm_candidates
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}

DEFAULT_OUTPUT_SIZE = 256
DEFAULT_CENTER_CROP_FRACTION = 0.70
DEFAULT_FFMPEG_FPS = 2
DEFAULT_DEDUP_HAMMING_THRESHOLD = 6  # bits
DEFAULT_PLEURAL_LINE_MAX_FRAC = 0.20  # keep frames with pleural line in top 20%

DHASH_SIZE = 8  # 8x8 + 1 = 64-bit hash


# ---------------------------------------------------------------------------
# Difference-hash (dhash) — minimal perceptual-hash implementation
# ---------------------------------------------------------------------------

def dhash(img_arr: np.ndarray, hash_size: int = DHASH_SIZE) -> int:
    """
    Compute a 64-bit difference hash (dhash) of a grayscale image.

    Standard dhash construction (Krawetz, 2013):
        1. Resize the image to (hash_size + 1) x hash_size
        2. Compare adjacent horizontal pixels: bit_i = (left > right)
        3. Pack the 64 bits into an integer

    Two images are considered "near-duplicate" if the Hamming distance
    between their hashes is <= DEFAULT_DEDUP_HAMMING_THRESHOLD.
    """
    # Resize to 9x8 using PIL (works on uint8)
    if img_arr.dtype != np.uint8:
        img_arr = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(img_arr).resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    ).convert("L")
    pixels = np.asarray(pil, dtype=np.int32)

    # Horizontal difference: bit set when left pixel > right pixel
    diff = pixels[:, :-1] > pixels[:, 1:]
    # Pack 64 bits into a Python int
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def center_crop_fraction(img: np.ndarray, frac: float = DEFAULT_CENTER_CROP_FRACTION) -> np.ndarray:
    """Keep the central `frac` of the image (removes vendor UI bands)."""
    h, w = img.shape[:2]
    nh = int(h * frac)
    nw = int(w * frac)
    r0 = (h - nh) // 2
    c0 = (w - nw) // 2
    return img[r0:r0 + nh, c0:c0 + nw]


def letterbox_resize(img: np.ndarray, size: int = DEFAULT_OUTPUT_SIZE) -> np.ndarray:
    """
    Resize an image to a square `size x size` canvas, preserving aspect
    ratio with zero-padding (letterbox). Avoids the distortion that a
    plain resize introduces on non-square ultrasound clips.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=img.dtype)

    scale = min(size / h, size / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    if img.dtype != np.uint8:
        scaled = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        scaled = img
    pil = Image.fromarray(scaled).resize((nw, nh), Image.LANCZOS).convert("L")
    resized = np.asarray(pil, dtype=np.uint8)

    canvas = np.zeros((size, size), dtype=np.uint8)
    r0 = (size - nh) // 2
    c0 = (size - nw) // 2
    canvas[r0:r0 + nh, c0:c0 + nw] = resized
    return canvas


def load_image_grayscale(path: Path) -> Optional[np.ndarray]:
    """Load any image file as a uint8 grayscale numpy array."""
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.uint8)
        return arr
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# ffmpeg frame extraction
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    fps: int = DEFAULT_FFMPEG_FPS,
) -> List[Path]:
    """
    Run ffmpeg to extract grayscale frames at `fps` Hz from a video.
    Returns the list of extracted frame paths.

    Output naming: {video_stem}_frame_NNNN.png (zero-padded for sort order).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"{video_path.stem}_frame_%04d.png"

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",  # overwrite existing
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(pattern),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg failed on %s: %s", video_path, exc)
        return []
    except FileNotFoundError:
        logger.error(
            "ffmpeg binary not found on PATH. Install via: "
            "sudo apt install ffmpeg  (or equivalent)"
        )
        return []

    return sorted(output_dir.glob(f"{video_path.stem}_frame_*.png"))


# ---------------------------------------------------------------------------
# Pleural line filter
# ---------------------------------------------------------------------------

def _detect_pleural_row_fraction(img: np.ndarray) -> Optional[float]:
    """
    Lightweight reimplementation of `_detect_pleural_line` from
    anatomy_bank.py, returning the pleural line position as a
    fraction of image height (0.0 = top, 1.0 = bottom).

    Inlined here so the ingest script doesn't have to import from
    src.anatomy_bank (which would pull in the whole src/ package).
    """
    h, w = img.shape
    if h == 0 or w == 0:
        return None

    img_f = img.astype(np.float32) / 255.0
    search_top = int(0.08 * h)
    search_bot = int(0.40 * h)
    if search_bot <= search_top:
        return 0.17

    row_means = img_f[search_top:search_bot, :].mean(axis=1)
    # Light smoothing
    if len(row_means) >= 5:
        kernel = np.ones(5) / 5
        row_means = np.convolve(row_means, kernel, mode="same")

    peak_row = search_top + int(np.argmax(row_means))
    return peak_row / h


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------

@dataclass
class IngestStats:
    scanned: int = 0
    extracted_from_video: int = 0
    deduped: int = 0
    rejected_pleural: int = 0
    accepted: int = 0


def discover_media(staging_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Walk the staging directory and return (images, videos)."""
    images: List[Path] = []
    videos: List[Path] = []
    for p in staging_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in VIDEO_EXTS:
            videos.append(p)
    return sorted(images), sorted(videos)


def write_manifest_row(
    manifest_path: Path,
    candidate: Path,
    source: Path,
    pleural_frac: Optional[float],
    sha256: str,
) -> None:
    """Append one row to the manifest CSV (creating the file with header if needed)."""
    is_new = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "filename",
                "source_relpath",
                "pleural_line_frac",
                "sha256",
            ])
        writer.writerow([
            candidate.name,
            str(source),
            f"{pleural_frac:.4f}" if pleural_frac is not None else "",
            sha256,
        ])


def ingest_one_image(
    img_arr: np.ndarray,
    source: Path,
    output_dir: Path,
    seen_hashes: Set[int],
    seen_sha: Set[str],
    stats: IngestStats,
    output_size: int,
    pleural_max_frac: float,
    dedup_threshold: int,
    crop_frac: float,
    manifest_path: Path,
    staging_root: Path,
) -> Optional[Path]:
    """Process a single image array → optional saved candidate path."""
    stats.scanned += 1

    # Crop UI overlays + letterbox resize
    cropped = center_crop_fraction(img_arr, frac=crop_frac)
    canvas = letterbox_resize(cropped, size=output_size)

    # Exact-byte SHA256 dedup (catches identical files saved twice)
    sha = hashlib.sha256(canvas.tobytes()).hexdigest()
    if sha in seen_sha:
        stats.deduped += 1
        return None
    seen_sha.add(sha)

    # Perceptual dhash dedup (catches near-duplicates from different sources)
    h = dhash(canvas)
    for prev in seen_hashes:
        if hamming_distance(h, prev) <= dedup_threshold:
            stats.deduped += 1
            return None
    seen_hashes.add(h)

    # Pleural-line filter — keep only frames where the pleural line is in
    # the upper portion (heuristic for lower-zone / diaphragm-visible views)
    pleural_frac = _detect_pleural_row_fraction(canvas)
    if pleural_frac is None or pleural_frac > pleural_max_frac:
        stats.rejected_pleural += 1
        return None

    # Save candidate with deterministic filename based on SHA prefix
    candidate_name = f"diap_cand_{sha[:12]}.png"
    candidate_path = output_dir / candidate_name
    Image.fromarray(canvas).save(candidate_path, optimize=True)

    # Write manifest row with relative source path for attribution
    try:
        rel_source = source.relative_to(staging_root)
    except ValueError:
        rel_source = source
    write_manifest_row(
        manifest_path,
        candidate_path,
        rel_source,
        pleural_frac,
        sha,
    )
    stats.accepted += 1
    return candidate_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=Path("data/diaphragm_staging"),
        help="Directory containing manually-sourced media files (recursive).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diaphragm_candidates"),
        help="Output directory for processed 256x256 candidates and manifest.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FFMPEG_FPS,
        help="Frame extraction rate for videos (Hz). Default 2 = ~one frame per breath.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=DEFAULT_OUTPUT_SIZE,
        help=f"Output image size (default {DEFAULT_OUTPUT_SIZE}).",
    )
    parser.add_argument(
        "--crop-frac",
        type=float,
        default=DEFAULT_CENTER_CROP_FRACTION,
        help=f"Center-crop fraction before resize (default {DEFAULT_CENTER_CROP_FRACTION}).",
    )
    parser.add_argument(
        "--pleural-max-frac",
        type=float,
        default=DEFAULT_PLEURAL_LINE_MAX_FRAC,
        help=(
            "Reject frames whose detected pleural line is below this fraction "
            "of image height. Default 0.20 keeps frames with the pleural line "
            "in the upper 20%, which is the lower-BLUE/PLAPS/Diaphragm view "
            "geometry."
        ),
    )
    parser.add_argument(
        "--dedup-threshold",
        type=int,
        default=DEFAULT_DEDUP_HAMMING_THRESHOLD,
        help=f"Hamming distance threshold for dhash dedup (default {DEFAULT_DEDUP_HAMMING_THRESHOLD}).",
    )
    parser.add_argument(
        "--no-pleural-filter",
        action="store_true",
        help="Disable the pleural-line filter (keep every cropped frame).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-source progress.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    staging = args.staging.resolve()
    output = args.output.resolve()

    if not staging.exists():
        print(f"ERROR: staging directory does not exist: {staging}", file=sys.stderr)
        print(
            "Create it and place sourced .png/.mp4 files inside, e.g.:\n"
            f"    mkdir -p {staging}/pocus_atlas\n"
            "    # save downloaded files there",
            file=sys.stderr,
        )
        return 1

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    # Fresh manifest each run — re-running the script regenerates the
    # manifest from scratch (the candidate PNGs themselves are content-
    # addressed by SHA, so they're naturally idempotent).
    if manifest_path.exists():
        manifest_path.unlink()

    print(f"Staging dir:  {staging}")
    print(f"Output dir:   {output}")
    print(f"Manifest CSV: {manifest_path}")
    print()

    images, videos = discover_media(staging)
    print(f"Discovered: {len(images)} images, {len(videos)} videos in staging")

    if args.no_pleural_filter:
        pleural_max = 1.0  # accept everything
    else:
        pleural_max = args.pleural_max_frac

    seen_sha: Set[str] = set()
    seen_hashes: Set[int] = set()
    stats = IngestStats()

    # ── Process standalone images ──
    for img_path in images:
        if args.verbose:
            print(f"  [img] {img_path}")
        arr = load_image_grayscale(img_path)
        if arr is None:
            continue
        ingest_one_image(
            img_arr=arr,
            source=img_path,
            output_dir=output,
            seen_hashes=seen_hashes,
            seen_sha=seen_sha,
            stats=stats,
            output_size=args.output_size,
            pleural_max_frac=pleural_max,
            dedup_threshold=args.dedup_threshold,
            crop_frac=args.crop_frac,
            manifest_path=manifest_path,
            staging_root=staging,
        )

    # ── Process videos via ffmpeg into a temp subdir ──
    if videos:
        temp_frames_dir = output / "_temp_video_frames"
        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)
        temp_frames_dir.mkdir(parents=True)

        for video_path in videos:
            if args.verbose:
                print(f"  [vid] {video_path} (fps={args.fps})")
            frame_paths = extract_video_frames(
                video_path,
                temp_frames_dir,
                fps=args.fps,
            )
            stats.extracted_from_video += len(frame_paths)

            for fp in frame_paths:
                arr = load_image_grayscale(fp)
                if arr is None:
                    continue
                ingest_one_image(
                    img_arr=arr,
                    source=video_path,  # attribute back to the video
                    output_dir=output,
                    seen_hashes=seen_hashes,
                    seen_sha=seen_sha,
                    stats=stats,
                    output_size=args.output_size,
                    pleural_max_frac=pleural_max,
                    dedup_threshold=args.dedup_threshold,
                    crop_frac=args.crop_frac,
                    manifest_path=manifest_path,
                    staging_root=staging,
                )
                fp.unlink()  # save disk

        shutil.rmtree(temp_frames_dir, ignore_errors=True)

    # ── Summary ──
    print()
    print("=" * 60)
    print("Ingest summary")
    print("=" * 60)
    print(f"  Scanned (images + extracted frames): {stats.scanned}")
    print(f"  Extracted from videos:                {stats.extracted_from_video}")
    print(f"  Deduplicated (SHA + dhash):           {stats.deduped}")
    print(f"  Rejected by pleural-line filter:      {stats.rejected_pleural}")
    print(f"  Accepted candidates:                  {stats.accepted}")
    print()
    print(f"Candidates written to: {output}")
    print(f"Manifest:              {manifest_path}")
    print()
    if stats.accepted > 0:
        print("Next step: open scripts/labeling_review.html in your browser")
        print("           and assign (pathology_class, zone_region) to each candidate.")
    else:
        print(
            "No candidates accepted. Check that:\n"
            "  - Files are present under the staging directory\n"
            "  - Files have one of the expected extensions: "
            f"{sorted(IMAGE_EXTS | VIDEO_EXTS)}\n"
            "  - The pleural-line filter isn't too strict (try "
            "--no-pleural-filter or raise --pleural-max-frac)"
        )

    # Emit a tiny summary JSON next to the manifest for tooling
    summary_path = output / "ingest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "scanned": stats.scanned,
                "extracted_from_video": stats.extracted_from_video,
                "deduped": stats.deduped,
                "rejected_pleural": stats.rejected_pleural,
                "accepted": stats.accepted,
                "config": {
                    "output_size": args.output_size,
                    "crop_frac": args.crop_frac,
                    "pleural_max_frac": pleural_max,
                    "dedup_threshold": args.dedup_threshold,
                    "fps": args.fps,
                },
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
