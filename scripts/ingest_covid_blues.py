#!/usr/bin/env python3
"""
COVID-BLUES auto-ingest — zones from filename, pathology from severity.csv

Specialised counterpart to `src.acquire_diaphragm_data`. The COVID-BLUES
dataset (already on disk at `data/_sources/covid_blues_meta/`) is unique
in two ways that let us skip the manual labeling step entirely:

  1. Each video filename encodes the BLUE protocol point:
       patient_<ID>_L1.mp4  Upper BLUE  Left
       patient_<ID>_L2.mp4  Lower BLUE  Left
       patient_<ID>_L3.mp4  PLAPS       Left
       patient_<ID>_R1.mp4  Upper BLUE  Right
       patient_<ID>_R2.mp4  Lower BLUE  Right
       patient_<ID>_R3.mp4  PLAPS       Right

  2. severity.csv has per-video annotations:
       - Severity Score (0.0 = normal, 3.0 = severe)
       - A-lines (yes/no) and B-lines (yes/no)
       - Free-text comments mentioning effusion / consolidation /
         thickened pleural line / etc.

This script combines both to emit a `diaphragm_labels.csv` directly,
in the same format the labeling UI produces, so the existing
`scripts/merge_diaphragm_labels.py` picks it up unchanged.

Pathology heuristic
-------------------
We map each video to one of the diaphragm-relevant MoCoLUS classes
using severity + A/B-line annotations + comment text:

  Comment contains "effusion" or "fluid"           → class 5 (effusion)
  Comment contains "consolidat"                    → class 4 (consolidation)
  Severity 3 + B-lines yes                         → class 6 (ARDS)
  Severity 2 + B-lines yes                         → class 3 (B-lines diffuse)
  Severity 1 + B-lines yes                         → class 2 (B-lines focal)
  Comment contains "thickened" or "broken pleura"  → class 8 (thickening)
  Severity 0 + A-lines yes + B-lines no            → class 0 (normal)
  All others                                       → skipped (uncertain)

The conservative defaults skip ambiguous videos rather than mislabel
them. You can override the heuristic per-video in the labeling UI
afterward if needed.

Zone mapping
------------
  L2 → ZoneRegion.LOWER_BLUE_L (1)
  L3 → ZoneRegion.PLAPS_L      (3)
  R2 → ZoneRegion.LOWER_BLUE_R (2)
  R3 → ZoneRegion.PLAPS_R      (4)

L1/R1 (upper zones) are skipped — they don't help the diaphragm bank.
Note: COVID-BLUES does NOT include true diaphragm-view scans
(zone_region 5/6) because the BLUE protocol stops at PLAPS. The
DiaphragmAnatomyBank's 5/6 cells will fall back to the parent
LesionAnatomyBank's class-only sampling for those zones.

Usage
-----
    python3 scripts/ingest_covid_blues.py \\
        --videos-dir data/_sources/covid_blues_meta/lus_videos \\
        --severity   data/_sources/covid_blues_meta/severity.csv \\
        --output     data/diaphragm_candidates \\
        --fps 2 \\
        --max-frames-per-video 8

After this runs:
    cat data/diaphragm_candidates/diaphragm_labels.csv  # auto-generated
    python3 scripts/merge_diaphragm_labels.py \\
        --labels         data/diaphragm_candidates/diaphragm_labels.csv \\
        --candidates-dir data/diaphragm_candidates \\
        --source-dataset COVID-BLUES

Licensing
---------
COVID-BLUES is CC BY-NC-ND 4.0. The "no derivatives" clause means
training ML models on this data is in a legal gray area. Cite the
upstream paper (Wiedemann et al. 2025, IEEE J Biomed Health Inform)
if you use it. See DIAPHRAGM_DATA.md for the full licensing notes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

# Reuse the helpers from the generic ingest pipeline so we don't duplicate
# the dhash / center-crop / letterbox / pleural-line logic.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.acquire_diaphragm_data import (  # noqa: E402
    center_crop_fraction,
    letterbox_resize,
    dhash,
    hamming_distance,
    extract_video_frames,
    load_image_grayscale,
    _detect_pleural_row_fraction,
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_CENTER_CROP_FRACTION,
    DEFAULT_DEDUP_HAMMING_THRESHOLD,
    DEFAULT_PLEURAL_LINE_MAX_FRAC,
)

logger = logging.getLogger(__name__)

# ─ Zone mapping for COVID-BLUES BLUE points ─
BLUE_POINT_TO_ZONE_REGION = {
    "L2": 1,  # LOWER_BLUE_L
    "L3": 3,  # PLAPS_L
    "R2": 2,  # LOWER_BLUE_R
    "R3": 4,  # PLAPS_R
}
# L1, R1 (upper zones) are intentionally absent — they don't go in the
# diaphragm bank (their entries fall back to the existing class-only
# LesionAnatomyBank).

# ─ MoCoLUS pathology class names (must match clinical_frames.py) ─
PATHOLOGY_CLASS_NAMES = {
    0: "normal_a_profile",
    2: "b_lines_focal",
    3: "b_lines_diffuse",
    4: "consolidation",
    5: "pleural_effusion",
    6: "ards_white_lung",
    8: "pleural_thickening",
}


@dataclass
class IngestStats:
    videos_scanned: int = 0
    videos_skipped_zone: int = 0      # not L2/L3/R2/R3
    videos_skipped_unsevered: int = 0  # no severity row
    videos_skipped_uncertain: int = 0  # heuristic couldn't classify
    frames_extracted: int = 0
    frames_deduped: int = 0
    frames_rejected_pleural: int = 0
    frames_accepted: int = 0
    accepted_per_class_zone: Dict[Tuple[int, int], int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Severity → pathology class heuristic
# ---------------------------------------------------------------------------

def classify_pathology(
    severity_str: str,
    a_lines: str,
    b_lines: str,
    comment: str,
) -> Optional[int]:
    """
    Map a COVID-BLUES severity row to a MoCoLUS pathology class.

    Returns None if the row is too ambiguous to classify confidently.

    Heuristic priority order (most to least specific):
      1. Effusion / consolidation keywords in comments → class 5/4
      2. Pleural thickening keywords (only when not dominated by B-lines)
      3. Severity 3 + ANY → class 6 (ARDS / white lung)
      4. Severity 2 + B-lines → class 3 (diffuse B-lines)
      5. Severity 1 + B-lines → class 2 (focal B-lines)
      6. Severity 0 + A=yes + B=yes → class 2 (focal, mild B-lines on
         otherwise normal aerated lung)
      7. Severity 0 + A=no + B=no with "liver" in comment → class 0
         (normal — the probe is over the diaphragm showing liver)
      8. Severity 0/1/2 + A=yes + B=no → class 8 (pleural changes
         without B-lines, mild thickening / early consolidation)
      9. Severity 0 + A=yes + B=no → class 0 (normal A-profile)
     10. otherwise → None (skip)
    """
    comment_lower = (comment or "").lower()
    a = (a_lines or "").strip().lower()
    b = (b_lines or "").strip().lower()
    try:
        severity = float(severity_str) if severity_str else 0.0
    except ValueError:
        severity = 0.0

    # ── 1. Strong-signal keywords first ──
    if "effusion" in comment_lower or "fluid" in comment_lower:
        return 5  # PLEURAL_EFFUSION

    if "consolidat" in comment_lower:
        return 4  # CONSOLIDATION

    if "thickened" in comment_lower or "broken pleura" in comment_lower:
        # Pleural thickening — but only when no dominant B-line feature
        if "b-line" not in comment_lower or severity <= 1.0:
            return 8  # PLEURAL_THICKENING

    # ── 2. Severity 3 = severe → ARDS regardless of A/B ──
    # Comments at sev=3 typically describe "white lung" / "fused B-lines"
    # which is exactly the ARDS pattern.
    if severity >= 3.0:
        return 6  # ARDS_WHITE_LUNG

    # ── 3. B-line spectrum at moderate severity ──
    if severity >= 2.0 and b == "yes":
        return 3  # B_LINES_DIFFUSE

    if severity >= 1.0 and b == "yes":
        return 2  # B_LINES_FOCAL

    # ── 4. Mild B-lines on normal aerated lung (sev=0) ──
    # 35 cases at sev=0 + A=yes + B=yes — these are 1-2 B-lines on a
    # normal exam, which is class 2 (focal B-lines).
    if severity == 0.0 and a == "yes" and b == "yes":
        return 2  # B_LINES_FOCAL

    # ── 5. Diaphragm/liver view (sev=0 + A=no + B=no) ──
    # 14 cases where comments say "mostly liver in the video" — the probe
    # is over the costophrenic recess showing the liver beneath the
    # diaphragm. The lung itself is normal; the structural overlay handles
    # the diaphragm + liver anatomy. Class 0 is the right label.
    if severity == 0.0 and a == "no" and b == "no":
        if "liver" in comment_lower or "mostly" in comment_lower:
            return 0  # NORMAL_A_PROFILE
        # Without a confirming comment, this is too ambiguous to label
        return None

    # ── 6. Pleural-only abnormality (no B-lines at non-zero severity) ──
    # 19 cases at sev=1 + A=yes + B=no, 12 at sev=2 + A=yes + B=no.
    # The exam is abnormal but B-lines are absent — most likely pleural
    # thickening or very early subpleural consolidation. Class 8 is the
    # closest match in MoCoLUS pathology space.
    if severity >= 1.0 and b == "no" and a == "yes":
        return 8  # PLEURAL_THICKENING

    # ── 7. Normal A-profile fallback ──
    if severity == 0.0 and a == "yes" and b == "no":
        return 0  # NORMAL_A_PROFILE

    # Too ambiguous — better to skip than mislabel
    return None


# ---------------------------------------------------------------------------
# Video filename → zone parsing
# ---------------------------------------------------------------------------

def parse_blue_point(filename: str) -> Optional[str]:
    """
    Extract the BLUE point token (e.g. "L2") from a video filename like
    "patient_42_L2.mp4" or "patient_15_L2_2.mp4" (the "_2" suffix marks
    a second take of the same point).
    """
    stem = Path(filename).stem  # e.g. "patient_42_L2_2"
    parts = stem.split("_")
    # Walk parts looking for an L1/L2/L3/R1/R2/R3 token
    for p in parts:
        token = p.upper()
        if token in {"L1", "L2", "L3", "R1", "R2", "R3"}:
            return token
    return None


# ---------------------------------------------------------------------------
# Severity CSV loader
# ---------------------------------------------------------------------------

def load_severity_csv(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Read severity.csv and return a dict keyed by `video_file` (without
    the .mp4 extension), e.g. "patient_42_L2" → {row dict}.
    """
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("video_file", "").strip()
            if key:
                out[key] = row
    return out


# ---------------------------------------------------------------------------
# Per-video frame extraction + ingest
# ---------------------------------------------------------------------------

def ingest_video(
    video_path: Path,
    pathology_class: int,
    zone_region: int,
    output_dir: Path,
    temp_dir: Path,
    seen_hashes: Set[int],
    seen_sha: Set[str],
    stats: IngestStats,
    fps: int,
    output_size: int,
    crop_frac: float,
    pleural_max_frac: float,
    dedup_threshold: int,
    max_frames_per_video: int,
    labels_writer,
) -> None:
    """Extract → crop → dedup → filter → label rows for a single video."""
    frame_paths = extract_video_frames(video_path, temp_dir, fps=fps)
    stats.frames_extracted += len(frame_paths)

    accepted_this_video = 0
    for fp in frame_paths:
        if accepted_this_video >= max_frames_per_video:
            fp.unlink(missing_ok=True)
            continue

        arr = load_image_grayscale(fp)
        fp.unlink(missing_ok=True)
        if arr is None:
            continue

        cropped = center_crop_fraction(arr, frac=crop_frac)
        canvas = letterbox_resize(cropped, size=output_size)

        sha = hashlib.sha256(canvas.tobytes()).hexdigest()
        if sha in seen_sha:
            stats.frames_deduped += 1
            continue
        seen_sha.add(sha)

        h = dhash(canvas)
        is_dupe = False
        for prev in seen_hashes:
            if hamming_distance(h, prev) <= dedup_threshold:
                stats.frames_deduped += 1
                is_dupe = True
                break
        if is_dupe:
            continue
        seen_hashes.add(h)

        pleural_frac = _detect_pleural_row_fraction(canvas)
        if pleural_frac is None or pleural_frac > pleural_max_frac:
            stats.frames_rejected_pleural += 1
            continue

        candidate_name = f"diap_cand_blues_{sha[:12]}.png"
        candidate_path = output_dir / candidate_name
        Image.fromarray(canvas).save(candidate_path, optimize=True)

        # Emit a labeled row directly into the labels CSV — no manual
        # labeling pass required for COVID-BLUES.
        labels_writer.writerow([
            candidate_name,
            pathology_class,
            zone_region,
            "labeled",
        ])

        stats.frames_accepted += 1
        accepted_this_video += 1
        key = (pathology_class, zone_region)
        stats.accepted_per_class_zone[key] = stats.accepted_per_class_zone.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=ROOT / "data" / "_sources" / "covid_blues_meta" / "lus_videos",
        help="Directory containing patient_*_X.mp4 files.",
    )
    parser.add_argument(
        "--severity",
        type=Path,
        default=ROOT / "data" / "_sources" / "covid_blues_meta" / "severity.csv",
        help="Path to severity.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "diaphragm_candidates",
        help="Output directory for candidates + labels CSV.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frame extraction rate (Hz). Default 2 = ~one frame per breath.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=8,
        help="Cap on accepted frames per video to avoid one video dominating.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="If > 0, cap total videos processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-size", type=int, default=DEFAULT_OUTPUT_SIZE,
    )
    parser.add_argument(
        "--crop-frac", type=float, default=DEFAULT_CENTER_CROP_FRACTION,
    )
    parser.add_argument(
        "--pleural-max-frac", type=float, default=DEFAULT_PLEURAL_LINE_MAX_FRAC,
    )
    parser.add_argument(
        "--dedup-threshold", type=int, default=DEFAULT_DEDUP_HAMMING_THRESHOLD,
    )
    parser.add_argument(
        "--no-pleural-filter",
        action="store_true",
        help="Disable pleural-line filter (keep every cropped frame).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if not args.videos_dir.exists():
        print(f"ERROR: videos dir not found: {args.videos_dir}", file=sys.stderr)
        return 1
    if not args.severity.exists():
        print(f"ERROR: severity CSV not found: {args.severity}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    labels_csv_path = args.output / "diaphragm_labels.csv"

    print(f"Videos dir:    {args.videos_dir}")
    print(f"Severity CSV:  {args.severity}")
    print(f"Output dir:    {args.output}")
    print(f"Labels CSV:    {labels_csv_path}")
    print(f"FPS:           {args.fps}")
    print(f"Max per video: {args.max_frames_per_video}")
    print()

    severity_rows = load_severity_csv(args.severity)
    print(f"Loaded {len(severity_rows)} severity rows")

    all_videos = sorted(args.videos_dir.glob("patient_*.mp4"))
    print(f"Discovered {len(all_videos)} videos")
    if args.max_videos > 0:
        all_videos = all_videos[: args.max_videos]
        print(f"  (capped to first {len(all_videos)} for this run)")

    pleural_max = 1.0 if args.no_pleural_filter else args.pleural_max_frac

    stats = IngestStats()
    seen_sha: Set[str] = set()
    seen_hashes: Set[int] = set()

    temp_dir = args.output / "_temp_video_frames"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # Open the labels CSV in append mode if it exists, otherwise create
    # with header. We always write our diap_cand_blues_*.png filenames so
    # the merge script can dedup against any existing entries.
    labels_csv_existed = labels_csv_path.exists()
    labels_file = open(labels_csv_path, "a", newline="")
    labels_writer = csv.writer(labels_file)
    if not labels_csv_existed:
        labels_writer.writerow(["filename", "pathology_class", "zone_region", "status"])

    try:
        for i, video_path in enumerate(all_videos):
            stats.videos_scanned += 1

            # Parse zone from filename
            blue_point = parse_blue_point(video_path.name)
            if blue_point not in BLUE_POINT_TO_ZONE_REGION:
                stats.videos_skipped_zone += 1
                continue
            zone_region = BLUE_POINT_TO_ZONE_REGION[blue_point]

            # Look up severity row (key uses stem without "_2" if present —
            # severity.csv has one row per *unique* point per patient)
            severity_key = video_path.stem
            row = severity_rows.get(severity_key)
            if row is None:
                # Try stripping a trailing "_N" suffix (second take)
                if "_" in severity_key:
                    stripped = "_".join(severity_key.split("_")[:3])
                    row = severity_rows.get(stripped)
            if row is None:
                stats.videos_skipped_unsevered += 1
                continue

            # Classify pathology
            pathology_class = classify_pathology(
                severity_str=row.get("Severity Score", ""),
                a_lines=row.get("A-lines", ""),
                b_lines=row.get("B-lines", ""),
                comment=row.get("comments", ""),
            )
            if pathology_class is None:
                stats.videos_skipped_uncertain += 1
                continue

            if args.verbose:
                print(
                    f"  [{i+1}/{len(all_videos)}] {video_path.name} "
                    f"→ class={pathology_class} ({PATHOLOGY_CLASS_NAMES.get(pathology_class, '?')}), "
                    f"zone={zone_region} ({blue_point})"
                )
            elif (i + 1) % 25 == 0:
                print(f"  processed {i + 1}/{len(all_videos)} videos, "
                      f"accepted {stats.frames_accepted} frames")

            ingest_video(
                video_path=video_path,
                pathology_class=pathology_class,
                zone_region=zone_region,
                output_dir=args.output,
                temp_dir=temp_dir,
                seen_hashes=seen_hashes,
                seen_sha=seen_sha,
                stats=stats,
                fps=args.fps,
                output_size=args.output_size,
                crop_frac=args.crop_frac,
                pleural_max_frac=pleural_max,
                dedup_threshold=args.dedup_threshold,
                max_frames_per_video=args.max_frames_per_video,
                labels_writer=labels_writer,
            )
    finally:
        labels_file.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ─ Summary ─
    print()
    print("=" * 60)
    print("COVID-BLUES ingest summary")
    print("=" * 60)
    print(f"  Videos scanned:               {stats.videos_scanned}")
    print(f"  Videos skipped (upper zone):  {stats.videos_skipped_zone}")
    print(f"  Videos skipped (no severity): {stats.videos_skipped_unsevered}")
    print(f"  Videos skipped (uncertain):   {stats.videos_skipped_uncertain}")
    print(f"  Frames extracted:             {stats.frames_extracted}")
    print(f"  Frames deduped:               {stats.frames_deduped}")
    print(f"  Frames rejected (pleural):    {stats.frames_rejected_pleural}")
    print(f"  Frames accepted:              {stats.frames_accepted}")
    print()
    print(f"Coverage by (class, zone):")
    for (cls_id, zone_id), count in sorted(stats.accepted_per_class_zone.items()):
        cls_name = PATHOLOGY_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        zone_name = {1: "LOWER_BLUE_L", 2: "LOWER_BLUE_R",
                     3: "PLAPS_L", 4: "PLAPS_R",
                     5: "DIAPHRAGM_L", 6: "DIAPHRAGM_R"}.get(zone_id, f"zone_{zone_id}")
        print(f"  ({cls_name:>22}, {zone_name:>14}): {count}")

    print()
    if stats.frames_accepted > 0:
        print("Next step:")
        print(f"  python3 scripts/merge_diaphragm_labels.py \\")
        print(f"      --labels         {labels_csv_path} \\")
        print(f"      --candidates-dir {args.output} \\")
        print(f"      --source-dataset COVID-BLUES")
    else:
        print("No candidates accepted — nothing to merge.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
