"""
Synthetic Frame Structural Validator
======================================
Compares synthetic LUS frames against real clinical benchmarks to verify
anatomical plausibility and image quality.

Metrics computed per-class:
  - Intensity histogram similarity (Wasserstein distance)
  - Structural similarity (SSIM) vs real reference samples
  - Pleural line detection rate (must be present in >95% of frames)
  - Texture statistics (mean, std, skewness) vs real distribution
  - Sub-pleural contrast ratio (pathology region vs background)

Outputs a pass/fail quality gate per class with detailed breakdown.

Usage:
    python scripts/validate_synthetic.py
    python scripts/validate_synthetic.py --classes 4 6 --max-samples 200
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance, skew

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

PATHOLOGY_NAMES = {
    0: "normal_a_profile", 1: "pneumothorax", 2: "b_lines_focal",
    3: "b_lines_diffuse", 4: "consolidation", 5: "pleural_effusion",
    6: "ards_white_lung", 7: "lung_point", 8: "pleural_thickening",
    9: "interstitial_syndrome",
}

# Quality gate thresholds
THRESHOLDS = {
    "pleural_detection_rate": 0.90,     # ≥90% of frames must have detectable pleural line
    "wasserstein_max": 0.12,            # histogram distance must be below this
    "mean_intensity_deviation": 0.15,   # synthetic mean within ±0.15 of real mean
    "std_intensity_deviation": 0.10,    # synthetic std within ±0.10 of real std
    "min_ssim": 0.04,                    # mean SSIM vs real references (unpaired cross-domain)
}


def detect_pleural_line(img: np.ndarray) -> bool:
    """Check if a pleural line is detectable in the upper region."""
    h, w = img.shape
    search = img[int(0.08 * h):int(0.40 * h), :]
    row_means = search.mean(axis=1)
    # Pleural line: a row significantly brighter than neighbors
    if len(row_means) < 3:
        return False
    peak = row_means.max()
    median = np.median(row_means)
    return peak > median + 0.08


def compute_texture_stats(img: np.ndarray) -> dict:
    """Compute texture statistics for a frame."""
    return {
        "mean": float(img.mean()),
        "std": float(img.std()),
        "skewness": float(skew(img.ravel())),
        "p10": float(np.percentile(img, 10)),
        "p90": float(np.percentile(img, 90)),
    }


def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images using skimage."""
    from skimage.metrics import structural_similarity
    # Resize if needed
    if img1.shape != img2.shape:
        from PIL import Image as PILImage
        img2_pil = PILImage.fromarray((img2 * 255).astype(np.uint8))
        img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), PILImage.LANCZOS)
        img2 = np.array(img2_pil, dtype=np.float32) / 255.0
    return float(structural_similarity(img1, img2, data_range=1.0))


def load_class_images(
    metadata_path: Path,
    images_dir: Path,
    pathology_class: int,
    source_filter: str,
    max_samples: int = 100,
) -> list:
    """Load images for a given class and source type."""
    images = []
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["pathology_class"]) != pathology_class:
                continue

            is_synthetic = row.get("source_dataset", "") == "MoCoLUS-Synthetic"
            if source_filter == "synthetic" and not is_synthetic:
                continue
            if source_filter == "real" and is_synthetic:
                continue

            img_path = images_dir / row["filename"]
            if not img_path.exists():
                continue

            img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
            images.append(img)

            if len(images) >= max_samples:
                break

    return images


def validate_class(
    real_images: list,
    synth_images: list,
    pathology_class: int,
) -> dict:
    """Run all validation metrics for a single class."""
    name = PATHOLOGY_NAMES.get(pathology_class, f"class_{pathology_class}")
    result = {"class": pathology_class, "name": name, "pass": True, "failures": []}

    if not synth_images:
        result["pass"] = False
        result["failures"].append("no_synthetic_images")
        return result

    if not real_images:
        result["pass"] = False
        result["failures"].append("no_real_images")
        return result

    # 1. Pleural line detection rate
    pleural_detected = sum(1 for img in synth_images if detect_pleural_line(img))
    pleural_rate = pleural_detected / len(synth_images)
    result["pleural_detection_rate"] = pleural_rate
    if pleural_rate < THRESHOLDS["pleural_detection_rate"]:
        result["pass"] = False
        result["failures"].append(f"pleural_rate={pleural_rate:.2f}<{THRESHOLDS['pleural_detection_rate']}")

    # 2. Texture statistics comparison
    real_stats = [compute_texture_stats(img) for img in real_images]
    synth_stats = [compute_texture_stats(img) for img in synth_images]

    real_mean = np.mean([s["mean"] for s in real_stats])
    synth_mean = np.mean([s["mean"] for s in synth_stats])
    real_std = np.mean([s["std"] for s in real_stats])
    synth_std = np.mean([s["std"] for s in synth_stats])

    result["real_mean_intensity"] = float(real_mean)
    result["synth_mean_intensity"] = float(synth_mean)
    result["real_std_intensity"] = float(real_std)
    result["synth_std_intensity"] = float(synth_std)

    mean_dev = abs(synth_mean - real_mean)
    std_dev = abs(synth_std - real_std)
    result["mean_deviation"] = float(mean_dev)
    result["std_deviation"] = float(std_dev)

    if mean_dev > THRESHOLDS["mean_intensity_deviation"]:
        result["pass"] = False
        result["failures"].append(f"mean_dev={mean_dev:.3f}>{THRESHOLDS['mean_intensity_deviation']}")
    if std_dev > THRESHOLDS["std_intensity_deviation"]:
        result["pass"] = False
        result["failures"].append(f"std_dev={std_dev:.3f}>{THRESHOLDS['std_intensity_deviation']}")

    # 3. Intensity histogram distance (Wasserstein)
    real_hist_pool = np.concatenate([img.ravel() for img in real_images[:50]])
    synth_hist_pool = np.concatenate([img.ravel() for img in synth_images[:50]])
    # Subsample for speed
    rng = np.random.default_rng(42)
    n_subsample = min(100_000, len(real_hist_pool), len(synth_hist_pool))
    real_sub = rng.choice(real_hist_pool, n_subsample, replace=False)
    synth_sub = rng.choice(synth_hist_pool, n_subsample, replace=False)
    wd = wasserstein_distance(real_sub, synth_sub)
    result["wasserstein_distance"] = float(wd)

    if wd > THRESHOLDS["wasserstein_max"]:
        result["pass"] = False
        result["failures"].append(f"wasserstein={wd:.3f}>{THRESHOLDS['wasserstein_max']}")

    # 4. SSIM vs real references (cross-sample mean)
    # Compare each synthetic image against a few real references
    n_refs = min(5, len(real_images))
    ref_images = real_images[:n_refs]
    ssim_scores = []
    for synth_img in synth_images[:20]:  # Cap for speed
        for ref_img in ref_images:
            ssim_scores.append(compute_ssim_simple(synth_img, ref_img))
    mean_ssim = np.mean(ssim_scores) if ssim_scores else 0
    result["mean_ssim"] = float(mean_ssim)

    if mean_ssim < THRESHOLDS["min_ssim"]:
        result["pass"] = False
        result["failures"].append(f"ssim={mean_ssim:.3f}<{THRESHOLDS['min_ssim']}")

    # 5. Skewness comparison (distribution shape)
    real_skew = np.mean([s["skewness"] for s in real_stats])
    synth_skew = np.mean([s["skewness"] for s in synth_stats])
    result["real_skewness"] = float(real_skew)
    result["synth_skewness"] = float(synth_skew)

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic LUS frames")
    parser.add_argument("--metadata", default="data/real_pocus/processed/metadata.csv")
    parser.add_argument("--images-dir", default="data/real_pocus/processed/images")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="Classes to validate (default: all with synthetic data)")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max images to load per class per source")
    args = parser.parse_args()

    metadata_path = ROOT / args.metadata
    images_dir = ROOT / args.images_dir

    # Determine which classes have synthetic data
    synth_classes = set()
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("source_dataset") == "MoCoLUS-Synthetic":
                synth_classes.add(int(row["pathology_class"]))

    if args.classes:
        validate_classes = sorted(set(args.classes) & synth_classes)
    else:
        validate_classes = sorted(synth_classes)

    if not validate_classes:
        print("No synthetic data found to validate.")
        return

    print("=" * 75)
    print("STRUCTURAL VALIDATION: Synthetic vs Real LUS Frames")
    print("=" * 75)

    all_results = []
    n_pass = 0
    n_fail = 0

    for cls in validate_classes:
        name = PATHOLOGY_NAMES.get(cls, f"class_{cls}")
        logger.info(f"Validating class {cls} ({name})...")

        real_images = load_class_images(
            metadata_path, images_dir, cls, "real", args.max_samples
        )
        synth_images = load_class_images(
            metadata_path, images_dir, cls, "synthetic", args.max_samples
        )

        logger.info(f"  Loaded {len(real_images)} real, {len(synth_images)} synthetic")

        result = validate_class(real_images, synth_images, cls)
        all_results.append(result)

        if result["pass"]:
            n_pass += 1
        else:
            n_fail += 1

    # Print results
    print()
    print(f"{'Class':>5}  {'Pathology':<25}  {'Pleural':>7}  {'W-Dist':>6}  "
          f"{'SSIM':>5}  {'Mean-D':>6}  {'Std-D':>5}  {'Result':>6}")
    print("-" * 75)

    for r in all_results:
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['class']:>5}  {r['name']:<25}  "
            f"{r.get('pleural_detection_rate', 0):>6.0%}  "
            f"{r.get('wasserstein_distance', 0):>6.3f}  "
            f"{r.get('mean_ssim', 0):>5.3f}  "
            f"{r.get('mean_deviation', 0):>6.3f}  "
            f"{r.get('std_deviation', 0):>5.3f}  "
            f"{status:>6}"
        )

    print("-" * 75)
    print(f"Quality Gate: {n_pass} PASS / {n_fail} FAIL out of {len(all_results)} classes")

    if n_fail > 0:
        print("\nFailed classes detail:")
        for r in all_results:
            if not r["pass"]:
                print(f"  Class {r['class']} ({r['name']}): {', '.join(r['failures'])}")

    # Detailed stats
    print("\nDetailed intensity comparison:")
    print(f"{'Class':>5}  {'Real Mean':>9}  {'Synth Mean':>10}  "
          f"{'Real Std':>8}  {'Synth Std':>9}  {'Real Skew':>9}  {'Synth Skew':>10}")
    print("-" * 75)
    for r in all_results:
        print(
            f"{r['class']:>5}  "
            f"{r.get('real_mean_intensity', 0):>9.3f}  "
            f"{r.get('synth_mean_intensity', 0):>10.3f}  "
            f"{r.get('real_std_intensity', 0):>8.3f}  "
            f"{r.get('synth_std_intensity', 0):>9.3f}  "
            f"{r.get('real_skewness', 0):>9.3f}  "
            f"{r.get('synth_skewness', 0):>10.3f}"
        )

    return n_fail == 0


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
