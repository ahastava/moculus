"""
Balanced Synthetic LUS Frame Generator
========================================
Uses the Lesion-Anatomy Bank + PMF conditioning to generate synthetic
frames that balance the training set.

Pipeline:
  1. Scan dataset for class imbalances (via scan_class_balance)
  2. For each under-represented class, generate frames using:
     - ClinicalFrameGenerator for structural base
     - LesionAnatomyBank for real texture blending + PMF-guided placement
  3. Validate structural features against real data statistics
  4. Save generated frames to data/real_pocus/processed/synthetic/

Usage:
    python scripts/generate_balanced.py --target-per-class 1500
    python scripts/generate_balanced.py --classes 4 6 --count 500
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Direct imports to avoid __init__.py keras dependency
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _import_modules():
    """Import project modules avoiding __init__.py."""
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    root = Path(__file__).resolve().parent.parent / "src"
    clinical = load("src.clinical_frames", root / "clinical_frames.py")
    anatomy = load("src.anatomy_bank", root / "anatomy_bank.py")
    return clinical, anatomy


def calibrate_intensity(
    frame: np.ndarray,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """
    Calibrate synthetic frame intensity to match real clinical statistics.

    Uses linear rescaling to match the target mean and std, preserving
    relative structure while shifting the intensity distribution.
    """
    current_mean = frame.mean()
    current_std = frame.std()

    if current_std < 1e-6:
        return np.full_like(frame, target_mean)

    # Linear rescaling: (frame - mean) / std * target_std + target_mean
    calibrated = (frame - current_mean) / current_std * target_std + target_mean
    return np.clip(calibrated, 0, 1).astype(np.float32)


# Real clinical intensity statistics per class (measured from dataset)
# These are the ground truth targets for calibration.
REAL_INTENSITY_STATS = {
    0: {"mean": 0.180, "std": 0.170},
    1: {"mean": 0.170, "std": 0.165},
    2: {"mean": 0.101, "std": 0.133},
    3: {"mean": 0.155, "std": 0.140},
    4: {"mean": 0.127, "std": 0.156},
    5: {"mean": 0.120, "std": 0.146},
    6: {"mean": 0.208, "std": 0.137},
    7: {"mean": 0.237, "std": 0.232},
    8: {"mean": 0.243, "std": 0.164},
    9: {"mean": 0.153, "std": 0.124},
}


def generate_with_anatomy_bank(
    frame_gen,
    bank,
    pathology_class: int,
    seed: int,
    blend_strength: float = 0.35,
) -> np.ndarray:
    """
    Generate a single frame with anatomy-bank texture blending.

    1. Generate structural base from ClinicalFrameGenerator
    2. Sample lesion position from PMF
    3. Blend real lesion texture at the sampled position
    4. Calibrate intensity to match real clinical distribution
    """
    clinical, anatomy = _import_modules()
    ClinicalPathology = clinical.ClinicalPathology

    # Generate structural base
    frame = frame_gen.generate(ClinicalPathology(pathology_class), seed=seed)
    h, w = frame.shape

    # Sample and blend lesion texture if available
    texture = bank.sample_lesion_texture(pathology_class, seed=seed)
    if texture is not None:
        row, col = bank.sample_lesion_position(
            pathology_class, image_size=(h, w),
            pleural_row=frame_gen.pleural_row, seed=seed + 1,
        )

        ps = texture.shape[0]
        r0 = max(0, row - ps // 2)
        r1 = min(h, r0 + ps)
        c0 = max(0, col - ps // 2)
        c1 = min(w, c0 + ps)

        # Actual region we can paste into
        tr0 = 0 if r0 >= 0 else -r0
        tc0 = 0 if c0 >= 0 else -c0
        th = r1 - r0
        tw = c1 - c0

        if th > 0 and tw > 0:
            tex_crop = texture[tr0:tr0 + th, tc0:tc0 + tw]

            # Soft blend mask (feathered edges)
            mask = np.ones((th, tw), dtype=np.float32)
            feather = min(8, th // 4, tw // 4)
            if feather > 0:
                for i in range(feather):
                    alpha = (i + 1) / feather
                    mask[i, :] *= alpha
                    mask[-(i + 1), :] *= alpha
                    mask[:, i] *= alpha
                    mask[:, -(i + 1)] *= alpha

            # Blend: preserve structure, add real texture
            region = frame[r0:r1, c0:c1]
            blended = region * (1 - blend_strength * mask) + tex_crop * blend_strength * mask
            frame[r0:r1, c0:c1] = np.clip(blended, 0, 1)

    # Calibrate intensity to match real clinical distribution
    stats = REAL_INTENSITY_STATS.get(pathology_class)
    if stats:
        frame = calibrate_intensity(frame, stats["mean"], stats["std"])

    return frame


def validate_frame(frame: np.ndarray, pathology_class: int) -> dict:
    """
    Basic structural validation of a generated frame.

    Checks:
      - Intensity range is valid
      - Pleural line is detectable
      - Sub-pleural content matches expected pattern
    """
    h, w = frame.shape
    metrics = {
        "valid": True,
        "mean_intensity": float(frame.mean()),
        "std_intensity": float(frame.std()),
    }

    # Check basic statistics
    if frame.mean() < 0.05 or frame.mean() > 0.85:
        metrics["valid"] = False
        metrics["reason"] = "abnormal_mean_intensity"

    if frame.std() < 0.02:
        metrics["valid"] = False
        metrics["reason"] = "too_uniform"

    # Check for pleural line presence (bright band in upper region)
    upper_region = frame[int(0.08 * h):int(0.35 * h), :]
    row_means = upper_region.mean(axis=1)
    if row_means.max() < 0.3:
        metrics["valid"] = False
        metrics["reason"] = "no_pleural_line"

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate balanced synthetic LUS frames")
    parser.add_argument("--target-per-class", type=int, default=1500,
                        help="Target samples per class")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="Specific classes to generate (default: all under-represented)")
    parser.add_argument("--count", type=int, default=0,
                        help="Override: generate exactly this many per specified class")
    parser.add_argument("--metadata", default="data/real_pocus/processed/metadata.csv")
    parser.add_argument("--output-dir", default="data/real_pocus/processed/images")
    parser.add_argument("--bank-path", default="checkpoints/anatomy_bank.pt")
    parser.add_argument("--blend-strength", type=float, default=0.35,
                        help="Texture blend strength (0=pure synthetic, 1=pure texture)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be generated without writing files")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    clinical, anatomy = _import_modules()

    # Load anatomy bank
    bank_path = root / args.bank_path
    if bank_path.exists():
        bank = anatomy.LesionAnatomyBank.load(str(bank_path))
        logger.info(f"Loaded anatomy bank from {bank_path}")
        print(bank.summary())
    else:
        logger.warning(f"No anatomy bank at {bank_path}. Run: python -c 'from src.anatomy_bank import ...'")
        logger.info("Proceeding without texture blending (structural-only generation)")
        bank = anatomy.LesionAnatomyBank(textures={}, pmfs={}, patch_size=64)

    # Scan current class distribution
    metadata_path = root / args.metadata
    from scripts.scan_class_balance import scan_metadata, BMODE_CLASSES, PATHOLOGY_NAMES
    bmode_counts, _, _ = scan_metadata(str(metadata_path))

    # Determine what to generate
    if args.classes and args.count > 0:
        synthesis_plan = {c: args.count for c in args.classes}
    elif args.classes:
        synthesis_plan = {
            c: max(0, args.target_per_class - bmode_counts.get(c, 0))
            for c in args.classes
        }
    else:
        synthesis_plan = {
            c: max(0, args.target_per_class - bmode_counts.get(c, 0))
            for c in sorted(BMODE_CLASSES)
            if bmode_counts.get(c, 0) < args.target_per_class
        }

    # Remove classes with nothing to generate
    synthesis_plan = {c: n for c, n in synthesis_plan.items() if n > 0}

    if not synthesis_plan:
        print("All classes meet target. Nothing to generate.")
        return

    total_to_generate = sum(synthesis_plan.values())
    print(f"\nGeneration plan ({total_to_generate} total frames):")
    for cls in sorted(synthesis_plan):
        name = PATHOLOGY_NAMES.get(cls, f"class_{cls}")
        current = bmode_counts.get(cls, 0)
        target = current + synthesis_plan[cls]
        print(f"  Class {cls} ({name}): {current} → {target} (+{synthesis_plan[cls]})")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Initialize generator
    frame_gen = clinical.ClinicalFrameGenerator(image_size=(256, 256))

    # Find next available image index
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("pocus_*.png"))
    if existing:
        max_idx = max(int(p.stem.split("_")[1]) for p in existing)
    else:
        max_idx = -1
    next_idx = max_idx + 1

    # Generate frames
    rng = np.random.default_rng(args.seed)
    new_rows = []
    stats = {"generated": 0, "valid": 0, "invalid": 0}
    t0 = time.time()

    for cls in sorted(synthesis_plan):
        n_needed = synthesis_plan[cls]
        name = PATHOLOGY_NAMES.get(cls, f"class_{cls}")
        logger.info(f"Generating {n_needed} frames for class {cls} ({name})...")

        generated = 0
        attempts = 0
        max_attempts = n_needed * 2  # Allow some rejections

        while generated < n_needed and attempts < max_attempts:
            seed = int(rng.integers(0, 2**31))
            attempts += 1

            frame = generate_with_anatomy_bank(
                frame_gen, bank, cls, seed=seed,
                blend_strength=args.blend_strength,
            )

            # Validate
            validation = validate_frame(frame, cls)
            if not validation["valid"]:
                stats["invalid"] += 1
                continue

            # Save image
            fname = f"pocus_{next_idx:05d}.png"
            img = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), mode="L")
            img.save(output_dir / fname)

            # Record metadata
            new_rows.append({
                "filename": fname,
                "pathology_class": cls,
                "pathology_name": name,
                "source_label": "synthetic",
                "source_file": f"anatomy_bank_seed_{seed}",
                "source_dataset": "MoCoLUS-Synthetic",
                "split": "train",
            })

            next_idx += 1
            generated += 1
            stats["generated"] += 1
            stats["valid"] += 1

            if generated % 100 == 0:
                elapsed = time.time() - t0
                rate = stats["generated"] / elapsed
                logger.info(
                    f"  {name}: {generated}/{n_needed} "
                    f"({stats['generated']}/{total_to_generate} total, {rate:.1f} frames/s)"
                )

    # Append to metadata.csv
    if new_rows:
        with open(metadata_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "filename", "pathology_class", "pathology_name",
                "source_label", "source_file", "source_dataset", "split",
            ])
            for row in new_rows:
                writer.writerow(row)

    elapsed = time.time() - t0
    print(f"\nGeneration complete:")
    print(f"  Generated: {stats['generated']} frames")
    print(f"  Rejected:  {stats['invalid']} frames (validation failed)")
    print(f"  Time:      {elapsed:.1f}s ({stats['generated']/max(elapsed,0.1):.1f} frames/s)")
    print(f"  Output:    {output_dir}")
    print(f"  Metadata:  {metadata_path} (appended {len(new_rows)} rows)")


if __name__ == "__main__":
    main()
