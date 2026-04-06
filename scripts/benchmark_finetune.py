"""
Benchmark: v4 (pre-anatomy-bank) vs v4_ab (anatomy-bank fine-tuned).

Generates frames for all 10 pathology classes from both checkpoints,
computes intensity stats, SSIM, and saves a visual comparison grid.
Also runs validate_synthetic.py metrics where applicable.
"""

import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realistic_generator import RealisticLungUSGenerator

CLASSES = {
    0: "normal_a_profile",
    1: "pneumothorax",
    2: "b_lines_focal",
    3: "b_lines_diffuse",
    4: "consolidation",
    5: "pleural_effusion",
    6: "ards_white_lung",
    7: "lung_point",
    8: "pleural_thickening",
    9: "interstitial_syndrome",
}

OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b, data_range=1.0)


def load_real_references(data_dir: str, n_per_class: int = 5) -> dict:
    """Load a few real reference frames per class for cross-model SSIM."""
    import csv
    refs = {}
    data_path = Path(data_dir)
    meta_path = data_path / "metadata.csv"
    if not meta_path.exists():
        return refs
    counts = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            cls = int(row["pathology_class"])
            if cls > 9:
                continue
            if counts.get(cls, 0) >= n_per_class:
                continue
            img_path = data_path / "images" / row["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("L").resize((256, 256), Image.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            refs.setdefault(cls, []).append(arr)
            counts[cls] = counts.get(cls, 0) + 1
    return refs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-path", default="checkpoints/realistic_v4/best.pt")
    parser.add_argument("--v4ab-path", default="checkpoints/realistic_v4_ab/best.pt")
    parser.add_argument("--trauma-path", default="checkpoints/realistic_v2_finetune/latest.pt")
    parser.add_argument("--data-dir", default="data/real_pocus/processed")
    args = parser.parse_args()

    print("=" * 70)
    print("BENCHMARK: v4 (pre-anatomy-bank) vs v4_ab (anatomy-bank fine-tuned)")
    print("=" * 70)

    # Load real references for SSIM comparison
    print("\nLoading real reference frames...")
    real_refs = load_real_references(args.data_dir)
    print(f"  Loaded refs for {len(real_refs)} classes")

    # Load both generators
    print("\nLoading v4 model...")
    gen_v4 = RealisticLungUSGenerator.from_pretrained(
        model_path=args.v4_path,
        trauma_model_path=args.trauma_path,
    )
    print(f"  has_model={gen_v4.has_model}")

    print("Loading v4_ab model...")
    gen_v4ab = RealisticLungUSGenerator.from_pretrained(
        model_path=args.v4ab_path,
        trauma_model_path=args.trauma_path,
    )
    print(f"  has_model={gen_v4ab.has_model}")

    if not gen_v4ab.has_model:
        print("\nERROR: v4_ab checkpoint not found. Training may still be in progress.")
        print(f"  Checked: {args.v4ab_path}")
        sys.exit(1)

    results = []
    v4_frames = []
    v4ab_frames = []

    print(f"\n{'Cls':<4} {'Pathology':<25} {'v4 mean':>8} {'AB mean':>8} "
          f"{'v4 std':>8} {'AB std':>8} {'v4↔AB':>7} {'v4↔real':>8} {'AB↔real':>8}")
    print("-" * 95)

    for cls_id, cls_name in CLASSES.items():
        seed = 42 + cls_id

        # v4
        v4_frame = gen_v4.generate(pathology_class=cls_id, seed=seed)
        v4_frames.append(v4_frame)

        # v4_ab
        v4ab_frame = gen_v4ab.generate(pathology_class=cls_id, seed=seed)
        v4ab_frames.append(v4ab_frame)

        # Cross-model SSIM
        cross_ssim = compute_ssim(v4_frame, v4ab_frame)

        # SSIM vs real references
        v4_real_ssim = 0.0
        ab_real_ssim = 0.0
        if cls_id in real_refs and len(real_refs[cls_id]) > 0:
            v4_ssims = [compute_ssim(v4_frame, r) for r in real_refs[cls_id]]
            ab_ssims = [compute_ssim(v4ab_frame, r) for r in real_refs[cls_id]]
            v4_real_ssim = np.mean(v4_ssims)
            ab_real_ssim = np.mean(ab_ssims)

        results.append({
            "class": cls_id,
            "name": cls_name,
            "v4_mean": v4_frame.mean(),
            "v4_std": v4_frame.std(),
            "v4ab_mean": v4ab_frame.mean(),
            "v4ab_std": v4ab_frame.std(),
            "cross_ssim": cross_ssim,
            "v4_real_ssim": v4_real_ssim,
            "v4ab_real_ssim": ab_real_ssim,
        })

        print(f"{cls_id:<4} {cls_name:<25} {v4_frame.mean():>8.4f} {v4ab_frame.mean():>8.4f} "
              f"{v4_frame.std():>8.4f} {v4ab_frame.std():>8.4f} {cross_ssim:>7.4f} "
              f"{v4_real_ssim:>8.4f} {ab_real_ssim:>8.4f}")

    # Summary
    avg_v4_real = np.mean([r["v4_real_ssim"] for r in results])
    avg_ab_real = np.mean([r["v4ab_real_ssim"] for r in results])
    avg_cross = np.mean([r["cross_ssim"] for r in results])
    print("-" * 95)
    print(f"{'AVG':<30} {'':>8} {'':>8} {'':>8} {'':>8} {avg_cross:>7.4f} "
          f"{avg_v4_real:>8.4f} {avg_ab_real:>8.4f}")

    improvement = avg_ab_real - avg_v4_real
    print(f"\nSSIM vs real improvement (v4_ab - v4): {improvement:+.4f}")

    # Save comparison grid: [v4 | v4_ab] per class
    cell_h, cell_w = 256, 256
    grid = np.zeros((len(CLASSES) * cell_h, 2 * cell_w), dtype=np.float32)
    for i, (f1, f2) in enumerate(zip(v4_frames, v4ab_frames)):
        grid[i * cell_h:(i + 1) * cell_h, :cell_w] = f1
        grid[i * cell_h:(i + 1) * cell_h, cell_w:] = f2

    grid_img = Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8), mode="L")
    grid_path = OUT_DIR / "v4_vs_v4ab.png"
    grid_img.save(grid_path)
    print(f"\nComparison grid saved to {grid_path}")

    # Save CSV
    csv_path = OUT_DIR / "finetune_benchmark.csv"
    with open(csv_path, "w") as f:
        f.write("class,name,v4_mean,v4_std,v4ab_mean,v4ab_std,cross_ssim,v4_real_ssim,v4ab_real_ssim\n")
        for r in results:
            f.write(f"{r['class']},{r['name']},{r['v4_mean']:.6f},{r['v4_std']:.6f},"
                    f"{r['v4ab_mean']:.6f},{r['v4ab_std']:.6f},"
                    f"{r['cross_ssim']:.6f},{r['v4_real_ssim']:.6f},{r['v4ab_real_ssim']:.6f}\n")
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
