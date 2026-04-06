"""
Benchmark: Physics-only vs ControlNet DDPM frame generation.

Generates one frame per pathology class using both pipelines,
computes intensity stats and SSIM between them, and saves a
visual comparison grid.
"""

import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clinical_frames import ClinicalFrameGenerator, ClinicalPathology
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
    """Simplified SSIM for single-channel images."""
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b, data_range=1.0)


def main():
    print("=" * 60)
    print("BENCHMARK: Physics-Only vs ControlNet DDPM")
    print("=" * 60)

    # --- Physics-only generator ---
    physics_gen = ClinicalFrameGenerator(image_size=(256, 256))

    # --- ControlNet generator ---
    print("\nLoading ControlNet model...")
    t0 = time.time()
    realistic_gen = RealisticLungUSGenerator.from_pretrained(
        model_path="checkpoints/realistic_v4/best.pt",
        trauma_model_path="checkpoints/realistic_v2_finetune/latest.pt",
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")
    print(f"  has_model={realistic_gen.has_model}, "
          f"anatomy_bank={realistic_gen.anatomy_bank is not None}, "
          f"trauma={realistic_gen.trauma_model is not None}")

    results = []
    physics_frames = []
    controlnet_frames = []

    print(f"\n{'Class':<4} {'Pathology':<25} {'Phys mean':>10} {'CN mean':>10} "
          f"{'Phys std':>10} {'CN std':>10} {'SSIM':>8} {'CN time':>10}")
    print("-" * 90)

    for cls_id, cls_name in CLASSES.items():
        seed = 42 + cls_id

        # Physics-only
        physics_frame = physics_gen.generate(ClinicalPathology(cls_id), seed=seed)
        physics_frames.append(physics_frame)

        # ControlNet
        t0 = time.time()
        cn_frame = realistic_gen.generate(pathology_class=cls_id, seed=seed)
        cn_time = time.time() - t0
        controlnet_frames.append(cn_frame)

        # Metrics
        ssim = compute_ssim(physics_frame, cn_frame)
        p_mean, p_std = physics_frame.mean(), physics_frame.std()
        c_mean, c_std = cn_frame.mean(), cn_frame.std()

        results.append({
            "class": cls_id,
            "name": cls_name,
            "physics_mean": p_mean,
            "physics_std": p_std,
            "controlnet_mean": c_mean,
            "controlnet_std": c_std,
            "ssim": ssim,
            "controlnet_time_s": cn_time,
        })

        print(f"{cls_id:<4} {cls_name:<25} {p_mean:>10.4f} {c_mean:>10.4f} "
              f"{p_std:>10.4f} {c_std:>10.4f} {ssim:>8.4f} {cn_time:>9.1f}s")

    # Summary
    avg_ssim = np.mean([r["ssim"] for r in results])
    avg_time = np.mean([r["controlnet_time_s"] for r in results])
    print("-" * 90)
    print(f"{'AVG':<30} {'':>10} {'':>10} {'':>10} {'':>10} {avg_ssim:>8.4f} {avg_time:>9.1f}s")

    # Save comparison grid: [physics | controlnet] per class, 10 rows × 2 cols
    cell_h, cell_w = 256, 256
    grid = np.zeros((len(CLASSES) * cell_h, 2 * cell_w), dtype=np.float32)

    for i, (pf, cf) in enumerate(zip(physics_frames, controlnet_frames)):
        grid[i * cell_h:(i + 1) * cell_h, :cell_w] = pf
        grid[i * cell_h:(i + 1) * cell_h, cell_w:] = cf

    grid_img = Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8), mode="L")
    grid_path = OUT_DIR / "physics_vs_controlnet.png"
    grid_img.save(grid_path)
    print(f"\nComparison grid saved to {grid_path}")

    # Save CSV
    csv_path = OUT_DIR / "integration_benchmark.csv"
    with open(csv_path, "w") as f:
        f.write("class,name,physics_mean,physics_std,controlnet_mean,controlnet_std,ssim,controlnet_time_s\n")
        for r in results:
            f.write(f"{r['class']},{r['name']},{r['physics_mean']:.6f},{r['physics_std']:.6f},"
                    f"{r['controlnet_mean']:.6f},{r['controlnet_std']:.6f},"
                    f"{r['ssim']:.6f},{r['controlnet_time_s']:.2f}\n")
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
