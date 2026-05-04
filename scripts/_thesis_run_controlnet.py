"""Run ControlNet DDPM directly to generate clean, photorealistic AI frames
for the 5 thesis comparison classes. Sweeps seeds + zone variants per class
and picks the realisation with the strongest texture (highest std)."""
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.realistic_generator import RealisticLungUSGenerator

OUT = Path("thesis_figures")
CAND = OUT / "_gen_candidates"
CAND.mkdir(exist_ok=True, parents=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Per-class sweep: (class, output_name, seeds_to_try, zone_regions_to_try, cfg_override)
# Lower CFG = more model freedom, less guide lock-in.
TARGETS = [
    (0, "normal",        [7, 11, 19, 23, 31, 41],  [0],    None),
    (1, "ptx",           [13, 17, 29, 43, 53, 67], [0],    None),
    (3, "blines",        [21, 31, 47, 59, 71, 83], [0],    None),
    # consolidation: try lower cfg + UPPER zone (avoid PLAPS overlay)
    (4, "consolidation", [29, 37, 53, 61, 79, 97], [0],    3.5),
    # effusion: route through BASE model, NOT trauma fine-tune (it locks to guide).
    # zone=0 also skips the PLAPS diaphragm overlay → less geometric.
    (5, "effusion",      [37, 41, 59, 73, 89, 101, 113, 127], [0], 3.0),
]


def main():
    print(f"loading models on {DEVICE}…")
    gen = RealisticLungUSGenerator.from_pretrained(
        model_path="checkpoints/realistic_v4_ab/best.pt",
        trauma_model_path="checkpoints/realistic_v2_finetune/latest.pt",
        anatomy_bank_path="checkpoints/anatomy_bank.pt",
        diaphragm_bank_path="checkpoints/anatomy_bank_diaphragm.pt",
        device=DEVICE,
        num_inference_steps=50,
    )
    print(f"has_model={gen.has_model}, trauma={gen.trauma_model is not None}, "
          f"diaphragm={gen.diaphragm_model is not None}")

    if not gen.has_model:
        raise SystemExit("Base model failed to load — aborting.")

    for cls, name, seeds, zones, cfg_override in TARGETS:
        if cfg_override is not None:
            gen.CLASS_GUIDANCE_SCALE[cls] = cfg_override
        # For effusion: bypass the trauma model so the base v4_ab handles class 5.
        # The trauma fine-tune over-conditions on the guide for non-PTX classes.
        if name == "effusion":
            saved_trauma = gen.trauma_model
            gen.trauma_model = None
        print(f"\n=== class={cls} ({name})  cfg={gen._get_guidance_scale(cls)}  zones={zones} ===")

        best = None
        for zr in zones:
            for seed in seeds:
                with torch.inference_mode():
                    frame = gen.generate(pathology_class=cls, seed=seed, zone_region=zr)
                arr = np.clip(frame, 0.0, 1.0)
                std = float(arr.std())
                mean = float(arr.mean())
                tag = f"{name}_z{zr}_s{seed}"
                cand_path = CAND / f"{tag}_std{std:.3f}.png"
                Image.fromarray((arr * 255).astype(np.uint8)).save(cand_path)
                # Score: prefer std close to "real" range (~0.16), also prefer mean ~0.5
                # Penalise very low std (guide-locked) heavily.
                if std < 0.08:
                    score = -1.0  # rejected
                else:
                    score = std - 0.3 * abs(mean - 0.50)
                print(f"  z{zr} s{seed:>3d} mean={mean:.3f} std={std:.3f} score={score:.3f} -> {cand_path.name}")
                if best is None or score > best[0]:
                    best = (score, arr, tag, mean, std)

        if best is None or best[0] < 0:
            print(f"  !! no acceptable candidate for {name}, keeping best regardless")
            # Re-pick by raw std
            best = max(
                ((float(np.clip(gen.generate(pathology_class=cls, seed=s, zone_region=z), 0, 1).std()),
                  np.clip(gen.generate(pathology_class=cls, seed=s, zone_region=z), 0, 1), f"{name}_z{z}_s{s}", 0, 0)
                 for s in seeds[:2] for z in zones),
                key=lambda x: x[0],
            )

        score, arr, tag, mean, std = best
        out_path = OUT / f"generated_{name}.png"
        Image.fromarray((arr * 255).astype(np.uint8)).save(out_path)
        print(f"  picked: {tag} (mean={mean:.3f} std={std:.3f}) -> {out_path}")

        if name == "effusion":
            gen.trauma_model = saved_trauma


if __name__ == "__main__":
    main()
