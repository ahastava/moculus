"""Generate one photorealistic ControlNet sample per pathology class (10 total)
for downstream AI image pipelines (e.g. SDXL/refiner stylisation).

Per-class tuning:
  - Default route: realistic_v4_ab/best.pt with class-specific CFG.
  - Trauma classes {1, 5, 7} normally route to realistic_v2_finetune.
    For class 5 (effusion) and class 7 (lung_point) the trauma model
    over-locks to the structural-guide rib shadows, so we bypass it and
    soften the guide before inference.
  - Seed sweep + std/rib-softness scoring picks the most textured candidate.
"""
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.realistic_generator import RealisticLungUSGenerator
from src.clinical_frames import ClinicalPathology

OUT = Path("thesis_figures")
CAND = OUT / "_ai_set_candidates"
CAND.mkdir(exist_ok=True, parents=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (class, output_name, seeds, cfg_override, bypass_trauma, soften_guide)
PLAN = [
    (0, "normal",             [7, 31, 41, 53, 71, 89, 113],          None, False, False),
    (1, "pneumothorax",       [13, 17, 29, 53, 67, 89, 113],         None, False, False),
    (2, "blines_focal",       [11, 19, 37, 47, 71, 97, 131],          4.5, False, False),
    (3, "blines_diffuse",     [21, 31, 47, 59, 83, 113, 137],        None, False, False),
    (4, "consolidation",      [29, 37, 53, 61, 79, 97, 131],          3.5, False, False),
    (5, "pleural_effusion",   [89, 113, 127, 137, 173, 191, 211],     3.0, True,  True),
    (6, "ards_white_lung",    [11, 23, 47, 71, 97, 127, 151],         5.0, False, False),
    (7, "lung_point",         [19, 37, 59, 89, 113, 137, 167],        4.0, True,  True),
    (8, "pleural_thickening", [13, 31, 53, 79, 109, 139, 163],        4.5, False, False),
    (9, "interstitial",       [17, 41, 67, 89, 113, 149, 181],        4.0, False, False),
]


def soften_rib_shadows(guide: np.ndarray, mix: float = 0.55) -> np.ndarray:
    blurred = gaussian_filter(guide, sigma=8.0)
    mask = (guide < 0.35).astype(np.float32)
    mask = gaussian_filter(mask, sigma=4.0)
    softened = guide * (1 - mix * mask) + blurred * (mix * mask)
    return np.clip(softened, 0.0, 1.0).astype(np.float32)


def generate_one(gen, cls, seed, soften):
    """Generate one frame, optionally with softened rib shadows."""
    if not soften:
        return gen.generate(pathology_class=cls, seed=seed, zone_region=0)
    # Manual path with softened guide
    guide = gen.guide_gen.generate(ClinicalPathology(cls), seed=seed, zone_region=0)
    guide = gen._enhance_guide_with_anatomy(guide, cls, seed=seed, zone_region=0)
    guide = soften_rib_shadows(guide, mix=0.55)

    guide_t = torch.from_numpy(guide * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    labels = torch.tensor([cls], dtype=torch.long, device=DEVICE)
    zone_labels = torch.tensor([0], dtype=torch.long, device=DEVICE)

    from src.train_realistic import sample_images
    torch.manual_seed(seed)
    with torch.inference_mode():
        result = sample_images(
            gen._select_model(cls, 0),
            gen.scheduler,
            guide_t,
            labels,
            num_inference_steps=50,
            guidance_scale=gen._get_guidance_scale(cls),
            device=DEVICE,
            zone_labels=zone_labels,
        )
    return ((result[0, 0].cpu().float() + 1.0) / 2.0).clamp(0, 1).numpy()


def score_frame(arr: np.ndarray) -> float:
    std = float(arr.std())
    mean = float(arr.mean())
    # Rib-band uniformity proxy (how much vertical-strip variance vs. horizontal)
    rib_l = arr[:, 24:48].mean()
    rib_c = arr[:, 110:140].mean()
    rib_r = arr[:, 208:232].mean()
    rib_softness = 1.0 - (max(abs(rib_l - rib_c), abs(rib_r - rib_c)) / max(rib_c, 0.01))
    if std < 0.07:
        return -1.0  # guide-locked
    return std + 0.25 * rib_softness - 0.3 * abs(mean - 0.50)


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
    saved_trauma = gen.trauma_model

    print("\nclass routing:")
    for cls, name, *_ in PLAN:
        m = gen._select_model(cls, 0)
        print(f"  {cls} ({name:>20s}) -> {m.__class__.__name__} cfg={gen._get_guidance_scale(cls):.1f}")

    chosen = []
    for cls, name, seeds, cfg, bypass_trauma, soften in PLAN:
        if cfg is not None:
            gen.CLASS_GUIDANCE_SCALE[cls] = cfg
        gen.trauma_model = None if bypass_trauma else saved_trauma
        active = gen._select_model(cls, 0).__class__.__name__
        print(f"\n=== class={cls} ({name})  cfg={gen._get_guidance_scale(cls)}  "
              f"trauma={'bypass' if bypass_trauma else 'on'}  soften={soften}  active={active} ===")

        best = None
        for seed in seeds:
            arr = np.clip(generate_one(gen, cls, seed, soften), 0, 1)
            s = score_frame(arr)
            std, mean = float(arr.std()), float(arr.mean())
            cand = CAND / f"{cls}_{name}_s{seed}_std{std:.3f}.png"
            Image.fromarray((arr * 255).astype(np.uint8)).save(cand)
            print(f"  s{seed:>3d}  mean={mean:.3f} std={std:.3f} score={s:.3f}")
            if best is None or s > best[0]:
                best = (s, arr, seed, std, mean)

        s, arr, seed, std, mean = best
        out = OUT / f"ai_{name}.png"
        Image.fromarray((arr * 255).astype(np.uint8)).save(out)
        print(f"  -> picked seed={seed} std={std:.3f} mean={mean:.3f}  ->  {out}")
        chosen.append((cls, name, seed, std, mean, str(out)))

    # Build a 2x5 contact sheet for quick eyeballing
    tile = 256
    gap = 6
    sheet = Image.new("L", (5 * tile + 6 * gap, 2 * tile + 3 * gap + 40), 0)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((gap, gap), "MoCoLUS ControlNet — 10-class AI set (one sample per class)", fill=255, font=font)
    for i, (cls, name, seed, std, mean, path) in enumerate(chosen):
        row, col = divmod(i, 5)
        x = gap + col * (tile + gap)
        y = 30 + gap + row * (tile + gap)
        im = Image.open(path).resize((tile, tile))
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + 4), f"{cls} {name}", fill=255, font=font)
        draw.text((x + 4, y + tile - 18), f"s{seed} σ={std:.2f}", fill=255, font=font)
    sheet_path = OUT / "ai_set_overview.png"
    sheet.save(sheet_path)
    print(f"\nwrote contact sheet -> {sheet_path}")

    print("\n--- final set ---")
    for cls, name, seed, std, mean, path in chosen:
        print(f"  {path}  (class {cls}, seed {seed}, σ={std:.3f}, μ={mean:.3f})")


if __name__ == "__main__":
    main()
