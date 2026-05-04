"""Generate a more photorealistic effusion frame by softening the rib-shadow
structure in the guide before passing through ControlNet.

The structural guide for effusion bakes hard rectangular rib shadows that the
ControlNet preserves verbatim — clinically real effusion frames have softer,
convex-probe-averaged shadows. We Gaussian-blur the dark vertical strips so the
model is free to render natural speckle there."""
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
CAND = OUT / "_gen_candidates"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def soften_rib_shadows(guide: np.ndarray, mix: float = 0.55) -> np.ndarray:
    """Blend hard guide with a heavily-blurred copy in the dark-shadow regions.
    Pleural line and bright structures stay sharp; only the dark valleys get
    smeared so the model has more freedom to fill them with realistic speckle."""
    blurred = gaussian_filter(guide, sigma=8.0)
    mask = (guide < 0.35).astype(np.float32)
    mask = gaussian_filter(mask, sigma=4.0)
    softened = guide * (1 - mix * mask) + blurred * (mix * mask)
    return np.clip(softened, 0.0, 1.0).astype(np.float32)


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

    # Bypass trauma fine-tune — it locks too hard onto guide rib shadows.
    gen.trauma_model = None

    cls = int(ClinicalPathology.PLEURAL_EFFUSION)
    seeds = [89, 113, 127, 137, 151, 173, 191, 211]

    best = None
    for seed in seeds:
        # Generate guide and apply anatomy bank textures
        guide = gen.guide_gen.generate(ClinicalPathology(cls), seed=seed, zone_region=0)
        guide = gen._enhance_guide_with_anatomy(guide, cls, seed=seed, zone_region=0)
        # Soften rib shadows
        guide = soften_rib_shadows(guide, mix=0.55)

        guide_t = torch.from_numpy(guide * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
        labels = torch.tensor([cls], dtype=torch.long, device=DEVICE)
        zone_labels = torch.tensor([0], dtype=torch.long, device=DEVICE)

        from src.train_realistic import sample_images
        torch.manual_seed(seed)
        with torch.inference_mode():
            result = sample_images(
                gen.model,
                gen.scheduler,
                guide_t,
                labels,
                num_inference_steps=50,
                guidance_scale=3.0,  # soft enough to let model deviate
                device=DEVICE,
                zone_labels=zone_labels,
            )
        frame = ((result[0, 0].cpu().float() + 1.0) / 2.0).clamp(0, 1).numpy()
        std, mean = float(frame.std()), float(frame.mean())
        # Reject if guide-locked (very low std) or saturated
        rib_band_l = frame[:, 24:48].mean()
        rib_band_c = frame[:, 110:140].mean()
        # rib bands should be similar to centre brightness if shadows are softened
        rib_softness = 1.0 - abs(rib_band_l - rib_band_c) / max(rib_band_c, 0.01)
        score = std + 0.4 * rib_softness - 0.3 * abs(mean - 0.5)
        path = CAND / f"effusion_soft_s{seed}_std{std:.3f}_rs{rib_softness:.2f}.png"
        Image.fromarray((frame * 255).astype(np.uint8)).save(path)
        print(f"  s{seed:>3d} mean={mean:.3f} std={std:.3f} rib_softness={rib_softness:.2f} score={score:.3f}")
        if best is None or score > best[0]:
            best = (score, frame, seed)

    score, frame, seed = best
    out = OUT / "generated_effusion.png"
    Image.fromarray((frame * 255).astype(np.uint8)).save(out)
    print(f"\npicked seed={seed} score={score:.3f} -> {out}")


if __name__ == "__main__":
    main()
