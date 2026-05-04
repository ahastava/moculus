"""Generate one structural guide PNG per pathology class for thesis Fig 3.6."""
import numpy as np
from PIL import Image
from pathlib import Path

from src.clinical_frames import ClinicalFrameGenerator, ClinicalPathology

OUT = Path("thesis_figures")
OUT.mkdir(exist_ok=True)

NAMES = {
    0: "normal",
    1: "pneumothorax",
    2: "blines_focal",
    3: "blines_diffuse",
    4: "consolidation",
    5: "effusion",
    6: "ards",
    7: "lung_point",
    8: "pleural_thickening",
    9: "interstitial",
}

gen = ClinicalFrameGenerator()
for cls, name in NAMES.items():
    frame = gen.generate(ClinicalPathology(cls), seed=42 + cls)
    arr = np.clip(frame, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    out_path = OUT / f"structural_guide_{name}.png"
    img.save(out_path)
    print(f"wrote {out_path} shape={arr.shape} min={arr.min():.3f} max={arr.max():.3f}")
