"""Extract one representative AI-generated frame per pathology class from frame_cache.npz."""
import numpy as np
from PIL import Image
from pathlib import Path

OUT = Path("thesis_figures")

# (scenario, zone, output_name) — UPPER_BLUE_L is the canonical anterior view
TARGETS = [
    ("normal",                "UPPER_BLUE_L", "generated_normal"),
    ("left_pneumothorax",     "UPPER_BLUE_L", "generated_ptx"),
    ("pulmonary_edema",       "UPPER_BLUE_L", "generated_blines"),
    ("left_pneumonia",        "UPPER_BLUE_L", "generated_consolidation"),
    ("right_pleural_effusion","PLAPS_R",      "generated_effusion"),
]

cache = np.load("data/frame_cache.npz", allow_pickle=True)

for scenario, zone, name in TARGETS:
    key = f"{scenario}/{zone}/bmode"
    if key not in cache.files:
        print(f"MISSING {key}")
        continue
    stack = np.asarray(cache[key], dtype=np.float32)  # [N, 256, 256]
    # Take a mid-stack frame (avoid sliding extremes)
    frame = stack[stack.shape[0] // 2]
    arr = np.clip(frame, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    out_path = OUT / f"{name}.png"
    img.save(out_path)
    print(f"wrote {out_path} from {key} idx={stack.shape[0]//2}/{stack.shape[0]}")
