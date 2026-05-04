"""Score real-source POCUS frames per class and emit contact sheets so we can
pick visually clean exemplars for the thesis."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("data/real_pocus/processed")
META = ROOT / "metadata.csv"
IMGDIR = ROOT / "images"
OUT = Path("thesis_figures/_candidates")
OUT.mkdir(parents=True, exist_ok=True)

REAL_SOURCES = {
    "COVID-BLUES",
    "POCOVID-Net",
    "Kaggle-LUS-BALD",
    "COVIDx-US/Litfl",
    "COVIDx-US/UF",
    "COVIDx-US/Paper",
    "COVIDx-US/CoreUltrasound",
    "COVIDx-US/Radiopaedia",
}

TARGETS = {
    0: ("normal",        ["a-line", "a_line", "normal", "healthy"]),
    1: ("ptx",           ["pneumo", "ptx", "stratosphere", "barcode"]),
    3: ("blines",        ["b-line", "b_line", "bline", "edema", "comet"]),
    4: ("consolidation", ["consolid", "shred", "hepati", "bronchogram"]),
    5: ("effusion",      ["effusion", "anechoic", "fluid"]),
}


def score_image(path: Path) -> float:
    """Quick visual-quality heuristic: prefer good dynamic range AND a
    reasonable pleural line near top AND not noise-dominated."""
    try:
        im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    except Exception:
        return -1.0
    if im.size == 0 or im.std() < 5:
        return -1.0
    H, W = im.shape
    mean = im.mean()
    std = im.std()
    # Pleural band: look for a bright horizontal stripe in upper 30%
    upper = im[: int(H * 0.30), :]
    row_max = upper.max(axis=1)
    pleural_strength = float(row_max.max() - upper.mean())
    # Penalise images that are near-black or near-white overall
    brightness_pen = abs(mean - 95.0) / 95.0  # 95 ≈ typical POCUS midtone
    # Edge density via simple gradient
    gy = np.abs(np.diff(im, axis=0)).mean()
    gx = np.abs(np.diff(im, axis=1)).mean()
    edges = (gy + gx) / 2
    score = 0.4 * std + 0.5 * pleural_strength + 0.3 * edges - 30.0 * brightness_pen
    return float(score)


def filename_bonus(source_file: str, keywords) -> float:
    s = source_file.lower()
    return sum(3.0 for k in keywords if k in s)


def main():
    rows_by_class: dict[int, list[tuple[str, str, str]]] = {c: [] for c in TARGETS}
    with META.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                c = int(r["pathology_class"])
            except ValueError:
                continue
            if c not in TARGETS:
                continue
            if r["source_dataset"] not in REAL_SOURCES:
                continue
            rows_by_class[c].append((r["filename"], r["source_dataset"], r["source_file"]))

    chosen: dict[str, str] = {}
    for c, (name, kws) in TARGETS.items():
        cands = rows_by_class[c]
        # Sample at most 400 to keep this fast
        rng = np.random.default_rng(seed=c)
        if len(cands) > 400:
            idx = rng.choice(len(cands), 400, replace=False)
            cands = [cands[i] for i in idx]
        scored = []
        for fname, ds, sf in cands:
            p = IMGDIR / fname
            s = score_image(p)
            if s < 0:
                continue
            s += filename_bonus(sf, kws)
            scored.append((s, fname, ds, sf))
        scored.sort(reverse=True)
        top = scored[:9]
        # Build a 3×3 contact sheet
        tile = 256
        gap = 4
        canvas = Image.new("L", (tile * 3 + gap * 4, tile * 3 + gap * 4 + 60), 0)
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
            )
        except Exception:
            font = ImageFont.load_default()
        draw.text((gap, gap), f"Class {c} — {name}", fill=255, font=font)
        for i, (s, fname, ds, sf) in enumerate(top):
            r, col = divmod(i, 3)
            x = gap + col * (tile + gap)
            y = 30 + gap + r * (tile + gap)
            img = Image.open(IMGDIR / fname).convert("L").resize((tile, tile))
            canvas.paste(img, (x, y))
            draw.text((x + 2, y + 2), f"{i}: {fname}", fill=255, font=font)
            draw.text((x + 2, y + tile - 16), f"{s:.1f}", fill=255, font=font)
        sheet_path = OUT / f"candidates_{name}.png"
        canvas.save(sheet_path)
        # Auto-pick #0 for now
        if top:
            chosen[name] = top[0][1]
            print(f"[{name}] auto-pick = {top[0][1]} (score {top[0][0]:.1f}) src={top[0][3]}")
            print(f"          contact sheet: {sheet_path}")

    print("\n--- chosen ---")
    for name, fname in chosen.items():
        print(f"  real_{name}.png  <-  data/real_pocus/images/{fname}")


if __name__ == "__main__":
    main()
