"""Build a clinical-review PDF packet for the thesis training data.

Layout:
  - Cover page: project + dataset summary
  - One page per pathology class (10): clinical definition, 5x5 grid of top-25
    real-source training frames, AI-generated cross-validation row, sign-off
  - Final page: per-class sign-off + reviewer comments

Scoring favours frames with: good dynamic range, a visible pleural band in the
upper third, and clinically-relevant filenames (effusion, consolid, etc.).
"""
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = Path("data/real_pocus/processed")
META = ROOT / "metadata.csv"
IMGDIR = ROOT / "images"
OUT_DIR = Path("thesis_figures")
OUT_PDF = OUT_DIR / "clinical_review_packet.pdf"

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

# (class, short, ai_filename_stem, clinical_definition, blue_protocol_note)
CLASSES = [
    (0, "Normal A-profile", "ai_normal",
     "Bilateral A-lines (parallel hyperechoic reverberation artefacts at "
     "regular intervals deep to the pleural line) with preserved lung sliding.",
     "BLUE A-profile — normal aerated lung."),
    (1, "Pneumothorax", "ai_pneumothorax",
     "A-lines preserved but lung sliding ABSENT. M-mode shows the "
     "stratosphere/barcode sign (no granular layer below pleura).",
     "BLUE A'-profile — A-prime = pneumothorax until proven otherwise."),
    (2, "Focal B-lines", "ai_blines_focal",
     "1–2 vertical hyperechoic comet-tail artefacts arising from the pleura "
     "and extending to the bottom of the image, erasing A-lines locally.",
     "Localised interstitial syndrome — early CHF, focal atelectasis, "
     "or contusion."),
    (3, "Diffuse B-lines", "ai_blines_diffuse",
     "≥3 B-lines per intercostal space, bilateral and symmetric. Lung sliding "
     "preserved.",
     "BLUE B-profile — most commonly cardiogenic pulmonary edema."),
    (4, "Consolidation", "ai_consolidation",
     "Hepatised lung tissue (tissue-like echo pattern). May include shred "
     "sign, dynamic air bronchograms, or sub-pleural irregularity.",
     "C-profile or PLAPS-positive — pneumonia, atelectasis, or contusion."),
    (5, "Pleural Effusion", "ai_pleural_effusion",
     "Anechoic (or low-echoic) fluid collection above the diaphragm, often "
     "with the spine sign and quad sign on M-mode.",
     "PLAPS — supports pneumonia, CHF with effusion, or hemothorax."),
    (6, "ARDS / White Lung", "ai_ards_white_lung",
     "Confluent B-lines producing a 'white lung' appearance with patchy "
     "sparing, reduced lung sliding, irregular pleura.",
     "Non-cardiogenic pulmonary edema — heterogeneous distribution, "
     "distinguishes from cardiogenic B-profile."),
    (7, "Lung Point", "ai_lung_point",
     "Transition between sliding and non-sliding lung within a single "
     "intercostal window — pathognomonic for pneumothorax.",
     "100% specific for pneumothorax — typically lateral chest wall."),
    (8, "Pleural Thickening", "ai_pleural_thickening",
     "Irregular, thickened, or fragmented pleural line; often with subtle "
     "sub-pleural consolidations.",
     "Hallmark of viral pneumonia / COVID-19, chronic interstitial disease."),
    (9, "Interstitial Syndrome", "ai_interstitial",
     "Multiple B-lines combined with pleural-line irregularities, sub-pleural "
     "consolidations, or a heterogeneous appearance.",
     "Encompasses viral pneumonia, fibrosis, COVID-19 pattern."),
]

KEYWORDS = {
    0: ["a-line", "a_line", "normal", "healthy", "aerat"],
    1: ["pneumo", "ptx", "stratosphere", "barcode"],
    2: ["b-line", "b_line", "bline", "focal"],
    3: ["b-line", "b_line", "bline", "edema", "comet", "diffuse"],
    4: ["consolid", "shred", "hepati", "bronchogram", "pneu"],
    5: ["effusion", "anechoic", "fluid"],
    6: ["ards", "white", "confluent"],
    7: ["lung_point", "lung-point", "transition"],
    8: ["thicken", "irregul", "fragment"],
    9: ["interstit", "covid", "viral"],
}


def score_image(path: Path) -> float:
    try:
        im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    except Exception:
        return -1.0
    if im.size == 0 or im.std() < 5:
        return -1.0
    H, W = im.shape
    mean = im.mean()
    std = im.std()
    upper = im[: int(H * 0.30), :]
    pleural_strength = float(upper.max(axis=1).max() - upper.mean())
    brightness_pen = abs(mean - 95.0) / 95.0
    gy = float(np.abs(np.diff(im, axis=0)).mean())
    gx = float(np.abs(np.diff(im, axis=1)).mean())
    edges = (gy + gx) / 2
    return float(0.4 * std + 0.5 * pleural_strength + 0.3 * edges - 30.0 * brightness_pen)


def filename_bonus(source_file: str, kws) -> float:
    s = source_file.lower()
    return sum(3.0 for k in kws if k in s)


def collect_per_class():
    """Return dict[class_int] -> list of (filename, dataset, source_file)."""
    rows = {c: [] for c, *_ in CLASSES}
    with META.open() as f:
        for r in csv.DictReader(f):
            try:
                c = int(r["pathology_class"])
            except ValueError:
                continue
            if c not in rows:
                continue
            if r["source_dataset"] not in REAL_SOURCES:
                continue
            rows[c].append((r["filename"], r["source_dataset"], r["source_file"]))
    return rows


def pick_top(rows, kws, n_top=25, sample_pool=400):
    rng = np.random.default_rng(seed=42)
    pool = rows
    if len(pool) > sample_pool:
        idx = rng.choice(len(pool), sample_pool, replace=False)
        pool = [pool[i] for i in idx]
    scored = []
    for fname, ds, sf in pool:
        s = score_image(IMGDIR / fname)
        if s < 0:
            continue
        s += filename_bonus(sf, kws)
        scored.append((s, fname, ds, sf))
    scored.sort(reverse=True)
    return scored[:n_top]


def cover_page(pdf, all_rows):
    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.93, "MoCoLUS — Training Data Review",
            ha="center", va="center", fontsize=22, weight="bold")
    ax.text(0.5, 0.89, "Synthetic Lung POCUS Training Data",
            ha="center", va="center", fontsize=13, color="#444")

    ax.text(0.08, 0.83,
            "Author: Alex Hastava   •   Generated: 2026-05-04",
            ha="left", va="top", fontsize=10, color="#222")

    # Dataset summary table
    ax.text(0.08, 0.76, "Real-source training frames per class", weight="bold", fontsize=12)
    headers = ["Class", "Pathology", "n", "Sources"]
    col_x = [0.08, 0.16, 0.42, 0.50]
    for x, h in zip(col_x, headers):
        ax.text(x, 0.72, h, weight="bold", fontsize=9, color="#222")
    y = 0.695
    total = 0
    for cls, name, *_ in CLASSES:
        rows = all_rows.get(cls, [])
        n = len(rows)
        total += n
        ds_counts = {}
        for _, ds, _ in rows:
            ds_counts[ds] = ds_counts.get(ds, 0) + 1
        ranked = sorted(ds_counts.items(), key=lambda x: -x[1])
        top_n = 3
        parts = [f"{d.split('/')[-1]} ({c})" for d, c in ranked[:top_n]]
        if len(ranked) > top_n:
            parts.append(f"+{len(ranked) - top_n} more")
        sources = ", ".join(parts)
        ax.text(col_x[0], y, str(cls), fontsize=8.5)
        ax.text(col_x[1], y, name, fontsize=8.5)
        ax.text(col_x[2], y, str(n), fontsize=8.5)
        ax.text(col_x[3], y, sources, fontsize=7.0, color="#333")
        y -= 0.025
    ax.plot([0.08, 0.92], [y + 0.012, y + 0.012], color="#888", linewidth=0.5)
    ax.text(col_x[1], y - 0.005, "Total", weight="bold", fontsize=9)
    ax.text(col_x[2], y - 0.005, str(total), weight="bold", fontsize=9)

    ax.text(0.08, 0.18,
            "Annotate in Preview / Adobe / GoodNotes — highlight, strike-through,\n"
            "or drop comments anywhere useful.",
            ha="left", va="top", fontsize=9, color="#666", style="italic")

    pdf.savefig(fig); plt.close(fig)


def class_page(pdf, cls, name, ai_stem, definition, blue_note, top_frames):
    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    fig.patch.set_facecolor("white")

    # Header
    head = fig.add_axes([0.05, 0.90, 0.90, 0.08])
    head.set_xlim(0, 1); head.set_ylim(0, 1); head.axis("off")
    head.text(0.0, 1.0, f"Class {cls} — {name}",
              fontsize=18, weight="bold", va="top")
    import textwrap
    wrapped_def = "\n".join(textwrap.wrap(definition, width=110))
    head.text(0.0, 0.45, wrapped_def, fontsize=8.5, va="top",
              color="#222")
    head.text(0.0, 0.05, blue_note, fontsize=8.5, style="italic",
              va="top", color="#444")

    # Real frames: 5x5 grid
    grid_top = 0.86
    grid_bottom = 0.30
    n_rows, n_cols = 5, 5
    cell_h = (grid_top - grid_bottom) / n_rows
    cell_w = 0.18
    grid_left = 0.05
    section = fig.add_axes([0, 0, 1, 1])
    section.set_xlim(0, 1); section.set_ylim(0, 1); section.axis("off")
    section.text(0.05, grid_top + 0.005, "Real clinical training frames "
                 f"(top {n_rows*n_cols} by visual-quality score)",
                 weight="bold", fontsize=10, color="#222")

    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = fig.add_axes([
            grid_left + c * cell_w,
            grid_top - (r + 1) * cell_h,
            cell_w * 0.92,
            cell_h * 0.85,
        ])
        ax.set_xticks([]); ax.set_yticks([])
        if i < len(top_frames):
            score, fname, ds, sf = top_frames[i]
            try:
                im = np.asarray(Image.open(IMGDIR / fname).convert("L"))
                ax.imshow(im, cmap="gray", vmin=0, vmax=255, aspect="auto")
                # Source tag (compact)
                src = ds.split("/")[-1][:8]
                ax.text(0.02, 0.97, f"{i+1}", transform=ax.transAxes,
                        fontsize=6, color="white", va="top",
                        bbox=dict(facecolor="black", pad=1, edgecolor="none"))
                ax.text(0.98, 0.03, src, transform=ax.transAxes,
                        fontsize=5, color="white", ha="right", va="bottom",
                        bbox=dict(facecolor="black", pad=1, edgecolor="none"))
            except Exception as e:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center", fontsize=6)
        else:
            ax.set_facecolor("#f4f4f4")
        for spine in ax.spines.values():
            spine.set_linewidth(0.4); spine.set_color("#888")

    # AI-generated row
    ai_top = 0.27
    ai_bottom = 0.10
    ai_section = fig.add_axes([0, 0, 1, 1])
    ai_section.set_xlim(0, 1); ai_section.set_ylim(0, 1); ai_section.axis("off")
    ai_section.text(0.05, ai_top + 0.005,
                    "AI-generated cross-validation sample "
                    "(MoCoLUS ControlNet, this class)",
                    weight="bold", fontsize=10, color="#222")

    # Single large AI sample on left, structural guide for context on right
    ai_path = OUT_DIR / f"{ai_stem}.png"
    guide_path = OUT_DIR / f"structural_guide_{name.lower().replace(' ','_').replace('/','_').replace('-','_')}.png"
    # Map class-id to known structural-guide filename
    guide_map = {
        0: "structural_guide_normal.png",
        1: "structural_guide_pneumothorax.png",
        2: "structural_guide_blines_focal.png",
        3: "structural_guide_blines_diffuse.png",
        4: "structural_guide_consolidation.png",
        5: "structural_guide_effusion.png",
        6: "structural_guide_ards.png",
        7: "structural_guide_lung_point.png",
        8: "structural_guide_pleural_thickening.png",
        9: "structural_guide_interstitial.png",
    }
    guide_path = OUT_DIR / guide_map[cls]

    # Two side-by-side panels: structural guide  →  AI output
    pw = 0.18
    pgap = 0.02
    px0 = 0.05
    py = ai_bottom + 0.005
    ph = ai_top - ai_bottom - 0.025

    for j, (label, p) in enumerate([("Structural guide", guide_path),
                                    ("ControlNet output", ai_path)]):
        ax = fig.add_axes([px0 + j * (pw + pgap), py, pw, ph])
        ax.set_xticks([]); ax.set_yticks([])
        if p.exists():
            ax.imshow(np.asarray(Image.open(p).convert("L")),
                      cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_title(label, fontsize=8, pad=3)
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("#888")

    # Per-class sign-off footer
    foot = fig.add_axes([0.05, 0.025, 0.90, 0.07])
    foot.set_xlim(0, 1); foot.set_ylim(0, 1); foot.axis("off")
    foot.text(0.0, 1.0, f"Notes — Class {cls} ({name})",
              weight="bold", fontsize=9, va="top", color="#222")
    foot.text(0.0, 0.55, "Anything looks off, surprising, or worth chatting about?",
              fontsize=8.5, style="italic", va="top", color="#444")
    foot.text(0.0, 0.20, "_____________________________________________________"
              "______________________________________________",
              fontsize=8, color="#666", va="top")

    # Page number
    fig.text(0.95, 0.012, f"Class {cls+1}/10",
             ha="right", fontsize=7, color="#888")

    pdf.savefig(fig); plt.close(fig)


def signoff_page(pdf):
    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.94, "Discussion Summary", ha="center", fontsize=22, weight="bold")
    ax.text(0.5, 0.90, "Anything you want to flag, follow up on, or explore further",
            ha="center", fontsize=12, color="#444")

    y = 0.82
    ax.text(0.08, y, "Per-class quick takes (optional)", weight="bold", fontsize=12)
    y -= 0.03
    headers = ["Class", "Pathology", "Quick reaction / things to discuss"]
    cols = [0.08, 0.18, 0.50]
    for x, h in zip(cols, headers):
        ax.text(x, y, h, weight="bold", fontsize=9.5)
    y -= 0.025
    for cls, name, *_ in CLASSES:
        ax.text(cols[0], y, str(cls), fontsize=9)
        ax.text(cols[1], y, name, fontsize=9)
        ax.text(cols[2], y, "_" * 55, fontsize=8.5, color="#888")
        y -= 0.038

    y -= 0.02
    ax.text(0.08, y, "Bigger-picture thoughts", weight="bold", fontsize=11)
    y -= 0.03
    for _ in range(8):
        ax.text(0.08, y, "_" * 92, fontsize=9, color="#888")
        y -= 0.025

    y -= 0.04
    ax.text(0.08, y, "Reviewed by: " + "_" * 30, fontsize=10, color="#444")
    ax.text(0.55, y, "Date: " + "_" * 18, fontsize=10, color="#444")
    ax.text(0.08, y - 0.035,
            "Thanks — happy to set up a time to walk through any of this.",
            fontsize=9, style="italic", color="#666")

    pdf.savefig(fig); plt.close(fig)


def main():
    print("collecting metadata…")
    all_rows = collect_per_class()
    for cls, name, *_ in CLASSES:
        print(f"  class {cls} ({name}): {len(all_rows[cls])} real-source frames")

    print("\nscoring + picking top-25 per class…")
    top_per_class = {}
    for cls, name, *_ in CLASSES:
        rows = all_rows[cls]
        if not rows:
            top_per_class[cls] = []
            print(f"  class {cls}: empty — skipping")
            continue
        top = pick_top(rows, KEYWORDS[cls], n_top=25, sample_pool=400)
        top_per_class[cls] = top
        print(f"  class {cls} ({name}): scored {len(rows)} → top {len(top)}, "
              f"best score {top[0][0]:.1f} ({top[0][1]})")

    print(f"\nbuilding {OUT_PDF}…")
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    with PdfPages(OUT_PDF) as pdf:
        cover_page(pdf, all_rows)
        for cls, name, ai_stem, definition, blue_note in CLASSES:
            class_page(pdf, cls, name, ai_stem, definition, blue_note,
                       top_per_class[cls])
        signoff_page(pdf)

    sz_mb = OUT_PDF.stat().st_size / (1024 * 1024)
    print(f"\nwrote {OUT_PDF}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
