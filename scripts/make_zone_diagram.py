"""Render docs/images/blue_protocol_zones.png — the MoCoLUS 8-zone scan map.

Run: python3 scripts/make_zone_diagram.py
"""
from pathlib import Path as PPath

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path

OUTPUT = PPath(__file__).resolve().parent.parent / "docs" / "images" / "blue_protocol_zones.png"

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
TORSO_FILL = "#161b22"
TORSO_EDGE = "#5b6471"
MIDLINE = "#30363d"
BORDER = "#3d4654"
HEADER_BG = "#1c2128"

TIERS = [
    ("Upper BLUE",  "#3fb950", "2nd ICS, mid-clavicular line",     "Pneumothorax, interstitial edema"),
    ("Lower BLUE",  "#58a6ff", "4-5th ICS, anterior axillary",     "B-lines, lung sliding"),
    ("PLAPS",       "#d29922", "10-12th ICS, posterior axillary",  "Effusions, consolidation"),
    ("Diaphragm",   "#f85149", "Costophrenic angle",               "Effusion volume, diaphragm motion"),
]

# Anatomically-anchored zone positions on the silhouette
ZONES = [
    ("Upper BLUE R", -0.55,  1.30, 0),
    ("Upper BLUE L",  0.55,  1.30, 0),
    ("Lower BLUE R", -1.05,  0.40, 1),
    ("Lower BLUE L",  1.05,  0.40, 1),
    ("PLAPS R",      -1.05, -0.65, 2),
    ("PLAPS L",       1.05, -0.65, 2),
    ("Diaphragm R",  -0.95, -1.50, 3),
    ("Diaphragm L",   0.95, -1.50, 3),
]

LABEL_GUTTER = 1.75


def torso_path():
    """Anterior-view human silhouette as a closed cubic Bezier path."""
    verts = [
        (0.00, 3.10),
        # head: top-right quarter
        (0.42, 3.10), (0.50, 2.55), (0.40, 2.20),
        # head right → jaw
        (0.38, 2.05), (0.32, 1.95), (0.22, 1.85),
        # neck → trapezius → shoulder cap
        (0.45, 1.78), (0.90, 1.68), (1.22, 1.65),
        # shoulder cap (rounds out and down)
        (1.40, 1.62), (1.46, 1.48), (1.42, 1.30),
        # lateral chest down
        (1.40, 0.80), (1.32, 0.10), (1.20, -0.20),
        # waist (narrows)
        (1.10, -0.40), (1.02, -0.55), (1.02, -0.75),
        # hip flare back out
        (1.02, -1.05), (1.20, -1.30), (1.30, -1.60),
        # bottom-right corner
        (1.32, -1.85), (1.24, -2.00), (1.08, -2.02),
        # across the bottom (mirror)
        (0.55, -2.07), (-0.55, -2.07), (-1.08, -2.02),
        # bottom-left corner
        (-1.24, -2.00), (-1.32, -1.85), (-1.30, -1.60),
        # hip flare left
        (-1.20, -1.30), (-1.02, -1.05), (-1.02, -0.75),
        # waist left
        (-1.02, -0.55), (-1.10, -0.40), (-1.20, -0.20),
        # lateral chest left
        (-1.32, 0.10), (-1.40, 0.80), (-1.42, 1.30),
        # shoulder cap left
        (-1.46, 1.48), (-1.40, 1.62), (-1.22, 1.65),
        # shoulder → neck
        (-0.90, 1.68), (-0.45, 1.78), (-0.22, 1.85),
        # neck → jaw
        (-0.32, 1.95), (-0.38, 2.05), (-0.40, 2.20),
        # head left back to top
        (-0.50, 2.55), (-0.42, 3.10), (0.00, 3.10),
        (0.00, 3.10),
    ]
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def draw_torso(ax):
    ax.add_patch(PathPatch(torso_path(), facecolor=TORSO_FILL,
                           edgecolor=TORSO_EDGE, linewidth=1.6, zorder=1))
    # Subtle clavicle hint
    ax.plot([-1.05, -0.20], [1.62, 1.78], color=TORSO_EDGE,
            linewidth=0.8, alpha=0.6, zorder=2)
    ax.plot([0.20, 1.05], [1.78, 1.62], color=TORSO_EDGE,
            linewidth=0.8, alpha=0.6, zorder=2)
    # Sternum / midline reference
    ax.plot([0, 0], [-1.85, 1.78], linestyle=(0, (4, 4)),
            color=MIDLINE, linewidth=1.0, zorder=2)


def draw_zones(ax):
    for name, x, y, tier in ZONES:
        color = TIERS[tier][1]
        ax.add_patch(Circle((x, y), 0.13, facecolor=color, alpha=0.22,
                            edgecolor="none", zorder=3))
        ax.add_patch(Circle((x, y), 0.085, facecolor=color,
                            edgecolor=FG, linewidth=1.2, zorder=4))
        # Subtle leader line from dot to label gutter
        gutter = -LABEL_GUTTER if x < 0 else LABEL_GUTTER
        ax.plot([x, gutter], [y, y], color=MUTED, linewidth=0.5,
                alpha=0.45, zorder=2)
        if x < 0:
            ax.text(-LABEL_GUTTER - 0.05, y, name, color=FG, fontsize=10.5,
                    ha="right", va="center", fontweight="500", zorder=5)
        else:
            ax.text(LABEL_GUTTER + 0.05, y, name, color=FG, fontsize=10.5,
                    ha="left", va="center", fontweight="500", zorder=5)


def draw_orientation(ax):
    ax.text(0, 3.45, "H E A D", color=MUTED, fontsize=9.5,
            ha="center", va="center")
    ax.text(0, -2.40, "F E E T", color=MUTED, fontsize=9.5,
            ha="center", va="center")
    ax.text(-3.10, 0, "Patient\nRight", color=MUTED, fontsize=9,
            ha="center", va="center", linespacing=1.3)
    ax.text(3.10, 0, "Patient\nLeft", color=MUTED, fontsize=9,
            ha="center", va="center", linespacing=1.3)


def draw_legend(ax):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    ax.axis("off")

    n_rows = 5  # 1 header + 4 data
    col_x = [0.0, 0.27, 0.60, 1.0]
    headers = ["Zone", "Location", "Primary assessment"]

    # Header row background
    ax.add_patch(Rectangle((0, n_rows - 1), 1, 1, facecolor=HEADER_BG,
                           edgecolor="none", zorder=1, clip_on=False))

    # Internal grid: vertical separators
    for x in col_x[1:-1]:
        ax.plot([x, x], [0, n_rows], color=BORDER, linewidth=0.9, zorder=3)
    # Horizontal separators
    for i in range(1, n_rows):
        ax.plot([0, 1], [i, i], color=BORDER, linewidth=0.9, zorder=3)
    # Outer border
    ax.add_patch(Rectangle((0, 0), 1, n_rows, facecolor="none",
                           edgecolor=BORDER, linewidth=1.4, zorder=4,
                           clip_on=False))

    # Header text
    for c, text in enumerate(headers):
        x_center = (col_x[c] + col_x[c + 1]) / 2
        ax.text(x_center, n_rows - 0.5, text.upper(), color=MUTED,
                fontsize=9.5, ha="center", va="center",
                fontweight="600", zorder=5)

    # Data rows
    for i, (name, color, location, assessment) in enumerate(TIERS):
        row_y = n_rows - 1.5 - i  # center y of row i (i=0 is first data row)
        ax.scatter([col_x[0] + 0.04], [row_y], s=180, c=color,
                   edgecolors=FG, linewidths=1.0, zorder=5, clip_on=False)
        ax.text(col_x[0] + 0.07, row_y, name, color=FG, fontsize=10.5,
                ha="left", va="center", fontweight="500", zorder=5)
        ax.text(col_x[1] + 0.025, row_y, location, color=FG, fontsize=10,
                ha="left", va="center", zorder=5)
        ax.text(col_x[2] + 0.02, row_y, assessment, color=FG, fontsize=10,
                ha="left", va="center", zorder=5)


def main():
    fig = plt.figure(figsize=(10, 11), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[5.8, 2.0], hspace=0.10)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-2.8, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.text(0.5, 0.960, "MoCoLUS 8-Zone Lung Scan",
             color=FG, fontsize=18, fontweight="700", ha="center")
    fig.text(0.5, 0.935, "Extended BLUE protocol (Lichtenstein, 2008) with diaphragm views",
             color=MUTED, fontsize=10.5, ha="center", style="italic")

    draw_torso(ax)
    draw_zones(ax)
    draw_orientation(ax)

    legend_ax = fig.add_subplot(gs[1])
    draw_legend(legend_ax)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
