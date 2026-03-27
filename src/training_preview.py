"""
Synthetic Training Image Preview & Rendering Module
=====================================================
Generates and displays clinically accurate synthetic lung POCUS images
for each pathology type before projecting onto the simulator GUI.

Provides:
  - Side-by-side B-mode + M-mode rendering for each pathology
  - Full pathology comparison grid
  - Annotated clinical feature overlays
  - Export to PNG/PDF for review
  - Scenario-level full-exam preview

Usage:
    # Preview all pathologies
    python -m src.training_preview --all

    # Preview specific pathology
    python -m src.training_preview --pathology consolidation

    # Preview a clinical scenario
    python -m src.training_preview --scenario left_pneumonia

    # Export grid to file
    python -m src.training_preview --all --output training_grid.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works headless)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from .clinical_frames import (
        ClinicalFrameGenerator,
        ClinicalTemporalStack,
        ClinicalPathology,
        CLINICAL_SCENARIOS,
        CLINICAL_PATHOLOGY_NAMES,
    )
except ImportError:
    from clinical_frames import (
        ClinicalFrameGenerator,
        ClinicalTemporalStack,
        ClinicalPathology,
        CLINICAL_SCENARIOS,
        CLINICAL_PATHOLOGY_NAMES,
    )


# ---------------------------------------------------------------------------
# Annotation definitions (clinical features to highlight)
# ---------------------------------------------------------------------------

PATHOLOGY_ANNOTATIONS = {
    ClinicalPathology.NORMAL_A_PROFILE: {
        "title": "Normal (A-Profile)",
        "features": [
            "Pleural line: bright, smooth, regular",
            "A-lines: horizontal reverberations at equal intervals",
            "Lung sliding present (seashore M-mode)",
            "Normal findings at all BLUE points",
        ],
        "clinical_significance": "Rules out pneumothorax and pulmonary edema",
    },
    ClinicalPathology.PNEUMOTHORAX: {
        "title": "Pneumothorax (A-Profile, No Sliding)",
        "features": [
            "A-lines present (strong reverberations)",
            "NO lung sliding (stratosphere/barcode M-mode)",
            "Absent lung pulse",
            "Look for lung point to confirm",
        ],
        "clinical_significance": "A-lines + absent sliding = pneumothorax until proven otherwise",
    },
    ClinicalPathology.B_LINES_FOCAL: {
        "title": "Focal B-Lines (1-2 per field)",
        "features": [
            "1-2 vertical hyperechoic lines from pleura to bottom",
            "A-lines still visible between B-lines",
            "Lung sliding preserved",
            "May be normal variant in dependent zones",
        ],
        "clinical_significance": "≤2 B-lines per zone may be normal; correlate clinically",
    },
    ClinicalPathology.B_LINES_DIFFUSE: {
        "title": "Diffuse B-Lines (B-Profile)",
        "features": [
            "≥3 B-lines per intercostal space",
            "B-lines erase A-lines",
            "Extend from pleural line to screen bottom",
            "Lung sliding preserved (vs ARDS)",
        ],
        "clinical_significance": "Bilateral B-profile = pulmonary edema (cardiogenic)",
    },
    ClinicalPathology.CONSOLIDATION: {
        "title": "Lung Consolidation (Hepatization)",
        "features": [
            "Tissue-like echotexture (looks like liver)",
            "Air bronchograms: bright dots within consolidation",
            "Shred sign: irregular deep border",
            "Pleural line may be irregular",
        ],
        "clinical_significance": "Consolidation at PLAPS = pneumonia diagnosis (BLUE protocol)",
    },
    ClinicalPathology.PLEURAL_EFFUSION: {
        "title": "Pleural Effusion",
        "features": [
            "Anechoic (black) fluid collection",
            "Quad sign: fluid bounded by landmarks",
            "Compressed lung below effusion",
            "Spine sign in larger effusions",
        ],
        "clinical_significance": "Effusion guides thoracentesis; quantify by depth",
    },
    ClinicalPathology.ARDS_WHITE_LUNG: {
        "title": "ARDS / White Lung",
        "features": [
            "Confluent B-lines (no individual B-lines visible)",
            "Entire sub-pleural space appears white",
            "Irregular, thickened pleural line",
            "Reduced or absent lung sliding",
        ],
        "clinical_significance": "White lung with absent sliding + spared areas = ARDS",
    },
    ClinicalPathology.LUNG_POINT: {
        "title": "Lung Point (Pneumothorax Border)",
        "features": [
            "Transition zone: sliding ↔ no sliding",
            "Pathognomonic for pneumothorax",
            "Left: stratosphere (no sliding)",
            "Right: seashore (sliding present)",
        ],
        "clinical_significance": "100% specific for pneumothorax; marks pneumothorax extent",
    },
    ClinicalPathology.PLEURAL_THICKENING: {
        "title": "Pleural Thickening",
        "features": [
            "Irregular, thickened pleural line (>2mm)",
            "Fragmented pleural appearance",
            "Scattered sub-pleural abnormalities",
            "May have focal B-lines",
        ],
        "clinical_significance": "Suggests chronic inflammation, fibrosis, or prior infection",
    },
    ClinicalPathology.INTERSTITIAL_SYNDROME: {
        "title": "Interstitial Syndrome (e.g., COVID)",
        "features": [
            "Multiple B-lines with irregular pleura",
            "Sub-pleural consolidations (small, patchy)",
            "Pleural line fragmentation",
            "Bilateral, non-homogeneous distribution",
        ],
        "clinical_significance": "Pattern seen in viral pneumonia, pulmonary fibrosis",
    },
}


# ---------------------------------------------------------------------------
# Preview Renderer
# ---------------------------------------------------------------------------

class TrainingPreviewRenderer:
    """
    Renders synthetic training images with clinical annotations.

    Supports:
      - Single pathology preview (B-mode + M-mode + annotations)
      - Full comparison grid (all pathologies)
      - Scenario-level previews (8-zone exam layout)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        n_frames: int = 32,
        dpi: int = 150,
    ):
        self.image_size = image_size
        self.n_frames = n_frames
        self.dpi = dpi
        self.frame_gen = ClinicalFrameGenerator(image_size=image_size)
        self.stack_gen = ClinicalTemporalStack(
            frame_gen=self.frame_gen,
            n_frames=n_frames,
        )

    # ------------------------------------------------------------------
    # Single pathology preview
    # ------------------------------------------------------------------

    def render_single(
        self,
        pathology: ClinicalPathology,
        seed: int = 42,
        show_annotations: bool = True,
    ) -> plt.Figure:
        """
        Render a single pathology with B-mode, M-mode, and annotations.

        Returns matplotlib Figure (call fig.savefig() or plt.show()).
        """
        # Determine if this pathology has lung sliding
        has_sliding = pathology not in (
            ClinicalPathology.PNEUMOTHORAX,
            ClinicalPathology.ARDS_WHITE_LUNG,
        )

        # Generate temporal stack
        result = self.stack_gen.generate_stack(
            pathology=pathology,
            lung_sliding=has_sliding,
            seed=seed,
        )

        bmode = result["bmode_stack"][0]  # First frame
        mmode = result["mmode"]
        mmode_pattern = result["mmode_pattern"]

        # Create figure
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(1, 3, width_ratios=[3, 2, 3], wspace=0.3)

        # B-mode image
        ax_bmode = fig.add_subplot(gs[0])
        ax_bmode.imshow(bmode, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax_bmode.set_title("B-Mode", fontsize=12, fontweight="bold")
        ax_bmode.set_ylabel("Depth (px)")
        ax_bmode.set_xlabel("Lateral (px)")

        # Add depth scale bar
        depth_ticks = np.linspace(0, self.image_size[0], 5)
        depth_labels = [f"{d * 12 / self.image_size[0]:.0f} cm" for d in depth_ticks]
        ax_bmode.set_yticks(depth_ticks)
        ax_bmode.set_yticklabels(depth_labels)

        # Pleural line marker
        pleural_row = self.frame_gen.pleural_row
        ax_bmode.axhline(y=pleural_row, color="cyan", linewidth=0.5, alpha=0.5, linestyle="--")
        ax_bmode.text(5, pleural_row - 3, "pleura", color="cyan", fontsize=7, alpha=0.7)

        # M-mode image
        ax_mmode = fig.add_subplot(gs[1])
        ax_mmode.imshow(mmode, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax_mmode.set_title(f"M-Mode ({mmode_pattern})", fontsize=12, fontweight="bold")
        ax_mmode.set_ylabel("Depth (px)")
        ax_mmode.set_xlabel("Time (frames)")
        ax_mmode.axhline(y=pleural_row, color="cyan", linewidth=0.5, alpha=0.5, linestyle="--")

        # Annotations panel
        if show_annotations:
            ax_text = fig.add_subplot(gs[2])
            ax_text.axis("off")

            annotations = PATHOLOGY_ANNOTATIONS.get(pathology, {})
            title = annotations.get("title", pathology.name)
            features = annotations.get("features", [])
            significance = annotations.get("clinical_significance", "")

            text_y = 0.95
            ax_text.text(0.05, text_y, title, fontsize=14, fontweight="bold",
                        transform=ax_text.transAxes, va="top")
            text_y -= 0.10

            ax_text.text(0.05, text_y, "Clinical Features:", fontsize=11,
                        fontweight="bold", transform=ax_text.transAxes, va="top",
                        color="#2196F3")
            text_y -= 0.06

            for feature in features:
                ax_text.text(0.08, text_y, f"• {feature}", fontsize=9,
                            transform=ax_text.transAxes, va="top", wrap=True)
                text_y -= 0.07

            text_y -= 0.05
            ax_text.text(0.05, text_y, "Clinical Significance:", fontsize=11,
                        fontweight="bold", transform=ax_text.transAxes, va="top",
                        color="#4CAF50")
            text_y -= 0.06
            ax_text.text(0.08, text_y, significance, fontsize=9,
                        transform=ax_text.transAxes, va="top", wrap=True,
                        style="italic")

            text_y -= 0.10
            ax_text.text(0.05, text_y, f"M-Mode Pattern: {mmode_pattern}",
                        fontsize=10, transform=ax_text.transAxes, va="top",
                        color="#FF9800", fontweight="bold")
            text_y -= 0.06
            sliding_str = "Present" if has_sliding else "Absent"
            ax_text.text(0.05, text_y, f"Lung Sliding: {sliding_str}",
                        fontsize=10, transform=ax_text.transAxes, va="top",
                        color="#FF9800")

        fig.suptitle(f"Synthetic Lung POCUS — {pathology.name}",
                    fontsize=14, fontweight="bold", y=1.02)

        return fig

    # ------------------------------------------------------------------
    # Full comparison grid
    # ------------------------------------------------------------------

    def render_comparison_grid(
        self,
        seed: int = 42,
        pathologies: Optional[List[ClinicalPathology]] = None,
    ) -> plt.Figure:
        """
        Render a grid comparing all (or selected) pathology types.
        Each cell shows the B-mode frame with title.
        """
        if pathologies is None:
            pathologies = list(ClinicalPathology)

        n = len(pathologies)
        cols = min(5, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]

        axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

        for i, pathology in enumerate(pathologies):
            ax = axes_flat[i]
            frame = self.frame_gen.generate(pathology, seed=seed + i)
            ax.imshow(frame, cmap="gray", aspect="auto", vmin=0, vmax=1)

            title = PATHOLOGY_ANNOTATIONS.get(pathology, {}).get("title", pathology.name)
            # Truncate long titles
            if len(title) > 30:
                title = title[:27] + "..."
            ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
            ax.tick_params(labelsize=6)

            # Pleural line indicator
            ax.axhline(y=self.frame_gen.pleural_row, color="cyan",
                       linewidth=0.3, alpha=0.3, linestyle="--")

        # Hide unused axes
        for i in range(n, len(axes_flat)):
            axes_flat[i].axis("off")

        fig.suptitle("Synthetic Lung POCUS — All Pathology Types",
                    fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    # ------------------------------------------------------------------
    # B-mode + M-mode comparison grid
    # ------------------------------------------------------------------

    def render_bmode_mmode_grid(
        self,
        seed: int = 42,
        pathologies: Optional[List[ClinicalPathology]] = None,
    ) -> plt.Figure:
        """
        Render a grid with B-mode and M-mode side by side for each pathology.
        """
        if pathologies is None:
            pathologies = list(ClinicalPathology)

        n = len(pathologies)
        fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n))
        if n == 1:
            axes = [axes]

        for i, pathology in enumerate(pathologies):
            has_sliding = pathology not in (
                ClinicalPathology.PNEUMOTHORAX,
                ClinicalPathology.ARDS_WHITE_LUNG,
            )

            result = self.stack_gen.generate_stack(
                pathology=pathology,
                lung_sliding=has_sliding,
                seed=seed + i,
            )

            # B-mode
            ax_b = axes[i][0]
            ax_b.imshow(result["bmode_stack"][0], cmap="gray", aspect="auto", vmin=0, vmax=1)
            title = PATHOLOGY_ANNOTATIONS.get(pathology, {}).get("title", pathology.name)
            ax_b.set_title(f"B-Mode: {title}", fontsize=9, fontweight="bold")
            ax_b.axhline(y=self.frame_gen.pleural_row, color="cyan",
                        linewidth=0.3, alpha=0.4, linestyle="--")

            # M-mode
            ax_m = axes[i][1]
            ax_m.imshow(result["mmode"], cmap="gray", aspect="auto", vmin=0, vmax=1)
            pattern = result["mmode_pattern"]
            sliding_label = "sliding" if has_sliding else "no sliding"
            ax_m.set_title(f"M-Mode: {pattern} ({sliding_label})", fontsize=9, fontweight="bold")
            ax_m.axhline(y=self.frame_gen.pleural_row, color="cyan",
                        linewidth=0.3, alpha=0.4, linestyle="--")

        fig.suptitle("Synthetic Lung POCUS — B-Mode & M-Mode Comparison",
                    fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        return fig

    # ------------------------------------------------------------------
    # Scenario preview (8-zone exam)
    # ------------------------------------------------------------------

    def render_scenario(
        self,
        scenario_key: str,
        seed: int = 42,
    ) -> plt.Figure:
        """
        Render a full BLUE protocol exam for a clinical scenario.
        Shows all 8 zones with their assigned pathologies.
        """
        if scenario_key not in CLINICAL_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario_key}'. "
                           f"Available: {list(CLINICAL_SCENARIOS.keys())}")

        scenario = CLINICAL_SCENARIOS[scenario_key]

        # Resolve zone pathologies
        zone_names = [
            "UPPER_BLUE_L", "UPPER_BLUE_R",
            "LOWER_BLUE_L", "LOWER_BLUE_R",
            "PLAPS_L", "PLAPS_R",
            "DIAPHRAGM_L", "DIAPHRAGM_R",
        ]

        zone_pathologies = {}
        if "zones" in scenario:
            zone_pathologies = scenario["zones"]
        elif "all_zones" in scenario:
            for z in zone_names:
                zone_pathologies[z] = scenario["all_zones"]
        else:
            for z in zone_names:
                if z.endswith("_L") and "left_zones" in scenario:
                    zone_pathologies[z] = scenario["left_zones"]
                elif z.endswith("_R") and "right_zones" in scenario:
                    zone_pathologies[z] = scenario["right_zones"]
                else:
                    zone_pathologies[z] = ClinicalPathology.NORMAL_A_PROFILE

        # Create 4x2 grid (rows: zone levels, cols: left/right)
        fig, axes = plt.subplots(4, 2, figsize=(10, 16))
        zone_pairs = [
            ("UPPER_BLUE_L", "UPPER_BLUE_R"),
            ("LOWER_BLUE_L", "LOWER_BLUE_R"),
            ("PLAPS_L", "PLAPS_R"),
            ("DIAPHRAGM_L", "DIAPHRAGM_R"),
        ]

        for row_idx, (left_zone, right_zone) in enumerate(zone_pairs):
            for col_idx, zone_name in enumerate([left_zone, right_zone]):
                ax = axes[row_idx][col_idx]
                pathology = zone_pathologies.get(zone_name, ClinicalPathology.NORMAL_A_PROFILE)
                frame = self.frame_gen.generate(pathology, seed=seed + row_idx * 2 + col_idx)
                ax.imshow(frame, cmap="gray", aspect="auto", vmin=0, vmax=1)

                side = "Left" if zone_name.endswith("_L") else "Right"
                zone_short = zone_name.replace("_L", "").replace("_R", "")
                path_name = pathology.name.replace("_", " ").title()
                ax.set_title(f"{side} {zone_short}\n{path_name}", fontsize=9, fontweight="bold")
                ax.axhline(y=self.frame_gen.pleural_row, color="cyan",
                          linewidth=0.3, alpha=0.3, linestyle="--")

        fig.suptitle(f"BLUE Protocol Exam: {scenario['name']}\n{scenario['description']}",
                    fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        return fig

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save_figure(self, fig: plt.Figure, output_path: str) -> str:
        """Save figure to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(path)

    def export_all_previews(self, output_dir: str = "data/training_previews", seed: int = 42):
        """
        Export preview images for all pathologies and scenarios.

        Creates:
          output_dir/
            pathology_grid.png
            bmode_mmode_grid.png
            individual/{pathology_name}.png
            scenarios/{scenario_name}.png
        """
        out = Path(output_dir)

        # Full comparison grid
        fig = self.render_comparison_grid(seed=seed)
        self.save_figure(fig, out / "pathology_grid.png")
        print(f"  Saved: {out / 'pathology_grid.png'}")

        # B-mode + M-mode grid
        fig = self.render_bmode_mmode_grid(seed=seed)
        self.save_figure(fig, out / "bmode_mmode_grid.png")
        print(f"  Saved: {out / 'bmode_mmode_grid.png'}")

        # Individual pathology previews
        for pathology in ClinicalPathology:
            fig = self.render_single(pathology, seed=seed)
            name = pathology.name.lower()
            self.save_figure(fig, out / "individual" / f"{name}.png")
            print(f"  Saved: {out / 'individual' / f'{name}.png'}")

        # Scenario previews
        for scenario_key in CLINICAL_SCENARIOS:
            fig = self.render_scenario(scenario_key, seed=seed)
            self.save_figure(fig, out / "scenarios" / f"{scenario_key}.png")
            print(f"  Saved: {out / 'scenarios' / f'{scenario_key}.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preview synthetic lung POCUS training images"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Render comparison grid of all pathologies",
    )
    parser.add_argument(
        "--pathology", type=str, default=None,
        choices=[p.name.lower() for p in ClinicalPathology],
        help="Render a specific pathology preview",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=list(CLINICAL_SCENARIOS.keys()),
        help="Render a clinical scenario (8-zone exam)",
    )
    parser.add_argument(
        "--bmode-mmode", action="store_true",
        help="Render B-mode + M-mode comparison grid",
    )
    parser.add_argument(
        "--export-all", action="store_true",
        help="Export all preview images to data/training_previews/",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (PNG or PDF)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=[256, 256],
        help="Image dimensions (H W)",
    )
    args = parser.parse_args()

    renderer = TrainingPreviewRenderer(
        image_size=tuple(args.image_size),
    )

    if args.export_all:
        print("Exporting all training previews...")
        renderer.export_all_previews(seed=args.seed)
        print("Done!")
        return

    fig = None

    if args.pathology:
        pathology = ClinicalPathology[args.pathology.upper()]
        fig = renderer.render_single(pathology, seed=args.seed)
    elif args.scenario:
        fig = renderer.render_scenario(args.scenario, seed=args.seed)
    elif args.bmode_mmode:
        fig = renderer.render_bmode_mmode_grid(seed=args.seed)
    elif args.all:
        fig = renderer.render_comparison_grid(seed=args.seed)
    else:
        # Default: show comparison grid
        fig = renderer.render_comparison_grid(seed=args.seed)

    if fig is not None:
        if args.output:
            path = renderer.save_figure(fig, args.output)
            print(f"Saved to: {path}")
        else:
            # Save to default location
            path = renderer.save_figure(fig, "data/training_preview.png")
            print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
