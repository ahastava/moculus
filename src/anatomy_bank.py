"""
Lesion-Anatomy Bank
====================
Implements the DiffUltra concept of a Lesion-Anatomy Bank adapted for
MoCoLUS's existing structural generation pipeline.

Two components:
  1. **Lesion Foreground Bank**: Extracts and indexes real lesion textures
     from the processed POCUS dataset, organized by pathology class and
     spatial position relative to the pleural line.

  2. **Joint PMF Conditioner**: Builds a probability mass function
     P(ΔX, ΔY | class) that models where lesions appear relative to
     anatomical landmarks (pleural line). During generation, lesion
     placement is sampled from this PMF to ensure anatomical realism
     (e.g., consolidations anchor to the pleural line, effusions sit
     above the diaphragm).

Reference:
  Chou et al., "Ultrasound Image Synthesis Using Generative AI for
  Lung Ultrasound Detection" (ISBI 2025, arXiv: 2501.06356)

Usage:
    from src.anatomy_bank import LesionAnatomyBank

    bank = LesionAnatomyBank.from_dataset("data/real_pocus/processed")
    bank.save("checkpoints/anatomy_bank.pt")

    # During generation:
    bank = LesionAnatomyBank.load("checkpoints/anatomy_bank.pt")
    texture = bank.sample_lesion_texture(pathology_class=4, seed=42)
    dy, dx = bank.sample_lesion_position(pathology_class=4, seed=42)
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Pathology classes where lesion extraction is meaningful
# (these have spatially localized features relative to the pleural line)
LESION_CLASSES = {
    3: "b_lines_diffuse",
    4: "consolidation",
    5: "pleural_effusion",
    6: "ards_white_lung",
    8: "pleural_thickening",
    9: "interstitial_syndrome",
}

# Approximate pleural line position as fraction of image height.
# Real POCUS: pleural line is typically at 15-25% of image depth.
PLEURAL_LINE_FRAC = 0.17

# Spatial grid for PMF (rows x cols relative to pleural line)
PMF_GRID_ROWS = 8   # depth bins below pleural line
PMF_GRID_COLS = 8   # lateral bins


class LesionAnatomyBank:
    """
    Stores extracted lesion textures and spatial PMFs from real POCUS data.

    The bank indexes lesion foregrounds by:
      - pathology_class: which pathology
      - depth_bin: how far below the pleural line (0 = at pleura, 7 = deep)
      - lateral_bin: position along the lateral axis

    The joint PMF P(depth_bin, lateral_bin | class) encodes where each
    pathology type typically appears, ensuring synthetic lesions are placed
    in anatomically plausible locations.
    """

    def __init__(
        self,
        textures: Dict[int, List[np.ndarray]],
        pmfs: Dict[int, np.ndarray],
        patch_size: int = 64,
    ):
        """
        Args:
            textures: {class_id: [array of extracted patches]}
            pmfs: {class_id: [PMF_GRID_ROWS, PMF_GRID_COLS] probability array}
            patch_size: size of extracted texture patches
        """
        self.textures = textures
        self.pmfs = pmfs
        self.patch_size = patch_size

    @classmethod
    def from_dataset(
        cls,
        processed_dir: str,
        patch_size: int = 64,
        max_patches_per_class: int = 500,
    ) -> "LesionAnatomyBank":
        """
        Build the anatomy bank from the processed POCUS dataset.

        Scans all images in the dataset, detects the pleural line position,
        and extracts sub-pleural texture patches for lesion classes. Also
        builds the joint PMF from the spatial distribution of high-intensity
        regions in each pathology.

        Args:
            processed_dir: Path to data/real_pocus/processed/
            patch_size: Size of texture patches to extract
            max_patches_per_class: Cap on patches per class (memory limit)
        """
        processed = Path(processed_dir)
        metadata_path = processed / "metadata.csv"
        images_dir = processed / "images"

        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.csv at {metadata_path}")

        # Read metadata, group files by lesion class
        class_files: Dict[int, List[str]] = {c: [] for c in LESION_CLASSES}
        with open(metadata_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = int(row["pathology_class"])
                if cls in LESION_CLASSES:
                    class_files[cls].append(row["filename"])

        textures: Dict[int, List[np.ndarray]] = {c: [] for c in LESION_CLASSES}
        spatial_hits: Dict[int, np.ndarray] = {
            c: np.zeros((PMF_GRID_ROWS, PMF_GRID_COLS), dtype=np.float64)
            for c in LESION_CLASSES
        }

        for cls, files in class_files.items():
            n_extracted = 0
            for fname in files:
                if n_extracted >= max_patches_per_class:
                    break

                img_path = images_dir / fname
                if not img_path.exists():
                    continue

                img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
                h, w = img.shape

                # Detect pleural line: brightest horizontal band in upper 40%
                pleural_row = _detect_pleural_line(img)
                if pleural_row is None or pleural_row >= h * 0.5:
                    continue  # Skip if pleural line not found

                # Extract sub-pleural region
                sub_pleural = img[pleural_row:, :]
                sp_h, sp_w = sub_pleural.shape

                if sp_h < patch_size or sp_w < patch_size:
                    continue

                # Find high-intensity regions (lesion candidates)
                # Use adaptive threshold: regions brighter than mean + 0.5*std
                threshold = sub_pleural.mean() + 0.5 * sub_pleural.std()
                bright_mask = sub_pleural > threshold

                # Record spatial distribution for PMF
                for dr in range(PMF_GRID_ROWS):
                    for dc in range(PMF_GRID_COLS):
                        r0 = int(dr * sp_h / PMF_GRID_ROWS)
                        r1 = int((dr + 1) * sp_h / PMF_GRID_ROWS)
                        c0 = int(dc * sp_w / PMF_GRID_COLS)
                        c1 = int((dc + 1) * sp_w / PMF_GRID_COLS)
                        cell = bright_mask[r0:r1, c0:c1]
                        spatial_hits[cls][dr, dc] += cell.sum()

                # Extract texture patches from sub-pleural region
                patches = _extract_patches(sub_pleural, patch_size, max_per_image=3)
                for p in patches:
                    if n_extracted >= max_patches_per_class:
                        break
                    textures[cls].append(p)
                    n_extracted += 1

            logger.info(
                f"Class {cls} ({LESION_CLASSES[cls]}): "
                f"extracted {len(textures[cls])} patches from {len(files)} images"
            )

        # Build PMFs from spatial hit counts
        pmfs = {}
        for class_id in LESION_CLASSES:
            hits = spatial_hits[class_id]
            total = hits.sum()
            if total > 0:
                pmf = hits / total
            else:
                # Uniform fallback
                pmf = np.ones_like(hits) / hits.size

            # Apply class-specific anatomical priors
            pmf = _apply_anatomical_prior(pmf, class_id)
            pmfs[class_id] = pmf

        return LesionAnatomyBank(textures=textures, pmfs=pmfs, patch_size=patch_size)

    def sample_lesion_position(
        self,
        pathology_class: int,
        image_size: Tuple[int, int] = (256, 256),
        pleural_row: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Sample an anatomically plausible (row, col) position for a lesion.

        Returns pixel coordinates in the output image where the lesion
        should be anchored, sampled from the learned PMF.

        Args:
            pathology_class: Pathology type
            image_size: (H, W) of the target image
            pleural_row: Row of the pleural line (auto-detected if None)
            seed: Random seed

        Returns:
            (row, col) in pixel coordinates
        """
        rng = np.random.default_rng(seed)
        H, W = image_size

        if pleural_row is None:
            pleural_row = int(PLEURAL_LINE_FRAC * H)

        if pathology_class not in self.pmfs:
            # Non-lesion class: return center of sub-pleural region
            row = pleural_row + (H - pleural_row) // 2
            col = W // 2
            return row, col

        pmf = self.pmfs[pathology_class]
        flat_pmf = pmf.ravel()

        # Sample a grid cell from the PMF
        cell_idx = rng.choice(len(flat_pmf), p=flat_pmf)
        grid_row = cell_idx // PMF_GRID_COLS
        grid_col = cell_idx % PMF_GRID_COLS

        # Convert grid cell to pixel coordinates (sub-pleural region)
        sub_h = H - pleural_row
        row = pleural_row + int((grid_row + rng.random()) * sub_h / PMF_GRID_ROWS)
        col = int((grid_col + rng.random()) * W / PMF_GRID_COLS)

        row = min(max(pleural_row, row), H - 1)
        col = min(max(0, col), W - 1)

        return row, col

    def sample_lesion_texture(
        self,
        pathology_class: int,
        seed: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Sample a random lesion texture patch for the given pathology.

        Returns:
            [patch_size, patch_size] float32 array, or None if no patches exist.
        """
        rng = np.random.default_rng(seed)

        if pathology_class not in self.textures or not self.textures[pathology_class]:
            return None

        patches = self.textures[pathology_class]
        idx = rng.integers(0, len(patches))
        patch = patches[idx].copy()

        # Random augmentation: flip, rotate, slight intensity jitter
        if rng.random() > 0.5:
            patch = patch[:, ::-1].copy()
        if rng.random() > 0.5:
            patch = patch[::-1, :].copy()
        # Intensity jitter
        patch *= rng.uniform(0.85, 1.15)
        patch = np.clip(patch, 0, 1)

        return patch

    def save(self, path: str) -> None:
        """Save the anatomy bank to disk."""
        import torch
        save_dict = {
            "textures": {k: [t.astype(np.float16) for t in v] for k, v in self.textures.items()},
            "pmfs": self.pmfs,
            "patch_size": self.patch_size,
        }
        torch.save(save_dict, path)
        total_patches = sum(len(v) for v in self.textures.values())
        logger.info(f"Anatomy bank saved: {total_patches} patches → {path}")

    @classmethod
    def load(cls, path: str) -> "LesionAnatomyBank":
        """Load a saved anatomy bank."""
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        textures = {
            k: [t.astype(np.float32) for t in v]
            for k, v in data["textures"].items()
        }
        return cls(
            textures=textures,
            pmfs=data["pmfs"],
            patch_size=data["patch_size"],
        )

    def summary(self) -> str:
        """Print a summary of the bank contents."""
        lines = ["Lesion-Anatomy Bank Summary", "=" * 40]
        total = 0
        for cls in sorted(LESION_CLASSES):
            n = len(self.textures.get(cls, []))
            total += n
            pmf = self.pmfs.get(cls)
            entropy = -np.sum(pmf * np.log(pmf + 1e-10)) if pmf is not None else 0
            lines.append(
                f"  Class {cls} ({LESION_CLASSES[cls]:>25}): "
                f"{n:>4} patches, PMF entropy={entropy:.2f}"
            )
        lines.append(f"  Total: {total} patches, patch_size={self.patch_size}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _detect_pleural_line(img: np.ndarray) -> Optional[int]:
    """
    Detect the pleural line row in a grayscale LUS image.

    The pleural line is the brightest continuous horizontal band in the
    upper portion of the image (typically 10-35% depth).

    Returns the row index, or None if not detected.
    """
    h, w = img.shape
    search_top = int(0.08 * h)
    search_bot = int(0.40 * h)

    if search_bot <= search_top:
        return int(PLEURAL_LINE_FRAC * h)

    # Compute mean intensity per row in the search region
    row_means = img[search_top:search_bot, :].mean(axis=1)

    # Smooth to avoid noise peaks
    from scipy.ndimage import gaussian_filter1d
    row_means_smooth = gaussian_filter1d(row_means, sigma=2)

    # Pleural line = brightest row
    peak_row = search_top + int(np.argmax(row_means_smooth))
    return peak_row


def _extract_patches(
    region: np.ndarray,
    patch_size: int,
    max_per_image: int = 3,
) -> List[np.ndarray]:
    """
    Extract interesting texture patches from a sub-pleural region.

    Selects patches with high variance (textured, not empty).
    """
    h, w = region.shape
    patches = []

    if h < patch_size or w < patch_size:
        return patches

    # Sample candidate positions
    rng = np.random.default_rng()
    n_candidates = min(10, max_per_image * 3)

    for _ in range(n_candidates):
        if len(patches) >= max_per_image:
            break

        r = rng.integers(0, h - patch_size)
        c = rng.integers(0, w - patch_size)
        patch = region[r:r + patch_size, c:c + patch_size].copy()

        # Skip low-variance patches (empty/shadow regions)
        if patch.std() < 0.03:
            continue

        patches.append(patch.astype(np.float32))

    return patches


def _apply_anatomical_prior(pmf: np.ndarray, pathology_class: int) -> np.ndarray:
    """
    Apply class-specific anatomical priors to the PMF.

    These encode clinical knowledge about where lesions appear:
      - Consolidations: anchor to pleural line (top rows)
      - Effusions: dependent (bottom rows, gravity)
      - B-lines: arise from pleural line, extend vertically (all depths)
      - ARDS: diffuse across entire sub-pleural space
    """
    prior = np.ones_like(pmf)

    if pathology_class == 4:  # Consolidation
        # Must anchor to pleural line — heavily weight top rows
        for r in range(PMF_GRID_ROWS):
            prior[r, :] = max(0.1, 1.0 - 0.8 * (r / PMF_GRID_ROWS))

    elif pathology_class == 5:  # Pleural effusion
        # Gravity-dependent: weight bottom rows
        for r in range(PMF_GRID_ROWS):
            prior[r, :] = 0.2 + 0.8 * (r / PMF_GRID_ROWS)
        # Slight lateral centering (between rib shadows)
        for c in range(PMF_GRID_COLS):
            dist_from_center = abs(c - PMF_GRID_COLS / 2) / (PMF_GRID_COLS / 2)
            prior[:, c] *= max(0.3, 1.0 - 0.5 * dist_from_center)

    elif pathology_class == 6:  # ARDS
        # Diffuse — near uniform, slight pleural weighting
        prior[0, :] = 1.2
        prior[1, :] = 1.1

    elif pathology_class == 8:  # Pleural thickening
        # At the pleural line level
        prior[0, :] = 3.0
        prior[1, :] = 1.5
        for r in range(2, PMF_GRID_ROWS):
            prior[r, :] = 0.2

    elif pathology_class in (3, 9):  # Diffuse B-lines, interstitial
        # B-lines: uniform lateral, extend from pleura to bottom
        prior[0, :] = 1.5  # Originate at pleura

    # Combine data-driven PMF with anatomical prior (Bayesian fusion)
    combined = pmf * prior
    total = combined.sum()
    if total > 0:
        combined /= total
    else:
        combined = np.ones_like(pmf) / pmf.size

    return combined
