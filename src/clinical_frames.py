"""
Clinically Accurate Lung POCUS Frame Generator
================================================
Generates realistic synthetic B-mode frames for each lung POCUS pathology,
modeled on real ultrasound physics and clinical appearance.

Improvements over base LungUSArtifactSynthesizer:
  - Layered tissue anatomy (skin → subcutaneous fat → muscle → rib/pleura)
  - Rib acoustic shadowing
  - Tissue echotexture (scatterer-based speckle)
  - Depth-dependent attenuation (TGC curve)
  - New pathologies: pleural effusion, consolidation, lung point
  - Clinically accurate B-line ring-down artifacts
  - Proper A-line equidistant spacing with decreasing intensity
  - M-mode seashore vs stratosphere patterns

References:
  Lichtenstein DA. Lung Ultrasound in the Critically Ill (2016)
  Volpicelli G et al. International evidence-based recommendations
    for point-of-care lung ultrasound. Intensive Care Med (2012)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
from enum import IntEnum


# ---------------------------------------------------------------------------
# Clinical Pathology Enum (extended)
# ---------------------------------------------------------------------------

class ClinicalPathology(IntEnum):
    """Extended pathology set for clinically accurate POCUS training."""
    NORMAL_A_PROFILE = 0       # Normal: A-lines + lung sliding
    PNEUMOTHORAX = 1           # A-lines, NO lung sliding (stratosphere M-mode)
    B_LINES_FOCAL = 2          # Focal B-lines (1-2, e.g. early CHF or atelectasis)
    B_LINES_DIFFUSE = 3        # Diffuse B-lines (≥3, pulmonary edema / B-profile)
    CONSOLIDATION = 4          # Hepatized lung tissue (tissue-like echo pattern)
    PLEURAL_EFFUSION = 5       # Anechoic fluid above diaphragm
    ARDS_WHITE_LUNG = 6        # Confluent B-lines / "white lung"
    LUNG_POINT = 7             # Transition point: sliding ↔ no sliding (pneumothorax border)
    PLEURAL_THICKENING = 8     # Irregular, thickened pleural line
    INTERSTITIAL_SYNDROME = 9  # Multiple B-lines with pleural irregularities


CLINICAL_PATHOLOGY_NAMES = {int(p): p.name.lower() for p in ClinicalPathology}


# ---------------------------------------------------------------------------
# Tissue Layer Model
# ---------------------------------------------------------------------------

@dataclass
class TissueLayer:
    """A single tissue layer with ultrasound properties."""
    name: str
    thickness_m: float          # Layer thickness in meters
    echogenicity: float         # Mean brightness (0-1)
    scatter_density: float      # Scatterer density (0-1, higher = more texture)
    attenuation_db_per_cm: float  # Frequency-dependent attenuation
    speed_of_sound: float = 1540.0  # m/s


# Standard anterior chest wall tissue stack (from skin to pleura)
CHEST_WALL_LAYERS = [
    TissueLayer("skin",              thickness_m=0.002, echogenicity=0.7,  scatter_density=0.3, attenuation_db_per_cm=1.0),
    TissueLayer("subcutaneous_fat",  thickness_m=0.008, echogenicity=0.35, scatter_density=0.5, attenuation_db_per_cm=0.6),
    TissueLayer("intercostal_muscle",thickness_m=0.008, echogenicity=0.45, scatter_density=0.7, attenuation_db_per_cm=1.0),
]
# Total chest wall ~ 1.8 cm to pleural line (2.0 cm including pleura)

PLEURAL_DEPTH_M = 0.020  # Standard pleural line depth


# ---------------------------------------------------------------------------
# Clinical Frame Generator
# ---------------------------------------------------------------------------

class ClinicalFrameGenerator:
    """
    Generates clinically accurate B-mode ultrasound frames for lung POCUS.

    Each frame includes:
      1. Layered chest wall tissue with realistic echotexture
      2. Rib shadows (when applicable)
      3. Pleural line (bright, hyperechoic)
      4. Sub-pleural pathology-specific patterns
      5. Depth-dependent attenuation + TGC compensation
      6. Physically correct speckle noise model

    Usage:
        gen = ClinicalFrameGenerator()
        frame = gen.generate(ClinicalPathology.NORMAL_A_PROFILE)
        # frame is np.ndarray [256, 256] float32 in [0, 1]
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        depth_range_m: Tuple[float, float] = (0.0, 0.12),
        center_freq_mhz: float = 5.0,
        chest_wall_layers: Optional[List[TissueLayer]] = None,
        pleural_depth_m: float = PLEURAL_DEPTH_M,
    ):
        self.H, self.W = image_size
        self.depth_min, self.depth_max = depth_range_m
        self.freq_mhz = center_freq_mhz
        self.layers = chest_wall_layers or CHEST_WALL_LAYERS
        self.pleural_depth = pleural_depth_m

        # Derived
        self.depth_axis = np.linspace(self.depth_min, self.depth_max, self.H)
        self.lateral_axis = np.linspace(-0.04, 0.04, self.W)  # ±4 cm
        self.m_per_px_axial = (self.depth_max - self.depth_min) / self.H
        self.m_per_px_lateral = 0.08 / self.W
        self.pleural_row = int(self.pleural_depth / self.m_per_px_axial)

    # ------------------------------------------------------------------
    # Tissue echotexture
    # ------------------------------------------------------------------

    def _generate_tissue_texture(self, rng: np.random.Generator) -> np.ndarray:
        """
        Generate chest wall tissue layers with realistic echotexture.
        Models scatterer-based speckle for each tissue type.
        """
        image = np.zeros((self.H, self.W), dtype=np.float64)
        current_depth = 0.0

        for layer in self.layers:
            start_row = max(0, int(current_depth / self.m_per_px_axial))
            end_row = min(self.H, int((current_depth + layer.thickness_m) / self.m_per_px_axial))

            if start_row >= self.H:
                break

            n_rows = end_row - start_row
            if n_rows <= 0:
                current_depth += layer.thickness_m
                continue

            # Base echogenicity
            tissue = np.full((n_rows, self.W), layer.echogenicity)

            # Scatterer texture: point scatterers convolved with PSF
            # Use moderate density for coherent tissue texture, not grain
            n_scatterers = int(layer.scatter_density * n_rows * self.W * 0.15)
            if n_scatterers > 0:
                scatter_field = np.zeros((n_rows, self.W))
                rows_s = rng.integers(0, n_rows, n_scatterers)
                cols_s = rng.integers(0, self.W, n_scatterers)
                amplitudes = rng.rayleigh(0.10, n_scatterers)
                np.add.at(scatter_field, (rows_s, cols_s), amplitudes)

                # Broader PSF for smoother, more coherent tissue texture
                axial_sigma = 1.8
                lateral_sigma = 2.5 + 0.8 * (start_row / max(1, self.H))
                scatter_field = gaussian_filter(scatter_field, sigma=[axial_sigma, lateral_sigma])
                tissue += scatter_field

            # Fascial boundaries: bright lines between layers
            if start_row > 0 and n_rows >= 2:
                tissue[0, :] += 0.2  # Interface echo

            image[start_row:end_row, :] = tissue
            current_depth += layer.thickness_m

        return image

    # ------------------------------------------------------------------
    # Pleural line
    # ------------------------------------------------------------------

    def _draw_pleural_line(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        irregular: bool = False,
        thickness_px: int = 3,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Draw the pleural line: a bright hyperechoic horizontal interface.

        Args:
            irregular: If True, draw with slight undulations (pleural thickening).
            thickness_px: Line thickness in pixels.
            intensity: Peak brightness.
        """
        p_row = self.pleural_row

        if irregular:
            # Irregular pleural line (thickening, fragmentation)
            for col in range(self.W):
                offset = int(rng.normal(0, 1.5))
                thick = rng.integers(2, 5)
                row = p_row + offset
                r0 = max(0, row - thick // 2)
                r1 = min(self.H, row + thick // 2 + 1)
                local_intensity = intensity * rng.uniform(0.7, 1.0)
                image[r0:r1, col] = np.maximum(image[r0:r1, col], local_intensity)
        else:
            # Normal: smooth, bright, thin
            half = thickness_px // 2
            r0 = max(0, p_row - half)
            r1 = min(self.H, p_row + half + 1)
            image[r0:r1, :] = np.maximum(image[r0:r1, :], intensity)

        return image

    # ------------------------------------------------------------------
    # Rib shadows
    # ------------------------------------------------------------------

    def _add_rib_shadows(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        n_ribs: int = 2,
    ) -> np.ndarray:
        """
        Add acoustic shadows from ribs — dark vertical bands flanking
        the intercostal window, starting above the pleural line.

        Ribs appear as bright curved arcs with posterior acoustic shadowing.
        """
        rib_width_px = int(0.012 / self.m_per_px_lateral)  # ~12mm rib width
        # Place ribs at edges of the image (intercostal window is between them)
        margin = self.W // 8
        rib_centers = [margin, self.W - margin]

        rib_top_row = max(0, int(0.01 / self.m_per_px_axial))  # ribs start ~1cm deep

        for center in rib_centers:
            half_w = rib_width_px // 2
            c0 = max(0, center - half_w)
            c1 = min(self.W, center + half_w)

            # Rib cortex: bright curved arc
            for col in range(c0, c1):
                dist_from_center = abs(col - center) / max(1, half_w)
                # Curved surface: depth increases toward edges
                arc_offset = int(3 * dist_from_center ** 2)
                rib_row = rib_top_row + arc_offset

                # Bright cortex (2-3 px thick)
                r0 = max(0, rib_row)
                r1 = min(self.H, rib_row + 3)
                image[r0:r1, col] = np.maximum(image[r0:r1, col], 0.85)

                # Posterior acoustic shadow (everything below rib is dark)
                shadow_start = rib_row + 3
                if shadow_start < self.H:
                    # Gradual shadow with slight noise
                    shadow_strength = 0.05 + 0.03 * rng.random()
                    image[shadow_start:, col] *= shadow_strength

        return image

    # ------------------------------------------------------------------
    # A-lines (reverberation artifacts)
    # ------------------------------------------------------------------

    def _draw_a_lines(
        self,
        image: np.ndarray,
        n_reverberations: int = 4,
        base_intensity: float = 0.7,
    ) -> np.ndarray:
        """
        Draw A-lines: equidistant horizontal reverberation artifacts
        below the pleural line.

        Physics: Each A-line is at n * pleural_depth, with intensity
        decreasing as 1/n due to reflection losses.
        """
        for n in range(2, 2 + n_reverberations):
            refl_depth = n * self.pleural_depth
            if refl_depth >= self.depth_max:
                break

            r_idx = int(refl_depth / self.m_per_px_axial)
            if r_idx >= self.H - 1:
                break

            intensity = base_intensity / n
            # A-lines are slightly blurrier with each bounce
            thickness = min(2 + n // 2, 5)
            half = thickness // 2
            r0 = max(0, r_idx - half)
            r1 = min(self.H, r_idx + half + 1)
            image[r0:r1, :] = np.maximum(image[r0:r1, :], intensity)

        return image

    # ------------------------------------------------------------------
    # B-lines (comet-tail / ring-down artifacts)
    # ------------------------------------------------------------------

    def _draw_b_lines(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        n_b_lines: int = 3,
        width_mm: float = 3.0,
        intensity: float = 0.9,
    ) -> np.ndarray:
        """
        Draw B-lines: hyperechoic vertical artifacts arising from the
        pleural line and extending to the bottom of the image without
        fading (unlike comet-tail artifacts which fade).

        Clinical criteria for true B-lines (Volpicelli 2012):
          - Arise from the pleural line
          - Move with lung sliding
          - Extend to bottom of screen without fading
          - Are laser-like (narrow, well-defined)
          - Erase A-lines where they cross
        """
        # Distribute B-lines across the lateral width
        positions_m = rng.uniform(
            self.lateral_axis[self.W // 6],
            self.lateral_axis[-self.W // 6],
            n_b_lines,
        )
        # Sort for cleaner appearance
        positions_m.sort()

        width_px = max(2, int(width_mm * 0.001 / self.m_per_px_lateral))
        n_sub = self.H - self.pleural_row
        if n_sub <= 0:
            return image

        # Precompute depth ring-down modulation (vectorized)
        depth_fracs = np.linspace(0, 1, n_sub)
        ring_down = 0.88 + 0.12 * np.sin(2 * np.pi * depth_fracs * 10)

        for b_pos in positions_m:
            col_center = int((b_pos - self.lateral_axis[0]) / self.m_per_px_lateral)
            col_center = np.clip(col_center, width_px, self.W - width_px)

            # Lateral Gaussian profile (vectorized across columns)
            c0 = max(0, col_center - width_px * 3)
            c1 = min(self.W, col_center + width_px * 3)
            cols = np.arange(c0, c1)
            gauss = np.exp(-0.5 * ((cols - col_center) / max(1, width_px * 0.5)) ** 2)

            # B-line: [depth x lateral] intensity matrix
            b_patch = intensity * np.outer(ring_down, gauss)

            # Apply as max-blend (B-lines overlay, don't replace)
            region = image[self.pleural_row:, c0:c1]
            np.maximum(region, b_patch[:region.shape[0], :region.shape[1]], out=region)

            # Erase A-lines at crossing (clinical criterion)
            for n in range(2, 6):
                r_idx = int(n * self.pleural_depth / self.m_per_px_axial)
                if r_idx < self.H:
                    cc0 = max(0, col_center - width_px)
                    cc1 = min(self.W, col_center + width_px)
                    image[r_idx, cc0:cc1] = max(0.7, image[r_idx, col_center])

        return image

    # ------------------------------------------------------------------
    # Consolidation (hepatization)
    # ------------------------------------------------------------------

    def _draw_consolidation(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        extent_frac: float = 0.5,
    ) -> np.ndarray:
        """
        Draw lung consolidation: tissue-like echotexture below the
        pleural line (hepatized lung).

        Features:
          - Tissue-like echopattern (similar to liver)
          - Air bronchograms: bright punctate echoes within consolidated lung
          - Shred sign: irregular border at transition to aerated lung
          - No A-lines (aerated lung replaced)
        """
        consol_start = self.pleural_row + 2
        consol_depth = int(extent_frac * (self.H - self.pleural_row))
        consol_end = min(self.H, consol_start + consol_depth)

        # Tissue-like echotexture (hepatization)
        tissue_echo = rng.uniform(0.3, 0.5, (consol_end - consol_start, self.W))
        # Add scatterers for tissue texture
        n_scatterers = int(0.2 * (consol_end - consol_start) * self.W)
        rows_s = rng.integers(0, consol_end - consol_start, n_scatterers)
        cols_s = rng.integers(0, self.W, n_scatterers)
        amps = rng.rayleigh(0.12, n_scatterers)
        np.add.at(tissue_echo, (rows_s, cols_s), amps)
        tissue_echo = gaussian_filter(tissue_echo, sigma=[1.0, 1.5])

        image[consol_start:consol_end, :] = tissue_echo

        # Air bronchograms: bright punctate echoes (trapped air in bronchi)
        n_bronchograms = rng.integers(5, 15)
        for _ in range(n_bronchograms):
            br_row = rng.integers(consol_start + 5, max(consol_start + 6, consol_end - 5))
            br_col = rng.integers(self.W // 4, 3 * self.W // 4)
            br_size = rng.integers(1, 4)
            r0 = max(consol_start, br_row - br_size)
            r1 = min(consol_end, br_row + br_size)
            c0 = max(0, br_col - br_size)
            c1 = min(self.W, br_col + br_size)
            image[r0:r1, c0:c1] = rng.uniform(0.7, 0.95)

        # Shred sign: irregular border at bottom of consolidation
        for col in range(self.W):
            border_offset = int(rng.normal(0, 4))
            border_row = consol_end + border_offset
            if 0 < border_row < self.H - 2:
                # Ragged transition to aerated lung
                transition_width = rng.integers(2, 6)
                for dr in range(transition_width):
                    row = border_row + dr
                    if row < self.H:
                        fade = 1.0 - dr / transition_width
                        image[row, col] *= fade

        return image

    # ------------------------------------------------------------------
    # Pleural effusion
    # ------------------------------------------------------------------

    def _draw_pleural_effusion(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        effusion_depth_m: float = 0.025,
    ) -> np.ndarray:
        """
        Draw pleural effusion: anechoic (black) fluid collection
        above the diaphragm.

        Features:
          - Anechoic space between visceral and parietal pleura
          - Spine sign: vertebral bodies visible through fluid
          - Quad sign: bounded by pleural line above and lung below
          - May show internal echoes if complex/exudative
        """
        effusion_start = self.pleural_row + 2
        effusion_px = int(effusion_depth_m / self.m_per_px_axial)
        effusion_end = min(self.H, effusion_start + effusion_px)

        # Anechoic fluid (very dark with minimal internal echoes)
        fluid = np.full((effusion_end - effusion_start, self.W), 0.02)
        # Very faint internal swirling (fluid not perfectly anechoic)
        swirl = rng.normal(0, 0.01, fluid.shape)
        swirl = gaussian_filter(swirl, sigma=[3, 3])
        fluid += np.abs(swirl)
        fluid = np.clip(fluid, 0, 0.08)

        image[effusion_start:effusion_end, :] = fluid

        # Bright line at bottom of effusion (visceral pleura / compressed lung)
        if effusion_end < self.H - 2:
            image[effusion_end:effusion_end + 2, :] = 0.85

            # Compressed lung below: slightly echogenic with reduced A-lines
            lung_region = effusion_end + 2
            if lung_region < self.H:
                compressed = rng.uniform(0.15, 0.25, (self.H - lung_region, self.W))
                compressed = gaussian_filter(compressed, sigma=[1.5, 2.0])
                image[lung_region:, :] = compressed

        # Quad sign boundaries (bright lateral walls)
        for col in [self.W // 5, 4 * self.W // 5]:
            image[effusion_start:effusion_end, max(0, col-1):min(self.W, col+2)] = 0.6

        return image

    # ------------------------------------------------------------------
    # Lung point
    # ------------------------------------------------------------------

    def _draw_lung_point(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        transition_col: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw lung point: the transition zone between pneumothorax
        (no sliding) and normal lung (sliding).

        Left half: A-lines with no sliding (pneumothorax)
        Right half: Normal lung sliding pattern
        The transition column is where the visceral pleura reattaches.
        """
        if transition_col is None:
            transition_col = self.W // 2 + int(rng.normal(0, self.W // 10))
            transition_col = np.clip(transition_col, self.W // 4, 3 * self.W // 4)

        # Pneumothorax side (left): strong A-lines, no tissue texture below pleura
        for n in range(2, 6):
            refl_depth = n * self.pleural_depth
            r_idx = int(refl_depth / self.m_per_px_axial)
            if r_idx < self.H:
                intensity = 0.7 / n
                image[max(0, r_idx-1):min(self.H, r_idx+2), :transition_col] = \
                    np.maximum(image[max(0, r_idx-1):min(self.H, r_idx+2), :transition_col], intensity)

        # Normal side (right): subtle tissue texture below pleura
        sub_pleural_rows = self.H - self.pleural_row - 2
        if sub_pleural_rows > 0:
            tissue = rng.uniform(0.15, 0.3, (sub_pleural_rows, self.W - transition_col))
            tissue = gaussian_filter(tissue, sigma=[1.5, 2.0])
            image[self.pleural_row + 2:, transition_col:] = np.maximum(
                image[self.pleural_row + 2:, transition_col:], tissue
            )

        # Transition zone: slight pleural discontinuity
        trans_width = 5
        c0 = max(0, transition_col - trans_width)
        c1 = min(self.W, transition_col + trans_width)
        image[self.pleural_row - 1:self.pleural_row + 3, c0:c1] *= 0.5

        return image

    # ------------------------------------------------------------------
    # Noise & post-processing
    # ------------------------------------------------------------------

    def _apply_depth_attenuation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply depth-dependent attenuation and TGC (Time Gain Compensation).

        Modern US machines have effective TGC that largely compensates for
        attenuation. We model a mild residual depth effect for realism.
        """
        attenuation_db_per_cm = 0.3 * self.freq_mhz  # mild: ~1.5 dB/cm at 5 MHz
        depth_cm = self.depth_axis * 100  # (H,) array
        atten_db = attenuation_db_per_cm * depth_cm
        atten_linear = 10 ** (-atten_db / 20)

        # TGC compensation — nearly full compensation, slight residual
        t = np.linspace(0, 1, self.H)
        tgc_gain = 1.0 + 2.0 * t  # aggressive TGC ramp
        combined = (atten_linear * tgc_gain).reshape(-1, 1)
        image = image * combined

        return image

    def _apply_speckle_noise(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Apply clinically realistic speckle noise.
        Uses a mild multiplicative model + PSF smoothing for a clean,
        textured appearance matching modern US machines.
        """
        # Mild multiplicative speckle — enough for texture, not destructive
        speckle = rng.rayleigh(scale=0.55, size=image.shape)
        # Smooth the speckle field to create coherent texture patches
        speckle = gaussian_filter(speckle, sigma=[0.8, 1.0])
        image = image * speckle

        # Depth-dependent PSF smoothing (lateral resolution degrades with depth)
        step = max(1, self.H // 32)
        for row in range(0, self.H, step):
            sigma_lat = 0.6 + 1.0 * (row / self.H)
            end = min(row + step, self.H)
            image[row:end] = gaussian_filter(image[row:end], sigma=[0.3, sigma_lat])

        # Very light electronic noise floor
        image += rng.normal(0, 0.004, image.shape)

        return image

    def _log_compress(self, image: np.ndarray, dynamic_range_db: float = 55.0) -> np.ndarray:
        """Log compression with clinical-grade dynamic range and contrast."""
        image = np.clip(image, 1e-6, None)
        image_db = 20 * np.log10(image / (image.max() + 1e-10))
        image_db = np.clip(image_db, -dynamic_range_db, 0)
        compressed = (image_db + dynamic_range_db) / dynamic_range_db
        # Mild gamma to lift midtones (typical US display curve)
        compressed = np.power(compressed, 0.85).astype(np.float32)
        # Light smoothing for clean appearance
        compressed = gaussian_filter(compressed, sigma=0.35)
        return compressed

    # ------------------------------------------------------------------
    # Main pathology generators
    # ------------------------------------------------------------------

    def generate_normal_a_profile(self, rng: np.random.Generator) -> np.ndarray:
        """
        Normal lung: A-lines + lung sliding.
        BLUE protocol A-profile (bilateral) = no pulmonary edema.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng)
        image = self._draw_a_lines(image, n_reverberations=4, base_intensity=0.6)
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    def generate_pneumothorax(self, rng: np.random.Generator) -> np.ndarray:
        """
        Pneumothorax: A-lines, NO lung sliding (absent on M-mode = stratosphere).
        Strong, crisp A-lines due to highly reflective air-pleura interface.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng, intensity=1.0)
        # Stronger A-lines in pneumothorax (more reflective air interface)
        image = self._draw_a_lines(image, n_reverberations=5, base_intensity=0.8)
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    def generate_focal_b_lines(self, rng: np.random.Generator) -> np.ndarray:
        """
        Focal B-lines (1-2): can be normal variant or early interstitial disease.
        Up to 2 B-lines in a single intercostal space is within normal limits.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng)
        # A-lines still visible between B-lines
        image = self._draw_a_lines(image, n_reverberations=3, base_intensity=0.4)
        image = self._draw_b_lines(image, rng, n_b_lines=rng.integers(1, 3))
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    def generate_diffuse_b_lines(self, rng: np.random.Generator) -> np.ndarray:
        """
        Diffuse B-lines (≥3): B-profile, indicates pulmonary edema
        (cardiogenic) or other causes of interstitial syndrome.

        ≥3 B-lines in ≥2 zones bilaterally = interstitial syndrome.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng)
        # Multiple B-lines dominate — A-lines mostly erased
        image = self._draw_b_lines(image, rng, n_b_lines=rng.integers(3, 6))
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    def generate_consolidation(self, rng: np.random.Generator) -> np.ndarray:
        """
        Lung consolidation (hepatization): tissue-like pattern with
        air bronchograms and shred sign.

        Seen in pneumonia, atelectasis, or pulmonary infarction.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng, irregular=True)
        image = self._draw_consolidation(image, rng, extent_frac=rng.uniform(0.3, 0.6))
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    def generate_pleural_effusion(self, rng: np.random.Generator) -> np.ndarray:
        """
        Pleural effusion: anechoic fluid between visceral and parietal pleura.

        Quad sign: fluid bounded by pleural line above, lung/diaphragm below,
        and rib shadows laterally. Spine sign in larger effusions.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng)
        effusion_size = rng.uniform(0.015, 0.04)  # 1.5-4 cm
        image = self._draw_pleural_effusion(image, rng, effusion_depth_m=effusion_size)
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    def generate_ards_white_lung(self, rng: np.random.Generator) -> np.ndarray:
        """
        ARDS / white lung: confluent B-lines creating a uniformly
        bright appearance below the pleural line.

        Entire sub-pleural space appears "white" due to complete
        loss of A-lines and coalescence of B-lines.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng, irregular=True, intensity=0.85)
        # Many confluent B-lines
        n_lines = int(self.W * 0.12)
        image = self._draw_b_lines(image, rng, n_b_lines=n_lines, width_mm=5.0, intensity=0.85)
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    def generate_lung_point(self, rng: np.random.Generator) -> np.ndarray:
        """
        Lung point: pathognomonic sign for pneumothorax.
        Transition zone where pneumothorax meets normal pleural apposition.
        On M-mode, alternating seashore and stratosphere patterns.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng)
        image = self._draw_lung_point(image, rng)
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    def generate_pleural_thickening(self, rng: np.random.Generator) -> np.ndarray:
        """
        Pleural thickening: irregular, thickened pleural line with
        scattered sub-pleural abnormalities.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng, irregular=True, thickness_px=5, intensity=0.9)
        # Sub-pleural consolidations (small)
        image = self._draw_a_lines(image, n_reverberations=2, base_intensity=0.3)
        # A few B-lines from pleural irregularities
        image = self._draw_b_lines(image, rng, n_b_lines=rng.integers(1, 3), width_mm=2.0)
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    def generate_interstitial_syndrome(self, rng: np.random.Generator) -> np.ndarray:
        """
        Interstitial syndrome: multiple B-lines with pleural irregularities.
        Seen in pulmonary fibrosis, viral pneumonia (COVID), etc.
        """
        image = self._generate_tissue_texture(rng)
        image = self._draw_pleural_line(image, rng, irregular=True, thickness_px=4)
        image = self._draw_b_lines(image, rng, n_b_lines=rng.integers(4, 8), width_mm=2.5)
        # Small sub-pleural consolidations
        consol_cols = rng.integers(self.W // 4, 3 * self.W // 4, rng.integers(1, 4))
        for col in consol_cols:
            r0 = self.pleural_row + 3
            r1 = r0 + rng.integers(5, 15)
            c0 = max(0, col - rng.integers(3, 8))
            c1 = min(self.W, col + rng.integers(3, 8))
            image[r0:min(self.H, r1), c0:c1] = rng.uniform(0.4, 0.6)
        image = self._add_rib_shadows(image, rng)
        image = self._apply_depth_attenuation(image)
        image = self._apply_speckle_noise(image, rng)
        return self._log_compress(image)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def generate(
        self,
        pathology: ClinicalPathology,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a single clinically accurate B-mode frame.

        Args:
            pathology: Which clinical pathology to render.
            seed: Random seed for reproducibility.

        Returns:
            np.ndarray [H, W] float32 in [0, 1].
        """
        rng = np.random.default_rng(seed)

        generators = {
            ClinicalPathology.NORMAL_A_PROFILE: self.generate_normal_a_profile,
            ClinicalPathology.PNEUMOTHORAX: self.generate_pneumothorax,
            ClinicalPathology.B_LINES_FOCAL: self.generate_focal_b_lines,
            ClinicalPathology.B_LINES_DIFFUSE: self.generate_diffuse_b_lines,
            ClinicalPathology.CONSOLIDATION: self.generate_consolidation,
            ClinicalPathology.PLEURAL_EFFUSION: self.generate_pleural_effusion,
            ClinicalPathology.ARDS_WHITE_LUNG: self.generate_ards_white_lung,
            ClinicalPathology.LUNG_POINT: self.generate_lung_point,
            ClinicalPathology.PLEURAL_THICKENING: self.generate_pleural_thickening,
            ClinicalPathology.INTERSTITIAL_SYNDROME: self.generate_interstitial_syndrome,
        }

        return generators[pathology](rng)

    def generate_all(
        self, seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Generate one frame for each pathology type.

        Returns:
            Dict mapping pathology name → [H, W] float32 frame.
        """
        frames = {}
        for i, pathology in enumerate(ClinicalPathology):
            frames[pathology.name.lower()] = self.generate(pathology, seed=seed + i)
        return frames


# ---------------------------------------------------------------------------
# Temporal Stack with Clinical Accuracy
# ---------------------------------------------------------------------------

class ClinicalTemporalStack:
    """
    Generates temporal stacks with clinically accurate lung sliding
    and respiratory motion for each pathology.
    """

    def __init__(
        self,
        frame_gen: Optional[ClinicalFrameGenerator] = None,
        n_frames: int = 32,
        fps: float = 30.0,
        sliding_amplitude_px: float = 2.5,
        respiratory_rate_hz: float = 0.25,
    ):
        self.gen = frame_gen or ClinicalFrameGenerator()
        self.n_frames = n_frames
        self.fps = fps
        self.sliding_amp = sliding_amplitude_px
        self.resp_rate = respiratory_rate_hz

    def generate_stack(
        self,
        pathology: ClinicalPathology,
        lung_sliding: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a temporal B-mode stack + M-mode strip.

        Returns:
            Dict with:
                bmode_stack: [n_frames, H, W]
                mmode: [H, n_frames]
                mmode_pattern: "seashore" or "stratosphere"
        """
        rng = np.random.default_rng(seed)
        H, W = self.gen.H, self.gen.W
        stack = np.zeros((self.n_frames, H, W), dtype=np.float32)

        for i in range(self.n_frames):
            t = i / self.fps
            # Each frame has independent speckle realization
            frame = self.gen.generate(pathology, seed=(seed or 0) + i * 1000)

            if lung_sliding:
                frame = self._apply_sliding(frame, t, H, W)

            frame = self._apply_breathing(frame, t, H, W)
            stack[i] = frame

        # M-mode extraction (center column over time)
        mmode = stack[:, :, W // 2].T  # [H, n_frames]

        # Classify M-mode
        mmode_pattern = self._classify_mmode(mmode, H)

        return {
            "bmode_stack": stack,
            "mmode": mmode.astype(np.float32),
            "mmode_pattern": mmode_pattern,
        }

    def _apply_sliding(self, frame: np.ndarray, t: float, H: int, W: int) -> np.ndarray:
        """Apply lung sliding motion to sub-pleural rows."""
        import math
        pleural_row = self.gen.pleural_row
        shift = self.sliding_amp * math.sin(2 * math.pi * self.resp_rate * t)

        result = frame.copy()
        for row in range(pleural_row, H):
            depth_factor = 1.0 + 0.3 * ((row - pleural_row) / max(1, H - pleural_row))
            row_shift = shift * depth_factor
            int_shift = int(math.floor(row_shift))
            frac = row_shift - int_shift
            if abs(int_shift) < W:
                result[row] = (
                    (1 - frac) * np.roll(frame[row], int_shift) +
                    frac * np.roll(frame[row], int_shift + (1 if row_shift >= 0 else -1))
                )
        return result

    def _apply_breathing(self, frame: np.ndarray, t: float, H: int, W: int) -> np.ndarray:
        """Apply respiratory depth modulation."""
        import math
        shift_m = 0.003 * math.sin(2 * math.pi * self.resp_rate * t)
        shift_rows = int(round(shift_m / self.gen.m_per_px_axial))
        if shift_rows != 0:
            frame = np.roll(frame, shift_rows, axis=0)
            if shift_rows > 0:
                frame[:shift_rows, :] = 0.0
            else:
                frame[shift_rows:, :] = 0.0
        return frame

    def _classify_mmode(self, mmode: np.ndarray, H: int) -> str:
        """Classify M-mode as seashore or stratosphere."""
        pleural_row = self.gen.pleural_row
        sub_pleural = mmode[pleural_row:pleural_row + int(H * 0.3), :]
        supra_pleural = mmode[:pleural_row, :]
        temporal_var = np.var(sub_pleural, axis=1).mean()
        supra_var = np.var(supra_pleural, axis=1).mean()
        ratio = temporal_var / (supra_var + 1e-8)
        return "seashore" if ratio > 1.5 else "stratosphere"


# ---------------------------------------------------------------------------
# Extended Clinical Scenarios (using new pathology types)
# ---------------------------------------------------------------------------

CLINICAL_SCENARIOS = {
    "normal": {
        "name": "Normal Bilateral A-Profile",
        "description": "Bilateral lung sliding with A-lines at all BLUE points. Normal exam.",
        "all_zones": ClinicalPathology.NORMAL_A_PROFILE,
        "sliding": True,
    },
    "left_tension_pneumothorax": {
        "name": "Left Tension Pneumothorax",
        "description": "Absent lung sliding + A-lines on left (stratosphere M-mode). Right normal.",
        "left_zones": ClinicalPathology.PNEUMOTHORAX,
        "right_zones": ClinicalPathology.NORMAL_A_PROFILE,
        "left_sliding": False,
        "right_sliding": True,
    },
    "bilateral_pulmonary_edema": {
        "name": "Bilateral Pulmonary Edema (B-Profile)",
        "description": "≥3 B-lines bilaterally at all anterior zones. Lung sliding preserved.",
        "all_zones": ClinicalPathology.B_LINES_DIFFUSE,
        "sliding": True,
    },
    "left_pneumonia": {
        "name": "Left Pneumonia with Consolidation",
        "description": "Left PLAPS consolidation with ipsilateral B-lines. Right normal.",
        "zones": {
            "UPPER_BLUE_L": ClinicalPathology.B_LINES_FOCAL,
            "LOWER_BLUE_L": ClinicalPathology.B_LINES_DIFFUSE,
            "PLAPS_L": ClinicalPathology.CONSOLIDATION,
            "DIAPHRAGM_L": ClinicalPathology.PLEURAL_EFFUSION,
            "UPPER_BLUE_R": ClinicalPathology.NORMAL_A_PROFILE,
            "LOWER_BLUE_R": ClinicalPathology.NORMAL_A_PROFILE,
            "PLAPS_R": ClinicalPathology.NORMAL_A_PROFILE,
            "DIAPHRAGM_R": ClinicalPathology.NORMAL_A_PROFILE,
        },
        "sliding": True,
    },
    "ards": {
        "name": "ARDS (Diffuse White Lung)",
        "description": "Bilateral white lung with reduced sliding. Severe interstitial syndrome.",
        "all_zones": ClinicalPathology.ARDS_WHITE_LUNG,
        "sliding": False,
    },
    "right_pleural_effusion": {
        "name": "Right Pleural Effusion",
        "description": "Moderate right-sided pleural effusion with normal left lung.",
        "zones": {
            "UPPER_BLUE_R": ClinicalPathology.NORMAL_A_PROFILE,
            "LOWER_BLUE_R": ClinicalPathology.B_LINES_FOCAL,
            "PLAPS_R": ClinicalPathology.PLEURAL_EFFUSION,
            "DIAPHRAGM_R": ClinicalPathology.PLEURAL_EFFUSION,
            "UPPER_BLUE_L": ClinicalPathology.NORMAL_A_PROFILE,
            "LOWER_BLUE_L": ClinicalPathology.NORMAL_A_PROFILE,
            "PLAPS_L": ClinicalPathology.NORMAL_A_PROFILE,
            "DIAPHRAGM_L": ClinicalPathology.NORMAL_A_PROFILE,
        },
        "sliding": True,
    },
    "covid_pneumonia": {
        "name": "COVID-19 Viral Pneumonia",
        "description": "Bilateral pleural irregularities, subpleural consolidations, B-lines.",
        "all_zones": ClinicalPathology.INTERSTITIAL_SYNDROME,
        "sliding": True,
    },
    "lung_point_left": {
        "name": "Left Pneumothorax with Lung Point",
        "description": "Partial left pneumothorax — lung point visible at transition zone.",
        "zones": {
            "UPPER_BLUE_L": ClinicalPathology.PNEUMOTHORAX,
            "LOWER_BLUE_L": ClinicalPathology.LUNG_POINT,
            "PLAPS_L": ClinicalPathology.NORMAL_A_PROFILE,
            "DIAPHRAGM_L": ClinicalPathology.NORMAL_A_PROFILE,
            "UPPER_BLUE_R": ClinicalPathology.NORMAL_A_PROFILE,
            "LOWER_BLUE_R": ClinicalPathology.NORMAL_A_PROFILE,
            "PLAPS_R": ClinicalPathology.NORMAL_A_PROFILE,
            "DIAPHRAGM_R": ClinicalPathology.NORMAL_A_PROFILE,
        },
        "left_sliding": {"UPPER_BLUE_L": False, "LOWER_BLUE_L": False},
        "right_sliding": True,
    },
}
