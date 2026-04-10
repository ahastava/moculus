"""
MoCoLUS Point-of-Care Lung US Image Stack Generator
====================================================
Generates spatially-indexed temporal image stacks for ultrasound simulator
training, mapped to BLUE protocol lung exam zones with probe coordinate
tracking.

Architecture:
    Probe (x, y) on mannequin chest
         │
         ▼
    ZoneResolver ──► Anatomical Zone (BLUE point, PLAPS, etc.)
         │
         ▼
    ScenarioEngine ──► Pathology assignment per zone
         │
         ▼
    StackGenerator ──► Temporal B-mode stack + M-mode strip
         │
         ▼
    SimulatorExport ──► Packaged HDF5 with coordinate metadata

Coordinate System (patient supine, anterior view):
    Origin: sternal notch
    +X: patient's left (viewer's right)
    +Y: caudal (toward feet)
    +Z: posterior (into chest wall)
    Units: meters

BLUE Protocol Zones (per hemithorax):
    Upper BLUE point  — 2nd-3rd ICS, mid-clavicular line
    Lower BLUE point  — 4th-5th ICS, anterior axillary line
    PLAPS point       — posterolateral, above diaphragm
    Diaphragm point   — costophrenic angle region
"""

import math
import json
import numpy as np
import h5py
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from enum import IntEnum

try:
    from .lung_us_generator import (
        LungUSArtifactSynthesizer, DualIMUReading, IMUPose,
        GridConfig, PoseCorrector, PATHOLOGY_CLASSES, N_CLASSES,
    )
    from .clinical_frames import ClinicalFrameGenerator, ClinicalPathology
except ImportError:
    from lung_us_generator import (
        LungUSArtifactSynthesizer, DualIMUReading, IMUPose,
        GridConfig, PoseCorrector, PATHOLOGY_CLASSES, N_CLASSES,
    )
    from clinical_frames import ClinicalFrameGenerator, ClinicalPathology


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class LungZone(IntEnum):
    """BLUE protocol exam zones."""
    UPPER_BLUE_L = 0   # Left upper BLUE point
    UPPER_BLUE_R = 1   # Right upper BLUE point
    LOWER_BLUE_L = 2   # Left lower BLUE point
    LOWER_BLUE_R = 3   # Right lower BLUE point
    PLAPS_L = 4         # Left PLAPS point
    PLAPS_R = 5         # Right PLAPS point
    DIAPHRAGM_L = 6    # Left diaphragm
    DIAPHRAGM_R = 7    # Right diaphragm


class Pathology(IntEnum):
    """
    Maps 1:1 to ClinicalPathology in clinical_frames.py.
    Values MUST match ClinicalPathology enum for correct image rendering.
    """
    NORMAL_A_PROFILE = 0       # Normal: A-lines + lung sliding
    PNEUMOTHORAX = 1           # A-lines, NO lung sliding (stratosphere M-mode)
    B_LINES_FOCAL = 2          # Focal B-lines (1-2, early CHF / atelectasis)
    B_LINES_DIFFUSE = 3        # Diffuse B-lines (>=3, pulmonary edema / B-profile)
    CONSOLIDATION = 4          # Hepatized lung (air bronchograms, shred sign)
    PLEURAL_EFFUSION = 5       # Anechoic fluid above diaphragm
    ARDS_WHITE_LUNG = 6        # Confluent B-lines / "white lung"
    LUNG_POINT = 7             # Pneumothorax border (sliding <-> no sliding)
    PLEURAL_THICKENING = 8     # Irregular, thickened pleural line
    INTERSTITIAL_SYNDROME = 9  # Multiple B-lines + pleural irregularities


# ---------------------------------------------------------------------------
# Zone → ZoneRegion mapping
# ---------------------------------------------------------------------------
# Translates the 8 BLUE protocol LungZones into the 7 anatomical
# ZoneRegions used by the structural guide generator and the model's
# zone_embedding. Upper BLUE zones (L/R) collapse to a single
# UPPER_GENERIC slot since they share anatomy. Integer values match
# the ZoneRegion enum in clinical_frames.py — DO NOT renumber.

ZONE_REGION_MAP: Dict[LungZone, int] = {
    LungZone.UPPER_BLUE_L:  0,  # ZoneRegion.UPPER_GENERIC
    LungZone.UPPER_BLUE_R:  0,  # ZoneRegion.UPPER_GENERIC
    LungZone.LOWER_BLUE_L:  1,  # ZoneRegion.LOWER_BLUE_L
    LungZone.LOWER_BLUE_R:  2,  # ZoneRegion.LOWER_BLUE_R
    LungZone.PLAPS_L:       3,  # ZoneRegion.PLAPS_L
    LungZone.PLAPS_R:       4,  # ZoneRegion.PLAPS_R
    LungZone.DIAPHRAGM_L:   5,  # ZoneRegion.DIAPHRAGM_L
    LungZone.DIAPHRAGM_R:   6,  # ZoneRegion.DIAPHRAGM_R
}


# ---------------------------------------------------------------------------
# Chest Wall Coordinate Map
# ---------------------------------------------------------------------------

@dataclass
class ZoneAnchor:
    """
    Anatomical anchor point for a BLUE protocol zone on the chest wall.

    Coordinates in meters from sternal notch (patient supine, anterior view).
    radius_m defines the circular activation region around the anchor.
    """
    zone: LungZone
    x_m: float          # lateral (+ = patient left)
    y_m: float          # craniocaudal (+ = caudal)
    radius_m: float     # activation radius
    description: str = ""

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_m, self.y_m)


# Default anchor positions for an adult mannequin (average anatomy)
# Derived from BLUE protocol landmark descriptions
DEFAULT_ZONE_ANCHORS: List[ZoneAnchor] = [
    # --- Left hemithorax ---
    ZoneAnchor(
        zone=LungZone.UPPER_BLUE_L,
        x_m=0.08, y_m=0.06,
        radius_m=0.03,
        description="Left upper BLUE: 2nd-3rd ICS, mid-clavicular line",
    ),
    ZoneAnchor(
        zone=LungZone.LOWER_BLUE_L,
        x_m=0.12, y_m=0.14,
        radius_m=0.03,
        description="Left lower BLUE: 4th-5th ICS, anterior axillary line",
    ),
    ZoneAnchor(
        zone=LungZone.PLAPS_L,
        x_m=0.16, y_m=0.20,
        radius_m=0.035,
        description="Left PLAPS: posterolateral, above diaphragm",
    ),
    ZoneAnchor(
        zone=LungZone.DIAPHRAGM_L,
        x_m=0.10, y_m=0.24,
        radius_m=0.03,
        description="Left diaphragm: costophrenic angle",
    ),
    # --- Right hemithorax (mirrored X) ---
    ZoneAnchor(
        zone=LungZone.UPPER_BLUE_R,
        x_m=-0.08, y_m=0.06,
        radius_m=0.03,
        description="Right upper BLUE: 2nd-3rd ICS, mid-clavicular line",
    ),
    ZoneAnchor(
        zone=LungZone.LOWER_BLUE_R,
        x_m=-0.12, y_m=0.14,
        radius_m=0.03,
        description="Right lower BLUE: 4th-5th ICS, anterior axillary line",
    ),
    ZoneAnchor(
        zone=LungZone.PLAPS_R,
        x_m=-0.16, y_m=0.20,
        radius_m=0.035,
        description="Right PLAPS: posterolateral, above diaphragm",
    ),
    ZoneAnchor(
        zone=LungZone.DIAPHRAGM_R,
        x_m=-0.10, y_m=0.24,
        radius_m=0.03,
        description="Right diaphragm: costophrenic angle",
    ),
]


# ---------------------------------------------------------------------------
# Zone Resolver — Probe (x,y) → Anatomical Zone
# ---------------------------------------------------------------------------

class ZoneResolver:
    """
    Maps a probe's (x, y) position on the chest wall to the nearest
    BLUE protocol zone. Returns zone identity and a proximity weight
    (1.0 at center, 0.0 at edge of activation radius).
    """

    def __init__(self, anchors: Optional[List[ZoneAnchor]] = None):
        self.anchors = anchors or DEFAULT_ZONE_ANCHORS

    def resolve(
        self, probe_x_m: float, probe_y_m: float
    ) -> List[Tuple[LungZone, float]]:
        """
        Return all zones whose activation radius contains the probe,
        sorted by proximity (closest first). Each entry is (zone, weight).

        Weight is cosine-tapered: 1.0 at center → 0.0 at radius edge.
        """
        hits = []
        for anchor in self.anchors:
            dist = math.hypot(probe_x_m - anchor.x_m, probe_y_m - anchor.y_m)
            if dist <= anchor.radius_m:
                # Cosine taper: smooth falloff to zero at boundary
                weight = 0.5 * (1.0 + math.cos(math.pi * dist / anchor.radius_m))
                hits.append((anchor.zone, weight))

        hits.sort(key=lambda t: t[1], reverse=True)
        return hits

    def nearest(
        self, probe_x_m: float, probe_y_m: float
    ) -> Tuple[LungZone, float]:
        """Return the single nearest zone and its distance (meters)."""
        best_zone = None
        best_dist = float("inf")
        for anchor in self.anchors:
            dist = math.hypot(probe_x_m - anchor.x_m, probe_y_m - anchor.y_m)
            if dist < best_dist:
                best_dist = dist
                best_zone = anchor.zone
        return best_zone, best_dist

    def get_anchor(self, zone: LungZone) -> ZoneAnchor:
        """Look up the anchor for a given zone."""
        for anchor in self.anchors:
            if anchor.zone == zone:
                return anchor
        raise ValueError(f"No anchor for zone {zone}")

    def all_zone_positions(self) -> Dict[str, Tuple[float, float]]:
        """Return a dict of zone_name → (x, y) for visualization."""
        return {
            anchor.zone.name: anchor.center
            for anchor in self.anchors
        }


# ---------------------------------------------------------------------------
# Clinical Scenarios — Zone → Pathology Mapping
# ---------------------------------------------------------------------------

@dataclass
class ClinicalScenario:
    """
    Defines what pathology appears at each BLUE protocol zone.
    Models a specific clinical presentation for the simulator.
    """
    name: str
    description: str
    zone_pathology: Dict[LungZone, Pathology]
    # M-mode: whether lung sliding is present at each zone
    lung_sliding: Dict[LungZone, bool] = field(default_factory=dict)

    def get_pathology(self, zone: LungZone) -> Pathology:
        return self.zone_pathology.get(zone, Pathology.NORMAL_A_PROFILE)

    def has_sliding(self, zone: LungZone) -> bool:
        return self.lung_sliding.get(zone, True)


# ---------------------------------------------------------------------------
# Clinical Scenarios for BLUE Protocol Training
# ---------------------------------------------------------------------------
# Each scenario maps BLUE zones -> pathology + lung sliding state.
# Covers the major diagnostic categories a paramedic encounters:
#   - Normal exam
#   - Pneumothorax (left/right, with lung point)
#   - Pulmonary edema (CHF)
#   - ARDS / white lung
#   - Pneumonia (unilateral consolidation, bilateral)
#   - Pleural effusion (right-sided, bilateral)
#   - COVID-19 / viral pneumonia pattern
#   - Hemothorax (trauma)
#   - Mixed presentations
#
# References:
#   Lichtenstein DA. BLUE protocol (2008)
#   Volpicelli G et al. International LUS recommendations, ICM (2012)
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, ClinicalScenario] = {

    # ── NORMAL ──────────────────────────────────────────────────────
    "normal": ClinicalScenario(
        name="Normal Lung Exam",
        description=(
            "Bilateral A-profile with lung sliding at all zones. "
            "Normal BLUE protocol exam — no pneumothorax, no pulmonary "
            "edema, no consolidation, no effusion."
        ),
        zone_pathology={z: Pathology.NORMAL_A_PROFILE for z in LungZone},
        lung_sliding={z: True for z in LungZone},
    ),

    # ── PNEUMOTHORAX ────────────────────────────────────────────────
    "left_pneumothorax": ClinicalScenario(
        name="Left Tension Pneumothorax",
        description=(
            "Left A-prime profile: absent lung sliding + A-lines "
            "(stratosphere sign on M-mode). Right side normal. "
            "BLUE protocol: A-prime profile = pneumothorax until "
            "proven otherwise."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_L: Pathology.PNEUMOTHORAX,
            LungZone.LOWER_BLUE_L: Pathology.PNEUMOTHORAX,
            LungZone.PLAPS_L: Pathology.PNEUMOTHORAX,
            LungZone.DIAPHRAGM_L: Pathology.NORMAL_A_PROFILE,
            LungZone.UPPER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.PLAPS_R: Pathology.NORMAL_A_PROFILE,
            LungZone.DIAPHRAGM_R: Pathology.NORMAL_A_PROFILE,
        },
        lung_sliding={
            LungZone.UPPER_BLUE_L: False,
            LungZone.LOWER_BLUE_L: False,
            LungZone.PLAPS_L: False,
            LungZone.DIAPHRAGM_L: True,
            LungZone.UPPER_BLUE_R: True,
            LungZone.LOWER_BLUE_R: True,
            LungZone.PLAPS_R: True,
            LungZone.DIAPHRAGM_R: True,
        },
    ),

    "right_pneumothorax": ClinicalScenario(
        name="Right Tension Pneumothorax",
        description=(
            "Right A-prime profile: absent lung sliding + A-lines. "
            "Left side normal. Stratosphere M-mode sign on right."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_R: Pathology.PNEUMOTHORAX,
            LungZone.LOWER_BLUE_R: Pathology.PNEUMOTHORAX,
            LungZone.PLAPS_R: Pathology.PNEUMOTHORAX,
            LungZone.DIAPHRAGM_R: Pathology.NORMAL_A_PROFILE,
            LungZone.UPPER_BLUE_L: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_L: Pathology.NORMAL_A_PROFILE,
            LungZone.PLAPS_L: Pathology.NORMAL_A_PROFILE,
            LungZone.DIAPHRAGM_L: Pathology.NORMAL_A_PROFILE,
        },
        lung_sliding={
            LungZone.UPPER_BLUE_R: False,
            LungZone.LOWER_BLUE_R: False,
            LungZone.PLAPS_R: False,
            LungZone.DIAPHRAGM_R: True,
            LungZone.UPPER_BLUE_L: True,
            LungZone.LOWER_BLUE_L: True,
            LungZone.PLAPS_L: True,
            LungZone.DIAPHRAGM_L: True,
        },
    ),

    "left_pneumothorax_with_lung_point": ClinicalScenario(
        name="Left Pneumothorax with Lung Point",
        description=(
            "Partial left pneumothorax. Upper zones show absent sliding "
            "(A-prime). Lower BLUE shows lung point (pathognomonic). "
            "PLAPS normal. Lung point = 100% specific for pneumothorax."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_L: Pathology.PNEUMOTHORAX,
            LungZone.LOWER_BLUE_L: Pathology.LUNG_POINT,
            LungZone.PLAPS_L: Pathology.NORMAL_A_PROFILE,
            LungZone.DIAPHRAGM_L: Pathology.NORMAL_A_PROFILE,
            LungZone.UPPER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.PLAPS_R: Pathology.NORMAL_A_PROFILE,
            LungZone.DIAPHRAGM_R: Pathology.NORMAL_A_PROFILE,
        },
        lung_sliding={
            LungZone.UPPER_BLUE_L: False,
            LungZone.LOWER_BLUE_L: False,  # lung point has mixed sliding
            LungZone.PLAPS_L: True,
            LungZone.DIAPHRAGM_L: True,
            LungZone.UPPER_BLUE_R: True,
            LungZone.LOWER_BLUE_R: True,
            LungZone.PLAPS_R: True,
            LungZone.DIAPHRAGM_R: True,
        },
    ),

    # ── PULMONARY EDEMA ─────────────────────────────────────────────
    "pulmonary_edema": ClinicalScenario(
        name="Bilateral Pulmonary Edema (B-profile)",
        description=(
            "Diffuse bilateral B-lines (>=3 per zone) at all anterior "
            "zones. Lung sliding preserved. BLUE protocol B-profile = "
            "cardiogenic pulmonary edema (most common cause of acute "
            "bilateral B-lines in the ED)."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_L: Pathology.B_LINES_DIFFUSE,
            LungZone.UPPER_BLUE_R: Pathology.B_LINES_DIFFUSE,
            LungZone.LOWER_BLUE_L: Pathology.B_LINES_DIFFUSE,
            LungZone.LOWER_BLUE_R: Pathology.B_LINES_DIFFUSE,
            LungZone.PLAPS_L: Pathology.B_LINES_DIFFUSE,
            LungZone.PLAPS_R: Pathology.B_LINES_DIFFUSE,
            LungZone.DIAPHRAGM_L: Pathology.B_LINES_FOCAL,
            LungZone.DIAPHRAGM_R: Pathology.B_LINES_FOCAL,
        },
        lung_sliding={z: True for z in LungZone},
    ),

    "pulmonary_edema_with_effusion": ClinicalScenario(
        name="Pulmonary Edema with Bilateral Effusions",
        description=(
            "Severe CHF: diffuse B-lines anteriorly with bilateral "
            "pleural effusions at PLAPS and diaphragm zones. "
            "Classic decompensated heart failure presentation."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_L: Pathology.B_LINES_DIFFUSE,
            LungZone.UPPER_BLUE_R: Pathology.B_LINES_DIFFUSE,
            LungZone.LOWER_BLUE_L: Pathology.B_LINES_DIFFUSE,
            LungZone.LOWER_BLUE_R: Pathology.B_LINES_DIFFUSE,
            LungZone.PLAPS_L: Pathology.PLEURAL_EFFUSION,
            LungZone.PLAPS_R: Pathology.PLEURAL_EFFUSION,
            LungZone.DIAPHRAGM_L: Pathology.PLEURAL_EFFUSION,
            LungZone.DIAPHRAGM_R: Pathology.PLEURAL_EFFUSION,
        },
        lung_sliding={z: True for z in LungZone},
    ),

    # ── ARDS / WHITE LUNG ──────────────────────────────────────────
    "ards": ClinicalScenario(
        name="ARDS (Diffuse White Lung)",
        description=(
            "Bilateral confluent B-lines / white lung at anterior zones. "
            "Reduced or absent lung sliding in severe zones. "
            "ARDS = non-cardiogenic pulmonary edema with heterogeneous "
            "distribution and absent/reduced sliding."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_L: Pathology.ARDS_WHITE_LUNG,
            LungZone.UPPER_BLUE_R: Pathology.ARDS_WHITE_LUNG,
            LungZone.LOWER_BLUE_L: Pathology.ARDS_WHITE_LUNG,
            LungZone.LOWER_BLUE_R: Pathology.ARDS_WHITE_LUNG,
            LungZone.PLAPS_L: Pathology.ARDS_WHITE_LUNG,
            LungZone.PLAPS_R: Pathology.ARDS_WHITE_LUNG,
            LungZone.DIAPHRAGM_L: Pathology.B_LINES_DIFFUSE,
            LungZone.DIAPHRAGM_R: Pathology.B_LINES_DIFFUSE,
        },
        lung_sliding={
            LungZone.UPPER_BLUE_L: False,
            LungZone.UPPER_BLUE_R: False,
            LungZone.LOWER_BLUE_L: False,
            LungZone.LOWER_BLUE_R: False,
            LungZone.PLAPS_L: True,
            LungZone.PLAPS_R: True,
            LungZone.DIAPHRAGM_L: True,
            LungZone.DIAPHRAGM_R: True,
        },
    ),

    # ── PNEUMONIA / CONSOLIDATION ───────────────────────────────────
    "left_pneumonia": ClinicalScenario(
        name="Left Lower Lobe Pneumonia",
        description=(
            "Left A/B profile (anterior B-lines) + positive PLAPS "
            "(consolidation at posterior-lateral). BLUE protocol: "
            "anterior B or A/B profile + positive PLAPS = pneumonia."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_L: Pathology.B_LINES_FOCAL,
            LungZone.LOWER_BLUE_L: Pathology.B_LINES_DIFFUSE,
            LungZone.PLAPS_L: Pathology.CONSOLIDATION,
            LungZone.DIAPHRAGM_L: Pathology.CONSOLIDATION,
            LungZone.UPPER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.PLAPS_R: Pathology.NORMAL_A_PROFILE,
            LungZone.DIAPHRAGM_R: Pathology.NORMAL_A_PROFILE,
        },
        lung_sliding={z: True for z in LungZone},
    ),

    "right_pneumonia": ClinicalScenario(
        name="Right Lower Lobe Pneumonia",
        description=(
            "Right A/B profile + PLAPS consolidation with air "
            "bronchograms. Left side normal. Shred sign at consolidation "
            "border indicates inflammatory origin."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_R: Pathology.B_LINES_DIFFUSE,
            LungZone.PLAPS_R: Pathology.CONSOLIDATION,
            LungZone.DIAPHRAGM_R: Pathology.CONSOLIDATION,
            LungZone.UPPER_BLUE_L: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_L: Pathology.NORMAL_A_PROFILE,
            LungZone.PLAPS_L: Pathology.NORMAL_A_PROFILE,
            LungZone.DIAPHRAGM_L: Pathology.NORMAL_A_PROFILE,
        },
        lung_sliding={z: True for z in LungZone},
    ),

    "bilateral_pneumonia": ClinicalScenario(
        name="Bilateral Pneumonia (COVID-19 Pattern)",
        description=(
            "Bilateral interstitial syndrome with pleural irregularities "
            "and scattered consolidations. Patchy B-lines with pleural "
            "thickening — hallmark of viral/COVID pneumonia."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_L: Pathology.INTERSTITIAL_SYNDROME,
            LungZone.UPPER_BLUE_R: Pathology.INTERSTITIAL_SYNDROME,
            LungZone.LOWER_BLUE_L: Pathology.B_LINES_DIFFUSE,
            LungZone.LOWER_BLUE_R: Pathology.B_LINES_DIFFUSE,
            LungZone.PLAPS_L: Pathology.CONSOLIDATION,
            LungZone.PLAPS_R: Pathology.PLEURAL_THICKENING,
            LungZone.DIAPHRAGM_L: Pathology.B_LINES_FOCAL,
            LungZone.DIAPHRAGM_R: Pathology.CONSOLIDATION,
        },
        lung_sliding={
            LungZone.UPPER_BLUE_L: True,
            LungZone.UPPER_BLUE_R: True,
            LungZone.LOWER_BLUE_L: True,
            LungZone.LOWER_BLUE_R: True,
            LungZone.PLAPS_L: True,
            LungZone.PLAPS_R: True,
            LungZone.DIAPHRAGM_L: True,
            LungZone.DIAPHRAGM_R: True,
        },
    ),

    # ── PLEURAL EFFUSION ────────────────────────────────────────────
    "right_pleural_effusion": ClinicalScenario(
        name="Right Pleural Effusion",
        description=(
            "Isolated right-sided pleural effusion. Anechoic fluid at "
            "PLAPS and diaphragm zones. Quad sign and spine sign may "
            "be present. Anterior zones show normal A-profile."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.PLAPS_R: Pathology.PLEURAL_EFFUSION,
            LungZone.DIAPHRAGM_R: Pathology.PLEURAL_EFFUSION,
            LungZone.UPPER_BLUE_L: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_L: Pathology.NORMAL_A_PROFILE,
            LungZone.PLAPS_L: Pathology.NORMAL_A_PROFILE,
            LungZone.DIAPHRAGM_L: Pathology.NORMAL_A_PROFILE,
        },
        lung_sliding={z: True for z in LungZone},
    ),

    "bilateral_effusion": ClinicalScenario(
        name="Bilateral Pleural Effusions",
        description=(
            "Large bilateral pleural effusions. Effusion visible at "
            "all posterior and basal zones. Anterior zones may show "
            "B-lines from adjacent fluid. Compressed lung visible."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_L: Pathology.B_LINES_FOCAL,
            LungZone.UPPER_BLUE_R: Pathology.B_LINES_FOCAL,
            LungZone.LOWER_BLUE_L: Pathology.B_LINES_FOCAL,
            LungZone.LOWER_BLUE_R: Pathology.B_LINES_FOCAL,
            LungZone.PLAPS_L: Pathology.PLEURAL_EFFUSION,
            LungZone.PLAPS_R: Pathology.PLEURAL_EFFUSION,
            LungZone.DIAPHRAGM_L: Pathology.PLEURAL_EFFUSION,
            LungZone.DIAPHRAGM_R: Pathology.PLEURAL_EFFUSION,
        },
        lung_sliding={z: True for z in LungZone},
    ),

    # ── TRAUMA / HEMOTHORAX ─────────────────────────────────────────
    "left_hemothorax": ClinicalScenario(
        name="Left Hemothorax (Trauma)",
        description=(
            "Blunt chest trauma with left hemothorax. Effusion at "
            "PLAPS and diaphragm on left (blood appears anechoic "
            "acutely). Left anterior shows B-lines from contusion. "
            "Right side normal."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_L: Pathology.B_LINES_FOCAL,
            LungZone.LOWER_BLUE_L: Pathology.B_LINES_DIFFUSE,
            LungZone.PLAPS_L: Pathology.PLEURAL_EFFUSION,
            LungZone.DIAPHRAGM_L: Pathology.PLEURAL_EFFUSION,
            LungZone.UPPER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_R: Pathology.NORMAL_A_PROFILE,
            LungZone.PLAPS_R: Pathology.NORMAL_A_PROFILE,
            LungZone.DIAPHRAGM_R: Pathology.NORMAL_A_PROFILE,
        },
        lung_sliding={z: True for z in LungZone},
    ),

    # ── PNEUMONIA WITH EFFUSION (PARAPNEUMONIC) ─────────────────────
    "pneumonia_with_effusion": ClinicalScenario(
        name="Pneumonia with Parapneumonic Effusion",
        description=(
            "Right lower lobe consolidation with associated "
            "parapneumonic effusion. Consolidation shows air "
            "bronchograms. Effusion adjacent to consolidated lung."
        ),
        zone_pathology={
            LungZone.UPPER_BLUE_R: Pathology.B_LINES_FOCAL,
            LungZone.LOWER_BLUE_R: Pathology.B_LINES_DIFFUSE,
            LungZone.PLAPS_R: Pathology.CONSOLIDATION,
            LungZone.DIAPHRAGM_R: Pathology.PLEURAL_EFFUSION,
            LungZone.UPPER_BLUE_L: Pathology.NORMAL_A_PROFILE,
            LungZone.LOWER_BLUE_L: Pathology.NORMAL_A_PROFILE,
            LungZone.PLAPS_L: Pathology.NORMAL_A_PROFILE,
            LungZone.DIAPHRAGM_L: Pathology.NORMAL_A_PROFILE,
        },
        lung_sliding={z: True for z in LungZone},
    ),

    # ── COPD / ASTHMA (Normal US despite dyspnea) ──────────────────
    "copd_exacerbation": ClinicalScenario(
        name="COPD Exacerbation (A-profile)",
        description=(
            "Bilateral A-profile with lung sliding. Normal POCUS exam "
            "despite dyspnea. BLUE protocol: bilateral A-profile + "
            "no DVT = COPD/asthma. Key teaching: normal LUS does NOT "
            "mean normal lungs — air trapping is invisible on US."
        ),
        zone_pathology={z: Pathology.NORMAL_A_PROFILE for z in LungZone},
        lung_sliding={z: True for z in LungZone},
    ),
}


# ---------------------------------------------------------------------------
# Patient Case Generator — Randomized clinical presentations for testing
# ---------------------------------------------------------------------------

@dataclass
class PatientCase:
    """
    A randomized patient case for POCUS training / testing mode.
    Includes clinical context that a paramedic would have before scanning.
    """
    scenario_key: str
    scenario: ClinicalScenario
    seed: int                     # Unique seed for image generation
    # Patient demographics
    age: int = 65
    sex: str = "M"
    # Clinical presentation
    chief_complaint: str = ""
    history: str = ""
    vitals: Dict[str, str] = field(default_factory=dict)
    # Correct diagnosis (hidden in testing mode)
    diagnosis: str = ""
    diagnosis_explanation: str = ""
    # BLUE protocol expected conclusion
    blue_profile: str = ""

    @property
    def vitals_text(self) -> str:
        parts = []
        for k, v in self.vitals.items():
            parts.append(f"{k}: {v}")
        return "  |  ".join(parts)


# Clinical context templates per scenario
_CASE_TEMPLATES: Dict[str, Dict] = {
    "normal": dict(
        complaints=["Annual checkup", "Mild cough x3 days", "Chest wall pain after exercise"],
        histories=["No significant PMH", "Hx of mild asthma, well controlled", "Recent URI, improving"],
        diagnosis="Normal lung exam",
        explanation="Bilateral A-profile with lung sliding at all zones. No B-lines, no effusion, no consolidation. Normal BLUE protocol exam.",
        blue_profile="Bilateral A-profile with lung sliding",
        vitals_ranges={"HR": (65, 85), "BP": ("110/70", "130/80"), "SpO2": (96, 99), "RR": (14, 18)},
    ),
    "left_pneumothorax": dict(
        complaints=["Sudden left chest pain + dyspnea", "Left pleuritic chest pain after fall",
                     "Acute dyspnea, left side sharp pain"],
        histories=["Tall, thin male with sudden onset", "MVA with seatbelt injury",
                    "Blunt trauma to left chest"],
        diagnosis="Left tension pneumothorax",
        explanation="Left A-prime profile: A-lines present but NO lung sliding (stratosphere M-mode). Right side normal with sliding. A-prime profile = pneumothorax until proven otherwise. Requires emergent needle decompression.",
        blue_profile="Left A-prime, Right A-profile",
        vitals_ranges={"HR": (110, 135), "BP": ("85/55", "100/65"), "SpO2": (82, 90), "RR": (24, 34)},
    ),
    "right_pneumothorax": dict(
        complaints=["Sudden right chest pain + dyspnea", "Right-sided pleuritic pain after lifting",
                     "Acute shortness of breath"],
        histories=["Tall, thin male, spontaneous onset", "After central line placement",
                    "Penetrating trauma to right chest"],
        diagnosis="Right tension pneumothorax",
        explanation="Right A-prime profile: A-lines present but NO lung sliding (stratosphere M-mode). Left side normal. Requires emergent needle decompression.",
        blue_profile="Right A-prime, Left A-profile",
        vitals_ranges={"HR": (110, 135), "BP": ("85/55", "100/65"), "SpO2": (82, 90), "RR": (24, 34)},
    ),
    "left_pneumothorax_with_lung_point": dict(
        complaints=["Left chest pain, gradual dyspnea", "Mild left pleuritic pain"],
        histories=["Known COPD, recent bleb rupture", "Post-procedure (thoracentesis)"],
        diagnosis="Left partial pneumothorax (lung point present)",
        explanation="Upper zones: absent sliding (A-prime). Lower BLUE: lung point visible — pathognomonic for pneumothorax (100% specificity). Lung point location indicates extent of pneumothorax.",
        blue_profile="Left A-prime (upper) + lung point (lower)",
        vitals_ranges={"HR": (90, 115), "BP": ("100/65", "120/75"), "SpO2": (88, 94), "RR": (20, 28)},
    ),
    "pulmonary_edema": dict(
        complaints=["Worsening dyspnea x2 days", "Can't lie flat, orthopnea",
                     "Bilateral leg swelling + SOB"],
        histories=["Known CHF (EF 25%)", "Hx of CHF, missed medications x3 days",
                    "Known cardiomyopathy, dietary indiscretion"],
        diagnosis="Acute cardiogenic pulmonary edema",
        explanation="Bilateral B-profile: diffuse B-lines (>=3) at all anterior zones with lung sliding preserved. BLUE protocol: bilateral B-profile = pulmonary edema (most common cause). Treat with NIV, diuretics, vasodilators.",
        blue_profile="Bilateral B-profile with sliding",
        vitals_ranges={"HR": (100, 130), "BP": ("150/95", "190/110"), "SpO2": (82, 92), "RR": (24, 36)},
    ),
    "pulmonary_edema_with_effusion": dict(
        complaints=["Progressive dyspnea over 1 week", "Severe orthopnea, PND",
                     "Can't breathe lying down"],
        histories=["Known CHF (EF 20%), not taking meds", "End-stage cardiomyopathy",
                    "Dialysis patient, missed 2 sessions"],
        diagnosis="Severe CHF with bilateral pleural effusions",
        explanation="Anterior zones: diffuse B-lines (B-profile). Posterior/basal zones: bilateral pleural effusions visible as anechoic fluid. Effusions + B-lines = severe decompensated heart failure.",
        blue_profile="Bilateral B-profile + bilateral effusions",
        vitals_ranges={"HR": (105, 140), "BP": ("90/60", "170/100"), "SpO2": (78, 88), "RR": (28, 40)},
    ),
    "ards": dict(
        complaints=["Progressive respiratory failure", "Worsening hypoxia despite O2",
                     "Acute dyspnea in ICU patient"],
        histories=["Sepsis from abdominal source", "Aspiration event 24h ago",
                    "Pancreatitis, now in respiratory distress"],
        diagnosis="ARDS (Acute Respiratory Distress Syndrome)",
        explanation="Bilateral white lung: confluent B-lines at anterior zones with ABSENT or reduced sliding. Key difference from cardiogenic edema: absent sliding + non-homogeneous distribution + clinical context (sepsis/aspiration). Requires intubation + lung-protective ventilation.",
        blue_profile="Bilateral white lung, reduced sliding",
        vitals_ranges={"HR": (115, 145), "BP": ("80/50", "100/60"), "SpO2": (70, 85), "RR": (30, 42)},
    ),
    "left_pneumonia": dict(
        complaints=["Fever + productive cough x5 days", "Left-sided pleuritic chest pain + fever",
                     "Cough with rust-colored sputum"],
        histories=["Otherwise healthy", "Diabetic, smoker", "Recent aspiration risk (alcohol)"],
        diagnosis="Left lower lobe pneumonia",
        explanation="Left A/B profile (focal B-lines anteriorly) + positive PLAPS (consolidation with air bronchograms). BLUE protocol: A/B or B-profile + PLAPS = pneumonia. Air bronchograms confirm pneumonic consolidation.",
        blue_profile="Left A/B-profile + positive PLAPS",
        vitals_ranges={"HR": (95, 120), "BP": ("100/65", "130/80"), "SpO2": (88, 94), "RR": (22, 30)},
    ),
    "right_pneumonia": dict(
        complaints=["High fever + cough x3 days", "Right chest pain worse with breathing",
                     "Productive cough, febrile"],
        histories=["Elderly, nursing home resident", "Recent influenza",
                    "Immunocompromised (chemo)"],
        diagnosis="Right lower lobe pneumonia",
        explanation="Right lower zones: B-lines anteriorly + consolidation at PLAPS (air bronchograms + shred sign). BLUE protocol: B-profile + positive PLAPS = pneumonia.",
        blue_profile="Right A/B-profile + positive PLAPS",
        vitals_ranges={"HR": (95, 120), "BP": ("95/60", "125/75"), "SpO2": (86, 93), "RR": (22, 32)},
    ),
    "bilateral_pneumonia": dict(
        complaints=["Worsening dyspnea + dry cough x7 days", "Progressive hypoxia",
                     "Bilateral chest tightness + fever"],
        histories=["Recent travel, COVID exposure", "Unvaccinated, sick contacts",
                    "Known COVID-19 positive x5 days"],
        diagnosis="Bilateral viral pneumonia (COVID-19 pattern)",
        explanation="Bilateral interstitial syndrome: irregular pleural line + patchy B-lines + sub-pleural consolidations. COVID-19 LUS pattern is typically bilateral, patchy, and posterior-predominant with pleural thickening.",
        blue_profile="Bilateral interstitial pattern + scattered consolidations",
        vitals_ranges={"HR": (100, 130), "BP": ("100/65", "130/80"), "SpO2": (80, 90), "RR": (24, 36)},
    ),
    "right_pleural_effusion": dict(
        complaints=["Gradual dyspnea over 2 weeks", "Dullness to percussion on right",
                     "Pleuritic pain, worse lying on right side"],
        histories=["Known malignancy (lung cancer)", "Recent pneumonia",
                    "Liver cirrhosis with ascites (hepatic hydrothorax)"],
        diagnosis="Right-sided pleural effusion",
        explanation="Anterior zones: normal A-profile. PLAPS + diaphragm on right: anechoic fluid (pleural effusion). Quad sign visible. Need to determine etiology (transudate vs exudate).",
        blue_profile="Right A-profile anterior + PLAPS effusion",
        vitals_ranges={"HR": (80, 100), "BP": ("110/70", "135/85"), "SpO2": (90, 96), "RR": (18, 26)},
    ),
    "bilateral_effusion": dict(
        complaints=["Progressive dyspnea over weeks", "Can't lie flat",
                     "Bilateral chest heaviness"],
        histories=["End-stage renal disease on dialysis", "Known nephrotic syndrome",
                    "Hepatic cirrhosis, decompensated"],
        diagnosis="Bilateral pleural effusions",
        explanation="Anterior zones: focal B-lines (from adjacent fluid). Posterior/basal zones: large bilateral anechoic effusions with compressed lung visible. Consider causes: CHF, renal failure, hepatic, malignancy.",
        blue_profile="Focal anterior B-lines + bilateral posterior effusions",
        vitals_ranges={"HR": (85, 110), "BP": ("100/60", "140/85"), "SpO2": (88, 94), "RR": (20, 28)},
    ),
    "left_hemothorax": dict(
        complaints=["MVC with left chest wall tenderness", "Fall from height, left rib pain",
                     "Stab wound left chest"],
        histories=["Trauma: MVC, unrestrained", "Fall from 10 ft",
                    "Penetrating trauma, hemodynamically unstable"],
        diagnosis="Left hemothorax",
        explanation="Left PLAPS/diaphragm: anechoic fluid (blood appears anechoic acutely). Left anterior: B-lines from pulmonary contusion. In trauma context, effusion = hemothorax until proven otherwise. Requires chest tube.",
        blue_profile="Left B-lines anterior + left posterior effusion (blood)",
        vitals_ranges={"HR": (110, 140), "BP": ("75/45", "95/60"), "SpO2": (85, 93), "RR": (24, 34)},
    ),
    "pneumonia_with_effusion": dict(
        complaints=["Fever x1 week, now worsening SOB", "Persistent cough despite antibiotics",
                     "Febrile with increasing dyspnea"],
        histories=["Community-acquired pneumonia, not improving", "Diabetic with recent pneumonia",
                    "Elderly, febrile, pleuritic pain"],
        diagnosis="Pneumonia with parapneumonic effusion",
        explanation="Right PLAPS: consolidation with air bronchograms (pneumonia). Right diaphragm: effusion (parapneumonic). Parapneumonic effusion may need drainage if complicated (septations, loculation).",
        blue_profile="Right B-profile + PLAPS consolidation + basal effusion",
        vitals_ranges={"HR": (100, 125), "BP": ("95/60", "125/75"), "SpO2": (86, 93), "RR": (22, 32)},
    ),
    "copd_exacerbation": dict(
        complaints=["Acute worsening of chronic dyspnea", "Wheezing + SOB, can't speak full sentences",
                     "Dyspnea + increased sputum production"],
        histories=["Known COPD (GOLD stage 3)", "Severe asthma, recent URI trigger",
                    "30-pack-year smoker, known emphysema"],
        diagnosis="COPD/asthma exacerbation (normal LUS)",
        explanation="Bilateral A-profile with lung sliding — NORMAL lung ultrasound. This is expected in COPD/asthma because hyperinflation and air trapping are invisible on US. BLUE protocol: bilateral A-profile + sliding + no DVT = COPD/asthma. Key teaching point: normal LUS does NOT exclude obstructive lung disease.",
        blue_profile="Bilateral A-profile with sliding (normal)",
        vitals_ranges={"HR": (100, 125), "BP": ("140/85", "165/95"), "SpO2": (85, 92), "RR": (24, 34)},
    ),
}


def generate_patient_case(
    scenario_key: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> PatientCase:
    """
    Generate a randomized patient case for testing/training mode.

    If scenario_key is None, picks a random scenario.
    Each call produces unique images via a random seed.
    """
    if rng is None:
        rng = np.random.default_rng()

    available = list(SCENARIOS.keys())
    if scenario_key is None:
        scenario_key = rng.choice(available)
    elif scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_key}")

    scenario = SCENARIOS[scenario_key]
    template = _CASE_TEMPLATES.get(scenario_key, {})
    seed = int(rng.integers(0, 2**31))

    # Randomize demographics
    age = int(rng.integers(
        template.get("age_range", (35, 82))[0] if isinstance(template.get("age_range"), tuple)
        else 35,
        template.get("age_range", (35, 82))[1] if isinstance(template.get("age_range"), tuple)
        else 82,
    ))
    sex = rng.choice(["M", "F"])

    # Randomize clinical presentation
    complaints = template.get("complaints", ["Dyspnea"])
    histories = template.get("histories", ["No significant PMH"])
    complaint = str(rng.choice(complaints))
    history = str(rng.choice(histories))

    # Randomize vitals
    vitals = {}
    vr = template.get("vitals_ranges", {})
    if "HR" in vr:
        vitals["HR"] = f"{rng.integers(vr['HR'][0], vr['HR'][1]+1)} bpm"
    if "BP" in vr:
        # Pick one of the BP options
        bps = [vr["BP"][0], vr["BP"][1]]
        vitals["BP"] = str(rng.choice(bps))
    if "SpO2" in vr:
        vitals["SpO2"] = f"{rng.integers(vr['SpO2'][0], vr['SpO2'][1]+1)}%"
    if "RR" in vr:
        vitals["RR"] = f"{rng.integers(vr['RR'][0], vr['RR'][1]+1)}/min"
    vitals["Temp"] = f"{rng.uniform(36.2, 39.5):.1f} C"

    return PatientCase(
        scenario_key=scenario_key,
        scenario=scenario,
        seed=seed,
        age=age,
        sex=sex,
        chief_complaint=complaint,
        history=history,
        vitals=vitals,
        diagnosis=template.get("diagnosis", scenario.name),
        diagnosis_explanation=template.get("explanation", scenario.description),
        blue_profile=template.get("blue_profile", ""),
    )


# ---------------------------------------------------------------------------
# Temporal Image Stack Generator
# ---------------------------------------------------------------------------

@dataclass
class StackConfig:
    """Configuration for temporal image stack generation."""
    n_frames: int = 32              # Frames per stack (≈1 sec at 30 fps)
    fps: float = 30.0               # Playback frame rate
    image_size: Tuple[int, int] = (256, 256)
    depth_range_m: Tuple[float, float] = (0.0, 0.12)
    sliding_amplitude_px: float = 2.0   # Pleural line lateral shift (pixels)
    sliding_freq_hz: float = 0.25       # Respiratory rate ~15 bpm
    breathing_depth_mod: float = 0.003  # Depth modulation from breathing (m)


class TemporalStackGenerator:
    """
    Generates temporal B-mode image stacks with frame-to-frame variation
    for lung sliding, respiratory motion, and speckle decorrelation.

    Each stack is [n_frames, H, W] — a cine loop at a single probe position.
    Also generates M-mode strips from the temporal stack.
    """

    def __init__(self, config: Optional[StackConfig] = None):
        self.cfg = config or StackConfig()
        # Use the clinically accurate frame generator (not the basic one)
        self.clinical_gen = ClinicalFrameGenerator(
            image_size=self.cfg.image_size,
            depth_range_m=self.cfg.depth_range_m,
        )
        # Keep legacy synth as fallback
        self.synth = LungUSArtifactSynthesizer(
            image_size=self.cfg.image_size,
            depth_range_m=self.cfg.depth_range_m,
        )

    def generate_stack(
        self,
        pathology: Pathology,
        lung_sliding: bool = True,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a temporal B-mode stack for a given pathology and sliding state.

        Args:
            pathology: Which pathology to render.
            lung_sliding: Whether pleural line motion is present.
            seed: Random seed for reproducibility.

        Returns:
            np.ndarray [n_frames, H, W] in [0, 1] range.
        """
        rng = np.random.default_rng(seed)
        n = self.cfg.n_frames
        H, W = self.cfg.image_size
        stack = np.zeros((n, H, W), dtype=np.float32)

        for frame_idx in range(n):
            t_sec = frame_idx / self.cfg.fps

            # Generate clinically accurate frame with per-frame speckle variation
            frame_seed = rng.integers(0, 2**31) if seed is not None else None
            try:
                base = self.clinical_gen.generate(
                    ClinicalPathology(int(pathology)),
                    seed=frame_seed,
                )
            except Exception:
                # Fallback to basic generator if clinical gen fails
                base = self.synth.generate(int(pathology))

            if lung_sliding:
                base = self._apply_sliding(base, t_sec)

            # Respiratory depth modulation (subtle axial shift)
            base = self._apply_breathing(base, t_sec, rng)

            stack[frame_idx] = base

        return stack

    def _apply_sliding(self, image: np.ndarray, t_sec: float) -> np.ndarray:
        """
        Simulate lung sliding by laterally shifting rows below the pleural line.
        The pleural line itself oscillates, and sub-pleural tissue moves with it.
        """
        H, W = image.shape
        # Estimate pleural line row (roughly top 15-20% of image)
        pleural_row = int(H * 0.16)  # ~2cm of 12cm depth
        amp = self.cfg.sliding_amplitude_px
        freq = self.cfg.sliding_freq_hz

        # Sinusoidal lateral shift
        shift_px = amp * math.sin(2 * math.pi * freq * t_sec)

        # Apply sub-pixel shift to rows below pleura using interpolation
        shifted = image.copy()
        for row in range(pleural_row, H):
            # Shift increases slightly with depth (tissue drag)
            depth_factor = 1.0 + 0.3 * ((row - pleural_row) / (H - pleural_row))
            row_shift = shift_px * depth_factor

            # Sub-pixel shift via linear interpolation
            int_shift = int(math.floor(row_shift))
            frac = row_shift - int_shift

            if abs(int_shift) < W:
                shifted[row] = (
                    (1 - frac) * np.roll(image[row], int_shift) +
                    frac * np.roll(image[row], int_shift + (1 if row_shift >= 0 else -1))
                )

        return shifted

    def _apply_breathing(
        self, image: np.ndarray, t_sec: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Simulate respiratory axial motion (subtle vertical shift)."""
        H, W = image.shape
        breath_phase = math.sin(2 * math.pi * self.cfg.sliding_freq_hz * t_sec)
        shift_rows = int(round(breath_phase * self.cfg.breathing_depth_mod /
                                (self.cfg.depth_range_m[1] / H)))

        if shift_rows != 0:
            image = np.roll(image, shift_rows, axis=0)
            # Zero-fill the wrapped edge
            if shift_rows > 0:
                image[:shift_rows, :] = 0.0
            else:
                image[shift_rows:, :] = 0.0

        return image

    def generate_mmode(
        self,
        stack: np.ndarray,
        lateral_col: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract an M-mode strip from a temporal stack.

        M-mode displays a single scan line over time:
            - X axis = time (frames)
            - Y axis = depth

        Args:
            stack: [n_frames, H, W] temporal image stack.
            lateral_col: Which column to use. Defaults to center.

        Returns:
            np.ndarray [H, n_frames] — the M-mode strip.
        """
        n_frames, H, W = stack.shape
        if lateral_col is None:
            lateral_col = W // 2

        # M-mode: stack columns across time → [H, n_frames]
        mmode = stack[:, :, lateral_col].T  # [H, n_frames]
        return mmode.astype(np.float32)

    def classify_mmode_pattern(self, mmode: np.ndarray) -> str:
        """
        Classify M-mode strip as seashore sign (sliding) or
        stratosphere/barcode sign (no sliding).

        Uses variance of sub-pleural rows across time as the discriminator.
        """
        H, T = mmode.shape
        pleural_row = int(H * 0.16)

        # Variance of sub-pleural region across time
        sub_pleural = mmode[pleural_row:pleural_row + int(H * 0.3), :]
        temporal_var = np.var(sub_pleural, axis=1).mean()

        # Above pleural line should be static (tissue)
        supra_pleural = mmode[:pleural_row, :]
        supra_var = np.var(supra_pleural, axis=1).mean()

        # Seashore: high sub-pleural variance (granular/sandy)
        # Stratosphere: low sub-pleural variance (horizontal lines)
        ratio = temporal_var / (supra_var + 1e-8)

        if ratio > 1.5:
            return "seashore"  # Lung sliding present
        else:
            return "stratosphere"  # No lung sliding (pneumothorax)


# ---------------------------------------------------------------------------
# POC Image Stack Generator (Top-Level)
# ---------------------------------------------------------------------------

@dataclass
class ProbeReading:
    """A single probe position + orientation reading from the simulator."""
    x_m: float              # Lateral position on chest wall
    y_m: float              # Craniocaudal position on chest wall
    pressure: float = 1.0   # Contact pressure (0 = no contact, 1 = firm)
    imu: Optional[DualIMUReading] = None
    timestamp_s: float = 0.0

    @property
    def has_contact(self) -> bool:
        return self.pressure > 0.1


class POCImageStackGenerator:
    """
    Top-level point-of-care image stack generator for simulator use.

    Workflow:
        1. Receive probe position (x, y) on mannequin chest
        2. Resolve to BLUE protocol zone(s)
        3. Look up pathology from active clinical scenario
        4. Generate temporal B-mode stack + M-mode
        5. Package with coordinate metadata

    Usage:
        gen = POCImageStackGenerator(scenario="left_pneumothorax")

        # Probe placed at left upper BLUE point
        reading = ProbeReading(x_m=0.08, y_m=0.06)
        result = gen.generate(reading)

        bmode_stack = result["bmode_stack"]    # [n_frames, H, W]
        mmode_strip = result["mmode"]          # [H, n_frames]
        zone_name   = result["zone"]           # "UPPER_BLUE_L"
        pathology   = result["pathology"]      # "tension_pneumothorax"
    """

    # Default checkpoint paths for ControlNet DDPM
    _CKPT_ROOT = Path(__file__).resolve().parent.parent / "checkpoints"
    _DEFAULT_MODEL = "checkpoints/realistic_v4_ab/best.pt"
    _TRAUMA_MODEL = "checkpoints/realistic_v2_finetune/latest.pt"
    _ANATOMY_BANK = "checkpoints/anatomy_bank.pt"

    def __init__(
        self,
        scenario: str = "normal",
        stack_config: Optional[StackConfig] = None,
        zone_anchors: Optional[List[ZoneAnchor]] = None,
        grid_config: Optional[GridConfig] = None,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Available: {list(SCENARIOS.keys())}"
            )

        self.scenario = SCENARIOS[scenario]
        self.stack_cfg = stack_config or StackConfig()
        self.zone_resolver = ZoneResolver(zone_anchors)
        self.stack_gen = TemporalStackGenerator(self.stack_cfg)
        self.grid_config = grid_config or GridConfig()
        self.pose_corrector = PoseCorrector(self.grid_config)

        # Load ControlNet DDPM with anatomy bank for photorealistic generation
        self._realistic_gen = None
        self._load_realistic_generator()

    def _load_realistic_generator(self):
        """Load ControlNet DDPM with anatomy bank for photorealistic generation."""
        _root = Path(__file__).resolve().parent.parent
        model_path = _root / self._DEFAULT_MODEL
        if not model_path.exists():
            return
        try:
            from .realistic_generator import RealisticLungUSGenerator

            trauma_path = _root / self._TRAUMA_MODEL
            trauma_str = self._TRAUMA_MODEL if trauma_path.exists() else None

            self._realistic_gen = RealisticLungUSGenerator.from_pretrained(
                model_path=self._DEFAULT_MODEL,
                trauma_model_path=trauma_str,
            )
            has_model = self._realistic_gen.has_model
            has_bank = self._realistic_gen.anatomy_bank is not None
            print(f"[MoCoLUS] Loaded ControlNet DDPM (model={has_model}, "
                  f"anatomy_bank={has_bank}, trauma={trauma_str is not None})")
        except Exception as e:
            print(f"[MoCoLUS] Could not load realistic generator: {e}")
            self._realistic_gen = None

    def _generate_realistic_stack(
        self,
        pathology_class: int,
        lung_sliding: bool,
        seed: Optional[int] = None,
        zone_region: Optional[int] = None,
    ) -> Optional[Dict]:
        """Generate a full stack using the ControlNet DDPM pipeline.

        Returns the stack dict from RealisticLungUSGenerator.generate_stack(),
        or None if the realistic generator is unavailable.

        Args:
            pathology_class: ClinicalPathology integer (0-9).
            lung_sliding:    Whether the M-mode should show sliding (seashore)
                             or no-sliding (stratosphere).
            seed:            Reproducibility seed.
            zone_region:     Optional anatomical zone region (0-6). When None
                             or 0 (UPPER_GENERIC), the legacy class-only
                             pipeline runs and outputs are bit-identical to
                             pre-Phase-1 behavior. When 1-6, the structural
                             guide gains diaphragmatic anatomy and the model
                             receives a non-null zone_embedding.
        """
        if self._realistic_gen is None:
            return None
        try:
            return self._realistic_gen.generate_stack(
                pathology_class=pathology_class,
                n_frames=self.stack_cfg.n_frames,
                lung_sliding=lung_sliding,
                seed=seed,
                zone_region=zone_region,
            )
        except Exception as e:
            print(f"[MoCoLUS] Realistic generation failed: {e}")
            return None

    def generate(
        self,
        probe: ProbeReading,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generate a complete image stack for a probe position.

        Args:
            probe: Probe position and orientation on the chest wall.
            seed: Random seed for reproducibility.

        Returns:
            Dict with keys:
                bmode_stack: np.ndarray [n_frames, H, W]
                mmode: np.ndarray [H, n_frames]
                mmode_pattern: str ("seashore" or "stratosphere")
                zone: str (zone name)
                zone_weight: float (proximity to zone center)
                pathology: str (pathology name)
                pathology_class: int
                lung_sliding: bool
                probe_x_m: float
                probe_y_m: float
                contact_pressure: float
                timestamp_s: float
        """
        if not probe.has_contact:
            return self._no_contact_result(probe)

        # Resolve probe position to zone
        hits = self.zone_resolver.resolve(probe.x_m, probe.y_m)

        if hits:
            zone, weight = hits[0]  # Use closest zone
        else:
            zone, dist = self.zone_resolver.nearest(probe.x_m, probe.y_m)
            # Weight falls off with distance beyond activation radius
            anchor = self.zone_resolver.get_anchor(zone)
            weight = max(0.0, 1.0 - dist / (anchor.radius_m * 3.0))

        # Look up pathology for this zone in the active scenario
        pathology = self.scenario.get_pathology(zone)
        sliding = self.scenario.has_sliding(zone)

        # Translate the LungZone enum into a ZoneRegion integer for the
        # zone-aware structural guide and zone_embedding. Default to 0
        # (UPPER_GENERIC) for any zone not in the map → legacy behavior.
        zone_region = ZONE_REGION_MAP.get(zone, 0)

        # Try ControlNet DDPM pipeline (class-conditioned + anatomy bank)
        realistic = self._generate_realistic_stack(
            pathology_class=int(pathology),
            lung_sliding=sliding,
            seed=seed,
            zone_region=zone_region,
        )

        if realistic is not None:
            bmode_stack = realistic["bmode_stack"]
            mmode = realistic["mmode"]
            mmode_pattern = realistic.get("mmode_pattern", "unknown")
        else:
            # Fallback: physics-only generation
            bmode_stack = self.stack_gen.generate_stack(
                pathology=pathology,
                lung_sliding=sliding,
                seed=seed,
            )
            mmode = self.stack_gen.generate_mmode(bmode_stack)
            mmode_pattern = self.stack_gen.classify_mmode_pattern(mmode)

        # Apply pressure-based quality degradation
        if probe.pressure < 0.5:
            bmode_stack = self._degrade_low_pressure(bmode_stack, probe.pressure)

        return {
            "bmode_stack": bmode_stack,
            "mmode": mmode,
            "mmode_pattern": mmode_pattern,
            "zone": zone.name,
            "zone_id": int(zone),
            "zone_weight": weight,
            "pathology": PATHOLOGY_CLASSES[int(pathology)],
            "pathology_class": int(pathology),
            "lung_sliding": sliding,
            "probe_x_m": probe.x_m,
            "probe_y_m": probe.y_m,
            "contact_pressure": probe.pressure,
            "timestamp_s": probe.timestamp_s,
            "scenario": self.scenario.name,
            "n_frames": self.stack_cfg.n_frames,
            "fps": self.stack_cfg.fps,
            "image_size": list(self.stack_cfg.image_size),
        }

    def _no_contact_result(self, probe: ProbeReading) -> Dict:
        """Return a black (no signal) stack when probe has no contact."""
        H, W = self.stack_cfg.image_size
        n = self.stack_cfg.n_frames
        noise_level = 0.02  # Very faint electronic noise
        stack = np.random.normal(0, noise_level, (n, H, W)).astype(np.float32)
        stack = np.clip(stack, 0, 1)

        return {
            "bmode_stack": stack,
            "mmode": stack[:, :, W // 2].T,
            "mmode_pattern": "no_contact",
            "zone": "NONE",
            "zone_id": -1,
            "zone_weight": 0.0,
            "pathology": "no_contact",
            "pathology_class": -1,
            "lung_sliding": False,
            "probe_x_m": probe.x_m,
            "probe_y_m": probe.y_m,
            "contact_pressure": probe.pressure,
            "timestamp_s": probe.timestamp_s,
            "scenario": self.scenario.name,
            "n_frames": self.stack_cfg.n_frames,
            "fps": self.stack_cfg.fps,
            "image_size": list(self.stack_cfg.image_size),
        }

    def _degrade_low_pressure(
        self, stack: np.ndarray, pressure: float
    ) -> np.ndarray:
        """
        Simulate poor acoustic coupling from low contact pressure.
        Adds shadow artifacts and reduces signal strength.
        """
        # Scale signal by pressure (poor coupling = weaker signal)
        attenuation = 0.3 + 0.7 * pressure  # range [0.3, 1.0]
        stack = stack * attenuation

        # Add random shadow bands (air gaps in coupling gel)
        n_shadows = int((1.0 - pressure) * 5)
        if n_shadows > 0:
            rng = np.random.default_rng()
            W = stack.shape[2]
            for _ in range(n_shadows):
                col = rng.integers(0, W)
                width = rng.integers(3, 15)
                c_start = max(0, col - width // 2)
                c_end = min(W, col + width // 2)
                shadow_strength = rng.uniform(0.1, 0.5)
                stack[:, :, c_start:c_end] *= shadow_strength

        return stack

    def generate_full_exam(
        self,
        seeds: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict]:
        """
        Generate image stacks for all 8 BLUE protocol zones in the
        current scenario. Simulates a complete POCUS lung exam.

        Returns:
            Dict mapping zone_name → generation result dict.
        """
        results = {}
        for anchor in self.zone_resolver.anchors:
            zone_seed = None
            if seeds and anchor.zone.name in seeds:
                zone_seed = seeds[anchor.zone.name]

            probe = ProbeReading(
                x_m=anchor.x_m,
                y_m=anchor.y_m,
                pressure=1.0,
            )
            results[anchor.zone.name] = self.generate(probe, seed=zone_seed)

        return results


# ---------------------------------------------------------------------------
# Simulator Export — Package stacks as HDF5 for playback
# ---------------------------------------------------------------------------

class SimulatorExporter:
    """
    Exports generated image stacks to HDF5 format for ultrasound
    simulator playback systems.

    HDF5 Schema:
        /scenario                       attrs: name, description
        /zones/{zone_name}/
            bmode_stack     [n_frames, H, W]  float32
            mmode           [H, n_frames]     float32
            attrs:
                zone_id, pathology, pathology_class, lung_sliding,
                mmode_pattern, probe_x_m, probe_y_m, zone_weight
        /coordinate_map/
            anchors         [n_zones, 2]      float32 — (x, y) per zone
            radii           [n_zones]         float32
            zone_names      [n_zones]         string
        /params/
            n_frames, fps, image_size, depth_range_m
    """

    def export(
        self,
        exam_results: Dict[str, Dict],
        output_path: str,
        zone_resolver: Optional[ZoneResolver] = None,
        stack_config: Optional[StackConfig] = None,
    ) -> str:
        """
        Write a full exam to HDF5.

        Args:
            exam_results: Output of POCImageStackGenerator.generate_full_exam()
            output_path: Path for the HDF5 file.
            zone_resolver: For writing coordinate map metadata.
            stack_config: For writing acquisition parameters.

        Returns:
            Path to the created file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = stack_config or StackConfig()
        resolver = zone_resolver or ZoneResolver()

        with h5py.File(output_path, "w") as f:
            # Scenario metadata
            first = next(iter(exam_results.values()))
            f.attrs["scenario_name"] = first.get("scenario", "unknown")
            f.attrs["n_zones"] = len(exam_results)

            # Per-zone data
            zones_grp = f.create_group("zones")
            for zone_name, result in exam_results.items():
                zg = zones_grp.create_group(zone_name)
                zg.create_dataset(
                    "bmode_stack", data=result["bmode_stack"],
                    compression="gzip", compression_opts=4,
                )
                zg.create_dataset("mmode", data=result["mmode"])

                zg.attrs["zone_id"] = result["zone_id"]
                zg.attrs["pathology"] = result["pathology"]
                zg.attrs["pathology_class"] = result["pathology_class"]
                zg.attrs["lung_sliding"] = result["lung_sliding"]
                zg.attrs["mmode_pattern"] = result["mmode_pattern"]
                zg.attrs["probe_x_m"] = result["probe_x_m"]
                zg.attrs["probe_y_m"] = result["probe_y_m"]
                zg.attrs["zone_weight"] = result["zone_weight"]

            # Coordinate map for probe tracking
            coord_grp = f.create_group("coordinate_map")
            anchors = resolver.anchors
            n_zones = len(anchors)
            anchor_xy = np.array(
                [[a.x_m, a.y_m] for a in anchors], dtype=np.float32
            )
            radii = np.array([a.radius_m for a in anchors], dtype=np.float32)
            names = [a.zone.name for a in anchors]

            coord_grp.create_dataset("anchors", data=anchor_xy)
            coord_grp.create_dataset("radii", data=radii)
            coord_grp.attrs["zone_names"] = names

            # Acquisition parameters
            params_grp = f.create_group("params")
            params_grp.attrs["n_frames"] = cfg.n_frames
            params_grp.attrs["fps"] = cfg.fps
            params_grp.attrs["image_size"] = list(cfg.image_size)
            params_grp.attrs["depth_range_m"] = list(cfg.depth_range_m)

        return str(output_path)

    def export_coordinate_map_json(
        self,
        zone_resolver: Optional[ZoneResolver] = None,
        output_path: str = "coordinate_map.json",
    ) -> str:
        """
        Export the zone coordinate map as JSON for external simulator
        integration (e.g., Unity, web-based trainers).
        """
        resolver = zone_resolver or ZoneResolver()
        coord_map = {
            "coordinate_system": {
                "origin": "sternal_notch",
                "x_axis": "patient_left_positive",
                "y_axis": "caudal_positive",
                "units": "meters",
            },
            "zones": [],
        }

        for anchor in resolver.anchors:
            coord_map["zones"].append({
                "name": anchor.zone.name,
                "id": int(anchor.zone),
                "x_m": anchor.x_m,
                "y_m": anchor.y_m,
                "radius_m": anchor.radius_m,
                "description": anchor.description,
            })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(coord_map, f, indent=2)

        return str(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Generate lung US POCUS image stacks for simulator use"
    )
    parser.add_argument(
        "--scenario", type=str, default="normal",
        choices=list(SCENARIOS.keys()),
        help="Clinical scenario to simulate",
    )
    parser.add_argument(
        "--output", type=str, default="data/poc_exam.h5",
        help="Output HDF5 path",
    )
    parser.add_argument(
        "--n-frames", type=int, default=32,
        help="Frames per stack",
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=[256, 256],
        help="Image dimensions (H W)",
    )
    parser.add_argument(
        "--export-coords", type=str, default=None,
        help="Also export coordinate map JSON to this path",
    )
    parser.add_argument(
        "--probe-x", type=float, default=None,
        help="Single probe X position (m). If set, generates one stack only.",
    )
    parser.add_argument(
        "--probe-y", type=float, default=None,
        help="Single probe Y position (m).",
    )
    args = parser.parse_args()

    stack_cfg = StackConfig(
        n_frames=args.n_frames,
        image_size=tuple(args.image_size),
    )

    gen = POCImageStackGenerator(
        scenario=args.scenario,
        stack_config=stack_cfg,
    )

    exporter = SimulatorExporter()

    if args.probe_x is not None and args.probe_y is not None:
        # Single probe position
        print(f"Generating stack at ({args.probe_x}, {args.probe_y}) "
              f"for scenario: {args.scenario}")
        probe = ProbeReading(x_m=args.probe_x, y_m=args.probe_y)
        result = gen.generate(probe)
        print(f"  Zone: {result['zone']}")
        print(f"  Pathology: {result['pathology']}")
        print(f"  Lung sliding: {result['lung_sliding']}")
        print(f"  M-mode pattern: {result['mmode_pattern']}")
        print(f"  Stack shape: {result['bmode_stack'].shape}")

        # Export as single-zone exam
        exam = {result["zone"]: result}
        path = exporter.export(exam, args.output, stack_config=stack_cfg)
        print(f"  Exported to: {path}")
    else:
        # Full 8-zone exam
        print(f"Generating full BLUE protocol exam: {args.scenario}")
        t0 = time.time()
        exam = gen.generate_full_exam()
        elapsed = time.time() - t0

        print(f"  Generated {len(exam)} zones in {elapsed:.1f}s")
        for zone_name, result in exam.items():
            print(f"  {zone_name:20s} | {result['pathology']:25s} | "
                  f"sliding={result['lung_sliding']} | "
                  f"M-mode={result['mmode_pattern']}")

        path = exporter.export(
            exam, args.output,
            zone_resolver=gen.zone_resolver,
            stack_config=stack_cfg,
        )
        print(f"\n  Exported to: {path}")

    # Optional coordinate map JSON export
    if args.export_coords:
        coord_path = exporter.export_coordinate_map_json(
            zone_resolver=gen.zone_resolver,
            output_path=args.export_coords,
        )
        print(f"  Coordinate map: {coord_path}")
