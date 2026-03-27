"""
MoCoLUS ↔ POCUS CEWIT GUI Bridge
=================================
Adapter layer that converts MoCoLUS simulator output into formats
compatible with the POCUS CEWIT OpenGL GUI (US_Image_Reader).

Key conversions:
  - IMU: radians (simulator) ↔ degrees (GUI)
  - Images: numpy float32 [H, W] ↔ uint8 RGBA for OpenGL textures
  - Probe position: meters (simulator) ↔ slider units (GUI)
  - Data source: replaces MySQL/BLE with in-memory simulator calls
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .lung_us_generator import (
    IMUPose,
    DualIMUReading,
    GridConfig,
    LungUSArtifactSynthesizer,
    PATHOLOGY_CLASSES,
    N_CLASSES,
)
from .poc_image_stack import (
    POCImageStackGenerator,
    ZoneResolver,
    TemporalStackGenerator,
    SimulatorExporter,
    ProbeReading,
    LungZone,
    Pathology,
    StackConfig,
    SCENARIOS,
    DEFAULT_ZONE_ANCHORS,
    PatientCase,
    generate_patient_case,
)


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi


# ---------------------------------------------------------------------------
# IMU Conversion
# ---------------------------------------------------------------------------

def gui_angles_to_imu_pose(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> IMUPose:
    """
    Convert GUI slider angles (degrees) to simulator IMUPose (radians).

    The GUI uses:
      - Yaw: -180 to +180 degrees (Z-axis rotation)
      - Pitch: -45 to +45 degrees (X-axis rotation)
      - Roll: -90 to +90 degrees (Y-axis rotation)

    The GUI inverts pitch for live mode, so we account for that here.
    """
    return IMUPose(
        pitch=deg_to_rad(pitch_deg),
        roll=deg_to_rad(roll_deg),
        yaw=deg_to_rad(yaw_deg),
        dx=dx,
        dy=dy,
        dz=dz,
    )


def imu_pose_to_gui_angles(pose: IMUPose) -> Tuple[int, int, int]:
    """
    Convert simulator IMUPose (radians) to GUI slider angles (degrees, int).

    Returns:
        (yaw_deg, pitch_deg, roll_deg) as integers for slider values.
    """
    return (
        int(round(rad_to_deg(pose.yaw))),
        int(round(rad_to_deg(pose.pitch))),
        int(round(rad_to_deg(pose.roll))),
    )


# ---------------------------------------------------------------------------
# Image Conversion
# ---------------------------------------------------------------------------

def bmode_to_rgba(image: np.ndarray, colormap: str = "gray") -> np.ndarray:
    """
    Convert a simulator B-mode image [H, W] float32 in [0,1]
    to RGBA uint8 [H, W, 4] suitable for OpenGL texture upload.

    Args:
        image: B-mode image, shape [H, W], values in [0, 1].
        colormap: "gray" for standard US look, "hot" for heatmap.

    Returns:
        RGBA uint8 array [H, W, 4].
    """
    image = np.clip(image, 0.0, 1.0)
    gray = (image * 255).astype(np.uint8)

    if colormap == "gray":
        rgba = np.zeros((*image.shape, 4), dtype=np.uint8)
        rgba[:, :, 0] = gray  # R
        rgba[:, :, 1] = gray  # G
        rgba[:, :, 2] = gray  # B
        rgba[:, :, 3] = 255   # A (fully opaque)
    elif colormap == "hot":
        # Simple hot colormap for visualization
        rgba = np.zeros((*image.shape, 4), dtype=np.uint8)
        rgba[:, :, 0] = np.clip(gray * 3, 0, 255).astype(np.uint8)
        rgba[:, :, 1] = np.clip(gray * 3 - 255, 0, 255).astype(np.uint8)
        rgba[:, :, 2] = np.clip(gray * 3 - 510, 0, 255).astype(np.uint8)
        rgba[:, :, 3] = 255
    else:
        raise ValueError(f"Unknown colormap: {colormap}")

    return rgba


def bmode_to_qimage_data(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Convert B-mode [H, W] float32 to data ready for QImage construction.

    Returns:
        (rgba_data, width, height) — rgba_data is contiguous uint8 [H, W, 4].
    """
    rgba = bmode_to_rgba(image)
    return rgba, image.shape[1], image.shape[0]


def stack_frame_to_rgba(
    stack: np.ndarray, frame_idx: int
) -> np.ndarray:
    """Extract a single frame from a B-mode stack and convert to RGBA."""
    return bmode_to_rgba(stack[frame_idx])


# ---------------------------------------------------------------------------
# Probe Position Mapping
# ---------------------------------------------------------------------------

@dataclass
class ChestWallMapping:
    """
    Maps GUI shift slider values to chest wall coordinates in meters.

    The GUI sliders range from -100 to +100 for X and Y shifts.
    The chest wall coordinate system has origin at sternal notch:
      +X = patient's left, +Y = caudal.

    The mapping is configured for an adult mannequin where the
    relevant BLUE protocol zones span roughly:
      X: -0.20 to +0.20 m (40 cm total width)
      Y:  0.00 to +0.30 m (30 cm craniocaudal)
    """
    # Slider range
    slider_min: float = -100.0
    slider_max: float = 100.0

    # Chest wall coordinate range (meters)
    x_min_m: float = -0.20
    x_max_m: float = 0.20
    y_min_m: float = 0.0
    y_max_m: float = 0.30

    def slider_to_meters(
        self, shift_x: float, shift_y: float
    ) -> Tuple[float, float]:
        """
        Convert GUI shift slider values to chest wall coordinates (meters).

        Args:
            shift_x: GUI Shift X slider value (-100 to +100)
            shift_y: GUI Shift Y slider value (-100 to +100)

        Returns:
            (x_m, y_m) in chest wall coordinate frame.
        """
        # Normalize slider to [0, 1]
        sx_norm = (shift_x - self.slider_min) / (self.slider_max - self.slider_min)
        sy_norm = (shift_y - self.slider_min) / (self.slider_max - self.slider_min)

        x_m = self.x_min_m + sx_norm * (self.x_max_m - self.x_min_m)
        y_m = self.y_min_m + sy_norm * (self.y_max_m - self.y_min_m)

        return x_m, y_m

    def meters_to_slider(
        self, x_m: float, y_m: float
    ) -> Tuple[int, int]:
        """
        Convert chest wall coordinates to GUI slider values.

        Returns:
            (shift_x, shift_y) as integer slider values.
        """
        sx_norm = (x_m - self.x_min_m) / (self.x_max_m - self.x_min_m)
        sy_norm = (y_m - self.y_min_m) / (self.y_max_m - self.y_min_m)

        shift_x = self.slider_min + sx_norm * (self.slider_max - self.slider_min)
        shift_y = self.slider_min + sy_norm * (self.slider_max - self.slider_min)

        return int(round(shift_x)), int(round(shift_y))


# ---------------------------------------------------------------------------
# Simulator Session — Main integration point
# ---------------------------------------------------------------------------

class SimulatorSession:
    """
    Manages a live simulation session that the GUI can query.

    Replaces the MySQL database + BLE hardware path with direct
    simulator calls. The GUI calls update() with current slider values,
    and the session returns generated B-mode images and metadata.

    Usage:
        session = SimulatorSession(scenario="left_pneumothorax")

        # Called by GUI on each slider change / timer tick
        result = session.update(
            yaw_deg=10, pitch_deg=-5, roll_deg=3,
            shift_x=20, shift_y=40, frame_idx=0,
        )

        # result contains:
        #   rgba_image: np.ndarray [H, W, 4] for OpenGL texture
        #   zone_name: str
        #   pathology_name: str
        #   pathology_class: int
        #   lung_sliding: bool
        #   mmode_pattern: str
        #   probe_x_m, probe_y_m: float
    """

    def __init__(
        self,
        scenario: str = "normal",
        image_size: Tuple[int, int] = (256, 256),
        n_frames: int = 32,
    ):
        self.scenario_name = scenario
        self.image_size = image_size
        self.n_frames = n_frames

        self.stack_config = StackConfig(
            n_frames=n_frames,
            image_size=image_size,
        )
        self.poc_generator = POCImageStackGenerator(
            scenario=scenario,
            stack_config=self.stack_config,
        )
        self.chest_map = ChestWallMapping()

        # Cache: store the last generated stack to avoid regeneration on
        # frame-only changes (probe position unchanged)
        self._cache_key: Optional[Tuple] = None
        self._cache_result: Optional[Dict] = None

    @property
    def available_scenarios(self) -> list:
        return list(SCENARIOS.keys())

    @property
    def scenario_description(self) -> str:
        return SCENARIOS[self.scenario_name].description

    def set_scenario(self, scenario: str):
        """Switch to a different clinical scenario."""
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario_name = scenario
        self.poc_generator = POCImageStackGenerator(
            scenario=scenario,
            stack_config=self.stack_config,
        )
        self._cache_key = None
        self._cache_result = None

    def update(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        pressure: float = 1.0,
        frame_idx: int = 0,
    ) -> Dict:
        """
        Generate simulator output for the current GUI state.

        Args:
            yaw_deg, pitch_deg, roll_deg: Orientation from GUI sliders (degrees).
            shift_x, shift_y: Position from GUI shift sliders (-100 to +100).
            pressure: Contact pressure (0 to 1). Default 1.0 (firm contact).
            frame_idx: Which frame of the temporal stack to display.

        Returns:
            Dict with keys:
                rgba_image: np.ndarray [H, W, 4] uint8 — for OpenGL texture
                bmode_frame: np.ndarray [H, W] float32 — raw B-mode
                zone_name: str
                zone_id: int
                pathology_name: str
                pathology_class: int
                lung_sliding: bool
                mmode_pattern: str
                mmode_image: np.ndarray [H, W, 4] uint8 — M-mode as RGBA
                probe_x_m: float
                probe_y_m: float
                n_frames: int
                scenario: str
        """
        # Convert GUI position to chest wall coordinates
        probe_x_m, probe_y_m = self.chest_map.slider_to_meters(shift_x, shift_y)

        # Convert GUI angles to IMU pose
        imu_pose = gui_angles_to_imu_pose(yaw_deg, pitch_deg, roll_deg)
        dual_imu = DualIMUReading(
            probe_imu=imu_pose,
            vehicle_imu=IMUPose(),  # Static vehicle (no motion compensation)
        )

        # Build cache key from position and pressure (not frame_idx or angles,
        # since the stack is generated per-position)
        cache_key = (round(probe_x_m, 4), round(probe_y_m, 4), round(pressure, 2))

        if cache_key != self._cache_key or self._cache_result is None:
            # Generate new stack for this position
            probe_reading = ProbeReading(
                x_m=probe_x_m,
                y_m=probe_y_m,
                pressure=pressure,
                imu=dual_imu,
            )
            self._cache_result = self.poc_generator.generate(probe_reading)
            self._cache_key = cache_key

        result = self._cache_result

        # Clamp frame index
        n_frames = result["bmode_stack"].shape[0]
        frame_idx = max(0, min(frame_idx, n_frames - 1))

        # Extract current frame and convert to RGBA
        bmode_frame = result["bmode_stack"][frame_idx]
        rgba_image = bmode_to_rgba(bmode_frame)

        # Convert M-mode to RGBA
        mmode_rgba = bmode_to_rgba(result["mmode"])

        return {
            "rgba_image": rgba_image,
            "bmode_frame": bmode_frame,
            "zone_name": result["zone"],
            "zone_id": result["zone_id"],
            "pathology_name": result["pathology"],
            "pathology_class": result["pathology_class"],
            "lung_sliding": result["lung_sliding"],
            "mmode_pattern": result["mmode_pattern"],
            "mmode_image": mmode_rgba,
            "probe_x_m": probe_x_m,
            "probe_y_m": probe_y_m,
            "n_frames": n_frames,
            "scenario": result["scenario"],
        }

    def get_zone_map(self) -> Dict[str, Tuple[float, float]]:
        """Return all zone positions for overlay visualization."""
        return self.poc_generator.zone_resolver.all_zone_positions()

    def get_zone_slider_positions(self) -> Dict[str, Tuple[int, int]]:
        """Return all zone positions as GUI slider values for navigation."""
        zone_positions = self.get_zone_map()
        result = {}
        for name, (x_m, y_m) in zone_positions.items():
            sx, sy = self.chest_map.meters_to_slider(x_m, y_m)
            result[name] = (sx, sy)
        return result
