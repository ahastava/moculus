"""
MoCoLUS × zea Integration: Synthetic Lung Ultrasound Generator
==============================================================
Bridges zea's cognitive ultrasound pipeline with MoCoLUS's dual-IMU
motion compensation and gridded multiplexer assembly.

Architecture:
  IMU Pose + Pressure Grid ──► PoseCorrectedPipeline ──► DiffusionPrior
                                                              │
                              6-Class Lung Pathology ◄────────┘
                              (per grid cell, per frame)

Pathology Classes:
  0: Healthy
  1: Tension Pneumothorax
  2: A-lines (reverberation artifacts)
  3: B-lines (comet-tail artifacts)
  4: Lung Sliding
  5: ARDS (diffuse B-lines / white lung)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from pathlib import Path

# Set PyTorch as Keras backend BEFORE importing zea or keras
os.environ.setdefault("KERAS_BACKEND", "torch")

try:
    import keras
    import zea
    from zea.models.diffusion import DiffusionModel
    _HAS_ZEA = True
except ImportError:
    _HAS_ZEA = False


# ---------------------------------------------------------------------------
# Pathology labels
# ---------------------------------------------------------------------------

PATHOLOGY_CLASSES = {
    0: "normal_a_profile",
    1: "pneumothorax",
    2: "b_lines_focal",
    3: "b_lines_diffuse",
    4: "consolidation",
    5: "pleural_effusion",
    6: "ards_white_lung",
    7: "lung_point",
    8: "pleural_thickening",
    9: "interstitial_syndrome",
}

# Legacy 6-class mapping for backwards compatibility
PATHOLOGY_CLASSES_LEGACY = {
    0: "healthy", 1: "tension_pneumothorax", 2: "a_lines",
    3: "b_lines", 4: "lung_sliding", 5: "ards",
}

N_CLASSES = len(PATHOLOGY_CLASSES)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IMUPose:
    """6-DOF pose reading from a single IMU.
    Angles in radians, translations in meters.
    """
    pitch: float = 0.0   # probe tilt in elevation (θ)
    roll: float = 0.0    # probe rotation around beam axis (φ)
    yaw: float = 0.0     # in-plane rotation
    dx: float = 0.0      # lateral translation
    dy: float = 0.0      # axial translation (depth change)
    dz: float = 0.0      # elevation translation

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.pitch, self.roll, self.yaw, self.dx, self.dy, self.dz],
            dtype=torch.float32
        )

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "IMUPose":
        t = t.cpu().numpy()
        return cls(pitch=t[0], roll=t[1], yaw=t[2], dx=t[3], dy=t[4], dz=t[5])


@dataclass
class DualIMUReading:
    """Fused reading from primary (probe) + secondary (vehicle) IMU."""
    probe_imu: IMUPose   # Primary IMU attached to probe
    vehicle_imu: IMUPose  # Secondary IMU tracking vehicle motion

    @property
    def compensated_pose(self) -> IMUPose:
        """Subtract vehicle motion to get true probe orientation."""
        return IMUPose(
            pitch=self.probe_imu.pitch - self.vehicle_imu.pitch,
            roll=self.probe_imu.roll - self.vehicle_imu.roll,
            yaw=self.probe_imu.yaw - self.vehicle_imu.yaw,
            dx=self.probe_imu.dx - self.vehicle_imu.dx,
            dy=self.probe_imu.dy - self.vehicle_imu.dy,
            dz=self.probe_imu.dz - self.vehicle_imu.dz,
        )


@dataclass
class GridConfig:
    """Configuration for the pressure-sensing mat grid."""
    n_rows: int = 8          # Number of grid rows
    n_cols: int = 8          # Number of grid columns
    cell_width_m: float = 0.025    # 25mm per cell
    cell_height_m: float = 0.025
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    probe_aperture_deg: float = 60.0  # Total fan angle in degrees
    max_depth_m: float = 0.12         # 12 cm max imaging depth


@dataclass
class PipelineParams:
    """Parameters passed to zea Pipeline at inference time."""
    theta_range: Tuple[float, float] = (-0.524, 0.524)  # ±30° in radians
    rho_range: Tuple[float, float] = (0.0, 0.12)
    speed_of_sound: float = 1540.0


# ---------------------------------------------------------------------------
# IMU → Pipeline Parameter Converter
# ---------------------------------------------------------------------------

class PoseCorrector:
    """
    Converts a DualIMUReading + grid cell position into
    zea Pipeline parameters, compensating for vehicle motion.
    """

    def __init__(self, grid_config: GridConfig):
        self.grid = grid_config

    def get_pipeline_params(
        self,
        dual_imu: DualIMUReading,
        grid_cell: Tuple[int, int]
    ) -> PipelineParams:
        pose = dual_imu.compensated_pose

        # Fan angle adjusted by probe pitch
        half_aperture = math.radians(self.grid.probe_aperture_deg / 2)
        theta_center = pose.pitch
        theta_range = (
            theta_center - half_aperture,
            theta_center + half_aperture,
        )

        # Depth range adjusted by axial IMU translation
        depth_min = max(0.001, -pose.dy)  # small positive minimum
        depth_max = self.grid.max_depth_m + pose.dy
        rho_range = (depth_min, max(depth_min + 0.01, depth_max))

        return PipelineParams(
            theta_range=theta_range,
            rho_range=rho_range,
            speed_of_sound=1540.0,
        )

    def get_probe_world_position(
        self,
        dual_imu: DualIMUReading,
        grid_cell: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """Return (x, y, z) of probe center in world coordinates."""
        i, j = grid_cell
        pose = dual_imu.compensated_pose
        x = self.grid.origin_x_m + j * self.grid.cell_width_m + pose.dx
        y = self.grid.origin_y_m + i * self.grid.cell_height_m + pose.dz
        z = pose.dy
        return (x, y, z)


# ---------------------------------------------------------------------------
# Conditioning Encoder (for diffusion model)
# ---------------------------------------------------------------------------

class LungUSConditioningEncoder(nn.Module):
    """
    Encodes (pathology_class, imu_pose, grid_cell) into a
    conditioning vector for the diffusion model.

    Output shape: [B, D_cond]  where D_cond = 128
    """

    D_CLASS = 32
    D_POSE = 48
    D_GRID = 32
    D_COND = 128  # D_CLASS + D_POSE + D_GRID = 112 → projected to 128

    def __init__(self, n_classes: int = N_CLASSES, grid_rows: int = 8, grid_cols: int = 8):
        super().__init__()
        self.n_classes = n_classes
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # Class embedding (learned)
        self.class_embed = nn.Embedding(n_classes, self.D_CLASS)

        # IMU pose encoder: 6-DOF → D_POSE
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, self.D_POSE),
        )

        # Grid cell positional encoding (sinusoidal)
        self.grid_embed_dim = self.D_GRID

        # Projection to final conditioning dim
        self.proj = nn.Linear(
            self.D_CLASS + self.D_POSE + self.D_GRID,
            self.D_COND
        )

    def sinusoidal_grid_encode(
        self, grid_i: torch.Tensor, grid_j: torch.Tensor
    ) -> torch.Tensor:
        """Sinusoidal encoding of 2D grid position."""
        B = grid_i.shape[0]
        D = self.grid_embed_dim // 4  # 8 freqs for i, 8 for j
        freqs = torch.arange(D, device=grid_i.device).float()
        freqs = 2.0 * math.pi * freqs / self.grid_rows

        i_norm = grid_i.float() / self.grid_rows
        j_norm = grid_j.float() / self.grid_cols

        i_enc = torch.cat([
            torch.sin(i_norm.unsqueeze(-1) * freqs.unsqueeze(0)),
            torch.cos(i_norm.unsqueeze(-1) * freqs.unsqueeze(0)),
        ], dim=-1)  # [B, D_GRID//2]

        j_enc = torch.cat([
            torch.sin(j_norm.unsqueeze(-1) * freqs.unsqueeze(0)),
            torch.cos(j_norm.unsqueeze(-1) * freqs.unsqueeze(0)),
        ], dim=-1)  # [B, D_GRID//2]

        return torch.cat([i_enc, j_enc], dim=-1)  # [B, D_GRID]

    def forward(
        self,
        pathology_class: torch.Tensor,  # [B] long
        imu_pose: torch.Tensor,          # [B, 6] float
        grid_i: torch.Tensor,            # [B] long
        grid_j: torch.Tensor,            # [B] long
    ) -> torch.Tensor:
        class_emb = self.class_embed(pathology_class)          # [B, D_CLASS]
        pose_emb = self.pose_encoder(imu_pose)                 # [B, D_POSE]
        grid_emb = self.sinusoidal_grid_encode(grid_i, grid_j) # [B, D_GRID]

        combined = torch.cat([class_emb, pose_emb, grid_emb], dim=-1)
        return self.proj(combined)  # [B, D_COND]


# ---------------------------------------------------------------------------
# Physics-Based Lung US Artifact Synthesizer
# ---------------------------------------------------------------------------

class LungUSArtifactSynthesizer:
    """
    Generates physics-consistent lung US B-mode images using the
    ClinicalFrameGenerator for all 10 pathology classes.

    Delegates to ClinicalFrameGenerator for clinically accurate rendering.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        depth_range_m: Tuple[float, float] = (0.0, 0.12),
        center_freq_hz: float = 5e6,
    ):
        self.H, self.W = image_size
        try:
            from .clinical_frames import ClinicalFrameGenerator, ClinicalPathology
        except ImportError:
            from clinical_frames import ClinicalFrameGenerator, ClinicalPathology
        self._clinical_gen = ClinicalFrameGenerator(
            image_size=image_size,
            depth_range_m=depth_range_m,
            center_freq_mhz=center_freq_hz / 1e6,
        )
        self._ClinicalPathology = ClinicalPathology

    def generate(self, pathology_class: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate a B-mode frame for any of the 10 pathology classes."""
        pathology = self._ClinicalPathology(pathology_class)
        return self._clinical_gen.generate(pathology, seed=seed)


# ---------------------------------------------------------------------------
# zea Pipeline Builder (Motion-Compensated)
# ---------------------------------------------------------------------------

class MotionCompensatedPipeline:
    """
    Wraps a zea.Pipeline and updates its parameters from IMU readings
    before each inference call. Supports JIT compilation.
    """

    def __init__(self, jit: bool = True):
        self.pipeline = zea.Pipeline(
            [
                zea.ops.ScanConvert(order=2),
            ],
            jit_options="ops" if jit else None,
        )
        self._pose_corrector: Optional[PoseCorrector] = None

    def set_grid_config(self, grid_config: GridConfig):
        self._pose_corrector = PoseCorrector(grid_config)

    def forward(
        self,
        polar_data: torch.Tensor,   # [B, H_polar, W_polar, 1]
        dual_imu: DualIMUReading,
        grid_cell: Tuple[int, int],
    ) -> torch.Tensor:
        assert self._pose_corrector is not None, "Call set_grid_config() first"

        params = self._pose_corrector.get_pipeline_params(dual_imu, grid_cell)
        pipeline_params = self.pipeline.prepare_parameters(
            theta_range=list(params.theta_range),
            rho_range=list(params.rho_range),
        )

        # Convert torch → numpy for zea (Keras/numpy backend)
        np_data = polar_data.detach().cpu().numpy()
        result = self.pipeline(data=np_data, **pipeline_params)["data"]

        return torch.from_numpy(np.array(result))


# ---------------------------------------------------------------------------
# Main Generator Class
# ---------------------------------------------------------------------------

class MoCoLUSLungUSGenerator:
    """
    Top-level synthetic lung US generator for MoCoLUS.

    Combines:
    - Physics-based artifact synthesis (fast, no GPU required)
    - zea DiffusionModel for photorealistic refinement (GPU required)
    - Motion compensation via dual IMU
    - Grid cell spatial indexing

    Usage:
        generator = MoCoLUSLungUSGenerator.from_pretrained(
            model_path="path/to/lung_us_diffusion",
            grid_config=GridConfig(n_rows=8, n_cols=8),
        )

        image = generator.generate(
            pathology_class=3,  # B-lines
            dual_imu=DualIMUReading(probe_imu=..., vehicle_imu=...),
            grid_cell=(3, 5),
        )
    """

    def __init__(
        self,
        diffusion_model: Optional["DiffusionModel"],
        grid_config: GridConfig,
        image_size: Tuple[int, int] = (256, 256),
        device: str = "cuda",
        use_physics_fallback: bool = True,
    ):
        self.model = diffusion_model
        self.grid_config = grid_config
        self.image_size = image_size
        self.device = device
        self.use_physics_fallback = use_physics_fallback

        self.physics = LungUSArtifactSynthesizer(image_size=image_size)
        self.pipeline = MotionCompensatedPipeline(jit=True)
        self.pipeline.set_grid_config(grid_config)

        self.conditioning_encoder = LungUSConditioningEncoder(
            n_classes=N_CLASSES,
            grid_rows=grid_config.n_rows,
            grid_cols=grid_config.n_cols,
        ).to(device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        grid_config: GridConfig,
        device: str = "cuda",
    ) -> "MoCoLUSLungUSGenerator":
        """Load generator with pretrained diffusion model.

        Tries in order:
          1. zea preset directory (from save_to_preset)
          2. PyTorch checkpoint with EMA weights
          3. HuggingFace hub preset name
          4. Falls back to physics-only mode
        """
        model = None
        model_dir = Path(model_path)

        # Try loading from PyTorch checkpoint (our training output)
        if model_dir.is_dir():
            ckpt_path = model_dir / "latest.pt"
            if not ckpt_path.exists():
                ckpt_path = model_dir / "best.pt"
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    img_size = ckpt.get("image_size", 256)
                    model = DiffusionModel(
                        input_shape=(img_size, img_size, 1),
                        network_name="unet_time_conditional",
                        operator="identity",
                        guidance={"name": "dps", "params": {"disable_jit": True}},
                        network_kwargs={"widths": [32, 64, 96, 128], "block_depth": 2},
                    )
                    model.compile(optimizer="adam")
                    dummy_img = torch.randn(1, img_size, img_size, 1).to(device)
                    dummy_var = torch.ones(1, 1, 1, 1).to(device) * 0.5
                    with torch.no_grad():
                        model.network([dummy_img, dummy_var], training=False)
                    # Load EMA weights (better quality)
                    weights = ckpt.get("ema_shadow", ckpt.get("network", None))
                    if weights:
                        model.network.load_state_dict(weights)
                    print(f"[MoCoLUS] Loaded trained model from: {ckpt_path}")
                except Exception as e:
                    print(f"[MoCoLUS] Warning: Checkpoint load failed ({e})")
                    model = None

        # Try zea preset
        if model is None:
            try:
                model = DiffusionModel.from_preset(
                    model_path,
                    guidance={"name": "dps", "params": {"disable_jit": True}},
                )
                print(f"[MoCoLUS] Loaded zea preset from: {model_path}")
            except Exception as e:
                print(f"[MoCoLUS] Warning: Could not load diffusion model ({e}). "
                      f"Using physics-only mode.")
                model = None

        return cls(
            diffusion_model=model,
            grid_config=grid_config,
            device=device,
            use_physics_fallback=(model is None),
        )

    @torch.no_grad()
    def generate(
        self,
        pathology_class: int,
        dual_imu: DualIMUReading,
        grid_cell: Tuple[int, int],
        n_diffusion_steps: int = 50,
        use_physics_only: bool = False,
    ) -> np.ndarray:
        """
        Generate a single synthetic lung US image.

        Args:
            pathology_class: 0–5 (see PATHOLOGY_CLASSES)
            dual_imu: Dual IMU reading (probe + vehicle)
            grid_cell: (row, col) position on pressure mat
            n_diffusion_steps: DDIM steps (fewer = faster but lower quality)
            use_physics_only: Skip diffusion, return physics-only image

        Returns:
            np.ndarray of shape [H, W] in [0, 1] range
        """
        # Physics-based generation (always fast)
        physics_image = self.physics.generate(pathology_class, seed=None)  # [H, W]

        if use_physics_only or self.model is None:
            return physics_image

        # Build conditioning vector
        class_t = torch.tensor([pathology_class], dtype=torch.long, device=self.device)
        imu_t = dual_imu.compensated_pose.to_tensor().unsqueeze(0).to(self.device)
        gi_t = torch.tensor([grid_cell[0]], dtype=torch.long, device=self.device)
        gj_t = torch.tensor([grid_cell[1]], dtype=torch.long, device=self.device)

        # Encode physics image as conditioning input for posterior sampling
        # (uses physics image as a noisy measurement to guide diffusion)
        physics_tensor = torch.from_numpy(physics_image).unsqueeze(0).unsqueeze(-1)
        physics_tensor = physics_tensor.to(self.device)

        # Generate using diffusion prior (posterior sampling around physics image)
        polar_samples = self.model.sample(
            n_samples=1,
            n_steps=n_diffusion_steps,
            verbose=False,
        )

        # Convert polar output through motion-compensated scan conversion
        cartesian = self.pipeline.forward(
            polar_data=torch.from_numpy(np.array(polar_samples)),
            dual_imu=dual_imu,
            grid_cell=grid_cell,
        )

        return cartesian.squeeze().cpu().numpy()

    def generate_batch(
        self,
        pathology_class: int,
        dual_imu: DualIMUReading,
        grid_cell: Tuple[int, int],
        batch_size: int = 8,
        n_diffusion_steps: int = 50,
    ) -> np.ndarray:
        """Generate a batch of diverse samples for the same condition."""
        if self.model is None:
            # Physics-only: add small stochastic variation per sample
            batch = []
            for i in range(batch_size):
                img = self.physics.generate(pathology_class, seed=None)
                batch.append(img)
            return np.stack(batch)  # [B, H, W]

        samples = self.model.sample(
            n_samples=batch_size,
            n_steps=n_diffusion_steps,
            verbose=True,
        )
        samples = keras.ops.squeeze(samples, axis=-1)
        return np.array(samples)

    def generate_full_grid(
        self,
        pathology_map: np.ndarray,   # [n_rows, n_cols] pathology class per cell
        dual_imu: DualIMUReading,
        use_physics_only: bool = False,
    ) -> np.ndarray:
        """
        Generate synthetic images for all active grid cells.

        Args:
            pathology_map: [n_rows, n_cols] integer array of pathology classes
            dual_imu: Current IMU reading
            use_physics_only: Fast physics-only generation

        Returns:
            np.ndarray of shape [n_rows, n_cols, H, W]
        """
        nr, nc = self.grid_config.n_rows, self.grid_config.n_cols
        H, W = self.image_size
        output = np.zeros((nr, nc, H, W), dtype=np.float32)

        for i in range(nr):
            for j in range(nc):
                cls = int(pathology_map[i, j])
                output[i, j] = self.generate(
                    pathology_class=cls,
                    dual_imu=dual_imu,
                    grid_cell=(i, j),
                    use_physics_only=use_physics_only,
                )
        return output
