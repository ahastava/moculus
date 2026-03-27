"""
Lung US Dataset Builder
=======================
Creates zea-compatible HDF5 datasets from:
  - Synthetic physics-based images (LungUSArtifactSynthesizer)
  - Real acquired RF data (if available)
  - IMU recordings and pressure mat grid logs

HDF5 Schema:
  /data              [N, H, W, 1]    float32 — B-mode images (polar domain)
  /labels
    /pathology_class [N]             int32
    /class_name      [N]             string
  /metadata
    /imu_probe       [N, 6]          float32 — [pitch, roll, yaw, dx, dy, dz]
    /imu_vehicle     [N, 6]          float32
    /grid_cell       [N, 2]          int32   — [row, col]
    /frame_index     [N]             int32
  /params                             group — acquisition parameters
    /center_frequency                 float32 (Hz)
    /sampling_frequency               float32 (Hz)
    /probe_pitch                      float32 (m)
    /n_elements                       int32
    /speed_of_sound                   float32 (m/s)
    /theta_range                      [2] float32 (rad)
    /rho_range                        [2] float32 (m)
"""

import numpy as np
import h5py
import torch
import torch.utils.data
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass

try:
    from .lung_us_generator import (
        LungUSArtifactSynthesizer, DualIMUReading, IMUPose,
        GridConfig, PATHOLOGY_CLASSES, N_CLASSES,
    )
    from .clinical_frames import ClinicalFrameGenerator, ClinicalPathology
except ImportError:
    from lung_us_generator import (
        LungUSArtifactSynthesizer, DualIMUReading, IMUPose,
        GridConfig, PATHOLOGY_CLASSES, N_CLASSES,
    )
    from clinical_frames import ClinicalFrameGenerator, ClinicalPathology

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset spec
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Specification for the synthetic dataset to be generated."""
    n_samples_per_class: int = 500
    image_size: Tuple[int, int] = (256, 256)
    grid_config: GridConfig = None
    imu_noise_std: float = 0.02   # Radians — simulated IMU noise
    vehicle_motion_std: float = 0.05  # Simulated ambulance/heli motion
    output_path: str = "lung_us_moculus.h5"
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # train/val/test

    def __post_init__(self):
        if self.grid_config is None:
            self.grid_config = GridConfig()


# ---------------------------------------------------------------------------
# IMU Simulator
# ---------------------------------------------------------------------------

class IMUSimulator:
    """
    Simulates realistic dual-IMU recordings for MoCoLUS training data.
    Models three vehicle scenarios: ambulance ground, helicopter hover, static.
    """

    VEHICLE_PROFILES = {
        "static": {"motion_std": 0.005, "drift_rate": 0.001},
        "ambulance_ground": {"motion_std": 0.04, "drift_rate": 0.01},
        "helicopter_hover": {"motion_std": 0.08, "drift_rate": 0.02},
        "ambulance_highway": {"motion_std": 0.06, "drift_rate": 0.015},
    }

    def sample_dual_imu(
        self,
        vehicle_type: str = "ambulance_ground",
        rng: Optional[np.random.Generator] = None,
    ) -> DualIMUReading:
        if rng is None:
            rng = np.random.default_rng()

        profile = self.VEHICLE_PROFILES.get(vehicle_type, self.VEHICLE_PROFILES["static"])
        std = profile["motion_std"]
        drift = profile["drift_rate"]

        # Probe IMU: operator holds probe (moderate intentional + unintentional motion)
        probe = IMUPose(
            pitch=rng.normal(0, 0.12),       # ±~7° typical probe tilt
            roll=rng.normal(0, 0.05),
            yaw=rng.normal(0, 0.08),
            dx=rng.normal(0, 0.005),
            dy=rng.normal(0, 0.003),
            dz=rng.normal(0, 0.003),
        )

        # Vehicle IMU: unpredictable environmental motion
        vehicle = IMUPose(
            pitch=rng.normal(0, std),
            roll=rng.normal(0, std),
            yaw=rng.normal(0, std * 0.5),
            dx=rng.normal(0, drift),
            dy=rng.normal(0, drift),
            dz=rng.normal(0, drift * 0.5),
        )

        return DualIMUReading(probe_imu=probe, vehicle_imu=vehicle)

    def sample_batch(
        self,
        n: int,
        vehicle_type: str = "ambulance_ground",
        seed: Optional[int] = None,
    ) -> List[DualIMUReading]:
        rng = np.random.default_rng(seed)
        return [self.sample_dual_imu(vehicle_type, rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------

class LungUSDatasetBuilder:
    """
    Builds a zea-compatible HDF5 dataset of synthetic lung US images
    with paired IMU data and grid cell labels.
    """

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.clinical_gen = ClinicalFrameGenerator(image_size=spec.image_size)
        self.imu_sim = IMUSimulator()

    def _generate_imu_array(self, dual_imu: DualIMUReading) -> Tuple[np.ndarray, np.ndarray]:
        probe = np.array([
            dual_imu.probe_imu.pitch, dual_imu.probe_imu.roll,
            dual_imu.probe_imu.yaw, dual_imu.probe_imu.dx,
            dual_imu.probe_imu.dy, dual_imu.probe_imu.dz,
        ], dtype=np.float32)

        vehicle = np.array([
            dual_imu.vehicle_imu.pitch, dual_imu.vehicle_imu.roll,
            dual_imu.vehicle_imu.yaw, dual_imu.vehicle_imu.dx,
            dual_imu.vehicle_imu.dy, dual_imu.vehicle_imu.dz,
        ], dtype=np.float32)

        return probe, vehicle

    def build(
        self,
        vehicle_types: Optional[List[str]] = None,
        seed: int = 42,
        verbose: bool = True,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> str:
        """
        Generate the full dataset and write to HDF5.

        Args:
            vehicle_types: List of vehicle scenarios to include.
                           If None, uses all available types.
            seed: Random seed for reproducibility.
            verbose: Show progress bar.

        Returns:
            Path to the created HDF5 file.
        """
        if vehicle_types is None:
            vehicle_types = list(IMUSimulator.VEHICLE_PROFILES.keys())

        n_per_class = self.spec.n_samples_per_class
        n_vehicles = len(vehicle_types)
        n_per_class_vehicle = n_per_class // n_vehicles

        total_samples = N_CLASSES * n_per_class
        H, W = self.spec.image_size
        grid = self.spec.grid_config
        rng = np.random.default_rng(seed)

        output_path = Path(self.spec.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Building dataset: {total_samples} samples → {output_path}")

        with h5py.File(output_path, "w") as f:
            # Create datasets
            dset_data = f.create_dataset(
                "data", shape=(total_samples, H, W, 1), dtype="float32",
                compression="gzip", compression_opts=4,
            )
            dset_class = f.create_dataset(
                "labels/pathology_class", shape=(total_samples,), dtype="int32"
            )
            dset_imu_probe = f.create_dataset(
                "metadata/imu_probe", shape=(total_samples, 6), dtype="float32"
            )
            dset_imu_vehicle = f.create_dataset(
                "metadata/imu_vehicle", shape=(total_samples, 6), dtype="float32"
            )
            dset_grid = f.create_dataset(
                "metadata/grid_cell", shape=(total_samples, 2), dtype="int32"
            )
            dset_frame = f.create_dataset(
                "metadata/frame_index", shape=(total_samples,), dtype="int32"
            )

            # Write acquisition parameters
            params = f.create_group("params")
            params.create_dataset("center_frequency", data=5e6)
            params.create_dataset("sampling_frequency", data=40e6)
            params.create_dataset("probe_pitch", data=0.3e-3)
            params.create_dataset("n_elements", data=64)
            params.create_dataset("speed_of_sound", data=1540.0)
            params.create_dataset("theta_range", data=[-0.524, 0.524])
            params.create_dataset("rho_range", data=[0.0, 0.12])

            # Generate samples
            idx = 0
            iterator = range(N_CLASSES)
            if verbose:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Generating classes")

            for cls in iterator:
                for v_type in vehicle_types:
                    for _ in range(n_per_class_vehicle):
                        if idx >= total_samples:
                            break

                        # Sample IMU
                        dual_imu = self.imu_sim.sample_dual_imu(v_type, rng)
                        probe_arr, vehicle_arr = self._generate_imu_array(dual_imu)

                        # Random grid cell
                        grid_i = int(rng.integers(0, grid.n_rows))
                        grid_j = int(rng.integers(0, grid.n_cols))

                        # Generate image using clinical frame generator
                        pathology = ClinicalPathology(cls)
                        image = self.clinical_gen.generate(pathology, seed=int(rng.integers(0, 2**31)))  # [H, W]

                        # Write
                        dset_data[idx] = image[:, :, np.newaxis]
                        dset_class[idx] = cls
                        dset_imu_probe[idx] = probe_arr
                        dset_imu_vehicle[idx] = vehicle_arr
                        dset_grid[idx] = [grid_i, grid_j]
                        dset_frame[idx] = idx

                        idx += 1

                        if progress_callback and idx % 50 == 0:
                            progress_callback({
                                "progress": idx / total_samples,
                                "samples": idx,
                                "total": total_samples,
                                "message": f"{idx}/{total_samples} samples",
                            })

            # Record actual number written
            f.attrs["n_samples"] = idx
            f.attrs["n_classes"] = N_CLASSES
            f.attrs["class_names"] = [PATHOLOGY_CLASSES[i] for i in range(N_CLASSES)]
            f.attrs["image_size"] = [H, W]
            f.attrs["vehicle_types"] = vehicle_types

        logger.info(f"Dataset written: {idx} samples → {output_path}")
        return str(output_path)

    def build_splits(
        self,
        vehicle_types: Optional[List[str]] = None,
        seed: int = 42,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Build and split into train/val/test HDF5 files."""
        full_path = self.build(vehicle_types=vehicle_types, seed=seed,
                               progress_callback=progress_callback)

        train_r, val_r, test_r = self.spec.split_ratios
        with h5py.File(full_path, "r") as f:
            N = f.attrs["n_samples"]

        rng = np.random.default_rng(seed + 1)
        indices = rng.permutation(N)
        n_train = int(N * train_r)
        n_val = int(N * val_r)

        splits = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:],
        }

        base = Path(full_path).stem
        parent = Path(full_path).parent
        split_paths = {}

        for split_name, split_idx in splits.items():
            split_idx = np.sort(split_idx)
            out_path = parent / f"{base}_{split_name}.h5"

            with h5py.File(full_path, "r") as src, h5py.File(out_path, "w") as dst:
                n = len(split_idx)
                H, W = self.spec.image_size

                dst.create_dataset("data", data=src["data"][split_idx])
                dst.create_dataset("labels/pathology_class",
                                   data=src["labels/pathology_class"][split_idx])
                dst.create_dataset("metadata/imu_probe",
                                   data=src["metadata/imu_probe"][split_idx])
                dst.create_dataset("metadata/imu_vehicle",
                                   data=src["metadata/imu_vehicle"][split_idx])
                dst.create_dataset("metadata/grid_cell",
                                   data=src["metadata/grid_cell"][split_idx])

                # Copy params
                src.copy("params", dst)

                dst.attrs["n_samples"] = n
                dst.attrs["split"] = split_name
                dst.attrs.update({k: src.attrs[k] for k in src.attrs
                                   if k != "n_samples"})

            split_paths[split_name] = str(out_path)
            logger.info(f"  {split_name}: {n} samples → {out_path}")

        return split_paths


# ---------------------------------------------------------------------------
# zea-Compatible PyTorch Dataset
# ---------------------------------------------------------------------------

class LungUSZeaDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that wraps a zea-compatible HDF5 file.
    Returns (image, pathology_class, imu_probe, imu_vehicle, grid_cell).
    """

    def __init__(self, hdf5_path: str, augment: bool = False):
        import torch.utils.data
        self.path = hdf5_path
        self.augment = augment
        self._file = None  # Lazy open (important for multiprocessing)

        # Read metadata without keeping file open
        with h5py.File(hdf5_path, "r") as f:
            self.n = int(f.attrs["n_samples"])

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        import torch
        self._open()

        image = torch.from_numpy(
            self._file["data"][idx].astype(np.float32)
        ).squeeze(-1)  # [H, W]

        pathology = torch.tensor(
            int(self._file["labels/pathology_class"][idx]), dtype=torch.long
        )

        imu_probe = torch.from_numpy(
            self._file["metadata/imu_probe"][idx].astype(np.float32)
        )
        imu_vehicle = torch.from_numpy(
            self._file["metadata/imu_vehicle"][idx].astype(np.float32)
        )
        grid_cell = torch.from_numpy(
            self._file["metadata/grid_cell"][idx].astype(np.int32)
        )

        return image, pathology, imu_probe, imu_vehicle, grid_cell

    def __del__(self):
        if self._file is not None:
            self._file.close()




# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build MoCoLUS Lung US Dataset")
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/lung_us_moculus.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vehicle-types", nargs="+",
                        default=["static", "ambulance_ground", "helicopter_hover"])
    args = parser.parse_args()

    spec = DatasetSpec(
        n_samples_per_class=args.n_per_class,
        output_path=args.output,
    )

    builder = LungUSDatasetBuilder(spec)
    paths = builder.build_splits(
        vehicle_types=args.vehicle_types,
        seed=args.seed,
    )

    print("\nDataset splits created:")
    for split, path in paths.items():
        print(f"  {split}: {path}")
