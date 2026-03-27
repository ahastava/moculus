"""
Zea DiffusionModel Training for Lung POCUS
===========================================
Trains a zea DiffusionModel on synthetic lung US images from the
ClinicalFrameGenerator (10 pathologies). Uses the pretrained echonet-dynamic
weights as initialization (transfer learning: cardiac echo → lung US).

Since zea's train_step is TF-only, this implements a PyTorch training loop
that operates directly on the zea model's internal network.

Usage:
    python -m src.train_zea_diffusion                    # defaults
    python -m src.train_zea_diffusion --epochs 200       # more epochs
    python -m src.train_zea_diffusion --from-scratch      # no pretrained init
    python -m src.train_zea_diffusion --resume checkpoints/zea_lung_pocus/latest.pt
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import math
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Callable, Optional

import keras
from zea.models.diffusion import DiffusionModel

try:
    from .clinical_frames import ClinicalFrameGenerator, ClinicalPathology
except ImportError:
    from clinical_frames import ClinicalFrameGenerator, ClinicalPathology

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_PATHOLOGIES = len(ClinicalPathology)

# ---------------------------------------------------------------------------
# On-the-fly dataset that generates images from ClinicalFrameGenerator
# ---------------------------------------------------------------------------

class ClinicalFrameDataset(torch.utils.data.Dataset):
    """
    Generates training images on-the-fly using ClinicalFrameGenerator.
    Each epoch produces fresh stochastic images (different speckle, noise).
    """

    def __init__(
        self,
        n_per_class: int = 200,
        image_size: int = 256,
        augment: bool = True,
    ):
        self.n_per_class = n_per_class
        self.n_classes = N_PATHOLOGIES
        self.total = n_per_class * self.n_classes
        self.gen = ClinicalFrameGenerator(image_size=(image_size, image_size))
        self.augment = augment
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        cls = idx // self.n_per_class
        sample_idx = idx % self.n_per_class

        # Seed from epoch + sample for reproducibility but fresh each epoch
        seed = self._epoch * 100000 + cls * 10000 + sample_idx
        pathology = ClinicalPathology(cls)
        frame = self.gen.generate(pathology, seed=seed)  # [H, W] float32

        if self.augment:
            rng = np.random.default_rng(seed + 77777)
            # Random horizontal flip
            if rng.random() < 0.5:
                frame = frame[:, ::-1].copy()
            # Small brightness/contrast jitter
            frame = frame * rng.uniform(0.9, 1.1)
            frame = np.clip(frame + rng.uniform(-0.05, 0.05), 0, 1)

        # Zea expects [H, W, 1] float32
        image = torch.from_numpy(frame.astype(np.float32)).unsqueeze(-1)
        label = torch.tensor(cls, dtype=torch.long)
        return image, label


# ---------------------------------------------------------------------------
# Cosine noise schedule (for PyTorch training loop)
# ---------------------------------------------------------------------------

def cosine_schedule(n_steps: int, s: float = 0.008):
    steps = torch.arange(n_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return alpha_bar.float()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ZeaDiffusionTrainer:

    def __init__(
        self,
        image_size: int = 256,
        n_per_class: int = 200,
        batch_size: int = 4,
        lr: float = 1e-4,
        n_epochs: int = 100,
        n_diffusion_steps: int = 1000,
        use_pretrained: bool = True,
        output_dir: str = "checkpoints/zea_lung_pocus",
        resume_path: Optional[str] = None,
    ):
        self.image_size = image_size
        self.n_epochs = n_epochs
        self.n_steps = n_diffusion_steps
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build zea DiffusionModel
        self._build_model(image_size, use_pretrained)

        # Dataset
        self.train_dataset = ClinicalFrameDataset(
            n_per_class=n_per_class, image_size=image_size, augment=True
        )
        self.val_dataset = ClinicalFrameDataset(
            n_per_class=max(20, n_per_class // 5), image_size=image_size, augment=False
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=1, pin_memory=True,
        )

        # Noise schedule
        alpha_bar = cosine_schedule(n_diffusion_steps).to(self.device)
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()

        # Optimizer — operate on the zea model's internal torch network
        self.network = self.model.network
        torch_params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(torch_params, lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )
        self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        # EMA (mirrors zea's internal EMA)
        self.ema_decay = 0.999
        self.ema_shadow = {k: v.clone() for k, v in self.network.state_dict().items()}

        self.start_epoch = 0
        self.step = 0

        if resume_path:
            self._load_checkpoint(resume_path)

        total_params = sum(p.numel() for p in torch_params)
        logger.info(f"Device: {self.device}")
        logger.info(f"Network params: {total_params / 1e6:.2f}M")
        logger.info(f"Train samples: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")

    def _build_model(self, image_size: int, use_pretrained: bool):
        if use_pretrained:
            logger.info("Loading pretrained echonet-dynamic weights...")
            try:
                base = DiffusionModel.from_preset(
                    "diffusion-echonet-dynamic",
                    load_weights=True,
                    guidance={"name": "dps", "params": {"disable_jit": True}},
                )
                pretrained_state = base.network.state_dict()
                logger.info(f"Pretrained model input: {base.input_shape}")
            except Exception as e:
                logger.warning(f"Could not load pretrained weights: {e}")
                pretrained_state = None
        else:
            pretrained_state = None

        # Build model at our target resolution
        self.model = DiffusionModel(
            input_shape=(image_size, image_size, 1),
            network_name="unet_time_conditional",
            operator="identity",
            guidance={"name": "dps", "params": {"disable_jit": True}},
            network_kwargs={"widths": [32, 64, 96, 128], "block_depth": 2},
        )
        # Compile to build all layers
        self.model.compile(optimizer="adam")
        # Build the network by passing dummy data (needs [images, noise_variance])
        dummy_img = torch.randn(1, image_size, image_size, 1).to(self.device)
        dummy_var = torch.ones(1, 1, 1, 1).to(self.device) * 0.5
        with torch.no_grad():
            self.model.network([dummy_img, dummy_var], training=False)

        if pretrained_state is not None:
            # Transfer compatible weights (same-shape params only)
            model_state = self.model.network.state_dict()
            transferred = 0
            for k in model_state:
                if k in pretrained_state and model_state[k].shape == pretrained_state[k].shape:
                    model_state[k] = pretrained_state[k]
                    transferred += 1
            self.model.network.load_state_dict(model_state)
            logger.info(f"Transferred {transferred}/{len(model_state)} weight tensors from pretrained")

    def _diffusion_schedule(self, t_int):
        """Convert integer timestep to (noise_rate, signal_rate) for zea network.
        Uses cosine schedule: signal_rate = cos(t_frac * pi/2), noise_rate = sin(t_frac * pi/2)
        """
        t_frac = t_int.float() / self.n_steps  # [0, 1)
        # Match zea's diffusion_schedule: sinusoidal interpolation
        signal_rate = torch.cos(t_frac * math.pi / 2).view(-1, 1, 1, 1)
        noise_rate = torch.sin(t_frac * math.pi / 2).view(-1, 1, 1, 1)
        return noise_rate, signal_rate

    def _ema_update(self):
        for k, v in self.network.state_dict().items():
            self.ema_shadow[k] = self.ema_decay * self.ema_shadow[k] + (1 - self.ema_decay) * v

    def _train_epoch(self, epoch: int):
        self.network.train()
        self.train_dataset.set_epoch(epoch)
        total_loss = 0.0
        n_batches = 0

        for images, labels in self.train_loader:
            # images: [B, H, W, 1], labels: [B]
            images = images.to(self.device)
            B = images.shape[0]

            t = torch.randint(0, self.n_steps, (B,), device=self.device)
            noise = torch.randn_like(images)
            noise_rate, signal_rate = self._diffusion_schedule(t)
            x_t = signal_rate * images + noise_rate * noise

            with autocast("cuda", enabled=self.device.type == "cuda"):
                # Zea unet_time_conditional expects [images, noise_variance]
                noise_var = noise_rate ** 2  # [B, 1, 1, 1]
                pred_noise = self.network([x_t, noise_var], training=True)
                loss = F.mse_loss(pred_noise, noise)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._ema_update()

            total_loss += loss.item()
            n_batches += 1
            self.step += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self):
        self.network.eval()
        total_loss = 0.0
        n = 0
        for images, labels in self.val_loader:
            images = images.to(self.device)
            B = images.shape[0]
            t = torch.randint(0, self.n_steps, (B,), device=self.device)
            noise = torch.randn_like(images)
            noise_rate, signal_rate = self._diffusion_schedule(t)
            x_t = signal_rate * images + noise_rate * noise
            noise_var = noise_rate ** 2
            pred = self.network([x_t, noise_var], training=False)
            total_loss += F.mse_loss(pred, noise).item()
            n += 1
        return total_loss / max(n, 1)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "step": self.step,
            "network": self.network.state_dict(),
            "ema_shadow": self.ema_shadow,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "image_size": self.image_size,
            "n_steps": self.n_steps,
        }
        path = self.output_dir / f"epoch_{epoch:04d}.pt"
        torch.save(state, path)
        # Always save latest
        torch.save(state, self.output_dir / "latest.pt")
        if is_best:
            torch.save(state, self.output_dir / "best.pt")
        logger.info(f"Saved checkpoint: {path}")

    def _save_zea_preset(self):
        """Save as zea-compatible preset directory for from_preset() loading."""
        preset_dir = self.output_dir / "zea_preset"
        preset_dir.mkdir(parents=True, exist_ok=True)

        # Apply EMA weights before saving
        orig_state = self.network.state_dict()
        self.network.load_state_dict(self.ema_shadow)
        try:
            self.model.save_to_preset(str(preset_dir))
            logger.info(f"Saved zea preset: {preset_dir}")
        except Exception as e:
            logger.warning(f"Could not save zea preset: {e}")
            # Fallback: save just the weights
            torch.save(self.ema_shadow, preset_dir / "ema_weights.pt")
            logger.info(f"Saved EMA weights to {preset_dir / 'ema_weights.pt'}")
        finally:
            self.network.load_state_dict(orig_state)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        self.ema_shadow = ckpt["ema_shadow"]
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.step = ckpt["step"]
        logger.info(f"Resumed from {path} (epoch {ckpt['epoch']}, step {self.step})")

    def train(self, progress_callback: Optional[Callable[[dict], None]] = None):
        """
        Run training loop.

        Args:
            progress_callback: Optional callable invoked each epoch with
                {"epoch": int, "total_epochs": int, "train_loss": float,
                 "val_loss": float|None, "best_val": float}.
                Used by the web server to relay progress via SSE.
        """
        logger.info(f"Training for {self.n_epochs} epochs ({N_PATHOLOGIES} pathologies)")
        best_val = float("inf")

        for epoch in range(self.start_epoch, self.n_epochs):
            train_loss = self._train_epoch(epoch)
            self.scheduler.step()

            log_msg = f"Epoch {epoch:3d}/{self.n_epochs} | Train Loss: {train_loss:.5f}"
            val_loss = None

            if (epoch + 1) % 5 == 0:
                val_loss = self._validate()
                log_msg += f" | Val Loss: {val_loss:.5f}"
                is_best = val_loss < best_val
                if is_best:
                    best_val = val_loss
            else:
                is_best = False

            logger.info(log_msg)

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, is_best=is_best)

            if progress_callback:
                progress_callback({
                    "epoch": epoch,
                    "total_epochs": self.n_epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "loss": train_loss,
                    "best_val": best_val,
                })

        self._save_checkpoint(self.n_epochs - 1)
        self._save_zea_preset()

        logger.info(f"Training complete. Best val loss: {best_val:.5f}")
        logger.info(f"Model saved to: {self.output_dir}")
        logger.info(f"Zea preset: {self.output_dir / 'zea_preset'}")

    @torch.no_grad()
    def sample(self, n_samples: int = 4, n_inference_steps: int = 50):
        """Generate samples using zea's built-in DDIM sampler."""
        # Apply EMA weights for sampling
        orig_state = self.network.state_dict()
        self.network.load_state_dict(self.ema_shadow)
        self.network.eval()

        try:
            samples = self.model.sample(n_samples=n_samples, n_steps=n_inference_steps)
            if hasattr(samples, 'cpu'):
                samples = samples.cpu().numpy()
            else:
                samples = np.array(samples)
            return samples
        finally:
            self.network.load_state_dict(orig_state)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train zea DiffusionModel on Lung POCUS")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--n-per-class", type=int, default=200,
                        help="Training samples per pathology class per epoch")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train from scratch (no pretrained echonet init)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--output-dir", type=str, default="checkpoints/zea_lung_pocus")
    parser.add_argument("--sample-only", action="store_true",
                        help="Just generate samples from existing checkpoint")
    args = parser.parse_args()

    trainer = ZeaDiffusionTrainer(
        image_size=args.image_size,
        n_per_class=args.n_per_class,
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=args.epochs,
        n_diffusion_steps=args.diffusion_steps,
        use_pretrained=not args.from_scratch,
        output_dir=args.output_dir,
        resume_path=args.resume,
    )

    if args.sample_only:
        samples = trainer.sample(n_samples=8, n_inference_steps=50)
        logger.info(f"Generated {samples.shape[0]} samples, shape: {samples.shape}")
        # Save preview
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, ax in enumerate(axes.flat):
            if i < len(samples):
                ax.imshow(samples[i, :, :, 0], cmap="gray")
                ax.set_title(f"Sample {i}")
            ax.axis("off")
        fig.suptitle("Zea DiffusionModel Samples")
        out_path = Path(args.output_dir) / "samples.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"Samples saved to: {out_path}")
    else:
        trainer.train()
