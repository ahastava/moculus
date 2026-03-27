"""
MoCoLUS Lung US Diffusion Model Training
=========================================
Fine-tunes / trains from scratch a zea DiffusionModel on the synthetic
lung US dataset for all 6 pathology classes.

Key design choices:
- PyTorch backend via Keras 3
- Classifier-free guidance on pathology class + IMU pose
- DDIM sampling for fast inference
- Mixed precision (AMP) for NVIDIA ThinkStation PX
- TensorBoard logging + optional W&B

Usage (on ThinkStation via SSH):
    python train_diffusion.py --config configs/lung_us_diffusion.yaml
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional
import logging
import math

import keras

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "data": {
        "train_path": "data/lung_us_moculus_train.h5",
        "val_path": "data/lung_us_moculus_val.h5",
        "image_size": [256, 256],
        "n_classes": 10,
    },
    "model": {
        "preset": None,               # None = train from scratch
        "unet_channels": [64, 128, 256, 512],
        "unet_attn_resolutions": [16, 8],
        "cond_dim": 128,
    },
    "training": {
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "n_epochs": 100,
        "n_diffusion_steps": 1000,    # Training noise schedule steps
        "n_inference_steps": 50,      # DDIM steps at eval
        "guidance_scale": 3.0,        # CFG scale
        "p_uncond": 0.15,             # Prob of dropping conditioning (CFG training)
        "grad_clip": 1.0,
        "ema_decay": 0.999,
        "mixed_precision": True,
        "save_every": 10,
        "val_every": 5,
        "output_dir": "checkpoints/lung_us_diffusion",
        "device": "cuda",
    },
    "logging": {
        "tensorboard": True,
        "wandb": False,
        "project": "moculus-lung-us",
    }
}


# ---------------------------------------------------------------------------
# DDPM Noise Schedule
# ---------------------------------------------------------------------------

def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).
    Preferred over linear for medical image diffusion.
    """
    steps = torch.arange(n_steps + 1, dtype=torch.float64)
    alphas_bar = torch.cos(((steps / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return torch.clamp(betas, 0.0001, 0.9999).float()


class DDPMSchedule:
    """DDPM forward process: q(x_t | x_0)."""

    def __init__(self, n_steps: int = 1000, device: str = "cuda"):
        self.n_steps = n_steps
        betas = cosine_beta_schedule(n_steps).to(device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register = lambda name, val: setattr(self, name, val)
        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.sqrt_alphas_bar = alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1.0 - alphas_bar).sqrt()

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t from x_0: x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε"""
        a = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        return a * x0 + b * noise


# ---------------------------------------------------------------------------
# Minimal U-Net Denoiser (score network)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        d = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(d, device=t.device) / d)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.c_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_ch))
        self.conv2 = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(x)
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = h + self.c_proj(c_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class LungUSDenoiser(nn.Module):
    """
    Compact U-Net score network for lung US images.
    Conditioned on time embedding + (class, IMU pose, grid cell) vector.
    """

    def __init__(self, channels=(64, 128, 256, 512), cond_dim: int = 128):
        super().__init__()
        T_DIM = 256
        self.t_embed = SinusoidalTimeEmbed(T_DIM)

        # Encoder
        self.enc0 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_blocks.append(ResBlock(channels[i], channels[i], T_DIM, cond_dim))
            self.downs.append(nn.Conv2d(channels[i], channels[i+1], 4, 2, 1))

        # Bottleneck
        self.mid = ResBlock(channels[-1], channels[-1], T_DIM, cond_dim)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        rev_ch = list(reversed(channels))
        for i in range(len(channels) - 1):
            self.ups.append(nn.ConvTranspose2d(rev_ch[i], rev_ch[i+1], 4, 2, 1))
            self.dec_blocks.append(ResBlock(rev_ch[i+1] * 2, rev_ch[i+1], T_DIM, cond_dim))

        self.out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], 1, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,          # [B, 1, H, W]
        t: torch.Tensor,           # [B]
        cond: torch.Tensor,        # [B, cond_dim]
        null_cond: Optional[torch.Tensor] = None,  # for CFG
    ) -> torch.Tensor:
        t_emb = self.t_embed(t)

        h = self.enc0(x)
        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            h = block(h, t_emb, cond)
            skips.append(h)
            h = down(h)

        h = self.mid(h, t_emb, cond)

        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb, cond)

        return self.out(h)


# ---------------------------------------------------------------------------
# EMA Helper
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LungUSDiffusionTrainer:

    def __init__(self, config: dict):
        self.cfg = config
        self.tc = config["training"]
        self.device = torch.device(self.tc["device"] if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on: {self.device}")

        self._build_dataloaders()
        self._build_model()
        self._build_optimizer()
        self._setup_logging()

        self.schedule = DDPMSchedule(self.tc["n_diffusion_steps"], str(self.device))
        self.scaler = GradScaler("cuda", enabled=self.tc["mixed_precision"])
        self.ema = EMA(self.denoiser, self.tc["ema_decay"])

        self.step = 0
        self.epoch = 0

    def _build_dataloaders(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from lung_us_dataset import LungUSZeaDataset

        dc = self.cfg["data"]
        self.train_dataset = LungUSZeaDataset(dc["train_path"], augment=True)
        self.val_dataset = LungUSZeaDataset(dc["val_path"], augment=False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.tc["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.tc["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        logger.info(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")

    def _build_model(self):
        from lung_us_generator import LungUSConditioningEncoder, GridConfig

        mc = self.cfg["model"]
        dc = self.cfg["data"]

        self.cond_encoder = LungUSConditioningEncoder(
            n_classes=dc["n_classes"]
        ).to(self.device)

        self.denoiser = LungUSDenoiser(
            channels=mc["unet_channels"],
            cond_dim=mc["cond_dim"],
        ).to(self.device)

        # Null conditioning for classifier-free guidance
        self.null_cond = nn.Parameter(
            torch.zeros(mc["cond_dim"]).to(self.device)
        )

        total_params = sum(p.numel() for p in self.denoiser.parameters())
        logger.info(f"Denoiser parameters: {total_params / 1e6:.1f}M")

    def _build_optimizer(self):
        params = (
            list(self.denoiser.parameters()) +
            list(self.cond_encoder.parameters()) +
            [self.null_cond]
        )
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.tc["lr"],
            weight_decay=self.tc["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.tc["n_epochs"],
            eta_min=self.tc["lr"] * 0.01,
        )

    def _setup_logging(self):
        self.out_dir = Path(self.tc["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg["logging"]["tensorboard"]:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.out_dir / "tb_logs")
        else:
            self.writer = None

    def _compute_cond(
        self,
        pathology: torch.Tensor,
        imu_probe: torch.Tensor,
        imu_vehicle: torch.Tensor,
        grid_cell: torch.Tensor,
    ) -> torch.Tensor:
        # Compensated IMU: probe - vehicle
        imu_comp = imu_probe - imu_vehicle
        grid_i = grid_cell[:, 0]
        grid_j = grid_cell[:, 1]
        return self.cond_encoder(pathology, imu_comp, grid_i, grid_j)

    def _train_step(self, batch):
        image, pathology, imu_probe, imu_vehicle, grid_cell = [
            x.to(self.device) for x in batch
        ]

        image = image.unsqueeze(1)  # [B, 1, H, W]
        B = image.shape[0]

        # Random timesteps
        t = torch.randint(0, self.tc["n_diffusion_steps"], (B,), device=self.device)

        # Add noise
        noise = torch.randn_like(image)
        x_t = self.schedule.add_noise(image, noise, t)

        # Classifier-free guidance: randomly drop conditioning
        cond = self._compute_cond(pathology, imu_probe, imu_vehicle, grid_cell)
        drop_mask = torch.rand(B, device=self.device) < self.tc["p_uncond"]
        null_c = self.null_cond.unsqueeze(0).expand(B, -1)
        cond = torch.where(drop_mask[:, None], null_c, cond)

        with autocast("cuda", enabled=self.tc["mixed_precision"]):
            pred_noise = self.denoiser(x_t, t, cond)
            loss = F.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.denoiser.parameters()) + list(self.cond_encoder.parameters()),
            self.tc["grad_clip"]
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.update(self.denoiser)

        return loss.item()

    def train(self):
        logger.info(f"Starting training for {self.tc['n_epochs']} epochs")

        for epoch in range(self.tc["n_epochs"]):
            self.epoch = epoch
            self.denoiser.train()
            self.cond_encoder.train()

            total_loss = 0.0
            n_batches = 0

            for batch in self.train_loader:
                loss = self._train_step(batch)
                total_loss += loss
                n_batches += 1
                self.step += 1

                if self.step % 100 == 0:
                    avg = total_loss / n_batches
                    logger.info(f"Epoch {epoch} Step {self.step} | Loss: {avg:.4f}")
                    if self.writer:
                        self.writer.add_scalar("train/loss", avg, self.step)

            self.scheduler.step()

            if (epoch + 1) % self.tc["save_every"] == 0:
                self._save_checkpoint(epoch)

            if (epoch + 1) % self.tc["val_every"] == 0:
                val_loss = self._validate()
                logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
                if self.writer:
                    self.writer.add_scalar("val/loss", val_loss, epoch)

        # Save final
        self._save_checkpoint(self.tc["n_epochs"] - 1, final=True)

    def _validate(self) -> float:
        self.denoiser.eval()
        self.cond_encoder.eval()
        total = 0.0
        n = 0

        with torch.no_grad():
            for batch in self.val_loader:
                image, pathology, imu_probe, imu_vehicle, grid_cell = [
                    x.to(self.device) for x in batch
                ]
                image = image.unsqueeze(1)
                B = image.shape[0]
                t = torch.randint(0, self.tc["n_diffusion_steps"], (B,), device=self.device)
                noise = torch.randn_like(image)
                x_t = self.schedule.add_noise(image, noise, t)
                cond = self._compute_cond(pathology, imu_probe, imu_vehicle, grid_cell)
                pred = self.denoiser(x_t, t, cond)
                total += F.mse_loss(pred, noise).item()
                n += 1

        return total / max(n, 1)

    def _save_checkpoint(self, epoch: int, final: bool = False):
        name = "final.pt" if final else f"epoch_{epoch:04d}.pt"
        path = self.out_dir / name
        torch.save({
            "epoch": epoch,
            "step": self.step,
            "denoiser": self.denoiser.state_dict(),
            "cond_encoder": self.cond_encoder.state_dict(),
            "ema_shadow": self.ema.shadow,
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
        }, path)
        logger.info(f"Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        config.update(overrides)

    trainer = LungUSDiffusionTrainer(config)
    trainer.train()
