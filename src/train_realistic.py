"""
Pixel-Space Conditional DDPM for Realistic Lung POCUS
======================================================

Trains a class-conditional denoising diffusion model directly in pixel space
to generate clinically plausible lung ultrasound images.

Architecture
------------
- UNet2DModel (from diffusers) with 2 input channels:
    channel 0: noisy grayscale ultrasound image
    channel 1: structural guide from clinical_frames.py
- Class conditioning via learned embedding (10 pathology classes + 1 null)
- Cosine noise schedule (1000 training steps, DDIM 50 inference steps)
- Classifier-free guidance on both class label and structural guide

Why pixel-space (not latent-space)?
    The Stable Diffusion VAE is trained on natural photographs. When it
    encodes ultrasound speckle patterns and decodes them, the decoder maps
    speckle to natural-image textures (the "oil painting" artifact seen in
    the previous pipeline). Operating in pixel space avoids this entirely.

Data augmentation policy
    Ultrasound images have strict physical constraints. Only augmentations
    that produce clinically plausible images are applied:
    ✓ Horizontal flip    — left/right lung anatomy is roughly symmetric
    ✓ Brightness jitter  — simulates different gain settings (±15%)
    ✓ Contrast jitter    — simulates different TGC curves (±10%)
    ✓ Additive noise     — simulates electronic noise floor differences
    ✗ Vertical flip      — would invert near/far field (physically impossible)
    ✗ Rotation           — would tilt tissue layers (anatomically wrong)
    ✗ Color jitter       — ultrasound is inherently grayscale
    ✗ Random erasing     — would create holes in continuous tissue

Usage:
    python -m src.train_realistic --epochs 300 --batch-size 8
    python -m src.train_realistic --resume checkpoints/realistic/latest.pt
"""

import argparse
import copy
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

NUM_PATHOLOGY_CLASSES = 10
MMODE_CLASS_OFFSET = 11   # M-mode class n = n + 11 (classes 11-20)
NULL_CLASS = 21            # Unconditional class for classifier-free guidance
NUM_EMBEDDINGS = 22        # 10 bmode + 10 mmode + 1 null + 1 gap(10)

# Zone-region conditioning (Phase 1: zero-init no-op; Phase 5: LoRA-tuned).
# The integer values match `ZoneRegion` in clinical_frames.py — DO NOT
# renumber, the on-disk metadata column and the trained zone_embedding
# both depend on these.
NUM_ZONE_REGIONS = 9       # 7 region slots (0-6) + 1 null + 1 gap
ZONE_REGION_NULL = 7       # CFG dropout slot for the zone embedding


# ── Dataset ──────────────────────────────────────────────────────────────────

class RealPOCUSDataset(Dataset):
    """
    Loads real POCUS images with on-the-fly structural guide generation.

    Each sample returns:
        image:   [1, H, W] float32 in [-1, 1]  (real ultrasound frame)
        guide:   [1, H, W] float32 in [-1, 1]  (structural map from physics renderer)
        label:   int (pathology class 0-9)

    The structural guide is generated fresh each time from clinical_frames.py
    with a random seed. This means the guide won't pixel-align with the real
    image — and that's intentional. The model learns the *correlation* between
    structural features (e.g., bright band = pleural line) and their realistic
    appearance, not a pixel-to-pixel mapping.

    When anatomy_bank is provided, guides are enhanced with real lesion textures
    (50% probability during training) to close the train/inference gap — the
    inference pipeline always uses anatomy bank enhancement.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        anatomy_bank=None,
        anatomy_blend: float = 0.35,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.split = split

        # Load metadata
        import csv
        self.samples = []
        meta_path = self.data_dir / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No metadata.csv at {meta_path}. Run: python -m src.acquire_pocus_data"
            )
        with open(meta_path) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append({
                        "path": self.data_dir / "images" / row["filename"],
                        "label": int(row["pathology_class"]),
                    })

        if not self.samples:
            raise ValueError(f"No samples found for split '{split}'")

        # Initialize structural guide generator (lazy)
        self._guide_gen = None

        # Anatomy bank for guide enhancement (closes train/inference gap)
        self.anatomy_bank = anatomy_bank
        self.anatomy_blend = anatomy_blend

        # Class weights for balanced sampling
        counts = {}
        for s in self.samples:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        max_count = max(counts.values())
        self.sample_weights = [max_count / counts[s["label"]] for s in self.samples]

        logger.info(
            f"RealPOCUSDataset({split}): {len(self.samples)} images, "
            f"classes: {dict(sorted(counts.items()))}"
        )

    @property
    def guide_gen(self):
        if self._guide_gen is None:
            from .clinical_frames import ClinicalFrameGenerator
            self._guide_gen = ClinicalFrameGenerator(
                image_size=(self.image_size, self.image_size)
            )
        return self._guide_gen

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]

        # Load real image
        img = Image.open(sample["path"]).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.asarray(img, dtype=np.float32) / 255.0

        # Generate structural guide
        from .clinical_frames import ClinicalPathology
        if label >= MMODE_CLASS_OFFSET:
            # M-mode: the image IS the guide (both are the cached M-mode)
            guide = image.copy()
        else:
            # B-mode: generate a structural guide from the physics renderer
            try:
                guide = self.guide_gen.generate(ClinicalPathology(label))
            except (ValueError, KeyError):
                guide = np.zeros_like(image)

        # Anatomy bank enhancement (50% chance during training)
        if self.anatomy_bank is not None and self.augment and np.random.random() < 0.5:
            try:
                texture = self.anatomy_bank.sample_lesion_texture(label)
                if texture is not None:
                    row, col = self.anatomy_bank.sample_lesion_position(
                        label, image_size=self.image_size,
                        pleural_row=int(self.image_size * 0.17),
                    )
                    ph, pw = texture.shape[:2]
                    r0 = max(0, row - ph // 2)
                    r1 = min(self.image_size, r0 + ph)
                    c0 = max(0, col - pw // 2)
                    c1 = min(self.image_size, c0 + pw)
                    th, tw = r1 - r0, c1 - c0
                    tex_crop = texture[:th, :tw]
                    feather = 8
                    mask = np.ones_like(tex_crop)
                    for i in range(min(feather, min(th, tw) // 2)):
                        alpha = i / feather
                        mask[i, :] *= alpha
                        mask[-(i + 1), :] *= alpha
                        mask[:, i] *= alpha
                        mask[:, -(i + 1)] *= alpha
                    region = guide[r0:r1, c0:c1]
                    blend = self.anatomy_blend
                    guide[r0:r1, c0:c1] = np.clip(
                        region * (1 - blend * mask) + tex_crop * blend * mask, 0, 1
                    )
            except Exception:
                pass  # Non-critical; fall back to plain guide

        # Clinically valid augmentations
        if self.augment:
            # Horizontal flip (bilateral lung symmetry)
            if np.random.random() < 0.5:
                image = np.fliplr(image).copy()
                guide = np.fliplr(guide).copy()

            # Brightness jitter (simulates gain adjustment ±15%)
            brightness = 1.0 + np.random.uniform(-0.15, 0.15)
            image = np.clip(image * brightness, 0.0, 1.0)

            # Contrast jitter (simulates TGC variation ±10%)
            contrast = 1.0 + np.random.uniform(-0.10, 0.10)
            mean = image.mean()
            image = np.clip((image - mean) * contrast + mean, 0.0, 1.0)

            # Additive Gaussian noise (simulates electronic noise floor)
            noise_std = np.random.uniform(0.0, 0.02)
            image = np.clip(image + np.random.normal(0, noise_std, image.shape).astype(np.float32), 0.0, 1.0)

        # Normalize to [-1, 1] (standard for diffusion models)
        image = image * 2.0 - 1.0
        guide = guide * 2.0 - 1.0

        # Add channel dimension: [1, H, W]
        image = torch.from_numpy(image).unsqueeze(0)
        guide = torch.from_numpy(guide).unsqueeze(0)

        return image, guide, label


# ── M-mode Dataset ───────────────────────────────────────────────────────────

class SyntheticMmodeDataset(Dataset):
    """
    Generates M-mode training samples from the physics-based renderer.

    M-mode is critical for EMS POCUS:
      - Seashore sign: granular sub-pleural texture = lung sliding = normal
      - Stratosphere sign: horizontal lines throughout = no sliding = pneumothorax
      - Lung point M-mode: alternating seashore↔stratosphere = confirms PTX

    M-mode images look fundamentally different from B-mode (time on X-axis,
    depth on Y-axis, single scan line). The physics-based renderer produces
    accurate seashore/stratosphere patterns because these are structurally
    simpler than 2D B-mode anatomy.

    Class labels are offset by MMODE_CLASS_OFFSET (11) so the model learns
    separate embeddings for B-mode vs M-mode generation:
      B-mode class 0 (normal) = embedding 0
      M-mode class 0 (normal seashore) = embedding 11

    Each sample returns:
        image:  [1, H, W] float32 in [-1, 1]  (synthetic M-mode)
        guide:  [1, H, W] float32 in [-1, 1]  (same M-mode as guide)
        label:  int (pathology class + MMODE_CLASS_OFFSET)
    """

    # M-mode patterns by pathology class:
    #   normal(0)       → seashore (sliding present)
    #   pneumothorax(1) → stratosphere (no sliding)
    #   focal blines(2) → seashore + vertical streaks
    #   diffuse blines(3) → seashore + multiple streaks
    #   consolidation(4) → seashore (sliding present)
    #   effusion(5)     → sinusoid sign (floating lung)
    #   ards(6)         → seashore but irregular
    #   lung_point(7)   → seashore↔stratosphere alternation (pathognomonic!)
    #   thickening(8)   → seashore with thick pleural line
    #   interstitial(9) → seashore with irregular streaks

    SLIDING_MAP = {
        0: True,   # normal → seashore
        1: False,  # pneumothorax → stratosphere
        2: True,   # focal B-lines → seashore
        3: True,   # diffuse B-lines → seashore
        4: True,   # consolidation → seashore
        5: True,   # effusion → seashore (sinusoid variant)
        6: True,   # ARDS → seashore
        7: True,   # lung point → handled specially (alternating)
        8: True,   # thickening → seashore
        9: True,   # interstitial → seashore
    }

    def __init__(
        self,
        n_per_class: int = 200,
        image_size: int = 256,
        n_frames: int = 32,
    ):
        self.image_size = image_size
        self.n_per_class = n_per_class
        self.n_frames = n_frames
        self._stack_gen = None

        # Build sample list: (pathology_class, seed)
        self.samples = []
        for cls in range(NUM_PATHOLOGY_CLASSES):
            for i in range(n_per_class):
                self.samples.append((cls, cls * 10000 + i))

        logger.info(
            f"SyntheticMmodeDataset: {len(self.samples)} samples "
            f"({n_per_class}/class × {NUM_PATHOLOGY_CLASSES} classes)"
        )

    @property
    def stack_gen(self):
        if self._stack_gen is None:
            from .clinical_frames import ClinicalFrameGenerator, ClinicalTemporalStack
            frame_gen = ClinicalFrameGenerator(
                image_size=(self.image_size, self.image_size)
            )
            self._stack_gen = ClinicalTemporalStack(
                frame_gen=frame_gen,
                n_frames=self.n_frames,
            )
        return self._stack_gen

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls, seed = self.samples[idx]

        from .clinical_frames import ClinicalPathology
        sliding = self.SLIDING_MAP.get(cls, True)

        stack_data = self.stack_gen.generate_stack(
            ClinicalPathology(cls),
            lung_sliding=sliding,
            seed=seed,
        )

        mmode = stack_data["mmode"]  # [H, n_frames]

        # Resize M-mode to square (H×W) for consistent model input
        from PIL import Image as PILImage
        mmode_img = PILImage.fromarray(
            (np.clip(mmode, 0, 1) * 255).astype(np.uint8), mode="L"
        )
        mmode_img = mmode_img.resize(
            (self.image_size, self.image_size), PILImage.LANCZOS
        )
        mmode_arr = np.asarray(mmode_img, dtype=np.float32) / 255.0

        # Augment: brightness/contrast jitter (same as B-mode)
        brightness = 1.0 + np.random.uniform(-0.10, 0.10)
        mmode_arr = np.clip(mmode_arr * brightness, 0.0, 1.0)

        # For M-mode, image and guide are both the synthetic M-mode
        # (the model learns M-mode texture + structure together)
        image = mmode_arr * 2.0 - 1.0
        guide = mmode_arr * 2.0 - 1.0

        image = torch.from_numpy(image).unsqueeze(0)
        guide = torch.from_numpy(guide).unsqueeze(0)

        # Offset label for M-mode class space
        label = cls + MMODE_CLASS_OFFSET

        return image, guide, label


# ── EMA (Exponential Moving Average) ────────────────────────────────────────

class EMAModel:
    """
    Maintains an exponential moving average of model parameters.

    EMA smooths out training noise and produces more stable generations.
    Decay of 0.9999 means the EMA model is a weighted average of the last
    ~10,000 parameter updates.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Copy EMA weights into model (for inference).

        Skips any model parameters not present in the shadow dict —
        this lets us load a pre-zone-aware checkpoint into a model that
        has new parameters (e.g. `zone_embedding.weight`) without
        clobbering their initial values. Mirrors PyTorch's `strict=False`
        load semantics.
        """
        for name, param in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue  # New param not present in old checkpoint — keep init
            param.data.copy_(shadow)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        """Merge incoming state dict into the shadow rather than replacing it.

        This preserves any new parameters that exist on the live model but
        weren't in the checkpoint (e.g. `zone_embedding.weight` added in
        Phase 1). Without the merge, those new params would be silently
        dropped and the matching `apply()` call would have nothing to
        copy from.
        """
        for k, v in state_dict.items():
            self.shadow[k] = v.clone()


# ── ControlNet Model ─────────────────────────────────────────────────────────

class ZeroConv(nn.Module):
    """
    Zero-initialized 1×1 convolution.

    Starts at zero so guide features have NO influence at the beginning of
    training. As training progresses, the model gradually learns to use
    the structural guide — it cannot ignore it because features are
    injected at every decoder level, not just the input.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class GuideEncoder(nn.Module):
    """
    Encodes the structural guide (from clinical_frames.py) into multi-scale
    feature maps that match the U-Net's internal resolutions.

    Input:  [B, 1, 256, 256]  — structural guide (pleural line, A-lines, etc.)
    Output: list of features at each scale:
            [B, 64, 128, 128]   — local tissue layer features
            [B, 128, 64, 64]    — rib shadow / pleural line scale
            [B, 256, 32, 32]    — A-line spacing / B-line extent
            [B, 512, 16, 16]    — global pathology layout
    """

    def __init__(self, in_channels: int = 1, block_out_channels=(64, 128, 256, 512)):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_channels
        for out_ch in block_out_channels:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
            ))
            ch = out_ch

    def forward(self, guide):
        features = []
        x = guide
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class ControlNetPOCUS(nn.Module):
    """
    U-Net with ControlNet-style structural guide injection.

    Why ControlNet instead of channel concatenation:
        The previous approach concatenated the guide as channel 2 of the input.
        After 4 downsampling blocks, spatial information from the guide was
        mostly lost — the model could (and did) ignore it, producing images
        with correct texture but wrong spatial layout.

        ControlNet injects guide features at EVERY level of the decoder via
        zero-initialized convolutions. This forces the model to place
        pathology features where the guide indicates:
          - Pleural line at the correct depth
          - A-lines at equidistant intervals
          - B-lines at specified lateral positions
          - Consolidation / effusion in the correct regions

    Architecture:
        Main U-Net:      processes noisy image (1ch) with timestep + class
        Guide encoder:   CNN producing features at 128, 64, 32, 16 px
        Zero-conv links: inject guide features into U-Net skip connections
                         (initialized to zero → gradual influence)

    Forward:
        noise_pred = model(noisy_image, timestep, class_labels, guide)
    """

    def __init__(
        self,
        image_size: int = 256,
        num_class_embeds: int = NUM_EMBEDDINGS,
        num_zone_regions: int = NUM_ZONE_REGIONS,
    ):
        super().__init__()
        from diffusers import UNet2DModel

        self.block_out_channels = (64, 128, 256, 512)

        # Main U-Net: processes ONLY the noisy image (1 channel, not 2)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=self.block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            num_class_embeds=num_class_embeds,
        )

        # Guide encoder: structural guide → multi-scale features
        self.guide_encoder = GuideEncoder(
            in_channels=1,
            block_out_channels=self.block_out_channels,
        )

        # Zero-convolutions: one per down_block output + one for mid_block
        # Each down_block produces (layers_per_block + 1) residual outputs
        # for the corresponding up_block's skip connections
        self.zero_convs = nn.ModuleList()
        for ch in self.block_out_channels:
            # 2 resnet layers + 1 downsample = 3 skip connections per block
            for _ in range(3):
                self.zero_convs.append(ZeroConv(ch))
        # Mid-block zero conv
        self.mid_zero_conv = ZeroConv(self.block_out_channels[-1])

        # ── Zone-region embedding ──
        # Additive to the class embedding inside `forward`. Initialised to
        # ZERO so that loading a pre-zone-aware checkpoint with
        # `strict=False` produces bit-identical outputs for any zone label
        # (including the new lower-zone slots). Phase 5 LoRA training will
        # learn deltas from this identity. Read the actual time-embed dim
        # from the constructed UNet so we don't hardcode the value across
        # diffusers versions.
        time_embed_dim = self.unet.time_embedding.linear_2.out_features
        self.zone_embedding = nn.Embedding(num_zone_regions, time_embed_dim)
        nn.init.zeros_(self.zone_embedding.weight)

    def forward(self, sample, timestep, class_labels=None, guide=None, zone_labels=None):
        """
        Args:
            sample:       [B, 1, H, W] noisy image
            timestep:     [B] diffusion timestep
            class_labels: [B] pathology class (0-9 bmode, 11-20 mmode, 21 null)
            guide:        [B, 1, H, W] structural guide from clinical_frames.py
            zone_labels:  Optional [B] anatomical zone region (0-6 valid, 7
                          for CFG null). When None or all zeros, the zone
                          embedding is a no-op (zero vector by initialization
                          on a fresh model; learned delta after Phase 5 LoRA).

        Returns:
            Object with .sample = [B, 1, H, W] predicted noise
        """
        # Encode structural guide into multi-scale features
        if guide is not None:
            guide_features = self.guide_encoder(guide)
        else:
            guide_features = None

        # ── Replicate UNet2DModel forward with guide injection ──

        # Timestep embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        t_emb = self.unet.time_proj(timesteps.to(sample.device))
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.unet.time_embedding(t_emb)

        if self.unet.class_embedding is not None and class_labels is not None:
            class_emb = self.unet.class_embedding(class_labels)
            emb = emb + class_emb

        # Zone-region conditioning: additive on top of class+time embedding.
        # Defaults to a zero embedding (no-op) when not provided. Loading an
        # old checkpoint with `strict=False` keeps zone_embedding zero-init,
        # so this branch is a perfect identity until Phase 5 trains it.
        if zone_labels is not None:
            emb = emb + self.zone_embedding(zone_labels)

        # Initial convolution
        sample = self.unet.conv_in(sample)

        # Encoder (down blocks) — collect skip connections
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb
            )
            down_block_res_samples += res_samples

        # Mid block
        if self.unet.mid_block is not None:
            sample = self.unet.mid_block(sample, emb)

        # ── Inject guide features into skip connections ──
        if guide_features is not None:
            modified_res = list(down_block_res_samples)

            zc_idx = 0
            for level, ch in enumerate(self.block_out_channels):
                guide_feat = guide_features[level]

                # Find which skip connections belong to this level
                # Each down_block produces 3 skip connections (2 resnet + 1 downsample)
                # except the initial sample which is before the first block
                start = 1 + level * 3  # +1 for the initial conv_in output
                end = start + 3

                for j in range(start, min(end, len(modified_res))):
                    if modified_res[j].shape == guide_feat.shape:
                        modified_res[j] = modified_res[j] + self.zero_convs[zc_idx](guide_feat)
                    zc_idx += 1

            # Mid-block injection
            if sample.shape == guide_features[-1].shape:
                sample = sample + self.mid_zero_conv(guide_features[-1])

            down_block_res_samples = tuple(modified_res)

        # Decoder (up blocks)
        for upsample_block in self.unet.up_blocks:
            n_resnets = len(upsample_block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            sample = upsample_block(
                hidden_states=sample,
                res_hidden_states_tuple=res_samples,
                temb=emb,
            )

        # Output
        if self.unet.conv_norm_out:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        # Return in same format as UNet2DModel
        from dataclasses import dataclass

        @dataclass
        class Output:
            sample: torch.Tensor

        return Output(sample=sample)


def create_model(image_size: int = 256) -> nn.Module:
    """
    Create the ControlNet POCUS model.

    Architecture:
        Main path:    U-Net (1ch input) with timestep + class conditioning
        Control path: Guide encoder → zero-conv injection at every decoder level
        Total:        ~75M parameters (63M U-Net + 12M guide encoder + zero-convs)
    """
    model = ControlNetPOCUS(image_size=image_size, num_class_embeds=NUM_EMBEDDINGS)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ControlNet model created: {n_params / 1e6:.1f}M parameters")
    return model


# ── Noise schedule ───────────────────────────────────────────────────────────

def create_noise_scheduler(num_train_timesteps: int = 1000):
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).

    Cosine schedule is preferred over linear for lower-resolution images
    because it spends more of the diffusion process at intermediate noise
    levels, which is where the model learns the most useful structure.
    Linear schedule wastes steps at very low/high noise where there's
    little to learn.
    """
    from diffusers import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
    )


def create_inference_scheduler(num_train_timesteps: int = 1000):
    """DDIM scheduler for fast inference (50 steps instead of 1000)."""
    from diffusers import DDIMScheduler

    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
    )


# ── Sampling ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_images(
    model: nn.Module,
    scheduler,
    structural_guides: torch.Tensor,
    class_labels: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    device: str = "cuda",
    zone_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate realistic POCUS frames using DDIM sampling with CFG.

    Args:
        model:              Trained U-Net denoiser
        scheduler:          DDIM noise scheduler
        structural_guides:  [B, 1, H, W] guides from clinical_frames.py
        class_labels:       [B] pathology class indices (0-9)
        guidance_scale:     CFG weight. 1.0 = no guidance, >1.0 = stronger
                           class conditioning. 3-5 works well for medical images
                           (higher values exaggerate class features)
        device:             Target device
        zone_labels:        Optional [B] anatomical zone region indices (0-6).
                           When None, the model is called without zone
                           conditioning (legacy bit-identical behavior).
                           When provided, the conditional branch passes the
                           supplied zones and the unconditional branch uses
                           ZONE_REGION_NULL for CFG dropout.

    Returns:
        [B, 1, H, W] tensor of generated images in [-1, 1]
    """
    model.eval()
    B = structural_guides.shape[0]
    H, W = structural_guides.shape[2], structural_guides.shape[3]

    # Start from pure Gaussian noise
    images = torch.randn(B, 1, H, W, device=device)

    # Null class labels for unconditional branch of CFG
    null_labels = torch.full((B,), NULL_CLASS, dtype=torch.long, device=device)
    null_zone_labels = (
        torch.full((B,), ZONE_REGION_NULL, dtype=torch.long, device=device)
        if zone_labels is not None
        else None
    )

    scheduler.set_timesteps(num_inference_steps, device=device)

    for t in scheduler.timesteps:
        timestep = t.expand(B)

        # Conditional: noisy image + structural guide (ControlNet injection)
        noise_pred_cond = model(
            images, timestep,
            class_labels=class_labels,
            guide=structural_guides,
            zone_labels=zone_labels,
        ).sample

        if guidance_scale > 1.0:
            # Unconditional: no guide, null class, null zone
            noise_pred_uncond = model(
                images, timestep,
                class_labels=null_labels,
                guide=None,
                zone_labels=null_zone_labels,
            ).sample

            # CFG interpolation
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        # DDIM step
        images = scheduler.step(noise_pred, t, images).prev_sample

    return images


def _load_real_example(data_dir: Path, pathology_class: int, image_size: int = 256):
    """Load a real training image for a given class (for visual comparison)."""
    import csv
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        return np.zeros((image_size, image_size), dtype=np.float32)
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            if int(row["pathology_class"]) == pathology_class and row["split"] == "train":
                img = Image.open(data_dir / "images" / row["filename"]).convert("L")
                img = img.resize((image_size, image_size), Image.LANCZOS)
                return np.asarray(img, dtype=np.float32) / 255.0
    return np.zeros((image_size, image_size), dtype=np.float32)


def generate_sample_grid(
    model, scheduler, guide_gen, device, epoch, output_dir, num_classes=NUM_PATHOLOGY_CLASSES
):
    """
    Generate a clinician-friendly comparison grid:
      Column 1: Real clinical image (what we're training on)
      Column 2: AI-generated image (what the model produces)
      Column 3: Structural layout (physics-based anatomy guide)
    """
    from .clinical_frames import ClinicalPathology
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    data_dir = _ROOT / "data" / "real_pocus" / "processed"

    # Clinical descriptions for each pathology
    class_info = [
        ("Normal (A-Profile)",       "A-lines + lung sliding\nHealthy aerated lung"),
        ("Pneumothorax",             "Absent sliding, A-lines only\nRequires decompression"),
        ("Focal B-Lines (1-2)",      "Isolated vertical artifacts\nEarly pulmonary edema"),
        ("Diffuse B-Lines (≥3)",     "Confluent vertical artifacts\nPulmonary edema / fluid"),
        ("Consolidation",            "Hepatized tissue + air bronchograms\nPneumonia"),
        ("Pleural Effusion",         "Anechoic fluid collection\nHemothorax / CHF"),
        ("ARDS / White Lung",        "Complete B-line confluence\nSevere respiratory failure"),
        ("Lung Point",               "Sliding↔no-sliding transition\nConfirms pneumothorax"),
        ("Pleural Thickening",       "Irregular pleural line\nChronic / inflammatory"),
        ("Interstitial Syndrome",    "B-lines + subpleural changes\nViral pneumonia / fibrosis"),
    ]

    all_guides = []
    all_labels = []
    real_examples = []

    for cls_id in range(num_classes):
        fixed_seed = cls_id * 100 + 42
        try:
            guide = guide_gen.generate(ClinicalPathology(cls_id), seed=fixed_seed)
        except (ValueError, KeyError):
            guide = np.zeros((256, 256), dtype=np.float32)
        all_guides.append(
            torch.from_numpy(guide * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)
        )
        all_labels.append(cls_id)
        real_examples.append(_load_real_example(data_dir, cls_id))

    guides = torch.cat(all_guides, dim=0).to(device)
    labels = torch.tensor(all_labels, dtype=torch.long, device=device)

    samples = sample_images(model, scheduler, guides, labels, device=device)
    samples = (samples.cpu().float() + 1.0) / 2.0

    # Layout: 10 rows × 3 columns
    fig, axes = plt.subplots(num_classes, 3, figsize=(12, num_classes * 2.8))

    for cls_id in range(num_classes):
        name, desc = class_info[cls_id]

        # Column 1: Real clinical image
        ax = axes[cls_id, 0]
        ax.imshow(real_examples[cls_id], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{name}\n{desc}", fontsize=7, linespacing=1.3,
                       fontfamily="sans-serif", labelpad=10)

        # Column 2: AI-generated image
        ax = axes[cls_id, 1]
        ax.imshow(samples[cls_id, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])

        # Column 3: Structural guide
        ax = axes[cls_id, 2]
        guide_img = (guides[cls_id, 0].cpu().float() + 1.0) / 2.0
        ax.imshow(guide_img.numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_title("Real Clinical Image\n(training data)", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("AI Generated\n(model output)", fontsize=9, fontweight="bold")
    axes[0, 2].set_title("Structural Layout\n(anatomy guide)", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"B-mode Lung Ultrasound — Training Epoch {epoch}/200\n"
        f"Goal: AI-generated images should progressively match real clinical images",
        fontsize=11, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0.12, 0.0, 1.0, 0.97])
    path = Path(output_dir) / f"samples_epoch_{epoch:04d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved B-mode sample grid → {path}")


def generate_mmode_sample_grid(
    model, scheduler, device, epoch, output_dir
):
    """
    Generate clinician-friendly M-mode comparison grid.

    M-mode is critical for EMS POCUS — it shows lung sliding (or absence)
    as a single static image:
      Seashore sign    = sliding present  = normal lung
      Stratosphere sign = no sliding      = pneumothorax
      Lung point       = alternating      = confirms PTX boundary
    """
    from .clinical_frames import ClinicalPathology, ClinicalFrameGenerator, ClinicalTemporalStack
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    data_dir = _ROOT / "data" / "real_pocus" / "processed"
    frame_gen = ClinicalFrameGenerator(image_size=(256, 256))
    stack_gen = ClinicalTemporalStack(frame_gen=frame_gen, n_frames=32)

    # Key M-mode patterns for EMS with clinical descriptions
    mmode_cases = [
        (0,  True,  "Normal — Seashore Sign",
         "Granular texture below pleural line\n= lung sliding present = NOT pneumothorax"),
        (1,  False, "Pneumothorax — Stratosphere Sign",
         "Horizontal lines throughout\n= NO lung sliding = needs decompression"),
        (7,  True,  "Lung Point",
         "Alternating seashore/stratosphere\n= pathognomonic for pneumothorax border"),
        (5,  True,  "Pleural Effusion — Sinusoid Sign",
         "Undulating pleural line on fluid\n= free-flowing effusion"),
    ]

    fig, axes = plt.subplots(len(mmode_cases), 3, figsize=(14, len(mmode_cases) * 3.2))

    for row, (cls_id, sliding, title, desc) in enumerate(mmode_cases):
        fixed_seed = cls_id * 100 + 42
        stack_data = stack_gen.generate_stack(
            ClinicalPathology(cls_id), lung_sliding=sliding, seed=fixed_seed
        )
        mmode = stack_data["mmode"]

        from PIL import Image as PILImage
        mmode_img = PILImage.fromarray(
            (np.clip(mmode, 0, 1) * 255).astype(np.uint8), mode="L"
        )
        mmode_arr = np.asarray(
            mmode_img.resize((256, 256), PILImage.LANCZOS), dtype=np.float32
        ) / 255.0

        guide_t = torch.from_numpy(mmode_arr * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).to(device)
        label_t = torch.tensor([cls_id + MMODE_CLASS_OFFSET], dtype=torch.long, device=device)

        result = sample_images(model, scheduler, guide_t, label_t, device=device)
        generated = (result[0, 0].cpu().float() + 1.0) / 2.0

        # Column 1: Real M-mode example from training data
        real_mmode = _load_real_example(data_dir, cls_id + MMODE_CLASS_OFFSET)
        axes[row, 0].imshow(real_mmode, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{title}\n{desc}", fontsize=7,
                                 linespacing=1.3, fontfamily="sans-serif", labelpad=10)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        # Column 2: AI-generated M-mode
        axes[row, 1].imshow(generated.clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])

        # Column 3: Structural layout
        axes[row, 2].imshow(mmode_arr, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])

    axes[0, 0].set_title("Reference M-mode\n(training target)", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("AI Generated\n(model output)", fontsize=9, fontweight="bold")
    axes[0, 2].set_title("Structural Layout\n(anatomy guide)", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"M-mode Lung Ultrasound — Training Epoch {epoch}/200\n"
        f"X-axis = time, Y-axis = depth along single scan line",
        fontsize=11, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0.15, 0.0, 1.0, 0.96])
    path = Path(output_dir) / f"mmode_epoch_{epoch:04d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved M-mode sample grid → {path}")


# ── Training loop ────────────────────────────────────────────────────────────

def train(
    data_dir: str = "data/real_pocus/processed",
    output_dir: str = "checkpoints/realistic",
    epochs: int = 300,
    batch_size: int = 8,
    lr: float = 1e-4,
    image_size: int = 256,
    cfg_dropout: float = 0.10,
    ema_decay: float = 0.9999,
    resume_path: Optional[str] = None,
    finetune: bool = False,
    sample_every: int = 10,
    save_every: int = 25,
    lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
):
    """
    Main training loop.

    Classifier-free guidance training:
        With probability cfg_dropout, the class label is replaced with the
        null class (unconditional). Independently, with the same probability,
        the structural guide channel is zeroed. This teaches the model to
        generate both with and without conditioning, enabling CFG at inference.

    LoRA fine-tuning (lora=True):
        After loading the resume checkpoint, the model is wrapped with
        rank-`lora_rank` LoRA adapters on every attention Linear and
        time_emb_proj layer (~0.74M trainable params at rank 16). The
        optimizer is rebuilt over LoRA + zone_embedding params only,
        and a LoRA-only delta (~3 MB) is saved as `latest_lora.pt`
        alongside the full checkpoint each epoch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = _ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Data:   {data_dir}")
    logger.info(f"Output: {output_path}")

    # Load anatomy bank for guide enhancement (closes train/inference gap)
    anatomy_bank = None
    bank_path = _ROOT / "checkpoints" / "anatomy_bank.pt"
    if bank_path.exists():
        from .anatomy_bank import LesionAnatomyBank
        anatomy_bank = LesionAnatomyBank.load(str(bank_path))
        logger.info(f"Loaded anatomy bank from {bank_path}")

    # Dataset — loads both B-mode (classes 0-9) and pre-cached M-mode (classes 11-20)
    # M-mode images are already saved to disk with class labels 11-20 in metadata.csv
    dataset = RealPOCUSDataset(
        str(_ROOT / data_dir), split="train", image_size=image_size,
        anatomy_bank=anatomy_bank,
    )
    val_dataset = RealPOCUSDataset(
        str(_ROOT / data_dir), split="val", image_size=image_size, augment=False,
    )

    sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = create_model(image_size).to(device)
    noise_scheduler = create_noise_scheduler()
    inference_scheduler = create_inference_scheduler()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # EMA
    ema = EMAModel(model, decay=ema_decay)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Structural guide generator (for sample grids)
    from .clinical_frames import ClinicalFrameGenerator
    guide_gen = ClinicalFrameGenerator(image_size=(image_size, image_size))

    # Metrics CSV for remote monitoring (append mode)
    metrics_path = output_path / "metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w") as f:
            f.write("epoch,train_loss,val_loss,lr,best_val_loss\n")

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_path:
        ckpt = torch.load(_ROOT / resume_path, map_location=device, weights_only=False)
        # strict=False so we can resume into a model that has the new
        # zone_embedding parameter while loading a pre-Phase-1 checkpoint.
        # The merged EMAModel.load_state_dict (also strict-tolerant) handles
        # the same case for the EMA shadow.
        miss, unex = model.load_state_dict(ckpt["model"], strict=False)
        if miss or unex:
            logger.info(
                f"Resume loaded with strict=False (missing={len(miss)}, unexpected={len(unex)}). "
                f"This is expected when adding zone_embedding to a pre-zone checkpoint."
            )
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if finetune:
            # Fine-tune: keep model/EMA weights, reset everything else
            logger.info(
                f"Fine-tuning from {resume_path} (epoch {ckpt.get('epoch', '?')}). "
                f"Fresh optimizer, LR schedule, epoch counter."
            )
        else:
            # Full resume: restore optimizer state and epoch counter
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            logger.info(f"Resumed from epoch {start_epoch}")

    # ── LoRA injection (zone-aware fine-tune path) ──
    # Must happen AFTER the base checkpoint is loaded so the LoRA wrappers
    # see the correct base weights. Rebuilds the optimizer over only the
    # LoRA + zone_embedding params (~1% of the total) so the AdamW state
    # stays tiny and training is fast.
    lora_active = bool(lora)
    if lora_active:
        from .lora import (
            inject_lora,
            mark_only_lora_as_trainable,
            print_trainable_parameters,
        )

        matched = inject_lora(model, rank=lora_rank, alpha=lora_alpha)
        logger.info(f"LoRA injected into {len(matched)} Linear layers (rank={lora_rank}, alpha={lora_alpha})")

        trainable_count, total_count = mark_only_lora_as_trainable(
            model, train_zone_embed=True
        )
        print_trainable_parameters(model)

        # Rebuild optimizer over trainable params only — the LoRA delta
        # tolerates a higher LR than full fine-tuning thanks to the
        # implicit low-rank regularization.
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Use a faster-adapting EMA for small-data fine-tuning. Default
        # 0.9999 is too slow when only ~500-1000 new samples are added.
        ema = EMAModel(model, decay=0.999)

    # Training
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, guides, labels in loader:
            images = images.to(device)     # [B, 1, H, W]
            guides = guides.to(device)     # [B, 1, H, W]
            labels = labels.to(device)     # [B]

            # ── Classifier-free guidance dropout ──
            # Drop class label → null class
            drop_class = torch.rand(labels.shape[0], device=device) < cfg_dropout
            labels = torch.where(drop_class, torch.full_like(labels, NULL_CLASS), labels)

            # Drop structural guide → zeros
            drop_guide = torch.rand(labels.shape[0], 1, 1, 1, device=device) < cfg_dropout
            guides = torch.where(drop_guide, torch.zeros_like(guides), guides)

            # ── Forward diffusion (add noise) ──
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (images.shape[0],), device=device, dtype=torch.long,
            )
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # ── Predict noise (ControlNet: separate noisy image + guide) ──
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                noise_pred = model(noisy_images, timesteps, class_labels=labels, guide=guides).sample
                loss = F.mse_loss(noise_pred, noise)

            # ── Backward ──
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──
        val_loss = 0.0
        n_val = 0
        model.eval()
        with torch.no_grad():
            for images, guides, labels in val_loader:
                images = images.to(device)
                guides = guides.to(device)
                labels = labels.to(device)

                noise = torch.randn_like(images)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (images.shape[0],), device=device, dtype=torch.long,
                )
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    noise_pred = model(noisy_images, timesteps, class_labels=labels, guide=guides).sample
                    val_loss += F.mse_loss(noise_pred, noise).item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)

        cur_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch:4d}/{epochs} | "
            f"train_loss={avg_loss:.5f} | val_loss={avg_val:.5f} | "
            f"lr={cur_lr:.2e}"
        )

        # Append to metrics CSV (remote monitoring: tail -f metrics.csv)
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{avg_val:.6f},{cur_lr:.2e},{best_val_loss:.6f}\n")

        # ── Save samples ──
        if (epoch + 1) % sample_every == 0 or epoch == 0:
            # Swap in EMA weights for sampling
            orig_state = {n: p.clone() for n, p in model.named_parameters()}
            ema.apply(model)
            generate_sample_grid(
                model, inference_scheduler, guide_gen, device, epoch + 1, output_path
            )
            generate_mmode_sample_grid(
                model, inference_scheduler, device, epoch + 1, output_path
            )
            # Copy to latest for easy remote viewing
            import shutil
            shutil.copy2(
                output_path / f"samples_epoch_{epoch + 1:04d}.png",
                output_path / "latest_samples.png",
            )
            shutil.copy2(
                output_path / f"mmode_epoch_{epoch + 1:04d}.png",
                output_path / "latest_mmode.png",
            )
            # Restore training weights
            for n, p in model.named_parameters():
                p.data.copy_(orig_state[n])

        # ── Save checkpoint ──
        if (epoch + 1) % save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "best_val_loss": best_val_loss,
                "config": {
                    "image_size": image_size,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "cfg_dropout": cfg_dropout,
                },
            }
            torch.save(ckpt, output_path / f"epoch_{epoch + 1:04d}.pt")

        # Save latest + best
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "best_val_loss": best_val_loss,
            "config": {
                "image_size": image_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "cfg_dropout": cfg_dropout,
                "lora": lora_active,
                "lora_rank": lora_rank if lora_active else None,
                "lora_alpha": lora_alpha if lora_active else None,
            },
        }
        torch.save(ckpt, output_path / "latest.pt")

        # When training with LoRA, also save a tiny delta-only checkpoint
        # (~3 MB) that can be shipped without the full ~300 MB base model.
        if lora_active:
            from .lora import lora_state_dict
            delta = lora_state_dict(model)
            torch.save(
                {
                    "epoch": epoch,
                    "lora_state_dict": delta,
                    "config": ckpt["config"],
                },
                output_path / "latest_lora.pt",
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, output_path / "best.pt")
            if lora_active:
                from .lora import lora_state_dict
                torch.save(
                    {
                        "epoch": epoch,
                        "lora_state_dict": lora_state_dict(model),
                        "config": ckpt["config"],
                    },
                    output_path / "best_lora.pt",
                )
            logger.info(f"  ★ New best val_loss={best_val_loss:.5f}")

    logger.info("Training complete.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train realistic POCUS DDPM")
    parser.add_argument("--data-dir", default="data/real_pocus/processed")
    parser.add_argument("--output-dir", default="checkpoints/realistic")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cfg-dropout", type=float, default=0.10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", action="store_true",
                        help="Load model/EMA weights from --resume but reset epoch, optimizer, and LR schedule")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--lora",
        action="store_true",
        help=(
            "Train only LoRA adapters + zone_embedding instead of full model. "
            "Requires --resume to load a base checkpoint. Use for the Phase 5 "
            "zone-aware fine-tune where ~0.74M trainable params on top of the "
            "frozen ~69M base is enough to learn zone-region deltas."
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default 16). Higher → more capacity, larger delta.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default 32 = 2*rank).",
    )
    args = parser.parse_args()

    # Ensure output dir exists before setting up log file
    (_ROOT / args.output_dir).mkdir(parents=True, exist_ok=True)

    # Line-buffered file handler so `tail -f training.log` works over SSH
    file_handler = logging.FileHandler(_ROOT / args.output_dir / "training.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
    )

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        cfg_dropout=args.cfg_dropout,
        resume_path=args.resume,
        finetune=args.finetune,
        sample_every=args.sample_every,
        save_every=args.save_every,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
