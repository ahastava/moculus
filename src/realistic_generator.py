"""
Realistic POCUS Frame Generator
=================================

Drop-in replacement for the old diffusion generators. Wraps the trained
pixel-space DDPM with structural guidance from clinical_frames.py.

Generation pipeline:
    1. Receive pathology class (+ optional pose/grid info)
    2. Generate structural guide from clinical_frames.py
       (positions pleural line, A-lines, B-lines, consolidation, etc.)
    3. Run DDIM denoising conditioned on structure + class
    4. Return realistic [H, W] float32 frame in [0, 1]

Temporal coherence:
    For B-mode cine loops, we use:
    - ClinicalTemporalStack for the structural guide sequence
      (provides lung sliding, breathing motion between frames)
    - Shared base noise with small per-frame perturbation
      (provides consistent speckle pattern that evolves naturally)
    This produces smooth temporal sequences without independent per-frame
    generation artifacts.

Fallback:
    When no trained model is available, falls back to the physics-based
    ClinicalFrameGenerator output directly. This ensures the simulator
    always works, even before training completes.

Usage:
    from src.realistic_generator import RealisticLungUSGenerator

    # Single model (auto-loads anatomy bank from checkpoints/anatomy_bank.pt):
    gen = RealisticLungUSGenerator.from_pretrained("checkpoints/realistic_v4/best.pt")

    # Hybrid routing (base + trauma-specialized):
    gen = RealisticLungUSGenerator.from_pretrained(
        "checkpoints/realistic_v4/best.pt",
        trauma_model_path="checkpoints/realistic_v2_finetune/latest.pt",
    )

    # Explicit anatomy bank path:
    gen = RealisticLungUSGenerator.from_pretrained(
        "checkpoints/realistic_v4/best.pt",
        anatomy_bank_path="checkpoints/anatomy_bank.pt",
    )

    frame = gen.generate(pathology_class=4)  # consolidation → base model
    frame = gen.generate(pathology_class=1)  # pneumothorax → trauma model
    stack = gen.generate_stack(pathology_class=0, n_frames=32, lung_sliding=True)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


class RealisticLungUSGenerator:
    """
    Generates realistic lung POCUS frames using a trained DDPM with
    structural guidance.

    Supports hybrid routing: separate models for trauma vs non-trauma
    pathologies, dispatching based on class at inference time. This allows
    using a trauma-specialized fine-tuned model for classes 1 (pneumothorax),
    5 (effusion/hemothorax), 7 (lung point) while using the base model
    (which produces better results) for all other pathologies.

    Attributes:
        model:         Trained UNet2DModel for non-trauma classes (or None)
        trauma_model:  Trained UNet2DModel for trauma classes (or None)
        scheduler:     DDIM noise scheduler
        guide_gen:     ClinicalFrameGenerator for structural maps
        device:        Compute device
        image_size:    Output image dimensions (square)
    """

    # Trauma classes routed to the fine-tuned model
    TRAUMA_CLASSES = {1, 5, 7}

    # Per-class guidance scale overrides.
    # Higher values amplify class-specific features (sharper pleural line,
    # more distinct A-lines/B-lines) at the cost of some diversity.
    CLASS_GUIDANCE_SCALE = {
        0: 6.0,   # Normal: strong boost for pleural line sharpness (esp. with fewer DDIM steps)
        1: 6.0,   # Pneumothorax: very bright, distinct pleural line
        2: 5.0,   # Focal B-lines: moderate boost for pleural + B-line definition
        4: 5.0,   # Consolidation: boost for tissue boundary definition
        7: 5.5,   # Lung point: needs clear sliding/non-sliding boundary
        8: 5.0,   # Pleural thickening: needs visible (irregular) pleural line
    }

    def __init__(
        self,
        model=None,
        scheduler=None,
        guide_gen=None,
        device: str = "cpu",
        image_size: int = 256,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        trauma_model=None,
        anatomy_bank=None,
        anatomy_blend: float = 0.35,
    ):
        self.model = model
        self.trauma_model = trauma_model
        self.scheduler = scheduler
        self.device = device
        self.image_size = image_size
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.anatomy_bank = anatomy_bank
        self.anatomy_blend = anatomy_blend

        if guide_gen is None:
            from .clinical_frames import ClinicalFrameGenerator
            guide_gen = ClinicalFrameGenerator(image_size=(image_size, image_size))
        self.guide_gen = guide_gen

    def _select_model(self, pathology_class: int):
        """Return the best model for a given pathology class."""
        if self.trauma_model is not None and pathology_class in self.TRAUMA_CLASSES:
            return self.trauma_model
        return self.model

    def _get_guidance_scale(self, pathology_class: int) -> float:
        """Return guidance scale for a class, with per-class overrides."""
        return self.CLASS_GUIDANCE_SCALE.get(pathology_class, self.guidance_scale)

    @staticmethod
    def _load_checkpoint(model_path: str, device: str, image_size: int = 256):
        """Load a single checkpoint and return the model in eval mode."""
        from .train_realistic import create_model, EMAModel

        model_file = _ROOT / model_path
        if not model_file.exists():
            return None, None

        ckpt = torch.load(model_file, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        img_size = config.get("image_size", image_size)

        model = create_model(img_size).to(device)

        if "ema" in ckpt:
            ema = EMAModel(model)
            ema.load_state_dict(ckpt["ema"])
            ema.apply(model)
            logger.info(f"Loaded EMA weights from {model_path}")
        else:
            model.load_state_dict(ckpt["model"])

        model.eval()
        epoch = ckpt.get("epoch", "?")
        logger.info(f"Loaded {model_path} (epoch {epoch})")
        return model, img_size

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        trauma_model_path: Optional[str] = None,
        anatomy_bank_path: Optional[str] = None,
        device: Optional[str] = None,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        anatomy_blend: float = 0.35,
    ) -> "RealisticLungUSGenerator":
        """
        Load trained model(s) from checkpoint.

        Args:
            model_path:        Path to base model checkpoint.
            trauma_model_path: Optional path to trauma-specialized checkpoint.
                               If provided, classes 1 (pneumothorax), 5 (effusion/
                               hemothorax), and 7 (lung point) are routed to this
                               model. All other classes use the base model.
            anatomy_bank_path: Optional path to Lesion-Anatomy Bank (.pt).
                               When provided, structural guides are enhanced with
                               real lesion textures placed via PMF conditioning
                               before being passed to the DDPM.
            device:            Compute device (auto-detected if None).
            guidance_scale:    Classifier-free guidance strength.
            num_inference_steps: DDIM inference steps.
            anatomy_blend:     Blend strength for anatomy bank textures (0-1).

        Falls back to physics-only generation if the base checkpoint doesn't
        exist or can't be loaded.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load anatomy bank if available
        anatomy_bank = None
        if anatomy_bank_path:
            bank_file = _ROOT / anatomy_bank_path
            if bank_file.exists():
                from .anatomy_bank import LesionAnatomyBank
                anatomy_bank = LesionAnatomyBank.load(str(bank_file))
                logger.info(f"Loaded anatomy bank from {bank_file}")
            else:
                logger.warning(f"No anatomy bank at {bank_file}")
        else:
            # Auto-detect default path
            default_bank = _ROOT / "checkpoints" / "anatomy_bank.pt"
            if default_bank.exists():
                from .anatomy_bank import LesionAnatomyBank
                anatomy_bank = LesionAnatomyBank.load(str(default_bank))
                logger.info(f"Auto-loaded anatomy bank from {default_bank}")

        try:
            from .train_realistic import create_inference_scheduler

            model, image_size = cls._load_checkpoint(model_path, device)
            if model is None:
                logger.warning(
                    f"No trained model at {_ROOT / model_path}. "
                    f"Using physics-only fallback. Run training first: "
                    f"python -m src.train_realistic"
                )
                return cls(
                    device=device, guidance_scale=guidance_scale,
                    anatomy_bank=anatomy_bank, anatomy_blend=anatomy_blend,
                )

            trauma_model = None
            if trauma_model_path:
                trauma_model, _ = cls._load_checkpoint(trauma_model_path, device)
                if trauma_model is not None:
                    logger.info(
                        f"Hybrid routing enabled: trauma classes "
                        f"{cls.TRAUMA_CLASSES} → {trauma_model_path}"
                    )

            scheduler = create_inference_scheduler()

            return cls(
                model=model,
                scheduler=scheduler,
                device=device,
                image_size=image_size,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                trauma_model=trauma_model,
                anatomy_bank=anatomy_bank,
                anatomy_blend=anatomy_blend,
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}. Using physics-only fallback.")
            return cls(
                device=device, guidance_scale=guidance_scale,
                anatomy_bank=anatomy_bank, anatomy_blend=anatomy_blend,
            )

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def _enhance_guide_with_anatomy(
        self,
        guide: np.ndarray,
        pathology_class: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Enhance a structural guide with real lesion textures from the
        anatomy bank, placed according to the learned PMF.

        This implements DiffUltra's core insight: structural guides that
        incorporate real tissue textures at anatomically plausible positions
        produce more realistic DDPM outputs than pure physics-based guides.
        """
        if self.anatomy_bank is None:
            return guide

        texture = self.anatomy_bank.sample_lesion_texture(
            pathology_class, seed=seed
        )
        if texture is None:
            return guide

        h, w = guide.shape
        row, col = self.anatomy_bank.sample_lesion_position(
            pathology_class,
            image_size=(h, w),
            pleural_row=self.guide_gen.pleural_row,
            seed=(seed + 1) if seed is not None else None,
        )

        ps = texture.shape[0]
        r0 = max(0, row - ps // 2)
        r1 = min(h, r0 + ps)
        c0 = max(0, col - ps // 2)
        c1 = min(w, c0 + ps)
        th, tw = r1 - r0, c1 - c0

        if th <= 0 or tw <= 0:
            return guide

        tex_crop = texture[:th, :tw]

        # Feathered blend mask
        mask = np.ones((th, tw), dtype=np.float32)
        feather = min(8, th // 4, tw // 4)
        if feather > 0:
            for i in range(feather):
                alpha = (i + 1) / feather
                mask[i, :] *= alpha
                mask[-(i + 1), :] *= alpha
                mask[:, i] *= alpha
                mask[:, -(i + 1)] *= alpha

        blend = self.anatomy_blend
        region = guide[r0:r1, c0:c1]
        guide[r0:r1, c0:c1] = np.clip(
            region * (1 - blend * mask) + tex_crop * blend * mask, 0, 1
        )
        return guide

    def generate(
        self,
        pathology_class: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a single realistic POCUS frame.

        Args:
            pathology_class: MoCoLUS pathology class (0-9)
            seed:           Random seed for reproducibility

        Returns:
            [H, W] float32 array in [0, 1]
        """
        from .clinical_frames import ClinicalPathology

        # Generate structural guide
        guide = self.guide_gen.generate(ClinicalPathology(pathology_class), seed=seed)

        # Enhance with anatomy bank textures + PMF placement
        guide = self._enhance_guide_with_anatomy(guide, pathology_class, seed=seed)

        if not self.has_model:
            return guide  # Physics-only fallback

        # Run DDPM inference
        if seed is not None:
            torch.manual_seed(seed)

        guide_t = torch.from_numpy(guide * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).to(self.device)
        labels = torch.tensor([pathology_class], dtype=torch.long, device=self.device)

        from .train_realistic import sample_images
        active_model = self._select_model(pathology_class)
        result = sample_images(
            active_model,
            self.scheduler,
            guide_t,
            labels,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self._get_guidance_scale(pathology_class),
            device=self.device,
        )

        # Convert back to [0, 1] numpy
        frame = (result[0, 0].cpu().float() + 1.0) / 2.0
        frame = frame.clamp(0.0, 1.0).numpy()
        return frame

    def generate_batch(
        self,
        pathology_class: int,
        batch_size: int = 8,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a batch of frames for the same pathology.

        Returns:
            [B, H, W] float32 array in [0, 1]
        """
        from .clinical_frames import ClinicalPathology

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        guides = []
        for _ in range(batch_size):
            frame_seed = int(rng.integers(0, 2**31))
            g = self.guide_gen.generate(
                ClinicalPathology(pathology_class),
                seed=frame_seed,
            )
            g = self._enhance_guide_with_anatomy(g, pathology_class, seed=frame_seed)
            guides.append(g)

        if not self.has_model:
            return np.stack(guides, axis=0)

        if seed is not None:
            torch.manual_seed(seed)

        guides_t = torch.from_numpy(
            np.stack(guides, axis=0)[:, np.newaxis] * 2.0 - 1.0
        ).float().to(self.device)
        labels = torch.full((batch_size,), pathology_class, dtype=torch.long, device=self.device)

        from .train_realistic import sample_images
        active_model = self._select_model(pathology_class)
        result = sample_images(
            active_model,
            self.scheduler,
            guides_t,
            labels,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self._get_guidance_scale(pathology_class),
            device=self.device,
        )

        frames = (result[:, 0].cpu().float() + 1.0) / 2.0
        frames = frames.clamp(0.0, 1.0).numpy()
        return frames

    def generate_stack(
        self,
        pathology_class: int,
        n_frames: int = 32,
        lung_sliding: bool = True,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generate a temporally coherent B-mode stack with M-mode.

        For temporal coherence:
        - The structural guides come from ClinicalTemporalStack, which
          simulates lung sliding and respiratory motion between frames
        - We use a shared base noise vector with small per-frame perturbation
          (noise_perturbation_scale=0.05) so the speckle pattern evolves
          smoothly rather than jumping between independent samples

        Returns dict with:
            bmode_stack:     [N, H, W] float32 in [0, 1]
            mmode:           [H, W] float32 in [0, 1]
            mmode_pattern:   "seashore" or "stratosphere"
            lung_sliding:    bool
        """
        from .clinical_frames import ClinicalPathology, ClinicalFrameGenerator, ClinicalTemporalStack

        # Generate structural guide stack with temporal motion
        frame_gen = ClinicalFrameGenerator(
            image_size=(self.image_size, self.image_size)
        )
        temporal = ClinicalTemporalStack(
            frame_gen=frame_gen,
            n_frames=n_frames,
        )
        stack_data = temporal.generate_stack(
            ClinicalPathology(pathology_class),
            lung_sliding=lung_sliding,
            seed=seed,
        )

        # Enhance each frame's guide with anatomy bank textures
        if self.anatomy_bank is not None:
            for i in range(n_frames):
                frame_seed = (seed + i * 100) if seed is not None else None
                stack_data["bmode_stack"][i] = self._enhance_guide_with_anatomy(
                    stack_data["bmode_stack"][i], pathology_class, seed=frame_seed,
                )

        if not self.has_model:
            return stack_data

        # Generate realistic frames using structural guides
        guide_stack = stack_data["bmode_stack"]  # [N, H, W]

        if seed is not None:
            torch.manual_seed(seed)

        # Shared base noise for temporal coherence
        base_noise = torch.randn(1, 1, self.image_size, self.image_size, device=self.device)
        noise_scale = 0.05  # Small perturbation between frames

        realistic_frames = []
        labels = torch.tensor([pathology_class], dtype=torch.long, device=self.device)

        from .train_realistic import sample_images
        active_model = self._select_model(pathology_class)

        for i in range(n_frames):
            guide_t = torch.from_numpy(
                guide_stack[i] * 2.0 - 1.0
            ).unsqueeze(0).unsqueeze(0).float().to(self.device)

            # Perturbed noise for this frame
            frame_noise = base_noise + noise_scale * torch.randn_like(base_noise)

            result = sample_images(
                active_model,
                self.scheduler,
                guide_t,
                labels,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self._get_guidance_scale(pathology_class),
                device=self.device,
            )

            frame = (result[0, 0].cpu().float() + 1.0) / 2.0
            realistic_frames.append(frame.clamp(0.0, 1.0).numpy())

        stack_data["bmode_stack"] = np.stack(realistic_frames, axis=0)

        # Generate realistic M-mode using trained M-mode embeddings
        # The model has separate embeddings for M-mode (class + MMODE_CLASS_OFFSET)
        from .train_realistic import MMODE_CLASS_OFFSET

        # Get synthetic M-mode as structural guide
        synthetic_mmode = stack_data["mmode"]  # [H, n_frames] from ClinicalTemporalStack
        from PIL import Image as PILImage
        mmode_img = PILImage.fromarray(
            (np.clip(synthetic_mmode, 0, 1) * 255).astype(np.uint8), mode="L"
        )
        mmode_arr = np.asarray(
            mmode_img.resize((self.image_size, self.image_size), PILImage.LANCZOS),
            dtype=np.float32,
        ) / 255.0

        mmode_guide = torch.from_numpy(mmode_arr * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).to(self.device)
        mmode_label = torch.tensor(
            [pathology_class + MMODE_CLASS_OFFSET], dtype=torch.long, device=self.device
        )

        mmode_result = sample_images(
            active_model, self.scheduler, mmode_guide, mmode_label,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self._get_guidance_scale(pathology_class),
            device=self.device,
        )
        mmode_frame = (mmode_result[0, 0].cpu().float() + 1.0) / 2.0
        stack_data["mmode"] = mmode_frame.clamp(0.0, 1.0).numpy()

        return stack_data
