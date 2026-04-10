"""
Minimal LoRA (Low-Rank Adaptation) implementation for ControlNetPOCUS
=====================================================================

Motivation
----------
For zone-aware fine-tuning (Phase 5) we want to train a small delta on
top of the existing `realistic_v2_finetune` checkpoint without touching
its ~69M parameters. LoRA adds rank-r factorised updates to selected
Linear layers, trains only those (plus the `zone_embedding` added in
Phase 1), and saves a tiny ~3 MB delta file.

Why not `peft`
--------------
`peft.get_peft_model` uses wrapping hooks that assume a standard
HuggingFace forward pass. Our `ControlNetPOCUS.forward` re-implements
the UNet2DModel forward pass to inject guide features at every decoder
level via zero-convs. `peft`'s module scanner silently misses those
injection sites on some diffusers versions, and the checkpoint format
changes in ways that break the existing EMA loader. Writing ~300 lines
of LoRA ourselves avoids both issues and keeps the delta file portable.

Design
------
`LoRALinear` wraps an existing `nn.Linear` with a frozen base weight
plus two trainable low-rank matrices. The forward pass computes

    y = base(x) + dropout(x) @ A.T @ B.T * (alpha / rank)

where
    A is [rank, in_features],  Kaiming-initialised (uniform, a=sqrt(5))
    B is [out_features, rank], zero-initialised (adapter = identity)

Zero-init on B means the adapter contributes nothing until training
begins — loading a fresh LoRA into an existing checkpoint produces
bit-identical outputs until at least one optimizer step.

`inject_lora` walks `model.named_modules()`, finds Linear layers whose
names match any of the provided glob-style patterns (via `fnmatch`),
and swaps them in-place with `LoRALinear` wrappers. It logs every
matched module name and raises if fewer than `min_matches` Linear
layers are found — this is a defensive check against `diffusers`
version drift renaming the attention layers under us.

Target patterns (default)
-------------------------
All attention Q/K/V/out projections in the 4 attention blocks
(down_blocks.{2,3}, up_blocks.{0,1}, mid_block) AND all `time_emb_proj`
Linear layers in every resnet across the whole U-Net.

The `time_emb_proj` inclusion is deliberate and important: these layers
receive the `(time + class + zone)` embedding and feed it into every
resnet. LoRA-wrapping them lets the model learn zone-specific resnet
modulations *through the embedding path* without having to touch the
resnet convs directly (which would cost 41 MB+ of params to wrap).

Trainable parameter budget
--------------------------
For the ControlNetPOCUS architecture (69.2M params) at rank 16:

    44 attention Linears  +  22 time_emb_proj Linears  =  66 targets
    → ~0.74 M trainable params  (~1.1% of total)

This is standard LoRA territory for a small-data fine-tune.
"""

from __future__ import annotations

import fnmatch
import logging
import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default target patterns for ControlNetPOCUS
# ---------------------------------------------------------------------------
# Each pattern is a glob matched against the dotted module name returned
# by `model.named_modules()`. Verified against the diffusers UNet2DModel
# as of 2026-04; log all matched module names at injection time so
# version drift shows up loudly during training.

DEFAULT_LORA_TARGETS: Tuple[str, ...] = (
    # ─ Attention Q/K/V/out projections in the 5 attention blocks ─
    "unet.down_blocks.2.attentions.*.to_q",
    "unet.down_blocks.2.attentions.*.to_k",
    "unet.down_blocks.2.attentions.*.to_v",
    "unet.down_blocks.2.attentions.*.to_out.0",
    "unet.down_blocks.3.attentions.*.to_q",
    "unet.down_blocks.3.attentions.*.to_k",
    "unet.down_blocks.3.attentions.*.to_v",
    "unet.down_blocks.3.attentions.*.to_out.0",
    "unet.mid_block.attentions.*.to_q",
    "unet.mid_block.attentions.*.to_k",
    "unet.mid_block.attentions.*.to_v",
    "unet.mid_block.attentions.*.to_out.0",
    "unet.up_blocks.0.attentions.*.to_q",
    "unet.up_blocks.0.attentions.*.to_k",
    "unet.up_blocks.0.attentions.*.to_v",
    "unet.up_blocks.0.attentions.*.to_out.0",
    "unet.up_blocks.1.attentions.*.to_q",
    "unet.up_blocks.1.attentions.*.to_k",
    "unet.up_blocks.1.attentions.*.to_v",
    "unet.up_blocks.1.attentions.*.to_out.0",
    # ─ time_emb_proj Linears in every resnet ─
    # These feed the (time + class + zone) embedding into the resnets.
    # LoRA-wrapping them is how the zone embedding actually influences
    # the convolutional path without touching the convs themselves.
    "unet.*.resnets.*.time_emb_proj",
    "unet.mid_block.resnets.*.time_emb_proj",
)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Linear layer with a frozen base and a trainable rank-`r` adapter.

    forward(x) = base(x) + dropout(x) @ A.T @ B.T * (alpha / rank)

    Initialisation follows the original LoRA paper (Hu et al. 2021):
        A ~ Kaiming uniform (a = sqrt(5))
        B = 0                → adapter starts as identity (no-op)

    The zero-init on B is what guarantees that freshly-injected LoRA
    leaves model outputs bit-identical until the first optimizer step.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze the base weights and any base biases.
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Trainable low-rank matrices. Match the base layer's dtype/device
        # so LoRA can be dropped into a mixed-precision model without a
        # device-mismatch headache.
        base_weight = base_layer.weight
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_features, dtype=base_weight.dtype, device=base_weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=base_weight.dtype, device=base_weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        dropout_x = self.dropout(x)
        # Shape: x @ A.T  →  [..., rank], then @ B.T  →  [..., out_features]
        delta = dropout_x @ self.lora_A.T @ self.lora_B.T
        return base_out + delta * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base_layer.in_features}, "
            f"out_features={self.base_layer.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )


# ---------------------------------------------------------------------------
# Helpers for in-place module replacement
# ---------------------------------------------------------------------------

def _get_submodule(root: nn.Module, dotted_name: str) -> nn.Module:
    """Resolve a dotted module path (e.g. 'unet.down_blocks.2.attentions.0.to_q')
    to the actual submodule. Handles numeric indices for ModuleList/Sequential."""
    mod: nn.Module = root
    for part in dotted_name.split("."):
        if part.isdigit():
            mod = mod[int(part)]  # type: ignore[index]
        else:
            mod = getattr(mod, part)
    return mod


def _set_submodule(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    """In-place replace the submodule at `dotted_name` with `replacement`."""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = replacement  # type: ignore[index]
    else:
        setattr(parent, last, replacement)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_lora(
    model: nn.Module,
    target_patterns: Iterable[str] = DEFAULT_LORA_TARGETS,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    min_matches: int = 16,
    verbose: bool = False,
) -> List[str]:
    """
    Replace every `nn.Linear` whose dotted name matches any of the
    provided glob patterns with a `LoRALinear` wrapper.

    Args:
        model:            The model to modify in-place.
        target_patterns:  Iterable of `fnmatch` glob patterns.
        rank:             LoRA rank (default 16).
        alpha:            LoRA alpha scaling factor (default 32).
        dropout:          Dropout applied on the LoRA input path.
        min_matches:      Raise if fewer than this many Linear layers
                          match. Defensive against diffusers version drift.
        verbose:          If True, log every matched module name at INFO.

    Returns:
        The list of matched dotted module names.
    """
    patterns = list(target_patterns)

    # Snapshot the candidate list first — we can't mutate named_modules()
    # mid-iteration without risking a visitor mismatch.
    candidates: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear):
            candidates.append((name, mod))

    matched: List[str] = []
    for name, _linear in candidates:
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            matched.append(name)

    if len(matched) < min_matches:
        logger.error(
            "inject_lora: only %d Linear modules matched patterns %s. "
            "Need at least %d. First 30 candidate names:\n  %s",
            len(matched),
            patterns,
            min_matches,
            "\n  ".join(name for name, _ in candidates[:30]),
        )
        raise RuntimeError(
            f"inject_lora: matched {len(matched)} Linear layers, "
            f"need at least {min_matches}. The diffusers version may "
            f"have renamed attention layers. Enable DEBUG logging to "
            f"see all candidate module names."
        )

    # Swap each matched module in-place
    for name in matched:
        base = _get_submodule(model, name)
        wrapper = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)  # type: ignore[arg-type]
        _set_submodule(model, name, wrapper)
        if verbose:
            logger.info("  wrapped %s -> LoRALinear(rank=%d)", name, rank)

    logger.info(
        "inject_lora: wrapped %d Linear layers at rank=%d, alpha=%d, dropout=%.2f",
        len(matched),
        rank,
        alpha,
        dropout,
    )
    return matched


# ---------------------------------------------------------------------------
# Trainable-parameter masking
# ---------------------------------------------------------------------------

def mark_only_lora_as_trainable(
    model: nn.Module,
    train_zone_embed: bool = True,
    train_class_embed: bool = False,
    extra_trainable_name_substrings: Iterable[str] = (),
) -> Tuple[int, int]:
    """
    Freeze every parameter in `model` except:
        - LoRA adapter weights (any param name containing 'lora_A' or 'lora_B')
        - The zone_embedding weight (if train_zone_embed=True)
        - The unet.class_embedding weight (if train_class_embed=True)
        - Any parameter whose name contains one of
          `extra_trainable_name_substrings`.

    Returns:
        (trainable_param_count, total_param_count)
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    extras = tuple(extra_trainable_name_substrings)
    trainable = 0
    total = 0

    for name, param in model.named_parameters():
        total += param.numel()

        is_lora = "lora_A" in name or "lora_B" in name
        is_zone = train_zone_embed and "zone_embedding" in name
        is_class = train_class_embed and "unet.class_embedding" in name
        is_extra = any(ext in name for ext in extras)

        if is_lora or is_zone or is_class or is_extra:
            param.requires_grad = True
            trainable += param.numel()

    return trainable, total


# ---------------------------------------------------------------------------
# Delta-only serialisation
# ---------------------------------------------------------------------------

def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a state dict containing ONLY the LoRA delta weights and the
    zone_embedding weight. Typical file size: ~3-5 MB.

    Use `torch.save(lora_state_dict(model), 'latest_lora.pt')` for a
    portable, tiny delta that can be shipped alongside the full base
    checkpoint. Load with `load_lora_state_dict`.
    """
    sd: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "zone_embedding" in name:
            sd[name] = param.detach().cpu().clone()
    return sd


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Load a LoRA-only state dict (produced by `lora_state_dict`) into a
    model that has already been `inject_lora`-wrapped.

    Args:
        model:      The LoRA-wrapped model.
        state_dict: The dict returned by `lora_state_dict` or loaded
                    from disk via `torch.load`.
        strict:     If True, raise if any expected key is missing or
                    any incoming key is unexpected. Default False.

    Returns:
        (missing, unexpected) key lists, matching the nn.Module convention.
    """
    model_sd = model.state_dict()
    missing: List[str] = []
    unexpected: List[str] = []

    for name, tensor in state_dict.items():
        if name not in model_sd:
            unexpected.append(name)
            continue
        target = model_sd[name]
        if target.shape != tensor.shape:
            if strict:
                raise RuntimeError(
                    f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)} "
                    f"vs model {tuple(target.shape)}"
                )
            unexpected.append(name)
            continue
        # In-place copy — state_dict() returns references, not clones,
        # so this writes straight into the live model parameters.
        target.copy_(tensor.to(target.device))

    # Missing: LoRA / zone params on the model but absent from the checkpoint.
    for name, _ in model.named_parameters():
        is_delta = "lora_A" in name or "lora_B" in name or "zone_embedding" in name
        if is_delta and name not in state_dict:
            missing.append(name)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict load failed. missing={missing} unexpected={unexpected}"
        )

    return missing, unexpected


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_trainable_parameters(model: nn.Module) -> None:
    """
    Log the trainable / total parameter counts for sanity checking.
    Emits both a logger.info line (for training logs) and a print
    (so it shows up in interactive shells even without log config).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    msg = (
        f"Trainable params: {trainable / 1e6:.3f}M "
        f"of {total / 1e6:.2f}M total ({pct:.2f}%)"
    )
    logger.info(msg)
    print(msg)
