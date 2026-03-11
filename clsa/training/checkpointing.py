"""Checkpoint utilities for saving and loading model state between phases.

Phase 1 trains individual TransformerForCausalLM models. Their backbone
weights need to be extracted and loaded into the CLSA cognitive modules
for Phase 2. Phases 2 and 3 save/load the full CLSA model.

Training state checkpoints (model + optimizer + scaler + epoch) are saved
after each epoch so training can resume from where it left off.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler

from clsa.config.clsa_config import ModuleType
from clsa.model import CLSA

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint with optional metadata.

    Args:
        model: any nn.Module (TransformerForCausalLM or CLSA).
        path: file path to save to (e.g. "checkpoints/phase1_logic.pt").
        metadata: optional dict of training info (epoch, loss, etc.)
            saved alongside the model state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {"model_state_dict": model.state_dict()}
    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into a model.

    Args:
        model: the model to load weights into.
        path: path to the checkpoint file.
        strict: whether to require exact key matching.

    Returns:
        The metadata dict from the checkpoint (empty dict if none).
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=strict
    )
    if missing:
        logger.warning("Missing keys when loading %s: %s", path, missing)
    if unexpected:
        logger.warning("Unexpected keys when loading %s: %s", path, unexpected)

    logger.info("Loaded checkpoint from %s", path)
    return checkpoint.get("metadata", {})


def save_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save full training state so training can resume after interruption.

    Saves model weights, optimizer state, grad scaler state, and the
    completed epoch number.

    Args:
        model: the model being trained.
        optimizer: the optimizer (contains momentum buffers, etc.).
        scaler: the AMP grad scaler.
        epoch: the epoch that just completed (0-indexed).
        path: file path to save to.
        metadata: optional extra info (phase, module type, losses, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
    }
    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)
    logger.info("Saved training state (epoch %d) to %s", epoch, path)


def load_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    path: str | Path,
    device: str = "cpu",
) -> int:
    """Load full training state and return the next epoch to run.

    Args:
        model: the model to load weights into.
        optimizer: the optimizer to restore state into.
        scaler: the AMP grad scaler to restore.
        path: path to the training state checkpoint.
        device: device to map tensors to.

    Returns:
        The next epoch to start from (completed_epoch + 1).
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    completed_epoch = checkpoint["epoch"]
    logger.info(
        "Resumed training state from %s (completed epoch %d)",
        path,
        completed_epoch,
    )
    return completed_epoch + 1


def save_phase1_backbone(
    causal_lm_model: nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Extract and save just the backbone weights from a TransformerForCausalLM.

    Phase 1 trains a full causal LM (backbone + lm_head). For Phase 2,
    we only need the backbone (the transformer layers), since CLSA
    cognitive modules replace the LM head with probabilistic output heads.

    The saved state dict uses the bare Transformer naming (no "model." prefix),
    matching what CognitiveModule.backbone expects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The TransformerForCausalLM has model.embed_tokens, model.layers, model.norm.
    # The CognitiveModule.backbone is a bare Transformer with embed_tokens, layers, norm.
    # So we strip the "model." prefix.
    backbone_state = {}
    for key, value in causal_lm_model.state_dict().items():
        if key.startswith("model."):
            backbone_key = key[len("model."):]
            backbone_state[backbone_key] = value

    checkpoint = {"model_state_dict": backbone_state}
    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)
    logger.info(
        "Saved Phase 1 backbone (%d params) to %s",
        len(backbone_state),
        path,
    )


def load_phase1_backbones_into_clsa(
    clsa_model: CLSA,
    backbone_paths: dict[ModuleType, str | Path],
) -> None:
    """Load Phase 1 backbone weights into the corresponding CLSA cognitive modules.

    Args:
        clsa_model: the CLSA model whose cognitive modules will be updated.
        backbone_paths: mapping from module type to the path of its
            Phase 1 backbone checkpoint.
    """
    for module_type, path in backbone_paths.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Phase 1 backbone checkpoint not found: {path}"
            )

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model_state_dict"]

        module = clsa_model.get_module(module_type)
        missing, unexpected = module.backbone.load_state_dict(
            state_dict, strict=False
        )

        if missing:
            logger.warning(
                "Missing keys loading %s backbone from %s: %s",
                module_type.value, path, missing,
            )
        if unexpected:
            logger.warning(
                "Unexpected keys loading %s backbone from %s: %s",
                module_type.value, path, unexpected,
            )

        logger.info(
            "Loaded Phase 1 %s backbone from %s (%d parameters)",
            module_type.value,
            path,
            len(state_dict),
        )
