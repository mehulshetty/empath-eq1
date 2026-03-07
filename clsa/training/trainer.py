"""Training loops for CLSA's three-phase training methodology.

Section 5.1: CLSA training follows three phases:
  Phase 1: Module-specific pre-training (establish module identity)
  Phase 2: Communication training (learn inter-module interaction)
  Phase 3: Guided fine-tuning with structural guardrails
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.model import CLSA
from clsa.modules.cognitive_module import CognitiveModule
from clsa.modules.transformer import TransformerForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters for CLSA training."""

    # Common
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    log_interval: int = 10

    # Phase 1: module pre-training
    phase1_epochs: int = 3
    phase1_lr: float = 5e-5

    # Phase 2: communication training
    phase2_epochs: int = 5
    phase2_lr: float = 1e-4

    # Phase 3: end-to-end fine-tuning
    phase3_epochs: int = 3
    phase3_lr: float = 2e-5

    # Device
    device: str = "cpu"
    dtype: str = "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype]


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


class Phase1Trainer:
    """Phase 1: Module-specific pre-training.

    Each cognitive module is trained independently on domain-specific data.
    We train the full TransformerForCausalLM (backbone + LM head) on
    domain data, then transfer the backbone weights to the CognitiveModule.

    This phase establishes each module's specialized identity before they
    learn to communicate.
    """

    def __init__(
        self,
        module_type: ModuleType,
        model: TransformerForCausalLM,
        config: TrainingConfig,
    ):
        self.module_type = module_type
        self.model = model.to(config.device)
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.phase1_lr,
            weight_decay=config.weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.config.device)
            labels = batch.get("labels", input_ids).to(self.config.device)

            output = self.model(input_ids, labels=labels)
            loss = output["loss"]

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    "Phase 1 [%s] batch %d: loss=%.4f",
                    self.module_type.value,
                    batch_idx,
                    loss.item(),
                )

        return total_loss / max(num_batches, 1)

    def train(self, dataloader: DataLoader) -> None:
        """Run full Phase 1 training."""
        logger.info(
            "Starting Phase 1 training for %s module", self.module_type.value
        )
        for epoch in range(self.config.phase1_epochs):
            avg_loss = self.train_epoch(dataloader)
            logger.info(
                "Phase 1 [%s] epoch %d: avg_loss=%.4f",
                self.module_type.value,
                epoch,
                avg_loss,
            )


class Phase2Trainer:
    """Phase 2: Communication training.

    Module core weights are frozen. Only the cross-attention layers,
    shared latent projection, and decoder are trained. The training data
    must require multiple cognitive faculties simultaneously.
    """

    def __init__(self, model: CLSA, config: TrainingConfig):
        self.model = model.to(config.device)
        self.config = config

        # Freeze cognitive module backbones
        for mt_str in model.modules_dict:
            module = model.modules_dict[mt_str]
            freeze_module(module.backbone)

        # Only train: cross-attention, prob heads, deliberation infra, decoder
        trainable = [
            p for p in model.parameters() if p.requires_grad
        ]
        logger.info(
            "Phase 2: %d trainable parameters (of %d total)",
            sum(p.numel() for p in trainable),
            sum(p.numel() for p in model.parameters()),
        )

        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.phase2_lr,
            weight_decay=config.weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch. Returns dict of average losses."""
        self.model.train()
        totals = {"total": 0.0, "task": 0.0, "orthogonality": 0.0, "diversity": 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.config.device)
            labels = batch.get("labels", input_ids).to(self.config.device)

            output = self.model(input_ids, labels=labels)
            loss = output["loss"]["total"]

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            for key in totals:
                totals[key] += output["loss"][key].item()
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    "Phase 2 batch %d: total=%.4f task=%.4f orth=%.4f div=%.4f",
                    batch_idx,
                    output["loss"]["total"].item(),
                    output["loss"]["task"].item(),
                    output["loss"]["orthogonality"].item(),
                    output["loss"]["diversity"].item(),
                )

        return {k: v / max(num_batches, 1) for k, v in totals.items()}

    def train(self, dataloader: DataLoader) -> None:
        """Run full Phase 2 training."""
        logger.info("Starting Phase 2 communication training")
        for epoch in range(self.config.phase2_epochs):
            avg_losses = self.train_epoch(dataloader)
            logger.info("Phase 2 epoch %d: %s", epoch, avg_losses)


class Phase3Trainer:
    """Phase 3: Guided fine-tuning with structural guardrails.

    All parameters are unfrozen and the system is fine-tuned end-to-end.
    Auxiliary losses (orthogonality, specialization, diversity) prevent
    module identity collapse.
    """

    def __init__(self, model: CLSA, config: TrainingConfig):
        self.model = model.to(config.device)
        self.config = config

        # Unfreeze everything
        unfreeze_module(model)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.phase3_lr,
            weight_decay=config.weight_decay,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Phase 3: all %d parameters trainable", total_params)

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch. Returns dict of average losses."""
        self.model.train()
        totals = {"total": 0.0, "task": 0.0, "orthogonality": 0.0, "diversity": 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.config.device)
            labels = batch.get("labels", input_ids).to(self.config.device)

            output = self.model(input_ids, labels=labels)
            loss = output["loss"]["total"]

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            for key in totals:
                totals[key] += output["loss"][key].item()
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    "Phase 3 batch %d: total=%.4f task=%.4f orth=%.4f div=%.4f",
                    batch_idx,
                    output["loss"]["total"].item(),
                    output["loss"]["task"].item(),
                    output["loss"]["orthogonality"].item(),
                    output["loss"]["diversity"].item(),
                )

        return {k: v / max(num_batches, 1) for k, v in totals.items()}

    def train(self, dataloader: DataLoader) -> None:
        """Run full Phase 3 training."""
        logger.info("Starting Phase 3 guided fine-tuning")
        for epoch in range(self.config.phase3_epochs):
            avg_losses = self.train_epoch(dataloader)
            logger.info("Phase 3 epoch %d: %s", epoch, avg_losses)
