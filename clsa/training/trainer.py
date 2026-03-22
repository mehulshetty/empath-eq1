"""Training loops for CLSA's three-phase training methodology.

Section 5.1: CLSA training follows three phases:
  Phase 1: Module-specific pre-training (establish module identity)
  Phase 2: Communication training (learn inter-module interaction)
  Phase 3: Guided fine-tuning with structural guardrails
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.model import CLSA
from clsa.modules.cognitive_module import CognitiveModule
from clsa.modules.transformer import TransformerForCausalLM
from clsa.training.checkpointing import save_training_state, load_training_state

logger = logging.getLogger(__name__)


def clear_memory(device: str) -> None:
    """Clear GPU memory cache for MPS backend only.

    On CUDA this is counterproductive -- the caching allocator reuses
    memory far more efficiently when left alone.
    """
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


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
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype]

    @property
    def use_amp(self) -> bool:
        return self.device == "cuda" and self.dtype != "float32"

    @property
    def amp_dtype(self) -> torch.dtype:
        """Dtype for autocast. Falls back to float32 when AMP is off."""
        if self.use_amp:
            return self.torch_dtype
        return torch.float32


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def masked_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for causal LM training with prompt masking support."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@dataclass
class ModuleSpecializationProbe:
    """Phase 1 probe task used to retain module-specific competence in Phase 3."""

    module_type: ModuleType
    lm_head_weight: torch.Tensor
    dataloader: DataLoader
    name: str
    _iterator: object | None = field(default=None, init=False, repr=False)

    def next_batch(self) -> dict[str, torch.Tensor]:
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)

    def loss_on_model(self, model: CLSA, device: str) -> torch.Tensor:
        batch = self.next_batch()
        non_blocking = device == "cuda"
        input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
        labels = batch["labels"].to(device, non_blocking=non_blocking)
        attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)

        hidden_states = model.get_module(self.module_type).backbone(input_ids, attention_mask)
        logits = F.linear(hidden_states, self.lm_head_weight.to(hidden_states.dtype))
        return masked_causal_lm_loss(logits, labels)


class Phase3SpecializationRetainer:
    """Periodically probes modules on Phase 1 tasks during Phase 3."""

    def __init__(
        self,
        probes: list[ModuleSpecializationProbe],
        *,
        interval: int = 4,
        device: str = "cpu",
    ):
        self.probes = probes
        self.interval = max(1, interval)
        self.device = device
        for probe in self.probes:
            probe.lm_head_weight = probe.lm_head_weight.detach().to(device)

    def maybe_compute(self, model: CLSA, step: int) -> torch.Tensor:
        if not self.probes or step % self.interval != 0:
            return torch.tensor(0.0, device=self.device)

        losses = [probe.loss_on_model(model, self.device) for probe in self.probes]
        return torch.stack(losses).mean()


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
        self.scaler = GradScaler(enabled=config.use_amp)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        device = self.config.device
        non_blocking = device == "cuda"

        pbar = tqdm(dataloader, desc=f"Phase 1 [{self.module_type.value}]")
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            labels = batch["labels"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)

            with autocast(device, dtype=self.config.amp_dtype, enabled=self.config.use_amp):
                output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.detach()
            num_batches += 1

            if num_batches % self.config.log_interval == 0:
                avg = (total_loss / num_batches).item()
                pbar.set_postfix(loss=f"{avg:.4f}")

            del input_ids, labels, attention_mask, output, loss
            if device == "mps":
                clear_memory(device)

        return (total_loss / max(num_batches, 1)).item()

    def train(
        self,
        dataloader: DataLoader,
        checkpoint_dir: str | None = None,
        start_epoch: int = 0,
    ) -> None:
        """Run full Phase 1 training.

        Args:
            dataloader: training data.
            checkpoint_dir: if set, save training state after each epoch
                to this directory so training can resume.
            start_epoch: epoch to start from (used when resuming).
        """
        logger.info(
            "Starting Phase 1 training for %s module (epoch %d/%d)",
            self.module_type.value,
            start_epoch,
            self.config.phase1_epochs,
        )
        for epoch in range(start_epoch, self.config.phase1_epochs):
            avg_loss = self.train_epoch(dataloader)
            logger.info(
                "Phase 1 [%s] epoch %d: avg_loss=%.4f",
                self.module_type.value,
                epoch,
                avg_loss,
            )
            if checkpoint_dir:
                save_training_state(
                    self.model,
                    self.optimizer,
                    self.scaler,
                    epoch,
                    f"{checkpoint_dir}/phase1_{self.module_type.value}_epoch{epoch}.pt",
                    metadata={
                        "phase": 1,
                        "module": self.module_type.value,
                        "epoch": epoch,
                        "avg_loss": avg_loss,
                    },
                )

    def resume(self, path: str) -> int:
        """Load training state from a checkpoint. Returns the next epoch."""
        return load_training_state(
            self.model, self.optimizer, self.scaler, path, self.config.device
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
        self.scaler = GradScaler(enabled=config.use_amp)

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch. Returns dict of average losses."""
        self.model.train()
        totals = {"total": 0.0, "task": 0.0, "orthogonality": 0.0, "diversity": 0.0}
        num_batches = 0
        device = self.config.device
        non_blocking = device == "cuda"

        pbar = tqdm(dataloader, desc="Phase 2")
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            labels = batch["labels"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)

            with autocast(device, dtype=self.config.amp_dtype, enabled=self.config.use_amp):
                output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output["loss"]["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key in totals:
                totals[key] += output["loss"][key].detach().item()
            num_batches += 1

            if num_batches % self.config.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{totals['total'] / num_batches:.4f}",
                    task=f"{totals['task'] / num_batches:.4f}",
                )

            del input_ids, labels, attention_mask, output, loss
            if device == "mps":
                clear_memory(device)

        return {k: v / max(num_batches, 1) for k, v in totals.items()}

    def train(
        self,
        dataloader: DataLoader,
        checkpoint_dir: str | None = None,
        start_epoch: int = 0,
    ) -> None:
        """Run full Phase 2 training.

        Args:
            dataloader: training data.
            checkpoint_dir: if set, save training state after each epoch.
            start_epoch: epoch to start from (used when resuming).
        """
        logger.info(
            "Starting Phase 2 communication training (epoch %d/%d)",
            start_epoch,
            self.config.phase2_epochs,
        )
        for epoch in range(start_epoch, self.config.phase2_epochs):
            avg_losses = self.train_epoch(dataloader)
            logger.info("Phase 2 epoch %d: %s", epoch, avg_losses)
            if checkpoint_dir:
                save_training_state(
                    self.model,
                    self.optimizer,
                    self.scaler,
                    epoch,
                    f"{checkpoint_dir}/phase2_epoch{epoch}.pt",
                    metadata={"phase": 2, "epoch": epoch, **avg_losses},
                )

    def resume(self, path: str) -> int:
        """Load training state from a checkpoint. Returns the next epoch."""
        return load_training_state(
            self.model, self.optimizer, self.scaler, path, self.config.device
        )


class Phase3Trainer:
    """Phase 3: Guided fine-tuning with structural guardrails.

    All parameters are unfrozen and the system is fine-tuned end-to-end.
    In addition to the main CLSA loss, optional Phase 1 probe batches can be
    replayed through each module backbone with frozen Phase 1 LM heads to
    preserve module-specific competence.
    """

    def __init__(
        self,
        model: CLSA,
        config: TrainingConfig,
        specialization_retainer: Phase3SpecializationRetainer | None = None,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.specialization_retainer = specialization_retainer

        # Unfreeze everything
        unfreeze_module(model)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.phase3_lr,
            weight_decay=config.weight_decay,
        )
        self.scaler = GradScaler(enabled=config.use_amp)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Phase 3: all %d parameters trainable", total_params)
        if self.specialization_retainer is not None:
            logger.info(
                "Phase 3 specialization retention enabled with %d module probes (interval=%d)",
                len(self.specialization_retainer.probes),
                self.specialization_retainer.interval,
            )

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch. Returns dict of average losses."""
        self.model.train()
        totals = {
            "total": 0.0,
            "task": 0.0,
            "orthogonality": 0.0,
            "specialization": 0.0,
            "diversity": 0.0,
        }
        num_batches = 0
        device = self.config.device
        non_blocking = device == "cuda"

        pbar = tqdm(dataloader, desc="Phase 3")
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            labels = batch["labels"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)

            with autocast(device, dtype=self.config.amp_dtype, enabled=self.config.use_amp):
                output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                specialization = torch.tensor(0.0, device=device)
                if self.specialization_retainer is not None:
                    specialization = self.specialization_retainer.maybe_compute(
                        self.model, num_batches + 1
                    )
                loss = (
                    output["loss"]["total"]
                    + self.model.config.specialization_beta * specialization
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            totals["total"] += loss.detach().item()
            totals["task"] += output["loss"]["task"].detach().item()
            totals["orthogonality"] += output["loss"]["orthogonality"].detach().item()
            totals["specialization"] += specialization.detach().item()
            totals["diversity"] += output["loss"]["diversity"].detach().item()
            num_batches += 1

            if num_batches % self.config.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{totals['total'] / num_batches:.4f}",
                    task=f"{totals['task'] / num_batches:.4f}",
                    spec=f"{totals['specialization'] / num_batches:.4f}",
                )

            del input_ids, labels, attention_mask, output, specialization, loss
            if device == "mps":
                clear_memory(device)

        return {k: v / max(num_batches, 1) for k, v in totals.items()}

    def train(
        self,
        dataloader: DataLoader,
        checkpoint_dir: str | None = None,
        start_epoch: int = 0,
    ) -> None:
        """Run full Phase 3 training.

        Args:
            dataloader: training data.
            checkpoint_dir: if set, save training state after each epoch.
            start_epoch: epoch to start from (used when resuming).
        """
        logger.info(
            "Starting Phase 3 guided fine-tuning (epoch %d/%d)",
            start_epoch,
            self.config.phase3_epochs,
        )
        for epoch in range(start_epoch, self.config.phase3_epochs):
            avg_losses = self.train_epoch(dataloader)
            logger.info("Phase 3 epoch %d: %s", epoch, avg_losses)
            if checkpoint_dir:
                save_training_state(
                    self.model,
                    self.optimizer,
                    self.scaler,
                    epoch,
                    f"{checkpoint_dir}/phase3_epoch{epoch}.pt",
                    metadata={"phase": 3, "epoch": epoch, **avg_losses},
                )

    def resume(self, path: str) -> int:
        """Load training state from a checkpoint. Returns the next epoch."""
        return load_training_state(
            self.model, self.optimizer, self.scaler, path, self.config.device
        )
