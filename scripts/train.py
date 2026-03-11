"""CLSA training script.

Single entry point for all three training phases:

  Phase 1 - Module-specific pre-training:
    uv run python scripts/train.py phase1 --module logic --epochs 3 --device cuda
    uv run python scripts/train.py phase1 --module eq --epochs 3 --device cuda

  Phase 2 - Communication training:
    uv run python scripts/train.py phase2 \\
        --logic-checkpoint checkpoints/phase1_logic.pt \\
        --eq-checkpoint checkpoints/phase1_eq.pt \\
        --epochs 5

  Phase 3 - End-to-end fine-tuning:
    uv run python scripts/train.py phase3 \\
        --checkpoint checkpoints/phase2.pt \\
        --epochs 3

Common options (all phases):
    --device cpu|cuda|mps
    --batch-size 8
    --max-length 512
    --max-samples 100  (for debugging, limits dataset size)
    --output-dir checkpoints/
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add project root to path so we can import clsa
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.config.transformer_config import TransformerConfig
from clsa.model import CLSA
from clsa.modules.transformer import TransformerForCausalLM
from clsa.modules.weight_loading import (
    download_smollm2_weights,
    load_smollm2_into_causal_lm,
    load_smollm2_into_transformer,
)
from clsa.training.checkpointing import (
    load_checkpoint,
    load_phase1_backbones_into_clsa,
    load_training_state,
    save_checkpoint,
    save_phase1_backbone,
)
from clsa.training.data import build_phase1_dataloader, build_phase2_dataloader
from clsa.training.trainer import (
    Phase1Trainer,
    Phase2Trainer,
    Phase3Trainer,
    TrainingConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across all phases."""
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                        help="Training precision (bfloat16/float16 enable AMP on CUDA)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (for debugging)")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (0 = main process only)")
    parser.add_argument("--resume", default=None,
                        help="Path to a training state checkpoint to resume from")


def load_tokenizer() -> AutoTokenizer:
    """Load the SmolLM2 tokenizer."""
    logger.info("Loading tokenizer from %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# -- Phase 1 ------------------------------------------------------------------


def run_phase1(args: argparse.Namespace) -> None:
    """Phase 1: Module-specific pre-training.

    Trains a TransformerForCausalLM on domain-specific data for a single
    module, then saves the backbone weights for use in Phase 2.
    """
    module_type = ModuleType(args.module)
    tokenizer = load_tokenizer()

    # Build dataloader
    dataloader = build_phase1_dataloader(
        module_type=module_type,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    logger.info("Phase 1 dataloader: %d batches", len(dataloader))

    # Initialize model from pretrained SmolLM2
    config = TransformerConfig()
    model = TransformerForCausalLM(config)
    load_smollm2_into_causal_lm(model)
    model.enable_gradient_checkpointing()

    # Configure training
    training_config = TrainingConfig(
        device=args.device,
        dtype=args.dtype,
        phase1_epochs=args.epochs,
        phase1_lr=args.lr or 5e-5,
        log_interval=args.log_interval,
    )

    # Train
    trainer = Phase1Trainer(module_type, model, training_config)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume(args.resume)
        logger.info("Resuming Phase 1 from epoch %d", start_epoch)

    trainer.train(
        dataloader,
        checkpoint_dir=args.output_dir,
        start_epoch=start_epoch,
    )

    # Save backbone checkpoint (strips the LM head)
    output_path = Path(args.output_dir) / f"phase1_{module_type.value}.pt"
    save_phase1_backbone(
        model,
        output_path,
        metadata={
            "phase": 1,
            "module": module_type.value,
            "epochs": args.epochs,
        },
    )
    logger.info("Phase 1 complete. Backbone saved to %s", output_path)


# -- Phase 2 ------------------------------------------------------------------


def run_phase2(args: argparse.Namespace) -> None:
    """Phase 2: Communication training.

    Loads Phase 1 backbones into a CLSA model, freezes them, and trains
    the cross-attention layers, probabilistic heads, and decoder.
    """
    tokenizer = load_tokenizer()

    dataloader = build_phase2_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    logger.info("Phase 2 dataloader: %d batches", len(dataloader))

    # Initialize CLSA model
    clsa_config = CLSAConfig()
    model = CLSA(clsa_config)

    # Load Phase 1 backbones
    backbone_paths = {}
    if args.logic_checkpoint:
        backbone_paths[ModuleType.LOGIC] = args.logic_checkpoint
    if args.eq_checkpoint:
        backbone_paths[ModuleType.EQ] = args.eq_checkpoint

    if backbone_paths:
        load_phase1_backbones_into_clsa(model, backbone_paths)
    else:
        # No Phase 1 checkpoints provided: initialize from pretrained SmolLM2
        logger.info("No Phase 1 checkpoints provided, using pretrained SmolLM2")
        model_dir = download_smollm2_weights(MODEL_ID)
        for mt_str in model.modules_dict:
            module = model.modules_dict[mt_str]
            load_smollm2_into_transformer(module.backbone, model_dir)
        load_smollm2_into_transformer(model.decoder.backbone, model_dir)

    # Configure training
    training_config = TrainingConfig(
        device=args.device,
        dtype=args.dtype,
        phase2_epochs=args.epochs,
        phase2_lr=args.lr or 1e-4,
        log_interval=args.log_interval,
    )

    # Train
    trainer = Phase2Trainer(model, training_config)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume(args.resume)
        logger.info("Resuming Phase 2 from epoch %d", start_epoch)

    trainer.train(
        dataloader,
        checkpoint_dir=args.output_dir,
        start_epoch=start_epoch,
    )

    # Save full CLSA checkpoint
    output_path = Path(args.output_dir) / "phase2.pt"
    save_checkpoint(
        model,
        output_path,
        metadata={"phase": 2, "epochs": args.epochs},
    )
    logger.info("Phase 2 complete. CLSA model saved to %s", output_path)


# -- Phase 3 ------------------------------------------------------------------


def run_phase3(args: argparse.Namespace) -> None:
    """Phase 3: End-to-end fine-tuning with structural guardrails.

    Loads a Phase 2 CLSA checkpoint and fine-tunes all parameters
    with orthogonality, specialization, and diversity losses.
    """
    tokenizer = load_tokenizer()

    dataloader = build_phase2_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    logger.info("Phase 3 dataloader: %d batches", len(dataloader))

    # Initialize and load Phase 2 checkpoint
    clsa_config = CLSAConfig()
    model = CLSA(clsa_config)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, strict=False)
    else:
        logger.warning("No Phase 2 checkpoint provided, training from scratch")

    # Configure training
    training_config = TrainingConfig(
        device=args.device,
        dtype=args.dtype,
        phase3_epochs=args.epochs,
        phase3_lr=args.lr or 2e-5,
        log_interval=args.log_interval,
    )

    # Train
    trainer = Phase3Trainer(model, training_config)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume(args.resume)
        logger.info("Resuming Phase 3 from epoch %d", start_epoch)

    trainer.train(
        dataloader,
        checkpoint_dir=args.output_dir,
        start_epoch=start_epoch,
    )

    # Save final checkpoint
    output_path = Path(args.output_dir) / "phase3.pt"
    save_checkpoint(
        model,
        output_path,
        metadata={"phase": 3, "epochs": args.epochs},
    )
    logger.info("Phase 3 complete. Final CLSA model saved to %s", output_path)


# -- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CLSA training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    # Phase 1
    p1 = subparsers.add_parser("phase1", help="Module-specific pre-training")
    p1.add_argument("--module", required=True, choices=["logic", "eq"],
                     help="Which cognitive module to train")
    p1.add_argument("--epochs", type=int, default=3)
    add_common_args(p1)

    # Phase 2
    p2 = subparsers.add_parser("phase2", help="Communication training")
    p2.add_argument("--logic-checkpoint", default=None,
                     help="Path to Phase 1 logic backbone checkpoint")
    p2.add_argument("--eq-checkpoint", default=None,
                     help="Path to Phase 1 EQ backbone checkpoint")
    p2.add_argument("--epochs", type=int, default=5)
    add_common_args(p2)

    # Phase 3
    p3 = subparsers.add_parser("phase3", help="End-to-end fine-tuning")
    p3.add_argument("--checkpoint", default=None,
                     help="Path to Phase 2 CLSA checkpoint")
    p3.add_argument("--epochs", type=int, default=3)
    add_common_args(p3)

    args = parser.parse_args()

    if args.phase == "phase1":
        run_phase1(args)
    elif args.phase == "phase2":
        run_phase2(args)
    elif args.phase == "phase3":
        run_phase3(args)


if __name__ == "__main__":
    main()
