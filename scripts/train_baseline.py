"""Baseline training script for monolithic model comparison.

Fine-tunes a standard HuggingFace causal LM on the combined CLSA training
data (Logic + EQ + Phase 2 multi-faculty). This produces the monolithic
baselines needed to demonstrate the see-saw effect that CLSA eliminates.

Usage:
  uv run python scripts/train_baseline.py \\
      --model HuggingFaceTB/SmolLM2-135M --epochs 3 --device cuda

  uv run python scripts/train_baseline.py \\
      --model HuggingFaceTB/SmolLM2-360M --epochs 3 --device cuda

The training loop mirrors the CLSA optimization settings (optimizer, LR,
grad clipping, AMP). Its data path now mirrors the current CLSA corpus
format as well: structured Phase 1 prompt/target supervision plus the
existing Phase 2/3 multi-faculty text corpus.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clsa.training.data import build_combined_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_grad_norm: float,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    non_blocking = device == "cuda"

    pbar = tqdm(dataloader, desc="Baseline", mininterval=30)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
        labels = batch["labels"].to(device, non_blocking=non_blocking)
        attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)

        with autocast(device, dtype=amp_dtype, enabled=use_amp):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach().item()
        num_batches += 1

        if num_batches % log_interval == 0:
            avg = total_loss / num_batches
            pbar.set_postfix(loss=f"{avg:.4f}")

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train monolithic baseline for CLSA comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID (e.g. HuggingFaceTB/SmolLM2-135M)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="checkpoints")

    args = parser.parse_args()

    use_amp = args.device == "cuda" and args.dtype != "float32"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = dtype_map[args.dtype] if use_amp else torch.float32

    # Derive a short name for the checkpoint file
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    output_path = Path(args.output_dir) / f"baseline_{model_short}.pt"

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = model.to(args.device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.1fM)", param_count, param_count / 1e6)

    logger.info("Building combined dataloader")
    dataloader = build_combined_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    logger.info("Dataloader: %d batches", len(dataloader))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        avg_loss = train_epoch(
            model, dataloader, optimizer, scaler,
            args.device, use_amp, amp_dtype, args.max_grad_norm,
            args.log_interval,
        )
        logger.info("Epoch %d: avg_loss=%.4f", epoch, avg_loss)

        epoch_path = Path(args.output_dir) / f"baseline_{model_short}_epoch{epoch}.pt"
        epoch_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "metadata": {
                "model_id": args.model,
                "epoch": epoch,
                "avg_loss": avg_loss,
            },
        }, epoch_path)
        logger.info("Saved training state to %s", epoch_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": {
            "model_id": args.model,
            "epochs": args.epochs,
            "avg_loss": avg_loss,
        },
    }, output_path)
    logger.info("Baseline training complete. Model saved to %s", output_path)


if __name__ == "__main__":
    main()
