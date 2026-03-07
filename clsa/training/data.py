"""Data loading utilities for CLSA training.

Provides dataset builders for each training phase:
  Phase 1: Domain-specific data per module (logic corpora, empathy dialogues)
  Phase 2/3: Multi-faculty data requiring both logic and EQ simultaneously

All builders return standard PyTorch DataLoaders yielding dicts with
"input_ids" and "labels" keys, matching what the trainer classes expect.
"""

import logging
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from clsa.config.clsa_config import ModuleType

logger = logging.getLogger(__name__)

# Default HuggingFace dataset IDs per module type.
# These can be overridden via CLI args.
DEFAULT_DATASETS = {
    ModuleType.LOGIC: {
        "path": "allenai/ai2_arc",
        "name": "ARC-Easy",
        "split": "train",
    },
    ModuleType.EQ: {
        "path": "facebook/empathetic_dialogues",
        "name": None,
        "split": "train",
    },
}

# Phase 2 uses data that requires both faculties.
DEFAULT_PHASE2_DATASET = {
    "path": "Amod/mental_health_counseling_conversations",
    "name": None,
    "split": "train",
}


class TokenizedDataset(Dataset):
    """Wraps a list of tokenized examples for use with DataLoader."""

    def __init__(self, examples: list[dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def _collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences to the same length.

    Pads input_ids with pad_token_id and labels with -100 (ignored by
    cross-entropy loss).
    """
    max_len = max(ex["input_ids"].size(0) for ex in batch)

    padded_ids = []
    padded_labels = []

    for ex in batch:
        seq_len = ex["input_ids"].size(0)
        pad_len = max_len - seq_len

        ids = torch.cat([
            ex["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
        ])
        labels = torch.cat([
            ex["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),
        ])

        padded_ids.append(ids)
        padded_labels.append(labels)

    return {
        "input_ids": torch.stack(padded_ids),
        "labels": torch.stack(padded_labels),
    }


def _format_arc_example(example: dict) -> str:
    """Format an ARC (AI2 Reasoning Challenge) example as a text prompt.

    Turns a multiple-choice science question into a completion task:
    "Question: ... Choices: A) ... B) ... Answer: X"
    """
    question = example["question"]
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = example["answerKey"]

    choices_str = " ".join(
        f"{label}) {text}" for label, text in zip(labels, texts)
    )
    return f"Question: {question} Choices: {choices_str} Answer: {answer_key}"


def _format_empathetic_example(example: dict) -> str:
    """Format an EmpatheticDialogues example as a text prompt.

    Concatenates the situation context with the utterance to create
    a completion task grounded in emotional context.
    """
    context = example.get("situation", "")
    utterance = example.get("utterance", "")
    if context:
        return f"Context: {context}\nResponse: {utterance}"
    return utterance


def _format_counseling_example(example: dict) -> str:
    """Format a mental health counseling conversation.

    These conversations naturally require both logical structure
    (identifying the problem, reasoning about solutions) and
    emotional intelligence (empathy, validation, sensitivity).
    """
    context = example.get("Context", "")
    response = example.get("Response", "")
    return f"Patient: {context}\nCounselor: {response}"


def _tokenize_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize a list of text strings into training examples.

    For causal LM training, labels are the same as input_ids
    (the model learns to predict the next token at each position).
    """
    examples = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].squeeze(0)
        if ids.numel() < 2:
            continue
        examples.append({"input_ids": ids, "labels": ids.clone()})
    return examples


def build_phase1_dataloader(
    module_type: ModuleType,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    max_samples: int | None = None,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    dataset_split: str | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for Phase 1 module-specific pre-training.

    Args:
        module_type: which module to load data for (LOGIC or EQ).
        tokenizer: the tokenizer to use (SmolLM2).
        batch_size: training batch size.
        max_length: maximum sequence length after tokenization.
        max_samples: optional cap on number of examples (for debugging).
        dataset_path: override the default HF dataset path.
        dataset_name: override the default HF dataset config name.
        dataset_split: override the default split.
        num_workers: DataLoader worker count.

    Returns:
        DataLoader yielding {"input_ids": ..., "labels": ...} dicts.
    """
    defaults = DEFAULT_DATASETS[module_type]
    path = dataset_path or defaults["path"]
    name = dataset_name or defaults["name"]
    split = dataset_split or defaults["split"]

    logger.info("Loading Phase 1 %s dataset: %s (%s)", module_type.value, path, name)
    ds = load_dataset(path, name, split=split, trust_remote_code=True)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Format examples based on module type
    if module_type == ModuleType.LOGIC:
        texts = [_format_arc_example(ex) for ex in ds]
    elif module_type == ModuleType.EQ:
        texts = [_format_empathetic_example(ex) for ex in ds]
    else:
        raise ValueError(f"Phase 1 data not configured for {module_type}")

    logger.info("Tokenizing %d examples (max_length=%d)", len(texts), max_length)
    examples = _tokenize_texts(texts, tokenizer, max_length)
    logger.info("Produced %d tokenized examples", len(examples))

    dataset = TokenizedDataset(examples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(_collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=num_workers,
    )


def build_phase2_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    max_samples: int | None = None,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    dataset_split: str | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for Phase 2/3 multi-faculty training.

    Uses data that inherently requires both logical reasoning and
    emotional intelligence (e.g. counseling conversations).

    Args:
        tokenizer: the tokenizer to use.
        batch_size: training batch size.
        max_length: maximum sequence length.
        max_samples: optional cap on number of examples.
        dataset_path: override the default HF dataset path.
        dataset_name: override the default HF dataset config name.
        dataset_split: override the default split.
        num_workers: DataLoader worker count.

    Returns:
        DataLoader yielding {"input_ids": ..., "labels": ...} dicts.
    """
    defaults = DEFAULT_PHASE2_DATASET
    path = dataset_path or defaults["path"]
    name = dataset_name or defaults["name"]
    split = dataset_split or defaults["split"]

    logger.info("Loading Phase 2/3 dataset: %s", path)
    ds = load_dataset(path, name, split=split, trust_remote_code=True)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts = [_format_counseling_example(ex) for ex in ds]

    logger.info("Tokenizing %d examples (max_length=%d)", len(texts), max_length)
    examples = _tokenize_texts(texts, tokenizer, max_length)
    logger.info("Produced %d tokenized examples", len(examples))

    dataset = TokenizedDataset(examples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(_collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=num_workers,
    )
