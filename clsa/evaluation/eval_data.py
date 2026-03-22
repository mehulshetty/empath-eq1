"""Answer-level evaluation data for objective held-out probes.

Unlike the original token-level evaluation, these probes emit prompt/target
examples with explicit scoring types. They are used only for objective held-out
checks (Logic and EQ domain competence), not for the core shared-prompt
see-saw benchmark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import load_dataset

logger = logging.getLogger(__name__)

GO_EMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@dataclass(frozen=True)
class ProbeExample:
    """One answer-level held-out probe item."""

    id: str
    dataset: str
    domain: str
    prompt: str
    scoring_type: str
    target: str | tuple[str, ...]
    allowed_labels: tuple[str, ...] = ()
    metadata: dict | None = None


def load_logic_probe_examples(max_examples: int | None = None) -> list[ProbeExample]:
    """Load answer-level logic probes from held-out splits."""

    configs = [
        ("allenai/ai2_arc", "ARC-Challenge", "test"),
        ("Rowan/hellaswag", None, "validation"),
    ]
    raw = []
    for path, name, split in configs:
        logger.info("Loading logic probe dataset: %s (%s) [%s]", path, name, split)
        ds = load_dataset(path, name, split=split)
        raw.append((path, ds))

    total_available = sum(len(ds) for _, ds in raw)
    examples: list[ProbeExample] = []

    for dataset_name, ds in raw:
        if max_examples is not None:
            cap = max(1, int(max_examples * len(ds) / total_available))
            ds = ds.select(range(min(cap, len(ds))))

        if dataset_name == "allenai/ai2_arc":
            examples.extend(_build_arc_probe_examples(ds))
        else:
            examples.extend(_build_hellaswag_probe_examples(ds))

    return examples


def load_eq_probe_examples(max_examples: int | None = None) -> list[ProbeExample]:
    """Load answer-level EQ probes from held-out splits."""

    configs = [
        ("google-research-datasets/go_emotions", "simplified", "test"),
        ("dair-ai/emotion", "split", "test"),
    ]
    raw = []
    for path, name, split in configs:
        logger.info("Loading EQ probe dataset: %s (%s) [%s]", path, name, split)
        ds = load_dataset(path, name, split=split)
        raw.append((path, ds))

    total_available = sum(len(ds) for _, ds in raw)
    examples: list[ProbeExample] = []

    for dataset_name, ds in raw:
        if max_examples is not None:
            cap = max(1, int(max_examples * len(ds) / total_available))
            ds = ds.select(range(min(cap, len(ds))))

        if dataset_name == "google-research-datasets/go_emotions":
            examples.extend(_build_goemotions_probe_examples(ds))
        else:
            examples.extend(_build_emotion_probe_examples(ds))

    return examples


def _build_arc_probe_examples(ds) -> list[ProbeExample]:
    examples = []
    for i, ex in enumerate(ds):
        choices = ex["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choices_block = "\n".join(
            f"{label}) {text}" for label, text in zip(labels, texts)
        )
        prompt = (
            "Answer the following science question. Respond with only the single "
            "correct choice letter.\n\n"
            f"Question: {ex['question']}\nChoices:\n{choices_block}\n\nAnswer:"
        )
        examples.append(
            ProbeExample(
                id=f"arc_{i}",
                dataset="arc_challenge",
                domain="logic",
                prompt=prompt,
                scoring_type="choice",
                target=str(ex["answerKey"]).strip().upper(),
                allowed_labels=tuple(labels),
                metadata={"source_split": "test"},
            )
        )
    return examples


def _build_hellaswag_probe_examples(ds) -> list[ProbeExample]:
    examples = []
    for i, ex in enumerate(ds):
        endings = ex["endings"]
        letters = [chr(65 + idx) for idx in range(len(endings))]
        choices_block = "\n".join(
            f"{letter}) {ending}" for letter, ending in zip(letters, endings)
        )
        prompt = (
            "Choose the best continuation. Respond with only the single correct "
            "choice letter.\n\n"
            f"Context: {ex['ctx']}\nChoices:\n{choices_block}\n\nAnswer:"
        )
        target_letter = chr(65 + int(ex["label"]))
        examples.append(
            ProbeExample(
                id=f"hellaswag_{i}",
                dataset="hellaswag",
                domain="logic",
                prompt=prompt,
                scoring_type="choice",
                target=target_letter,
                allowed_labels=tuple(letters),
                metadata={"source_split": "validation"},
            )
        )
    return examples


def _build_goemotions_probe_examples(ds) -> list[ProbeExample]:
    examples = []
    allowed = tuple(GO_EMOTIONS_LABELS)
    label_set = set(allowed)

    for i, ex in enumerate(ds):
        labels = tuple(
            sorted(
                GO_EMOTIONS_LABELS[idx]
                for idx in ex["labels"]
                if 0 <= idx < len(GO_EMOTIONS_LABELS)
            )
        ) or ("neutral",)
        prompt = (
            "Identify all emotions present in the text. Respond with a comma-separated "
            "list using only labels from this set:\n"
            + ", ".join(allowed)
            + f"\n\nText: {ex['text']}\n\nLabels:"
        )
        examples.append(
            ProbeExample(
                id=f"goemotions_{i}",
                dataset="go_emotions",
                domain="eq",
                prompt=prompt,
                scoring_type="label_set",
                target=labels,
                allowed_labels=tuple(label for label in allowed if label in label_set),
                metadata={"source_split": "test"},
            )
        )
    return examples


def _build_emotion_probe_examples(ds) -> list[ProbeExample]:
    examples = []
    allowed = tuple(EMOTION_LABELS)
    for i, ex in enumerate(ds):
        label_idx = int(ex["label"])
        target = EMOTION_LABELS[label_idx] if 0 <= label_idx < len(EMOTION_LABELS) else "unknown"
        prompt = (
            "Classify the primary emotion in the text. Respond with exactly one label "
            "from this set:\n"
            + ", ".join(allowed)
            + f"\n\nText: {ex['text']}\n\nEmotion:"
        )
        examples.append(
            ProbeExample(
                id=f"emotion_{i}",
                dataset="emotion",
                domain="eq",
                prompt=prompt,
                scoring_type="single_label",
                target=target,
                allowed_labels=allowed,
                metadata={"source_split": "test"},
            )
        )
    return examples
