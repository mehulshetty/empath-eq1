"""Objective answer-level scoring helpers for held-out probes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from clsa.evaluation.eval_data import ProbeExample


@dataclass
class ProbeScore:
    """Parsed prediction plus score for one answer-level probe."""

    parsed_prediction: str | tuple[str, ...]
    score: float
    metric: str


def score_probe_response(example: ProbeExample, response_text: str) -> ProbeScore:
    """Score a generated response against an objective held-out probe target."""

    if example.scoring_type == "choice":
        prediction = extract_choice_label(response_text, example.allowed_labels)
        target = str(example.target)
        return ProbeScore(
            parsed_prediction=prediction,
            score=1.0 if prediction == target else 0.0,
            metric="exact_match",
        )

    if example.scoring_type == "single_label":
        prediction = extract_single_label(response_text, example.allowed_labels)
        target = normalize_label(str(example.target))
        return ProbeScore(
            parsed_prediction=prediction,
            score=1.0 if prediction == target else 0.0,
            metric="exact_match",
        )

    if example.scoring_type == "label_set":
        prediction = extract_label_set(response_text, example.allowed_labels)
        target = tuple(sorted(normalize_label(label) for label in example.target))
        return ProbeScore(
            parsed_prediction=prediction,
            score=set_f1_score(prediction, target),
            metric="set_f1",
        )

    raise ValueError(f"Unknown scoring_type: {example.scoring_type}")


def extract_choice_label(text: str, allowed_labels: tuple[str, ...]) -> str:
    """Extract the first valid multiple-choice label from model output."""

    allowed = {label.upper() for label in allowed_labels}
    for char in re.findall(r"[A-Z]", text.upper()):
        if char in allowed:
            return char
    return ""


def extract_single_label(text: str, allowed_labels: tuple[str, ...]) -> str:
    """Extract exactly one allowed label from free-form text."""

    normalized_text = normalize_text(text)
    allowed = [normalize_label(label) for label in allowed_labels]

    for label in sorted(allowed, key=len, reverse=True):
        pattern = rf"\b{re.escape(label)}\b"
        if re.search(pattern, normalized_text):
            return label
    return ""


def extract_label_set(text: str, allowed_labels: tuple[str, ...]) -> tuple[str, ...]:
    """Extract a set of allowed labels from comma-separated or free-form text."""

    normalized_text = normalize_text(text)
    extracted = []
    for label in sorted((normalize_label(v) for v in allowed_labels), key=len, reverse=True):
        pattern = rf"\b{re.escape(label)}\b"
        if re.search(pattern, normalized_text):
            extracted.append(label)
    return tuple(sorted(set(extracted)))


def set_f1_score(predicted: tuple[str, ...], target: tuple[str, ...]) -> float:
    """F1 on unordered label sets."""

    pred = set(predicted)
    gold = set(target)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_label(text: str) -> str:
    return normalize_text(text).replace("-", " ")
