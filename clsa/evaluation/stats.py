"""Statistical summaries and export helpers for evaluation results."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from clsa.evaluation.openai_judge import JudgeScore


@dataclass(frozen=True)
class MeanWithCI:
    mean: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


@dataclass(frozen=True)
class JudgeAggregate:
    logic: MeanWithCI
    eq: MeanWithCI
    combined: MeanWithCI
    hard_fail_rate: MeanWithCI
    n: int

    def to_dict(self) -> dict:
        return {
            "logic": self.logic.to_dict(),
            "eq": self.eq.to_dict(),
            "combined": self.combined.to_dict(),
            "hard_fail_rate": self.hard_fail_rate.to_dict(),
            "n": self.n,
        }


def summarize_judge_scores(
    scores: list[JudgeScore],
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> JudgeAggregate:
    logic_values = [score.logic_score for score in scores]
    eq_values = [score.eq_score for score in scores]
    combined_values = [score.combined_score for score in scores]
    fail_values = [1.0 if score.hard_fail else 0.0 for score in scores]

    return JudgeAggregate(
        logic=bootstrap_mean_ci(logic_values, bootstrap_samples, seed),
        eq=bootstrap_mean_ci(eq_values, bootstrap_samples, seed + 1),
        combined=bootstrap_mean_ci(combined_values, bootstrap_samples, seed + 2),
        hard_fail_rate=bootstrap_mean_ci(fail_values, bootstrap_samples, seed + 3),
        n=len(scores),
    )


def bootstrap_mean_ci(
    values: list[float],
    bootstrap_samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> MeanWithCI:
    if not values:
        return MeanWithCI(mean=0.0, ci_low=0.0, ci_high=0.0)

    if len(values) == 1:
        value = float(values[0])
        return MeanWithCI(mean=value, ci_low=value, ci_high=value)

    rng = random.Random(seed)
    n = len(values)
    sample_means = []
    for _ in range(bootstrap_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        sample_means.append(sum(sample) / n)
    sample_means.sort()

    alpha = 1.0 - confidence
    low_idx = int((alpha / 2) * len(sample_means))
    high_idx = int((1 - alpha / 2) * len(sample_means)) - 1
    mean_value = sum(values) / len(values)
    return MeanWithCI(
        mean=mean_value,
        ci_low=sample_means[max(0, low_idx)],
        ci_high=sample_means[min(len(sample_means) - 1, high_idx)],
    )


def paired_bootstrap_delta(
    a_values: list[float],
    b_values: list[float],
    bootstrap_samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> MeanWithCI:
    if len(a_values) != len(b_values):
        raise ValueError("Paired bootstrap requires equal-length samples")
    if not a_values:
        return MeanWithCI(mean=0.0, ci_low=0.0, ci_high=0.0)

    deltas = [a - b for a, b in zip(a_values, b_values)]
    return bootstrap_mean_ci(deltas, bootstrap_samples, seed, confidence)


def logic_variance_from_aggregates(aggregates: list[JudgeAggregate]) -> float:
    values = [aggregate.logic.mean for aggregate in aggregates]
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def eq_range_from_aggregates(aggregates: list[JudgeAggregate]) -> float:
    values = [aggregate.eq.mean for aggregate in aggregates]
    if not values:
        return 0.0
    return max(values) - min(values)


def write_compare_csv(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_summary_markdown(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
