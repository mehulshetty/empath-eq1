"""Objective held-out probe evaluation.

These probes answer a different question than the shared benchmark:

- Does the system still solve held-out logic tasks?
- Does the system still identify held-out EQ labels?

They are scored locally with answer-level metrics, not with an LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tqdm import tqdm

from clsa.evaluation.eval_data import ProbeExample
from clsa.evaluation.generation import GenerationConfig, GenerationControl, ResponseGenerator
from clsa.evaluation.scorers import ProbeScore, score_probe_response


@dataclass
class ProbeExampleResult:
    id: str
    dataset: str
    domain: str
    control_name: str
    control_value: float | str
    target: str | tuple[str, ...]
    parsed_prediction: str | tuple[str, ...]
    metric: str
    score: float
    prompt: str
    response: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "domain": self.domain,
            "control_name": self.control_name,
            "control_value": self.control_value,
            "target": self.target,
            "parsed_prediction": self.parsed_prediction,
            "metric": self.metric,
            "score": self.score,
            "prompt": self.prompt,
            "response": self.response,
        }


@dataclass
class ProbeDatasetSummary:
    dataset: str
    domain: str
    metric: str
    mean_score: float
    n: int

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "domain": self.domain,
            "metric": self.metric,
            "mean_score": self.mean_score,
            "n": self.n,
        }


@dataclass
class ProbeReport:
    model_family: str
    model_label: str
    control_name: str
    control_value: float | str
    overall_score: float
    dataset_summaries: list[ProbeDatasetSummary]
    example_results: list[ProbeExampleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_family": self.model_family,
            "model_label": self.model_label,
            "control_name": self.control_name,
            "control_value": self.control_value,
            "overall_score": self.overall_score,
            "dataset_summaries": [summary.to_dict() for summary in self.dataset_summaries],
            "example_results": [result.to_dict() for result in self.example_results],
        }


def evaluate_generator_on_probes(
    generator: ResponseGenerator,
    examples: list[ProbeExample],
    control: GenerationControl,
    config: GenerationConfig,
) -> ProbeReport:
    """Generate responses for probe items and score them locally."""

    results: list[ProbeExampleResult] = []
    iterator = examples
    if config.use_tqdm:
        desc = f"Probe {generator.model_label}:{control.name}={control.value}"
        iterator = tqdm(examples, desc=desc, mininterval=5)

    for example in iterator:
        generated = generator.generate_one(example, control, config)
        scored: ProbeScore = score_probe_response(example, generated.response)
        results.append(
            ProbeExampleResult(
                id=example.id,
                dataset=example.dataset,
                domain=example.domain,
                control_name=control.name,
                control_value=control.value,
                target=example.target,
                parsed_prediction=scored.parsed_prediction,
                metric=scored.metric,
                score=scored.score,
                prompt=example.prompt,
                response=generated.response,
            )
        )

    dataset_summaries = _summarize_by_dataset(results)
    overall_score = (
        sum(result.score for result in results) / len(results) if results else 0.0
    )
    return ProbeReport(
        model_family=generator.model_family,
        model_label=generator.model_label,
        control_name=control.name,
        control_value=control.value,
        overall_score=overall_score,
        dataset_summaries=dataset_summaries,
        example_results=results,
    )


def _summarize_by_dataset(results: list[ProbeExampleResult]) -> list[ProbeDatasetSummary]:
    grouped: dict[tuple[str, str, str], list[ProbeExampleResult]] = {}
    for result in results:
        key = (result.dataset, result.domain, result.metric)
        grouped.setdefault(key, []).append(result)

    summaries = []
    for (dataset, domain, metric), rows in sorted(grouped.items()):
        mean_score = sum(row.score for row in rows) / len(rows)
        summaries.append(
            ProbeDatasetSummary(
                dataset=dataset,
                domain=domain,
                metric=metric,
                mean_score=mean_score,
                n=len(rows),
            )
        )
    return summaries
