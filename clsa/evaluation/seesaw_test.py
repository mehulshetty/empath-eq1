"""Shared-prompt see-saw evaluation for the CLSA hypothesis test."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from clsa.evaluation.benchmark import BenchmarkExample
from clsa.evaluation.generation import (
    GeneratedResponse,
    GenerationConfig,
    GenerationControl,
    ResponseGenerator,
    generate_shared_benchmark,
)
from clsa.evaluation.openai_judge import JudgeScore, OpenAIJudge
from clsa.evaluation.stats import (
    JudgeAggregate,
    eq_range_from_aggregates,
    logic_variance_from_aggregates,
    summarize_judge_scores,
)

logger = logging.getLogger(__name__)


@dataclass
class SeeSawPointResult:
    """Aggregate result for one control point in a sweep."""

    control_name: str
    control_value: float | str
    aggregate: JudgeAggregate

    def to_dict(self) -> dict:
        return {
            "control_name": self.control_name,
            "control_value": self.control_value,
            "aggregate": self.aggregate.to_dict(),
        }


@dataclass
class SeeSawReport:
    """Full see-saw evaluation report across a control sweep."""

    model_label: str
    points: list[SeeSawPointResult]

    @property
    def logic_variance(self) -> float:
        return logic_variance_from_aggregates([point.aggregate for point in self.points])

    @property
    def eq_range(self) -> float:
        return eq_range_from_aggregates([point.aggregate for point in self.points])

    def to_dict(self) -> dict:
        return {
            "model_label": self.model_label,
            "points": [point.to_dict() for point in self.points],
            "logic_variance": self.logic_variance,
            "eq_range": self.eq_range,
            "pass": self.logic_variance < 0.01,
        }

    def summary(self) -> str:
        lines = ["See-Saw Evaluation Report", "=" * 40, f"Model: {self.model_label}"]
        for point in self.points:
            agg = point.aggregate
            lines.append(
                f"  {point.control_name}={point.control_value}"
                f" | logic={agg.logic.mean:.3f}"
                f" [{agg.logic.ci_low:.3f}, {agg.logic.ci_high:.3f}]"
                f" | EQ={agg.eq.mean:.3f}"
                f" [{agg.eq.ci_low:.3f}, {agg.eq.ci_high:.3f}]"
                f" | combined={agg.combined.mean:.3f}"
                f" | hard_fail={agg.hard_fail_rate.mean:.3f}"
            )
        lines.append(f"Logic score variance: {self.logic_variance:.6f}")
        lines.append(f"EQ score range: {self.eq_range:.3f}")
        lines.append("PASS" if self.logic_variance < 0.01 else "FAIL (see-saw detected)")
        return "\n".join(lines)


class SeeSawEvaluator:
    """Judge-based shared-prompt see-saw evaluator."""

    def __init__(
        self,
        benchmark: list[BenchmarkExample],
        judge: OpenAIJudge,
        generation_config: GenerationConfig,
        bootstrap_samples: int = 1000,
        bootstrap_seed: int = 0,
    ):
        self.benchmark = benchmark
        self.benchmark_by_id = {example.id: example for example in benchmark}
        self.judge = judge
        self.generation_config = generation_config
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed

    def run_sweep(
        self,
        generator: ResponseGenerator,
        controls: list[GenerationControl],
    ) -> tuple[SeeSawReport, list[GeneratedResponse], list[JudgeScore]]:
        point_results: list[SeeSawPointResult] = []
        all_responses: list[GeneratedResponse] = []
        all_scores: list[JudgeScore] = []

        for offset, control in enumerate(controls):
            logger.info(
                "Running shared-benchmark sweep point %s=%s for %s",
                control.name,
                control.value,
                generator.model_label,
            )
            responses = generate_shared_benchmark(
                generator, self.benchmark, control, self.generation_config
            )
            scores = self.judge.score_many(self.benchmark_by_id, responses)
            aggregate = summarize_judge_scores(
                scores,
                bootstrap_samples=self.bootstrap_samples,
                seed=self.bootstrap_seed + offset,
            )
            point_results.append(
                SeeSawPointResult(
                    control_name=control.name,
                    control_value=control.value,
                    aggregate=aggregate,
                )
            )
            all_responses.extend(responses)
            all_scores.extend(scores)

        report = SeeSawReport(model_label=generator.model_label, points=point_results)
        logger.info("\n%s", report.summary())
        return report, all_responses, all_scores
