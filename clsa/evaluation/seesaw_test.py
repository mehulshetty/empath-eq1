"""See-Saw Effect evaluation framework.

Section 6.1: The core hypothesis test for CLSA. Measures whether adjusting
one module's precision degrades another module's performance.

H0: In a monolithic LLM, increasing EQ emphasis degrades logical accuracy.
H1: In CLSA, increasing EQ precision does NOT degrade logical accuracy.

The test varies pi_EQ across a range while holding pi_L constant, and
independently measures logical accuracy and emotional appropriateness.
"""

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from clsa.config.clsa_config import ModuleType
from clsa.model import CLSA

logger = logging.getLogger(__name__)


@dataclass
class SeeSawResult:
    """Results from a single see-saw evaluation point."""

    eq_precision: float
    logic_precision: float
    logical_accuracy: float
    emotional_score: float
    combined_score: float  # Product, not average (penalizes trade-offs)
    deliberation_steps: float  # Average steps taken


@dataclass
class SeeSawReport:
    """Full see-saw evaluation report across the precision sweep."""

    results: list[SeeSawResult]

    @property
    def logic_variance(self) -> float:
        """Variance of logical accuracy across EQ precision settings.

        Low variance = CLSA hypothesis supported (logic is stable).
        High variance = see-saw effect present.
        """
        scores = [r.logical_accuracy for r in self.results]
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)

    @property
    def eq_range(self) -> float:
        """Range of emotional scores across the sweep."""
        scores = [r.emotional_score for r in self.results]
        return max(scores) - min(scores)

    def summary(self) -> str:
        lines = ["See-Saw Evaluation Report", "=" * 40]
        for r in self.results:
            lines.append(
                f"  pi_EQ={r.eq_precision:.1f} pi_L={r.logic_precision:.1f} "
                f"| logic={r.logical_accuracy:.3f} "
                f"| EQ={r.emotional_score:.3f} "
                f"| combined={r.combined_score:.3f} "
                f"| steps={r.deliberation_steps:.1f}"
            )
        lines.append(f"Logic accuracy variance: {self.logic_variance:.6f}")
        lines.append(f"EQ score range: {self.eq_range:.3f}")
        lines.append(
            "PASS" if self.logic_variance < 0.01 else "FAIL (see-saw detected)"
        )
        return "\n".join(lines)


class SeeSawEvaluator:
    """Runs the core see-saw hypothesis test.

    Sweeps pi_EQ across a range while holding pi_L constant, measuring
    logical accuracy and emotional appropriateness independently at
    each point.

    Users must provide scoring functions for their specific domain.
    """

    def __init__(
        self,
        model: CLSA,
        logic_scorer: callable,
        eq_scorer: callable,
        device: str = "cpu",
    ):
        """
        Args:
            model: trained CLSA model to evaluate.
            logic_scorer: function(logits, labels) -> float score in [0, 1].
            eq_scorer: function(logits, labels) -> float score in [0, 1].
            device: device for evaluation.
        """
        self.model = model.to(device)
        self.logic_scorer = logic_scorer
        self.eq_scorer = eq_scorer
        self.device = device

    @torch.no_grad()
    def evaluate_point(
        self,
        dataloader: DataLoader,
        eq_precision: float,
        logic_precision: float,
    ) -> SeeSawResult:
        """Evaluate at a single precision setting."""
        self.model.eval()

        total_logic = 0.0
        total_eq = 0.0
        total_steps = 0.0
        num_batches = 0

        precision_overrides = {
            ModuleType.EQ: eq_precision,
            ModuleType.LOGIC: logic_precision,
        }

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            output = self.model(
                input_ids,
                labels=labels,
                precision_overrides=precision_overrides,
            )

            total_logic += self.logic_scorer(output["logits"], labels)
            total_eq += self.eq_scorer(output["logits"], labels)
            total_steps += output["deliberation"]["steps"]
            num_batches += 1

        n = max(num_batches, 1)
        logic_acc = total_logic / n
        eq_score = total_eq / n

        return SeeSawResult(
            eq_precision=eq_precision,
            logic_precision=logic_precision,
            logical_accuracy=logic_acc,
            emotional_score=eq_score,
            combined_score=logic_acc * eq_score,
            deliberation_steps=total_steps / n,
        )

    def run_sweep(
        self,
        dataloader: DataLoader,
        eq_precision_range: list[float] | None = None,
        logic_precision: float = 1.0,
    ) -> SeeSawReport:
        """Run the full see-saw sweep.

        Args:
            dataloader: evaluation data requiring both logic and EQ.
            eq_precision_range: list of pi_EQ values to test.
                Defaults to [0.1, 0.5, 1.0, 2.0, 5.0, 10.0].
            logic_precision: held constant throughout the sweep.

        Returns:
            SeeSawReport with results at each precision point.
        """
        if eq_precision_range is None:
            eq_precision_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        results = []
        for eq_pi in eq_precision_range:
            logger.info("Evaluating pi_EQ=%.1f, pi_L=%.1f", eq_pi, logic_precision)
            result = self.evaluate_point(dataloader, eq_pi, logic_precision)
            results.append(result)
            logger.info(
                "  logic=%.3f eq=%.3f combined=%.3f",
                result.logical_accuracy,
                result.emotional_score,
                result.combined_score,
            )

        report = SeeSawReport(results=results)
        logger.info("\n%s", report.summary())
        return report
