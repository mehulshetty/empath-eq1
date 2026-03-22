"""Shared judge result types used by manual and model-based evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeScore:
    """One scored response on the shared multi-faculty benchmark."""

    benchmark_id: str
    domain: str
    model_family: str
    model_label: str
    control_name: str
    control_value: float | str
    logic_score: float
    eq_score: float
    combined_score: float
    hard_fail: bool
    hard_fail_reasons: list[str] = field(default_factory=list)
    logic_rationale: str = ""
    eq_rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "domain": self.domain,
            "model_family": self.model_family,
            "model_label": self.model_label,
            "control_name": self.control_name,
            "control_value": self.control_value,
            "logic_score": self.logic_score,
            "eq_score": self.eq_score,
            "combined_score": self.combined_score,
            "hard_fail": self.hard_fail,
            "hard_fail_reasons": self.hard_fail_reasons,
            "logic_rationale": self.logic_rationale,
            "eq_rationale": self.eq_rationale,
            "metadata": self.metadata,
        }
