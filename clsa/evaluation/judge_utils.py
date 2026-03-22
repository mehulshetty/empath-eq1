"""Shared helpers for judge backends."""

from __future__ import annotations

import json
from typing import Any

from clsa.evaluation.benchmark import BenchmarkExample
from clsa.evaluation.generation import GeneratedResponse
from clsa.evaluation.judge_types import JudgeScore


def judge_score_from_payload(
    payload: dict[str, Any],
    example: BenchmarkExample,
    response: GeneratedResponse,
    *,
    metadata: dict[str, Any] | None = None,
) -> JudgeScore:
    """Convert a raw judge payload into the shared score type."""

    logic_score = bounded_score(payload.get("logic_score"))
    eq_score = bounded_score(payload.get("eq_score"))
    hard_fail = bool(payload.get("hard_fail", False))
    hard_fail_reasons = string_list(payload.get("hard_fail_reasons"))
    logic_rationale = str(payload.get("logic_rationale", "")).strip()
    eq_rationale = str(payload.get("eq_rationale", "")).strip()

    return JudgeScore(
        benchmark_id=example.id,
        domain=example.domain,
        model_family=response.model_family,
        model_label=response.model_label,
        control_name=response.control_name,
        control_value=response.control_value,
        logic_score=logic_score,
        eq_score=eq_score,
        combined_score=logic_score * eq_score,
        hard_fail=hard_fail,
        hard_fail_reasons=hard_fail_reasons,
        logic_rationale=logic_rationale,
        eq_rationale=eq_rationale,
        metadata=dict(metadata or {}),
    )


def coerce_json(content: str) -> dict[str, Any]:
    """Parse JSON from plain or fenced text."""

    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {content}") from exc


def bounded_score(value: Any) -> float:
    """Validate a 0-1 score."""

    if value is None:
        raise ValueError("Judge score missing required numeric field")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"Judge score out of bounds: {score}")
    return score


def string_list(value: Any) -> list[str]:
    """Normalize a judge field into a list of non-empty strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []
