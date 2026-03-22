"""Manual judging export/import for the shared benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from clsa.evaluation.benchmark import BenchmarkExample
from clsa.evaluation.generation import GeneratedResponse
from clsa.evaluation.judge_types import JudgeScore


MANUAL_JUDGE_COLUMNS = [
    "annotation_id",
    "benchmark_id",
    "domain",
    "model_family",
    "model_label",
    "control_name",
    "control_value",
    "prompt",
    "response",
    "logic_constraints",
    "eq_criteria",
    "hard_fail_conditions",
    "reference_notes",
    "logic_score",
    "eq_score",
    "hard_fail",
    "hard_fail_reasons",
    "logic_rationale",
    "eq_rationale",
]


def export_manual_judge_csv(
    benchmark_by_id: dict[str, BenchmarkExample],
    responses: list[GeneratedResponse],
    output_path: str | Path,
) -> None:
    """Write a CSV template for manual scoring."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANUAL_JUDGE_COLUMNS)
        writer.writeheader()
        for response in responses:
            example = benchmark_by_id[response.benchmark_id]
            writer.writerow(
                {
                    "annotation_id": annotation_id(response),
                    "benchmark_id": response.benchmark_id,
                    "domain": response.domain,
                    "model_family": response.model_family,
                    "model_label": response.model_label,
                    "control_name": response.control_name,
                    "control_value": response.control_value,
                    "prompt": response.prompt,
                    "response": response.response,
                    "logic_constraints": " | ".join(example.logic_constraints),
                    "eq_criteria": " | ".join(example.eq_criteria),
                    "hard_fail_conditions": " | ".join(example.hard_fail_conditions),
                    "reference_notes": example.reference_notes or "",
                    "logic_score": "",
                    "eq_score": "",
                    "hard_fail": "",
                    "hard_fail_reasons": "",
                    "logic_rationale": "",
                    "eq_rationale": "",
                }
            )


def load_manual_judge_scores(
    path: str | Path,
    *,
    require_complete: bool = True,
) -> list[JudgeScore]:
    """Load a completed manual scoring CSV into JudgeScore objects."""

    rows = []
    seen_annotation_ids: set[str] = set()
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if _is_blank_row(row):
                continue

            annotation = (row.get("annotation_id") or "").strip()
            if annotation:
                if annotation in seen_annotation_ids:
                    raise ValueError(f"Duplicate annotation_id in manual CSV: {annotation}")
                seen_annotation_ids.add(annotation)

            if require_complete and (not row.get("logic_score") or not row.get("eq_score")):
                raise ValueError(
                    f"Manual annotations missing scores for annotation_id={row.get('annotation_id')}"
                )

            if not row.get("logic_score") or not row.get("eq_score"):
                continue

            logic_score = _parse_bounded_float(row["logic_score"], "logic_score")
            eq_score = _parse_bounded_float(row["eq_score"], "eq_score")
            hard_fail = _parse_bool(row.get("hard_fail", "false"))
            hard_fail_reasons = _parse_list_field(row.get("hard_fail_reasons", ""))

            rows.append(
                JudgeScore(
                    benchmark_id=row["benchmark_id"],
                    domain=row["domain"],
                    model_family=row["model_family"],
                    model_label=row["model_label"],
                    control_name=row["control_name"],
                    control_value=_parse_control_value(row["control_value"]),
                    logic_score=logic_score,
                    eq_score=eq_score,
                    combined_score=logic_score * eq_score,
                    hard_fail=hard_fail,
                    hard_fail_reasons=hard_fail_reasons,
                    logic_rationale=(row.get("logic_rationale") or "").strip(),
                    eq_rationale=(row.get("eq_rationale") or "").strip(),
                    metadata={
                        "annotation_id": row.get("annotation_id", "").strip(),
                        "judge_mode": "manual",
                        "response_text": row.get("response", ""),
                    },
                )
            )
    return rows


def annotation_id(response: GeneratedResponse) -> str:
    return (
        f"{response.model_label}__{response.control_name}__"
        f"{response.control_value}__{response.benchmark_id}"
    )


def _is_blank_row(row: dict[str, Any]) -> bool:
    return not any(str(value).strip() for value in row.values())


def _parse_bounded_float(raw: str, field: str) -> float:
    value = float(raw)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field} must be between 0 and 1, got {value}")
    return value


def _parse_bool(raw: str) -> bool:
    value = (raw or "").strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Could not parse boolean value: {raw!r}")


def _parse_list_field(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split("|") if part.strip()]


def _parse_control_value(raw: str) -> float | str:
    text = (raw or "").strip()
    if text == "":
        return text
    try:
        return float(text)
    except ValueError:
        return text
