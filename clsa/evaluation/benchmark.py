"""Shared benchmark schema and loader for the CLSA see-saw test.

The core CLSA claim must be evaluated on the same held-out prompts for every
model and control setting. This module provides the strongly-typed benchmark
record used throughout the generation, judging, and reporting pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkExample:
    """One held-out prompt for the shared multi-faculty benchmark."""

    id: str
    prompt: str
    domain: str
    logic_constraints: tuple[str, ...]
    eq_criteria: tuple[str, ...]
    hard_fail_conditions: tuple[str, ...] = ()
    reference_notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkExample":
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]).strip(),
            domain=str(data["domain"]).strip(),
            logic_constraints=tuple(_normalize_list(data.get("logic_constraints"))),
            eq_criteria=tuple(_normalize_list(data.get("eq_criteria"))),
            hard_fail_conditions=tuple(_normalize_list(data.get("hard_fail_conditions"))),
            reference_notes=_optional_str(data.get("reference_notes")),
        )

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "prompt": self.prompt,
            "domain": self.domain,
            "logic_constraints": list(self.logic_constraints),
            "eq_criteria": list(self.eq_criteria),
        }
        if self.hard_fail_conditions:
            data["hard_fail_conditions"] = list(self.hard_fail_conditions)
        if self.reference_notes:
            data["reference_notes"] = self.reference_notes
        return data

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Benchmark example must have a non-empty id")
        if not self.prompt:
            raise ValueError(f"Benchmark example {self.id} must have a prompt")
        if not self.domain:
            raise ValueError(f"Benchmark example {self.id} must have a domain")
        if not self.logic_constraints:
            raise ValueError(
                f"Benchmark example {self.id} must have at least one logic constraint"
            )
        if not self.eq_criteria:
            raise ValueError(
                f"Benchmark example {self.id} must have at least one EQ criterion"
            )


def load_benchmark(path: str | Path) -> list[BenchmarkExample]:
    """Load a JSONL benchmark file and validate every record."""

    records = []
    seen_ids: set[str] = set()

    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            record = BenchmarkExample.from_dict(raw)
            record.validate()
            if record.id in seen_ids:
                raise ValueError(f"Duplicate benchmark id {record.id} at line {line_number}")
            seen_ids.add(record.id)
            records.append(record)

    if not records:
        raise ValueError(f"No benchmark records found in {path}")

    return records


def save_benchmark(path: str | Path, examples: list[BenchmarkExample]) -> None:
    """Write benchmark examples back to JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for example in examples:
            example.validate()
            f.write(json.dumps(example.to_dict(), ensure_ascii=True) + "\n")


def benchmark_domains(examples: list[BenchmarkExample]) -> dict[str, int]:
    """Return a domain -> count summary for quick audits."""

    counts: dict[str, int] = {}
    for example in examples:
        counts[example.domain] = counts.get(example.domain, 0) + 1
    return counts


def _normalize_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Expected string or list[str], got {type(value)!r}")


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
