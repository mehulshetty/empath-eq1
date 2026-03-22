"""OpenAI-backed judge for the shared CLSA benchmark.

The judge scores each generated response on two independent axes:

- logic_score
- eq_score

The rubric is deterministic and cached by prompt/model/control hash so reruns
are reproducible and affordable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from clsa.evaluation.benchmark import BenchmarkExample
from clsa.evaluation.generation import GeneratedResponse


@dataclass
class JudgeScore:
    """Judge output for one benchmark response."""

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


class OpenAIJudge:
    """OpenAI judge with filesystem caching."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        rubric_path: str | Path = "benchmarks/judge_rubric.md",
        cache_dir: str | Path = "eval_results/judge_cache",
        api_key: str | None = None,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running judge-based evaluation."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.rubric_path = Path(rubric_path)
        self.rubric_text = self.rubric_path.read_text().strip()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def score_many(
        self,
        benchmark_by_id: dict[str, BenchmarkExample],
        responses: list[GeneratedResponse],
        show_progress: bool = True,
    ) -> list[JudgeScore]:
        iterator = responses
        if show_progress:
            iterator = tqdm(responses, desc="OpenAI judge", mininterval=5)

        results = []
        for response in iterator:
            example = benchmark_by_id[response.benchmark_id]
            results.append(self.score_one(example, response))
        return results

    def score_one(
        self,
        example: BenchmarkExample,
        response: GeneratedResponse,
    ) -> JudgeScore:
        cache_key = self._cache_key(example, response)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return self._load_cache(cache_path)

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation judge for a research benchmark.\n\n"
                        f"{self.rubric_text}\n\n"
                        "Use the rubric exactly. Return strict JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "benchmark_example": example.to_dict(),
                            "candidate_response": response.response,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
        )
        content = completion.choices[0].message.content or "{}"
        payload = _coerce_json(content)

        logic_score = _bounded_score(payload.get("logic_score"))
        eq_score = _bounded_score(payload.get("eq_score"))
        hard_fail = bool(payload.get("hard_fail", False))
        hard_fail_reasons = _string_list(payload.get("hard_fail_reasons"))
        logic_rationale = str(payload.get("logic_rationale", "")).strip()
        eq_rationale = str(payload.get("eq_rationale", "")).strip()

        score = JudgeScore(
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
            metadata={
                "judge_model": self.model,
                "cache_key": cache_key,
                "response_text": response.response,
            },
        )
        cache_path.write_text(json.dumps(score.to_dict(), indent=2, ensure_ascii=True))
        return score

    def _cache_key(self, example: BenchmarkExample, response: GeneratedResponse) -> str:
        payload = json.dumps(
            {
                "judge_model": self.model,
                "rubric": self.rubric_text,
                "benchmark_id": example.id,
                "model_family": response.model_family,
                "model_label": response.model_label,
                "control_name": response.control_name,
                "control_value": response.control_value,
                "response": response.response,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self, path: Path) -> JudgeScore:
        data = json.loads(path.read_text())
        return JudgeScore(
            benchmark_id=data["benchmark_id"],
            domain=data["domain"],
            model_family=data["model_family"],
            model_label=data["model_label"],
            control_name=data["control_name"],
            control_value=data["control_value"],
            logic_score=float(data["logic_score"]),
            eq_score=float(data["eq_score"]),
            combined_score=float(data["combined_score"]),
            hard_fail=bool(data["hard_fail"]),
            hard_fail_reasons=_string_list(data.get("hard_fail_reasons")),
            logic_rationale=str(data.get("logic_rationale", "")),
            eq_rationale=str(data.get("eq_rationale", "")),
            metadata=dict(data.get("metadata", {})),
        )


def _coerce_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {content}") from exc


def _bounded_score(value: Any) -> float:
    if value is None:
        raise ValueError("Judge score missing required numeric field")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"Judge score out of bounds: {score}")
    return score


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []
