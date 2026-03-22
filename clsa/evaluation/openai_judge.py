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
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from clsa.evaluation.benchmark import BenchmarkExample
from clsa.evaluation.generation import GeneratedResponse
from clsa.evaluation.judge_types import JudgeScore
from clsa.evaluation.judge_utils import coerce_json, judge_score_from_payload


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
        payload = coerce_json(content)

        score = judge_score_from_payload(
            payload,
            example,
            response,
            metadata={
                "judge_provider": "openai",
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
