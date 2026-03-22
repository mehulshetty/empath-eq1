"""You.com-backed judge for the shared CLSA benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm

from clsa.evaluation.benchmark import BenchmarkExample
from clsa.evaluation.generation import GeneratedResponse
from clsa.evaluation.judge_types import JudgeScore
from clsa.evaluation.judge_utils import coerce_json, judge_score_from_payload


class YouJudge:
    """Judge responses with a configured You.com custom agent."""

    def __init__(
        self,
        agent_id: str,
        rubric_path: str | Path = "benchmarks/judge_rubric.md",
        cache_dir: str | Path = "eval_results/judge_cache",
        api_key: str | None = None,
        endpoint: str = "https://api.you.com/v1/agents/runs",
        timeout_s: float = 120.0,
    ):
        api_key = api_key or os.getenv("YOU_API_KEY")
        if not api_key:
            raise RuntimeError(
                "YOU_API_KEY is not set. Export it before running You.com judge-based evaluation."
            )
        if not agent_id.strip():
            raise ValueError("You.com judge requires a non-empty agent ID")

        self.api_key = api_key
        self.agent_id = agent_id.strip()
        self.endpoint = endpoint
        self.timeout_s = timeout_s
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
            iterator = tqdm(responses, desc="You.com judge", mininterval=5)

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

        request_body = {
            "agent": self.agent_id,
            "input": self._build_input(example, response),
            "stream": False,
        }
        raw_response = self._post(request_body)
        payload = self._extract_payload(raw_response)

        score = judge_score_from_payload(
            payload,
            example,
            response,
            metadata={
                "judge_provider": "you",
                "judge_agent_id": self.agent_id,
                "judge_endpoint": self.endpoint,
                "cache_key": cache_key,
                "response_text": response.response,
            },
        )
        cache_path.write_text(json.dumps(score.to_dict(), indent=2, ensure_ascii=True))
        return score

    def _build_input(self, example: BenchmarkExample, response: GeneratedResponse) -> str:
        prompt_payload = {
            "benchmark_example": example.to_dict(),
            "candidate_response": response.response,
        }
        return (
            "You are an evaluation judge for a research benchmark.\n\n"
            f"{self.rubric_text}\n\n"
            "Use the rubric exactly. Return strict JSON only.\n\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"You.com judge request failed with status {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"You.com judge request failed: {exc.reason}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"You.com judge returned invalid JSON response: {raw}") from exc

    def _extract_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        direct = _find_score_payload(data)
        if direct is not None:
            return direct
        raise ValueError(
            "Could not locate judge JSON payload in You.com response. "
            f"Response preview: {json.dumps(data, ensure_ascii=False)[:1000]}"
        )

    def _cache_key(self, example: BenchmarkExample, response: GeneratedResponse) -> str:
        payload = json.dumps(
            {
                "judge_provider": "you",
                "judge_agent_id": self.agent_id,
                "judge_endpoint": self.endpoint,
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
            hard_fail_reasons=[
                str(reason).strip() for reason in data.get("hard_fail_reasons", []) if str(reason).strip()
            ],
            logic_rationale=str(data.get("logic_rationale", "")),
            eq_rationale=str(data.get("eq_rationale", "")),
            metadata=dict(data.get("metadata", {})),
        )


def _find_score_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if "logic_score" in value and "eq_score" in value:
            return value
        for nested in value.values():
            found = _find_score_payload(nested)
            if found is not None:
                return found
        return None

    if isinstance(value, list):
        for item in value:
            found = _find_score_payload(item)
            if found is not None:
                return found
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if "logic_score" not in text or "eq_score" not in text:
            return None
        try:
            parsed = coerce_json(text)
        except ValueError:
            return None
        return _find_score_payload(parsed)

    return None
