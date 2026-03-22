"""Paper-quality evaluation CLI for CLSA.

This script implements the surgical evaluation redesign:

- Shared prompt benchmark for the actual see-saw claim
- Generation-based comparison across all model families
- OpenAI judge scoring on independent Logic and EQ axes
- Objective held-out probes scored at the answer level
- Statistical summaries and export files for downstream plotting
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clsa.config.clsa_config import CLSAConfig
from clsa.config.transformer_config import TransformerConfig
from clsa.evaluation.benchmark import load_benchmark
from clsa.evaluation.eval_data import load_eq_probe_examples, load_logic_probe_examples
from clsa.evaluation.generation import (
    CLSAGenerator,
    EnsembleGenerator,
    GeneratedResponse,
    GenerationConfig,
    GenerationControl,
    HFCausalLMGenerator,
    SpecialistGenerator,
    default_baseline_controls,
    default_clsa_controls,
    default_ensemble_controls,
)
from clsa.evaluation.objective_probes import evaluate_generator_on_probes
from clsa.evaluation.openai_judge import JudgeScore, OpenAIJudge
from clsa.evaluation.seesaw_test import SeeSawEvaluator
from clsa.evaluation.stats import (
    JudgeAggregate,
    paired_bootstrap_delta,
    write_compare_csv,
    write_jsonl,
    write_summary_markdown,
)
from clsa.model import CLSA
from clsa.modules.transformer import TransformerForCausalLM
from clsa.training.checkpointing import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
BASELINE_135M_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
BASELINE_360M_MODEL_ID = "HuggingFaceTB/SmolLM2-360M"


def load_shared_tokenizer(model_id: str = MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_clsa_model(checkpoint_path: str, device: str) -> CLSA:
    model = CLSA(CLSAConfig())
    load_checkpoint(model, checkpoint_path, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_phase1_as_causal_lm(backbone_path: str, device: str) -> TransformerForCausalLM:
    model = TransformerForCausalLM(TransformerConfig())
    checkpoint = torch.load(backbone_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    causal_state = {f"model.{k}": v for k, v in state.items()}
    model.load_state_dict(causal_state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_hf_baseline(checkpoint_path: str, model_id: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def build_generation_config(args: argparse.Namespace, *, probe: bool = False) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=args.probe_max_new_tokens if probe else args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_on_eos=True,
        use_tqdm=not args.no_progress,
    )


def build_judge(args: argparse.Namespace) -> OpenAIJudge:
    return OpenAIJudge(
        model=args.judge_model,
        rubric_path=args.rubric_path,
        cache_dir=args.judge_cache_dir,
    )


def build_single_generator(args: argparse.Namespace):
    tokenizer = load_shared_tokenizer()

    if getattr(args, "clsa_checkpoint", None):
        model = load_clsa_model(args.clsa_checkpoint, args.device)
        return CLSAGenerator(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            model_label=args.generator_label or "clsa",
        ), GenerationControl(name="pi_eq", value=args.pi_eq)

    if getattr(args, "phase1_backbone", None):
        model = load_phase1_as_causal_lm(args.phase1_backbone, args.device)
        return SpecialistGenerator(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            model_label=args.generator_label or "phase1_specialist",
        ), GenerationControl(name="mode", value="default")

    if getattr(args, "baseline_checkpoint", None):
        if not args.baseline_model_id:
            raise ValueError("--baseline-model-id is required with --baseline-checkpoint")
        model = load_hf_baseline(args.baseline_checkpoint, args.baseline_model_id, args.device)
        return HFCausalLMGenerator(
            model=model,
            tokenizer=load_shared_tokenizer(args.baseline_model_id),
            device=args.device,
            model_label=args.generator_label or "baseline",
        ), GenerationControl(name="steering", value=args.steering)

    if getattr(args, "ensemble_logic_backbone", None) and getattr(args, "ensemble_eq_backbone", None):
        logic_model = load_phase1_as_causal_lm(args.ensemble_logic_backbone, args.device)
        eq_model = load_phase1_as_causal_lm(args.ensemble_eq_backbone, args.device)
        return EnsembleGenerator(
            logic_model=logic_model,
            eq_model=eq_model,
            tokenizer=tokenizer,
            device=args.device,
        ), GenerationControl(name="logic_weight", value=args.logic_weight)

    raise ValueError("No valid generator configuration provided")


def run_seesaw(args: argparse.Namespace) -> dict:
    benchmark = load_benchmark(args.benchmark_path)
    if args.max_examples is not None:
        benchmark = benchmark[: args.max_examples]

    tokenizer = load_shared_tokenizer()
    model = load_clsa_model(args.checkpoint, args.device)
    generator = CLSAGenerator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        model_label=args.generator_label or Path(args.checkpoint).stem,
    )
    judge = build_judge(args)
    evaluator = SeeSawEvaluator(
        benchmark=benchmark,
        judge=judge,
        generation_config=build_generation_config(args),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report, responses, scores = evaluator.run_sweep(generator, default_clsa_controls())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seesaw.json").write_text(json.dumps(report.to_dict(), indent=2))
    write_jsonl(output_dir / "seesaw_responses.jsonl", [row.to_dict() for row in responses])
    write_jsonl(output_dir / "seesaw_judge_outputs.jsonl", [row.to_dict() for row in scores])
    write_summary_markdown(output_dir / "seesaw_summary.md", report.summary())

    print(report.summary())
    return report.to_dict()


def run_probe(args: argparse.Namespace) -> dict:
    generator, control = build_single_generator(args)
    probe_config = build_generation_config(args, probe=True)
    logic_examples = load_logic_probe_examples(args.max_probe_examples)
    eq_examples = load_eq_probe_examples(args.max_probe_examples)

    logic_report = evaluate_generator_on_probes(generator, logic_examples, control, probe_config)
    eq_report = evaluate_generator_on_probes(generator, eq_examples, control, probe_config)

    results = {
        "logic": logic_report.to_dict(),
        "eq": eq_report.to_dict(),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe.json").write_text(json.dumps(results, indent=2))
    write_summary_markdown(output_dir / "probe_summary.md", _probe_summary_text(results))

    print(_probe_summary_text(results))
    return results


def run_compare(args: argparse.Namespace) -> dict:
    benchmark = load_benchmark(args.benchmark_path)
    if args.max_examples is not None:
        benchmark = benchmark[: args.max_examples]

    logic_probes = load_logic_probe_examples(args.max_probe_examples)
    eq_probes = load_eq_probe_examples(args.max_probe_examples)

    generation_config = build_generation_config(args)
    probe_config = build_generation_config(args, probe=True)
    judge = build_judge(args)
    evaluator = SeeSawEvaluator(
        benchmark=benchmark,
        judge=judge,
        generation_config=generation_config,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    shared_tokenizer = load_shared_tokenizer()

    compare_results: dict[str, dict] = {}
    raw_responses: list[GeneratedResponse] = []
    raw_scores: list[JudgeScore] = []
    csv_rows: list[dict] = []
    score_index: dict[tuple[str, str, str], list[JudgeScore]] = {}

    def evaluate_method(
        method_label: str,
        generator,
        shared_controls: list[GenerationControl],
        probe_control: GenerationControl,
    ) -> None:
        report, responses, scores = evaluator.run_sweep(generator, shared_controls)
        logic_probe = evaluate_generator_on_probes(generator, logic_probes, probe_control, probe_config)
        eq_probe = evaluate_generator_on_probes(generator, eq_probes, probe_control, probe_config)

        compare_results[method_label] = {
            "shared_benchmark": report.to_dict(),
            "objective_probes": {
                "logic": logic_probe.to_dict(),
                "eq": eq_probe.to_dict(),
            },
        }

        raw_responses.extend(responses)
        raw_scores.extend(scores)
        for point in report.points:
            csv_rows.append(_compare_row(method_label, point.control_name, point.control_value, point.aggregate))
        for score in scores:
            key = (method_label, score.control_name, str(score.control_value))
            score_index.setdefault(key, []).append(score)

    if args.logic_backbone:
        logic_specialist = SpecialistGenerator(
            model=load_phase1_as_causal_lm(args.logic_backbone, args.device),
            tokenizer=shared_tokenizer,
            device=args.device,
            model_label="phase1_logic",
        )
        default_control = GenerationControl(name="mode", value="default")
        evaluate_method("phase1_logic", logic_specialist, [default_control], default_control)
        del logic_specialist
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if args.eq_backbone:
        eq_specialist = SpecialistGenerator(
            model=load_phase1_as_causal_lm(args.eq_backbone, args.device),
            tokenizer=shared_tokenizer,
            device=args.device,
            model_label="phase1_eq",
        )
        default_control = GenerationControl(name="mode", value="default")
        evaluate_method("phase1_eq", eq_specialist, [default_control], default_control)
        del eq_specialist
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if args.baseline_135m:
        baseline_135m = HFCausalLMGenerator(
            model=load_hf_baseline(args.baseline_135m, BASELINE_135M_MODEL_ID, args.device),
            tokenizer=load_shared_tokenizer(BASELINE_135M_MODEL_ID),
            device=args.device,
            model_label="baseline_135m",
        )
        evaluate_method(
            "baseline_135m",
            baseline_135m,
            default_baseline_controls(),
            GenerationControl(name="steering", value="balanced"),
        )
        del baseline_135m
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if args.baseline_360m:
        baseline_360m = HFCausalLMGenerator(
            model=load_hf_baseline(args.baseline_360m, BASELINE_360M_MODEL_ID, args.device),
            tokenizer=load_shared_tokenizer(BASELINE_360M_MODEL_ID),
            device=args.device,
            model_label="baseline_360m",
        )
        evaluate_method(
            "baseline_360m",
            baseline_360m,
            default_baseline_controls(),
            GenerationControl(name="steering", value="balanced"),
        )
        del baseline_360m
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if args.logic_backbone and args.eq_backbone:
        ensemble = EnsembleGenerator(
            logic_model=load_phase1_as_causal_lm(args.logic_backbone, args.device),
            eq_model=load_phase1_as_causal_lm(args.eq_backbone, args.device),
            tokenizer=shared_tokenizer,
            device=args.device,
        )
        evaluate_method(
            "naive_ensemble",
            ensemble,
            default_ensemble_controls(),
            GenerationControl(name="logic_weight", value=0.5),
        )
        del ensemble
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if args.clsa_phase2:
        clsa_phase2 = CLSAGenerator(
            model=load_clsa_model(args.clsa_phase2, args.device),
            tokenizer=shared_tokenizer,
            device=args.device,
            model_label="clsa_phase2",
        )
        evaluate_method(
            "clsa_phase2",
            clsa_phase2,
            default_clsa_controls(),
            GenerationControl(name="pi_eq", value=1.0),
        )
        del clsa_phase2
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if args.clsa_phase3:
        clsa_phase3 = CLSAGenerator(
            model=load_clsa_model(args.clsa_phase3, args.device),
            tokenizer=shared_tokenizer,
            device=args.device,
            model_label="clsa_phase3",
        )
        evaluate_method(
            "clsa_phase3",
            clsa_phase3,
            default_clsa_controls(),
            GenerationControl(name="pi_eq", value=1.0),
        )
        del clsa_phase3
        if args.device == "cuda":
            torch.cuda.empty_cache()

    summary = _build_compare_summary(compare_results, score_index, args.bootstrap_samples, args.bootstrap_seed)
    full_payload = {
        "benchmark_path": args.benchmark_path,
        "judge_model": args.judge_model,
        "shared_benchmark_examples": len(benchmark),
        "compare": compare_results,
        "summary": summary,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compare.json").write_text(json.dumps(full_payload, indent=2))
    write_compare_csv(output_dir / "compare.csv", csv_rows)
    write_jsonl(output_dir / "judge_outputs.jsonl", [score.to_dict() for score in raw_scores])
    write_jsonl(output_dir / "responses.jsonl", [response.to_dict() for response in raw_responses])
    write_summary_markdown(output_dir / "compare_summary.md", _compare_summary_text(summary, compare_results))

    print(_compare_summary_text(summary, compare_results))
    return full_payload


def run_full(args: argparse.Namespace) -> dict:
    results = {}
    if args.clsa_phase3:
        seesaw_args = argparse.Namespace(**{**vars(args), "checkpoint": args.clsa_phase3})
        results["seesaw"] = run_seesaw(seesaw_args)

    results["compare"] = run_compare(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full.json").write_text(json.dumps(results, indent=2))
    return results


def _compare_row(
    method_label: str,
    control_name: str,
    control_value: float | str,
    aggregate: JudgeAggregate,
) -> dict:
    return {
        "method_label": method_label,
        "control_name": control_name,
        "control_value": control_value,
        "logic_mean": aggregate.logic.mean,
        "logic_ci_low": aggregate.logic.ci_low,
        "logic_ci_high": aggregate.logic.ci_high,
        "eq_mean": aggregate.eq.mean,
        "eq_ci_low": aggregate.eq.ci_low,
        "eq_ci_high": aggregate.eq.ci_high,
        "combined_mean": aggregate.combined.mean,
        "combined_ci_low": aggregate.combined.ci_low,
        "combined_ci_high": aggregate.combined.ci_high,
        "hard_fail_mean": aggregate.hard_fail_rate.mean,
        "n": aggregate.n,
    }


def _build_compare_summary(
    compare_results: dict[str, dict],
    score_index: dict[tuple[str, str, str], list[JudgeScore]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    best_points = {}
    for method_label, payload in compare_results.items():
        points = payload["shared_benchmark"]["points"]
        best = max(points, key=lambda point: point["aggregate"]["combined"]["mean"])
        best_points[method_label] = best

    deltas = {}
    if "clsa_phase3" in compare_results:
        clsa_default = score_index.get(("clsa_phase3", "pi_eq", "1.0"), [])
        for baseline_label in ["baseline_135m", "baseline_360m"]:
            baseline_default = score_index.get((baseline_label, "steering", "balanced"), [])
            if clsa_default and baseline_default:
                deltas[f"clsa_phase3_vs_{baseline_label}"] = _paired_delta_summary(
                    clsa_default,
                    baseline_default,
                    bootstrap_samples,
                    bootstrap_seed,
                )

    return {
        "best_points": best_points,
        "paired_deltas": deltas,
    }


def _paired_delta_summary(
    a_scores: list[JudgeScore],
    b_scores: list[JudgeScore],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    a_by_id = {score.benchmark_id: score for score in a_scores}
    b_by_id = {score.benchmark_id: score for score in b_scores}
    common_ids = sorted(set(a_by_id) & set(b_by_id))
    logic_a = [a_by_id[idx].logic_score for idx in common_ids]
    logic_b = [b_by_id[idx].logic_score for idx in common_ids]
    eq_a = [a_by_id[idx].eq_score for idx in common_ids]
    eq_b = [b_by_id[idx].eq_score for idx in common_ids]
    combined_a = [a_by_id[idx].combined_score for idx in common_ids]
    combined_b = [b_by_id[idx].combined_score for idx in common_ids]

    return {
        "logic_delta": paired_bootstrap_delta(logic_a, logic_b, bootstrap_samples, bootstrap_seed).to_dict(),
        "eq_delta": paired_bootstrap_delta(eq_a, eq_b, bootstrap_samples, bootstrap_seed + 1).to_dict(),
        "combined_delta": paired_bootstrap_delta(combined_a, combined_b, bootstrap_samples, bootstrap_seed + 2).to_dict(),
        "n": len(common_ids),
    }


def _probe_summary_text(results: dict) -> str:
    logic = results["logic"]
    eq = results["eq"]
    lines = ["Objective Probe Summary", "=" * 30]
    lines.append(f"Logic overall score: {logic['overall_score']:.3f}")
    for summary in logic["dataset_summaries"]:
        lines.append(
            f"  {summary['dataset']}: {summary['mean_score']:.3f} ({summary['metric']}, n={summary['n']})"
        )
    lines.append(f"EQ overall score: {eq['overall_score']:.3f}")
    for summary in eq["dataset_summaries"]:
        lines.append(
            f"  {summary['dataset']}: {summary['mean_score']:.3f} ({summary['metric']}, n={summary['n']})"
        )
    return "\n".join(lines)


def _compare_summary_text(summary: dict, compare_results: dict[str, dict]) -> str:
    lines = ["CLSA Comparative Evaluation Summary", "=" * 40]
    for method_label, best_point in summary["best_points"].items():
        agg = best_point["aggregate"]
        lines.append(
            f"{method_label}: best {best_point['control_name']}={best_point['control_value']}"
            f" | logic={agg['logic']['mean']:.3f}"
            f" | eq={agg['eq']['mean']:.3f}"
            f" | combined={agg['combined']['mean']:.3f}"
            f" | hard_fail={agg['hard_fail_rate']['mean']:.3f}"
        )
    if summary["paired_deltas"]:
        lines.append("")
        lines.append("Paired deltas (default controls):")
        for label, delta in summary["paired_deltas"].items():
            lines.append(
                f"{label}: logic_delta={delta['logic_delta']['mean']:.3f}, "
                f"eq_delta={delta['eq_delta']['mean']:.3f}, "
                f"combined_delta={delta['combined_delta']['mean']:.3f}"
            )
    return "\n".join(lines)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--benchmark-path", default="benchmarks/seesaw_benchmark.jsonl")
    parser.add_argument("--rubric-path", default="benchmarks/judge_rubric.md")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-cache-dir", default="eval_results/judge_cache")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap shared benchmark examples for quick runs")
    parser.add_argument("--max-probe-examples", type=int, default=None,
                        help="Cap held-out probe examples for quick runs")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--probe-max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")


def main():
    parser = argparse.ArgumentParser(
        description="Paper-quality CLSA evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ss = subparsers.add_parser("seesaw", help="Shared-prompt see-saw sweep for CLSA")
    p_ss.add_argument("--checkpoint", required=True, help="Path to CLSA checkpoint")
    p_ss.add_argument("--generator-label", default=None)
    add_common_args(p_ss)

    p_pr = subparsers.add_parser("probe", help="Objective held-out probes for one model config")
    p_pr.add_argument("--clsa-checkpoint", default=None)
    p_pr.add_argument("--phase1-backbone", default=None)
    p_pr.add_argument("--baseline-checkpoint", default=None)
    p_pr.add_argument("--baseline-model-id", default=None)
    p_pr.add_argument("--ensemble-logic-backbone", default=None)
    p_pr.add_argument("--ensemble-eq-backbone", default=None)
    p_pr.add_argument("--generator-label", default=None)
    p_pr.add_argument("--pi-eq", type=float, default=1.0)
    p_pr.add_argument("--steering", default="balanced")
    p_pr.add_argument("--logic-weight", type=float, default=0.5)
    add_common_args(p_pr)

    p_cmp = subparsers.add_parser("compare", help="Full benchmark + probe comparison")
    p_cmp.add_argument("--clsa-phase3", default=None)
    p_cmp.add_argument("--clsa-phase2", default=None)
    p_cmp.add_argument("--logic-backbone", default=None)
    p_cmp.add_argument("--eq-backbone", default=None)
    p_cmp.add_argument("--baseline-135m", default=None)
    p_cmp.add_argument("--baseline-360m", default=None)
    add_common_args(p_cmp)

    p_full = subparsers.add_parser("full", help="Run dedicated CLSA seesaw plus compare bundle")
    p_full.add_argument("--clsa-phase3", default=None)
    p_full.add_argument("--clsa-phase2", default=None)
    p_full.add_argument("--logic-backbone", default=None)
    p_full.add_argument("--eq-backbone", default=None)
    p_full.add_argument("--baseline-135m", default=None)
    p_full.add_argument("--baseline-360m", default=None)
    add_common_args(p_full)

    args = parser.parse_args()

    if args.command == "seesaw":
        run_seesaw(args)
    elif args.command == "probe":
        run_probe(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "full":
        run_full(args)


if __name__ == "__main__":
    main()
