"""Generation wrappers for all evaluation configurations.

The surgical redesign compares systems on shared prompts by generating real
responses before scoring. This module provides a uniform interface over:

- CLSA models with precision sweeps
- HuggingFace monolithic baselines with steering prompts
- Phase 1 specialists reconstructed as causal LMs
- Naive ensemble generation via interpolated logits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from clsa.config.clsa_config import ModuleType
from clsa.evaluation.benchmark import BenchmarkExample
from clsa.model import CLSA
from clsa.modules.transformer import TransformerForCausalLM


BASELINE_STEERING_PROMPTS = {
    "logic_max": (
        "Style instruction: prioritize maximum logical rigor, precision, and factual "
        "carefulness. Be direct and analytical. Do not add emotional cushioning unless "
        "it is necessary for clarity.\n\nTask:\n{prompt}\n\nResponse:"
    ),
    "logic_leaning": (
        "Style instruction: be mostly analytical and highly clear, while still remaining "
        "professional and minimally considerate.\n\nTask:\n{prompt}\n\nResponse:"
    ),
    "balanced": (
        "Style instruction: balance clear reasoning with emotional sensitivity. Be both "
        "accurate and compassionate.\n\nTask:\n{prompt}\n\nResponse:"
    ),
    "eq_leaning": (
        "Style instruction: be especially warm, validating, and tactful while still "
        "remaining accurate and responsible.\n\nTask:\n{prompt}\n\nResponse:"
    ),
    "eq_max": (
        "Style instruction: prioritize maximum empathy, emotional validation, gentleness, "
        "and social tact while still trying to stay accurate.\n\nTask:\n{prompt}\n\nResponse:"
    ),
}


@dataclass(frozen=True)
class GenerationControl:
    """One control point in a sweep."""

    name: str
    value: float | str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Shared generation settings across methods."""

    max_new_tokens: int = 160
    temperature: float = 0.0
    top_p: float = 1.0
    stop_on_eos: bool = True
    use_tqdm: bool = True


@dataclass
class GeneratedResponse:
    """Raw model output for one benchmark item and one control setting."""

    benchmark_id: str
    domain: str
    prompt: str
    response: str
    model_family: str
    model_label: str
    control_name: str
    control_value: float | str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "domain": self.domain,
            "prompt": self.prompt,
            "response": self.response,
            "model_family": self.model_family,
            "model_label": self.model_label,
            "control_name": self.control_name,
            "control_value": self.control_value,
            "metadata": self.metadata,
        }


class ResponseGenerator(Protocol):
    """Protocol implemented by all generators."""

    model_family: str
    model_label: str

    def generate_one(
        self,
        example: BenchmarkExample,
        control: GenerationControl,
        config: GenerationConfig,
    ) -> GeneratedResponse:
        ...


def default_clsa_controls() -> list[GenerationControl]:
    return [
        GenerationControl(name="pi_eq", value=v)
        for v in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ]


def default_baseline_controls() -> list[GenerationControl]:
    return [
        GenerationControl(name="steering", value=name)
        for name in ["logic_max", "logic_leaning", "balanced", "eq_leaning", "eq_max"]
    ]


def default_ensemble_controls() -> list[GenerationControl]:
    return [
        GenerationControl(name="logic_weight", value=v)
        for v in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    ]


def generate_shared_benchmark(
    generator: ResponseGenerator,
    examples: list[BenchmarkExample],
    control: GenerationControl,
    config: GenerationConfig,
) -> list[GeneratedResponse]:
    """Generate responses for one sweep point across the full benchmark."""

    iterator = examples
    if config.use_tqdm:
        desc = f"{generator.model_label}:{control.name}={control.value}"
        iterator = tqdm(examples, desc=desc, mininterval=5)

    outputs = []
    for example in iterator:
        outputs.append(generator.generate_one(example, control, config))
    return outputs


class HFCausalLMGenerator:
    """Generator backed by HuggingFace `generate()` for monolithic baselines."""

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        model_label: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_family = "monolithic_baseline"
        self.model_label = model_label

    @torch.no_grad()
    def generate_one(
        self,
        example: BenchmarkExample,
        control: GenerationControl,
        config: GenerationConfig,
    ) -> GeneratedResponse:
        prompt = _apply_steering_prompt(example.prompt, str(control.value))
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0.0,
            temperature=max(config.temperature, 1e-5),
            top_p=config.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GeneratedResponse(
            benchmark_id=example.id,
            domain=example.domain,
            prompt=example.prompt,
            response=response,
            model_family=self.model_family,
            model_label=self.model_label,
            control_name=control.name,
            control_value=control.value,
            metadata={"steered_prompt": prompt},
        )


class SpecialistGenerator:
    """Generator for Phase 1 specialists reconstructed as causal LMs."""

    def __init__(
        self,
        model: TransformerForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        model_label: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_family = "phase1_specialist"
        self.model_label = model_label

    @torch.no_grad()
    def generate_one(
        self,
        example: BenchmarkExample,
        control: GenerationControl,
        config: GenerationConfig,
    ) -> GeneratedResponse:
        prompt = example.prompt
        response = _manual_causal_generation(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt=prompt,
            config=config,
        )
        return GeneratedResponse(
            benchmark_id=example.id,
            domain=example.domain,
            prompt=prompt,
            response=response,
            model_family=self.model_family,
            model_label=self.model_label,
            control_name=control.name,
            control_value=control.value,
        )


class CLSAGenerator:
    """Generator for CLSA checkpoints using fixed prompt deliberation."""

    def __init__(
        self,
        model: CLSA,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        model_label: str,
        logic_precision: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_family = "clsa"
        self.model_label = model_label
        self.logic_precision = logic_precision

    @torch.no_grad()
    def generate_one(
        self,
        example: BenchmarkExample,
        control: GenerationControl,
        config: GenerationConfig,
    ) -> GeneratedResponse:
        encoded = self.tokenizer(example.prompt, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        precision_overrides = {
            ModuleType.LOGIC: self.logic_precision,
            ModuleType.EQ: float(control.value),
        }
        deliberation = self.model.deliberation(
            input_ids,
            attention_mask=attention_mask,
            precision_overrides=precision_overrides,
            temperature=0.0,
        )

        decoder_input_ids = input_ids.clone()
        prompt_len = decoder_input_ids.shape[1]

        for _ in range(config.max_new_tokens):
            decoder_attention = torch.ones_like(decoder_input_ids)
            decoder_out = self.model.decoder(
                decoder_input_ids,
                deliberation["all_states"],
                attention_mask=decoder_attention,
            )
            next_logits = decoder_out["logits"][:, -1, :]
            next_token = _select_next_token(next_logits, config)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            if (
                config.stop_on_eos
                and self.tokenizer.eos_token_id is not None
                and next_token.item() == self.tokenizer.eos_token_id
            ):
                break

        new_tokens = decoder_input_ids[0, prompt_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GeneratedResponse(
            benchmark_id=example.id,
            domain=example.domain,
            prompt=example.prompt,
            response=response,
            model_family=self.model_family,
            model_label=self.model_label,
            control_name=control.name,
            control_value=control.value,
            metadata={
                "logic_precision": self.logic_precision,
                "deliberation_steps": deliberation["steps"],
            },
        )


class EnsembleGenerator:
    """Naive two-model ensemble generator using interpolated logits."""

    def __init__(
        self,
        logic_model: TransformerForCausalLM,
        eq_model: TransformerForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        model_label: str = "naive_ensemble",
    ):
        self.logic_model = logic_model
        self.eq_model = eq_model
        self.tokenizer = tokenizer
        self.device = device
        self.model_family = "naive_ensemble"
        self.model_label = model_label

    @torch.no_grad()
    def generate_one(
        self,
        example: BenchmarkExample,
        control: GenerationControl,
        config: GenerationConfig,
    ) -> GeneratedResponse:
        prompt = example.prompt
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = encoded["input_ids"].clone()
        prompt_len = generated.shape[1]
        logic_weight = float(control.value)
        eq_weight = 1.0 - logic_weight

        for _ in range(config.max_new_tokens):
            attention = torch.ones_like(generated)
            logic_out = self.logic_model(generated, attention_mask=attention)
            eq_out = self.eq_model(generated, attention_mask=attention)
            next_logits = (
                logic_weight * logic_out["logits"][:, -1, :]
                + eq_weight * eq_out["logits"][:, -1, :]
            )
            next_token = _select_next_token(next_logits, config)
            generated = torch.cat([generated, next_token], dim=1)
            if (
                config.stop_on_eos
                and self.tokenizer.eos_token_id is not None
                and next_token.item() == self.tokenizer.eos_token_id
            ):
                break

        new_tokens = generated[0, prompt_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GeneratedResponse(
            benchmark_id=example.id,
            domain=example.domain,
            prompt=prompt,
            response=response,
            model_family=self.model_family,
            model_label=self.model_label,
            control_name=control.name,
            control_value=control.value,
            metadata={"logic_weight": logic_weight, "eq_weight": eq_weight},
        )


def _apply_steering_prompt(prompt: str, steering: str) -> str:
    template = BASELINE_STEERING_PROMPTS.get(steering)
    if template is None:
        raise ValueError(f"Unknown steering mode: {steering}")
    return template.format(prompt=prompt)


@torch.no_grad()
def _manual_causal_generation(
    model: TransformerForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompt: str,
    config: GenerationConfig,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    generated = encoded["input_ids"].clone()
    prompt_len = generated.shape[1]

    for _ in range(config.max_new_tokens):
        attention = torch.ones_like(generated)
        out = model(generated, attention_mask=attention)
        next_logits = out["logits"][:, -1, :]
        next_token = _select_next_token(next_logits, config)
        generated = torch.cat([generated, next_token], dim=1)
        if (
            config.stop_on_eos
            and tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            break

    new_tokens = generated[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _select_next_token(logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    if config.temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / config.temperature
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)
