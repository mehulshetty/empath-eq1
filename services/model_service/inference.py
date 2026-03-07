"""CLSA inference engine.

Handles model loading, tokenization, and generation. Separated from
the HTTP layer so it can be tested independently and potentially
swapped to a different serving protocol later.
"""

import logging

import torch
from transformers import AutoTokenizer

from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.deliberation.superposition import gaussian_entropy
from clsa.model import CLSA
from clsa.modules.weight_loading import download_smollm2_weights, load_smollm2_into_transformer

from services.model_service.schemas import (
    DeliberationInfo,
    GenerateRequest,
    GenerateResponse,
    PrecisionWeights,
)

logger = logging.getLogger(__name__)


class CLSAEngine:
    """Manages a CLSA model instance for inference."""

    def __init__(
        self,
        device: str = "cpu",
        model_id: str = "HuggingFaceTB/SmolLM2-135M",
    ):
        self.device = device
        self.model_id = model_id
        self.model: CLSA | None = None
        self.tokenizer = None

    def load(self) -> None:
        """Load the model and tokenizer. Call once at startup."""
        logger.info("Loading tokenizer from %s", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Initializing CLSA model")
        config = CLSAConfig()
        self.model = CLSA(config)

        # Load pretrained weights into each cognitive module's backbone
        logger.info("Loading pretrained weights into cognitive modules")
        model_dir = download_smollm2_weights(self.model_id)
        for mt_str in self.model.modules_dict:
            module = self.model.modules_dict[mt_str]
            load_smollm2_into_transformer(module.backbone, model_dir)

        # Load pretrained weights into the decoder backbone
        load_smollm2_into_transformer(self.model.decoder.backbone, model_dir)

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("CLSA model ready on %s", self.device)

    def _build_precision_overrides(
        self, weights: PrecisionWeights
    ) -> dict[ModuleType, float]:
        return {
            ModuleType.LOGIC: weights.logic,
            ModuleType.EQ: weights.eq,
        }

    @torch.no_grad()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Run inference on a prompt and return generated text.

        This uses a simple greedy/sampling loop. For each new token:
        1. Run the full CLSA deliberation on the current sequence
        2. Take the logits at the last position
        3. Sample or argmax the next token
        4. Append and repeat
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        precision_overrides = self._build_precision_overrides(request.precision)

        generated_ids = input_ids
        last_delib = None

        for _ in range(request.max_new_tokens):
            output = self.model(
                generated_ids,
                attention_mask=attention_mask,
                precision_overrides=precision_overrides,
                temperature=request.temperature,
            )
            last_delib = output["deliberation"]

            # Get logits at the last position
            next_logits = output["logits"][:, -1, :]

            if request.temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / max(request.temperature, 1e-8), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=1
            )

            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode only the newly generated tokens
        new_tokens = generated_ids[:, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        # Build deliberation info if requested
        deliberation = None
        if request.return_deliberation and last_delib is not None:
            entropy = gaussian_entropy(last_delib["final_logvar"]).mean().item()
            deliberation = DeliberationInfo(
                steps=last_delib["steps"],
                module_precisions={
                    mt: self.model.get_module(ModuleType(mt)).precision.item()
                    for mt in self.model.modules_dict
                },
                converged=last_delib["steps"] < self.model.config.max_deliberation_steps,
                final_entropy=entropy,
            )

        return GenerateResponse(text=text, deliberation=deliberation)
