"""Top-level CLSA model tying all components together.

This is the main entry point for using CLSA. It creates cognitive modules,
wires up the deliberation loop, and connects the projection decoder.
"""

import torch
import torch.nn as nn

from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.decoder.projection import ProjectionDecoder
from clsa.deliberation.deliberation_loop import DeliberationLoop
from clsa.modules.cognitive_module import CognitiveModule
from clsa.training.losses import total_clsa_loss


class CLSA(nn.Module):
    """Collaborative Latent Superposition Architecture.

    The full CLSA system: multiple cognitive modules deliberate in a
    shared latent space, producing a combined probabilistic representation
    that is decoded into natural language.

    Usage:
        config = CLSAConfig()
        model = CLSA(config)

        # Forward pass (training)
        output = model(input_ids, labels=labels)
        loss = output["loss"]["total"]

        # Inference with precision overrides (the "mixing board")
        output = model(
            input_ids,
            decoder_input_ids=decoder_ids,
            precision_overrides={ModuleType.EQ: 2.0, ModuleType.LOGIC: 1.0},
        )
    """

    def __init__(self, config: CLSAConfig):
        super().__init__()
        self.config = config

        # Create cognitive modules
        modules = {}
        for mt in config.active_modules:
            modules[mt] = CognitiveModule(mt, config)
        self.modules_dict = nn.ModuleDict(
            {mt.value: mod for mt, mod in modules.items()}
        )

        # Deliberation loop
        self.deliberation = DeliberationLoop(modules, config)

        # Projection decoder
        self.decoder = ProjectionDecoder(
            config.transformer, config.latent_dim
        )

    def get_module(self, module_type: ModuleType) -> CognitiveModule:
        """Access a specific cognitive module by type."""
        return self.modules_dict[module_type.value]

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        precision_overrides: dict[ModuleType, float] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: deliberation followed by decoding.

        Args:
            input_ids: (batch, seq_len) input tokens for the cognitive modules.
            decoder_input_ids: (batch, dec_seq_len) input tokens for the decoder.
                If None, uses input_ids (teacher forcing with same sequence).
            attention_mask: optional padding mask for input_ids.
            labels: (batch, dec_seq_len) target tokens for decoder loss.
            precision_overrides: optional dict to override module precision
                weights. This is the developer-facing "mixing board" API.
            temperature: sampling temperature for deliberation (0 = deterministic).

        Returns:
            Dict with:
                logits: (batch, dec_seq_len, vocab_size) decoder output logits.
                loss: dict of loss components (if labels provided).
                deliberation: full deliberation output dict.
        """
        # Run deliberation
        delib_output = self.deliberation(
            input_ids, attention_mask, precision_overrides, temperature
        )

        # Decode
        if decoder_input_ids is None:
            decoder_input_ids = input_ids

        decoder_output = self.decoder(
            decoder_input_ids,
            delib_output["all_states"],
            labels=labels,
        )

        result = {
            "logits": decoder_output["logits"],
            "deliberation": delib_output,
        }

        # Compute full CLSA loss if labels are provided
        if labels is not None:
            losses = total_clsa_loss(
                task_loss=decoder_output["loss"],
                module_hidden_states=delib_output["module_hidden_states"],
                combined_logvar=delib_output["final_logvar"],
                alpha=self.config.orthogonality_alpha,
                beta=self.config.specialization_beta,
                gamma=self.config.diversity_gamma,
            )
            result["loss"] = losses

        return result

    def set_precision(self, module_type: ModuleType, value: float) -> None:
        """Set a module's precision weight (the "dial" on the mixing board).

        This modifies the module's learned log-precision parameter directly.
        For temporary overrides during inference, use precision_overrides
        in the forward call instead.

        Args:
            module_type: which module to adjust.
            value: new precision value (must be positive).
        """
        if value <= 0:
            raise ValueError("Precision must be positive")
        module = self.get_module(module_type)
        module.log_precision.data = torch.tensor(value).log()

    def get_precisions(self) -> dict[str, float]:
        """Return current precision values for all modules."""
        return {
            mt: self.get_module(ModuleType(mt)).precision.item()
            for mt in self.modules_dict
        }
