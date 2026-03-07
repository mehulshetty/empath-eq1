"""CLSA Cognitive Module: a transformer backbone with a probabilistic output head.

Each cognitive module wraps a base Transformer and adds a projection that
outputs a Gaussian distribution (mean, log-variance) over the shared latent
space. This is the core building block described in Section 3.2 and 3.4 of
the CLSA proposal.

The module does NOT produce token logits. Instead, it produces a probability
distribution that participates in the deliberation loop.
"""

import torch
import torch.nn as nn

from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.config.transformer_config import TransformerConfig
from clsa.modules.transformer import Transformer


class ProbabilisticHead(nn.Module):
    """Projects transformer hidden states to Gaussian parameters.

    Given the final hidden state from the transformer backbone, produces
    a mean vector and a log-variance vector parameterizing a diagonal
    Gaussian over the latent space.

    Using log-variance (instead of raw variance) ensures numerical
    stability and allows the network to output any real number, which
    is then exponentiated to get a positive variance.
    """

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.mu_proj = nn.Linear(hidden_size, latent_dim)
        self.logvar_proj = nn.Linear(hidden_size, latent_dim)

        # Initialize log-variance projection to output near-zero values,
        # so initial variance is close to 1.0 (exp(0) = 1).
        nn.init.zeros_(self.logvar_proj.weight)
        nn.init.zeros_(self.logvar_proj.bias)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce Gaussian parameters from hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_size) from the transformer.

        Returns:
            mu: (batch, seq_len, latent_dim) mean of the distribution.
            logvar: (batch, seq_len, latent_dim) log-variance.
        """
        mu = self.mu_proj(hidden_states)
        logvar = self.logvar_proj(hidden_states)
        return mu, logvar


class CognitiveModule(nn.Module):
    """A single CLSA cognitive module.

    Wraps a Transformer backbone with a ProbabilisticHead. Each module
    is specialized for a cognitive function (Logic, EQ, etc.) through
    Phase 1 domain-specific pre-training.

    During deliberation, the module:
    1. Reads the current shared latent state (via input tokens or conditioning).
    2. Processes through its transformer backbone.
    3. Outputs a Gaussian distribution over the next latent state.

    The distribution is then combined with other modules' distributions
    via precision-weighted product of Gaussians in the deliberation loop.
    """

    def __init__(
        self,
        module_type: ModuleType,
        config: CLSAConfig,
    ):
        super().__init__()
        self.module_type = module_type
        self.config = config

        # Transformer backbone (initialized from pretrained SmolLM2 weights)
        self.backbone = Transformer(config.transformer)

        # Probabilistic output head
        self.prob_head = ProbabilisticHead(
            config.transformer.hidden_size, config.latent_dim
        )

        # Learnable precision weight for this module (the "dial").
        # Stored as log-precision so the actual precision is always positive.
        self.log_precision = nn.Parameter(torch.tensor(0.0))

    @property
    def precision(self) -> torch.Tensor:
        """Current precision value (inverse variance scaling factor)."""
        return self.log_precision.exp()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the module on token input and produce a latent distribution.

        Args:
            input_ids: (batch, seq_len) token indices.
            attention_mask: optional padding mask.

        Returns:
            Dict with:
                mu: (batch, seq_len, latent_dim) distribution mean.
                logvar: (batch, seq_len, latent_dim) distribution log-variance.
                hidden_states: (batch, seq_len, hidden_size) backbone output
                    (kept for cross-attention in the deliberation loop).
        """
        hidden_states = self.backbone(input_ids, attention_mask)
        mu, logvar = self.prob_head(hidden_states)

        return {
            "mu": mu,
            "logvar": logvar,
            "hidden_states": hidden_states,
        }

    def get_effective_variance(self, logvar: torch.Tensor) -> torch.Tensor:
        """Apply precision weighting to get the effective variance.

        As described in Section 3.5: sigma^2_effective = sigma^2 / pi
        where pi is the precision weight for this module.
        """
        variance = logvar.exp()
        return variance / self.precision
