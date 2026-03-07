"""The CLSA deliberation loop: the central reasoning mechanism.

Section 4.1: At each latent reasoning step, modules compute probabilistic
outputs, these are combined via distributional interference (product of
Gaussians), EQ applies continuous reweighting, and the shared state updates.

This module orchestrates the full deliberation process from initial input
to a final converged latent state ready for decoding.
"""

import torch
import torch.nn as nn

from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.deliberation.cross_attention import GatedCrossAttention
from clsa.deliberation.superposition import (
    gaussian_entropy,
    product_of_gaussians,
    sample_from_gaussian,
)
from clsa.modules.cognitive_module import CognitiveModule


class EQReweightingField(nn.Module):
    """Continuous reweighting field applied by the EQ module.

    Section 4.1 step 3: EQ has architectural privilege to apply a learned
    soft attention map over the combined distribution. This biases
    reasoning toward emotionally constructive trajectories without
    binary suppression.

    Operates on the combined mu/logvar, producing a multiplicative
    reweighting of the combined distribution's precision (sharpening
    some dimensions, broadening others).
    """

    def __init__(self, latent_dim: int, hidden_size: int):
        super().__init__()
        # Takes EQ hidden states and combined distribution parameters
        # to produce per-dimension reweighting factors
        self.net = nn.Sequential(
            nn.Linear(hidden_size + latent_dim * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(
        self,
        eq_hidden: torch.Tensor,
        combined_mu: torch.Tensor,
        combined_logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply EQ reweighting to the combined distribution.

        Args:
            eq_hidden: (batch, seq_len, hidden_size) EQ module's hidden states.
            combined_mu: (batch, seq_len, latent_dim)
            combined_logvar: (batch, seq_len, latent_dim)

        Returns:
            Reweighted (mu, logvar) pair. The mean is unchanged;
            only the variance is modulated.
        """
        context = torch.cat([eq_hidden, combined_mu, combined_logvar], dim=-1)
        # Log-scale reweighting factor, centered at 0 (no change by default)
        log_reweight = self.net(context)
        # Modulate the log-variance: adding to logvar divides the precision
        # Positive log_reweight = broader (less EQ confidence on this dim)
        # Negative log_reweight = sharper (more EQ confidence)
        reweighted_logvar = combined_logvar + log_reweight
        return combined_mu, reweighted_logvar


class TerminationClassifier(nn.Module):
    """Learned termination signal for the deliberation loop.

    Section 4.3: Predicts readiness for decoding based on the current
    shared latent state, cross-module attention stability, and
    distributional agreement.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        # Input: concatenation of combined mu and logvar
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Predict termination probability.

        Args:
            mu: (batch, seq_len, latent_dim)
            logvar: (batch, seq_len, latent_dim)

        Returns:
            Termination probability of shape (batch,), averaged over
            sequence positions.
        """
        x = torch.cat([mu, logvar], dim=-1)
        # Per-position termination logit, averaged over positions
        logits = self.net(x).squeeze(-1)  # (batch, seq_len)
        return torch.sigmoid(logits.mean(dim=-1))  # (batch,)


class DeliberationLoop(nn.Module):
    """Orchestrates the full CLSA deliberation process.

    Given a set of cognitive modules and an input, runs the iterative
    deliberation loop:
    1. Each module processes input through its backbone
    2. Modules communicate via gated cross-attention
    3. Probabilistic outputs are combined via product of Gaussians
    4. EQ applies continuous reweighting (if privileged)
    5. Termination is evaluated
    6. Repeat or output final latent state
    """

    def __init__(
        self,
        modules: dict[ModuleType, CognitiveModule],
        config: CLSAConfig,
    ):
        super().__init__()
        self.config = config
        self.cognitive_modules = nn.ModuleDict(
            {mt.value: mod for mt, mod in modules.items()}
        )

        hidden_size = config.transformer.hidden_size
        latent_dim = config.latent_dim

        # Build cross-attention layers between all module pairs
        self.cross_attentions = nn.ModuleDict()
        module_types = list(modules.keys())
        for receiver in module_types:
            for sender in module_types:
                if receiver == sender:
                    continue
                key = f"{receiver.value}_from_{sender.value}"
                self.cross_attentions[key] = GatedCrossAttention(
                    hidden_size=hidden_size,
                    num_heads=config.cross_attention_heads,
                    dropout=config.cross_attention_dropout,
                )

        # EQ reweighting field (only if EQ is active and privileged)
        self.eq_reweighting = None
        if ModuleType.EQ in modules and config.eq_privileged:
            self.eq_reweighting = EQReweightingField(latent_dim, hidden_size)

        # Termination classifier
        self.termination = TerminationClassifier(latent_dim)

        # Latent state projection: maps from latent_dim back to hidden_size
        # so the shared state can condition subsequent deliberation steps
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)

    def _run_cross_attention(
        self,
        hidden_states: dict[ModuleType, torch.Tensor],
    ) -> dict[ModuleType, torch.Tensor]:
        """Apply gated cross-attention between all module pairs.

        Each module's hidden states are updated by attending to every
        other module's states.
        """
        updated = {}
        for receiver_type, receiver_hidden in hidden_states.items():
            h = receiver_hidden
            for sender_type, sender_hidden in hidden_states.items():
                if sender_type == receiver_type:
                    continue
                key = f"{receiver_type.value}_from_{sender_type.value}"
                h = self.cross_attentions[key](h, sender_hidden)
            updated[receiver_type] = h
        return updated

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        precision_overrides: dict[ModuleType, float] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Run the full deliberation loop.

        Args:
            input_ids: (batch, seq_len) token indices.
            attention_mask: optional padding mask.
            precision_overrides: optional dict mapping module types to
                precision values, overriding the learned defaults.
                This is the "mixing board" API for developers.
            temperature: sampling temperature for the shared state update.
                0.0 = deterministic (use mean). 1.0 = full stochastic.

        Returns:
            Dict with:
                final_mu: (batch, seq_len, latent_dim) final combined mean.
                final_logvar: (batch, seq_len, latent_dim) final combined log-variance.
                final_state: (batch, seq_len, latent_dim) sampled/mode latent state.
                all_states: list of latent states from each deliberation step
                    (for the decoder to use as context).
                steps: number of deliberation steps taken.
                module_hidden_states: dict of final hidden states per module
                    (for loss computation).
        """
        # Step 1: Initial forward pass through all modules
        module_outputs = {}
        for mt_str, module in self.cognitive_modules.items():
            mt = ModuleType(mt_str)
            module_outputs[mt] = module(input_ids, attention_mask)

        # Collect hidden states for cross-attention
        hidden_states = {
            mt: out["hidden_states"] for mt, out in module_outputs.items()
        }

        all_states = []
        prev_state = None

        for step in range(self.config.max_deliberation_steps):
            # Step 2: Cross-attention between modules
            hidden_states = self._run_cross_attention(hidden_states)

            # Step 3: Collect probabilistic outputs with precision weighting
            mus = []
            logvars = []
            precisions = []

            for mt_str, module in self.cognitive_modules.items():
                mt = ModuleType(mt_str)
                out = module_outputs[mt]

                # Use override precision if provided, otherwise learned value
                if precision_overrides and mt in precision_overrides:
                    pi = torch.tensor(
                        precision_overrides[mt],
                        device=out["mu"].device,
                        dtype=out["mu"].dtype,
                    )
                else:
                    pi = module.precision

                mus.append(out["mu"])
                logvars.append(out["logvar"])
                precisions.append(pi)

            # Step 4: Combine via product of Gaussians
            combined_mu, combined_logvar = product_of_gaussians(
                mus, logvars, precisions
            )

            # Step 5: EQ continuous reweighting
            if self.eq_reweighting is not None:
                eq_hidden = hidden_states[ModuleType.EQ]
                combined_mu, combined_logvar = self.eq_reweighting(
                    eq_hidden, combined_mu, combined_logvar
                )

            # Step 6: Sample or take mode for state update
            current_state = sample_from_gaussian(
                combined_mu, combined_logvar, temperature
            )
            all_states.append(current_state)

            # Step 7: Check termination criteria
            should_stop = False

            # Convergence check: L2 distance between successive states
            if prev_state is not None:
                dist = (current_state - prev_state).pow(2).mean().sqrt()
                if dist < self.config.convergence_threshold:
                    should_stop = True

            # Entropy check
            entropy = gaussian_entropy(combined_logvar).mean()
            if entropy < self.config.entropy_threshold:
                should_stop = True

            # Learned termination
            term_prob = self.termination(combined_mu, combined_logvar)
            if term_prob.mean() > 0.5:
                should_stop = True

            if should_stop and step > 0:
                break

            prev_state = current_state

            # Step 8: Condition modules on the updated shared state for
            # the next deliberation step. Project the latent state back
            # to hidden_size and add it as a residual to each module's
            # hidden states.
            state_as_hidden = self.latent_to_hidden(current_state)
            for mt in hidden_states:
                hidden_states[mt] = hidden_states[mt] + state_as_hidden

            # Re-run probabilistic heads with updated hidden states
            for mt_str, module in self.cognitive_modules.items():
                mt = ModuleType(mt_str)
                mu, logvar = module.prob_head(hidden_states[mt])
                module_outputs[mt]["mu"] = mu
                module_outputs[mt]["logvar"] = logvar
                module_outputs[mt]["hidden_states"] = hidden_states[mt]

        return {
            "final_mu": combined_mu,
            "final_logvar": combined_logvar,
            "final_state": current_state,
            "all_states": all_states,
            "steps": step + 1,
            "module_hidden_states": {
                mt: out["hidden_states"] for mt, out in module_outputs.items()
            },
        }
