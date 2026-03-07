"""Projection Interface (Decoder) for CLSA.

Section 4.4: The final shared latent state is decoded into natural language
by a specialized decoder. Following the PLaT paradigm, the decoder is
trained separately from the cognitive modules to prevent decoding from
interfering with latent reasoning representations.

The decoder receives the full trajectory of shared latent states as context,
not just the final state, enabling richer output that reflects the full
deliberation history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from clsa.config.transformer_config import TransformerConfig
from clsa.modules.transformer import Transformer


class LatentStateEncoder(nn.Module):
    """Encodes a trajectory of latent states into a conditioning sequence.

    The deliberation loop produces a list of latent states, one per step.
    This module converts them into a sequence of hidden-size vectors that
    the decoder transformer can attend to.
    """

    def __init__(self, latent_dim: int, hidden_size: int, max_steps: int = 16):
        super().__init__()
        self.proj = nn.Linear(latent_dim, hidden_size)
        # Learned step embeddings so the decoder can distinguish early
        # vs late deliberation states
        self.step_embeddings = nn.Embedding(max_steps, hidden_size)

    def forward(
        self, latent_states: list[torch.Tensor]
    ) -> torch.Tensor:
        """Encode a trajectory of latent states.

        Args:
            latent_states: list of (batch, seq_len, latent_dim) tensors,
                one per deliberation step.

        Returns:
            (batch, num_steps * seq_len, hidden_size) conditioning sequence.
        """
        encoded = []
        for i, state in enumerate(latent_states):
            projected = self.proj(state)
            step_idx = torch.full(
                (1,), i, dtype=torch.long, device=state.device
            )
            step_emb = self.step_embeddings(step_idx).unsqueeze(1)
            encoded.append(projected + step_emb)

        # Concatenate along the sequence dimension
        return torch.cat(encoded, dim=1)


class ProjectionDecoder(nn.Module):
    """Decodes latent deliberation states into token logits.

    Uses a transformer backbone (can be initialized from SmolLM2) with
    cross-attention to the encoded latent trajectory. The decoder is
    autoregressive: it generates tokens one at a time, attending to both
    previously generated tokens and the latent state context.
    """

    def __init__(self, config: TransformerConfig, latent_dim: int):
        super().__init__()
        self.config = config

        # Latent trajectory encoder
        self.latent_encoder = LatentStateEncoder(
            latent_dim, config.hidden_size
        )

        # Decoder backbone (autoregressive transformer)
        self.backbone = Transformer(config)

        # Cross-attention from decoder to latent context
        self.latent_cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
            bias=False,
        )
        self.cross_attn_norm = nn.LayerNorm(config.hidden_size)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        latent_states: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode latent states into token logits.

        Args:
            input_ids: (batch, seq_len) decoder input token indices.
            latent_states: list of latent state tensors from deliberation.
            attention_mask: optional padding mask for input_ids.
            labels: optional target tokens for loss computation.

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        # Encode the latent trajectory into a conditioning sequence
        latent_context = self.latent_encoder(latent_states)

        # Run decoder backbone on input tokens
        hidden_states = self.backbone(input_ids, attention_mask)

        # Cross-attend to latent context
        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states, _ = self.latent_cross_attn(
            query=hidden_states,
            key=latent_context,
            value=latent_context,
        )
        hidden_states = residual + hidden_states

        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result
