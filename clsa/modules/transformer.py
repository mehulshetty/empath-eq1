"""Custom transformer implementation compatible with SmolLM2-135M weights.

Architecture: LLaMA-family with RMSNorm, SiLU, Rotary Position Embeddings,
and Grouped Query Attention. Written from scratch for full control over
internals, which CLSA needs for injecting cross-attention and probabilistic
output heads.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from clsa.config.transformer_config import TransformerConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not re-center activations (no mean
    subtraction), only re-scales. This is cheaper and works just as well
    in practice.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 100000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for rotary position embeddings.

    Returns (cos, sin) each of shape (seq_len, head_dim).
    """
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    # Frequency for each dimension pair: theta^(-2i/d) for i in [0, d/2)
    dim_pairs = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    freqs = 1.0 / (theta ** (dim_pairs / head_dim))
    # Outer product: (seq_len,) x (head_dim/2,) -> (seq_len, head_dim/2)
    angles = torch.outer(pos, freqs)
    # Duplicate each frequency for the paired dimensions
    angles = angles.repeat(1, 2)
    return angles.cos(), angles.sin()


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to query or key tensor.

    Args:
        x: (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Rotate pairs: for each pair (x0, x1), compute
    # (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
    x_paired = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1)
    x_rotated = x_paired.reshape(x.shape)
    return x * cos + x_rotated * sin


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with grouped key-value heads (GQA).

    Multiple query heads share the same key-value head, reducing memory
    and compute for KV projections while keeping full query expressivity.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to queries and keys
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Expand KV heads to match query heads for grouped attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )

        # Merge heads and project
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)


class FeedForward(nn.Module):
    """SiLU-gated feed-forward network (SwiGLU variant).

    Computes: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    The gate and up projections go from hidden_size to intermediate_size,
    then down projects back.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer layer: attention + feed-forward with pre-norm."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-norm feed-forward with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Transformer(nn.Module):
    """Full transformer model compatible with SmolLM2-135M weights.

    This is the base building block used by CLSA cognitive modules.
    It produces hidden states (no language model head), since CLSA
    modules output probabilistic distributions, not token logits.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Precompute rotary embedding tables (not a parameter, just a buffer)
        cos, sin = build_rope_cache(
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the transformer and return final hidden states.

        Args:
            input_ids: (batch, seq_len) token indices.
            attention_mask: optional mask for padding. If None, causal
                masking is applied automatically by SDPA.

        Returns:
            Hidden states of shape (batch, seq_len, hidden_size).
        """
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, self.rope_cos, self.rope_sin, attention_mask
            )

        return self.norm(hidden_states)


class TransformerForCausalLM(nn.Module):
    """Transformer with a language model head for causal (autoregressive) generation.

    Used for Phase 1 pre-training and standalone evaluation. The CLSA
    deliberation system uses the base Transformer (without this head) as
    the backbone for cognitive modules.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.model = Transformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: (batch, seq_len) token indices.
            attention_mask: optional padding mask.
            labels: (batch, seq_len) target token indices for loss.
                Positions with value -100 are ignored.

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        result = {"logits": logits, "hidden_states": hidden_states}

        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result
