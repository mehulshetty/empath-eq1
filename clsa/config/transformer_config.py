"""Configuration for the base transformer architecture.

These defaults match SmolLM2-135M so we can load its pretrained weights.
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # Vocabulary and embedding
    vocab_size: int = 49152
    hidden_size: int = 576
    max_position_embeddings: int = 8192

    # Transformer blocks
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3  # Grouped query attention (3:1 ratio)
    intermediate_size: int = 1536

    # Normalization and activation
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"

    # Rotary position embeddings
    rope_theta: float = 100000.0

    # Weight tying
    tie_word_embeddings: bool = True

    # Regularization
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # Initialization
    initializer_range: float = 0.0417

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per key-value head."""
        return self.num_attention_heads // self.num_key_value_heads
