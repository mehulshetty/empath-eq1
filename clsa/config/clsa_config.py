"""Configuration for the CLSA architecture.

Defines module types, deliberation parameters, and precision weights.
"""

from dataclasses import dataclass, field
from enum import Enum

from clsa.config.transformer_config import TransformerConfig


class ModuleType(str, Enum):
    """Cognitive module types supported by CLSA.

    The MVP uses LOGIC and EQ. CREATIVITY and RETRIEVAL are defined
    for forward compatibility.
    """

    LOGIC = "logic"
    EQ = "eq"
    CREATIVITY = "creativity"
    RETRIEVAL = "retrieval"


@dataclass
class CLSAConfig:
    """Top-level configuration for a CLSA system."""

    # Base transformer config shared by all modules
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    # Which modules are active in this CLSA instance
    active_modules: list[ModuleType] = field(
        default_factory=lambda: [ModuleType.LOGIC, ModuleType.EQ]
    )

    # Latent space dimensionality for probabilistic outputs.
    # Defaults to the transformer hidden size so module outputs
    # live in the same space as internal representations.
    latent_dim: int = 576

    # Deliberation loop settings
    max_deliberation_steps: int = 8
    convergence_threshold: float = 1e-3  # L2 distance for early stopping
    entropy_threshold: float = 0.1  # combined posterior entropy threshold

    # Precision weights (the "mixing board" dials).
    # Higher precision = sharper distribution = more influence.
    # Keys are ModuleType values, defaults to 1.0 for all modules.
    default_precision: float = 1.0

    # Soft orthogonality regularization coefficient (alpha in the paper)
    orthogonality_alpha: float = 0.1

    # Specialization loss coefficient (beta)
    specialization_beta: float = 0.05

    # Diversity loss coefficient (gamma)
    diversity_gamma: float = 0.01

    # Cross-attention settings for inter-module communication
    cross_attention_heads: int = 4
    cross_attention_dropout: float = 0.0

    # EQ module gets architectural privilege: it cross-attends to all
    # other modules at every deliberation step (see Section 3.2)
    eq_privileged: bool = True
