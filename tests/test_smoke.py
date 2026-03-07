"""Smoke tests for CLSA components.

Verifies that all components instantiate correctly and can run a
forward pass with small random inputs. Does not test correctness
of learned behavior, just structural integrity.
"""

import torch

from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.config.transformer_config import TransformerConfig
from clsa.model import CLSA
from clsa.modules.transformer import Transformer, TransformerForCausalLM
from clsa.deliberation.superposition import product_of_gaussians, sample_from_gaussian


# Use a tiny config for fast tests
TINY_CONFIG = TransformerConfig(
    vocab_size=256,
    hidden_size=64,
    max_position_embeddings=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
)

TINY_CLSA_CONFIG = CLSAConfig(
    transformer=TINY_CONFIG,
    latent_dim=64,
    active_modules=[ModuleType.LOGIC, ModuleType.EQ],
    max_deliberation_steps=3,
    cross_attention_heads=4,
)

BATCH = 2
SEQ_LEN = 16


def test_transformer_forward():
    model = Transformer(TINY_CONFIG)
    ids = torch.randint(0, TINY_CONFIG.vocab_size, (BATCH, SEQ_LEN))
    out = model(ids)
    assert out.shape == (BATCH, SEQ_LEN, TINY_CONFIG.hidden_size)


def test_causal_lm_forward():
    model = TransformerForCausalLM(TINY_CONFIG)
    ids = torch.randint(0, TINY_CONFIG.vocab_size, (BATCH, SEQ_LEN))
    out = model(ids, labels=ids)
    assert out["logits"].shape == (BATCH, SEQ_LEN, TINY_CONFIG.vocab_size)
    assert "loss" in out


def test_product_of_gaussians():
    mu1 = torch.randn(BATCH, SEQ_LEN, 64)
    mu2 = torch.randn(BATCH, SEQ_LEN, 64)
    logvar1 = torch.zeros(BATCH, SEQ_LEN, 64)
    logvar2 = torch.zeros(BATCH, SEQ_LEN, 64)
    pi1 = torch.tensor(1.0)
    pi2 = torch.tensor(2.0)

    combined_mu, combined_logvar = product_of_gaussians(
        [mu1, mu2], [logvar1, logvar2], [pi1, pi2]
    )
    assert combined_mu.shape == mu1.shape
    assert combined_logvar.shape == logvar1.shape

    # With equal variance, combined mean should be weighted toward
    # the higher-precision module (mu2 with pi=2)
    weight_2 = 2.0 / 3.0
    weight_1 = 1.0 / 3.0
    expected_mu = weight_1 * mu1 + weight_2 * mu2
    assert torch.allclose(combined_mu, expected_mu, atol=1e-5)


def test_clsa_forward():
    model = CLSA(TINY_CLSA_CONFIG)
    ids = torch.randint(0, TINY_CONFIG.vocab_size, (BATCH, SEQ_LEN))

    # Forward with labels (training mode)
    out = model(ids, labels=ids)
    assert out["logits"].shape == (BATCH, SEQ_LEN, TINY_CONFIG.vocab_size)
    assert "total" in out["loss"]
    assert out["deliberation"]["steps"] >= 1


def test_precision_overrides():
    model = CLSA(TINY_CLSA_CONFIG)
    ids = torch.randint(0, TINY_CONFIG.vocab_size, (BATCH, SEQ_LEN))

    # Forward with precision overrides
    out = model(
        ids,
        labels=ids,
        precision_overrides={ModuleType.EQ: 5.0, ModuleType.LOGIC: 1.0},
    )
    assert out["logits"].shape == (BATCH, SEQ_LEN, TINY_CONFIG.vocab_size)


def test_set_precision():
    model = CLSA(TINY_CLSA_CONFIG)
    model.set_precision(ModuleType.EQ, 3.0)
    precisions = model.get_precisions()
    assert abs(precisions["eq"] - 3.0) < 0.01
