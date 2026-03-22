import torch

from clsa.config.transformer_config import TransformerConfig
from clsa.modules.transformer import TransformerForCausalLM
from clsa.training.checkpointing import (
    extract_phase1_backbone_state,
    extract_phase1_lm_head_weight,
    save_phase1_backbone,
)


def _tiny_causal_lm() -> TransformerForCausalLM:
    config = TransformerConfig(
        vocab_size=256,
        hidden_size=64,
        max_position_embeddings=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    )
    return TransformerForCausalLM(config)


def test_extract_phase1_backbone_state_from_stripped_checkpoint(tmp_path):
    model = _tiny_causal_lm()
    checkpoint_path = tmp_path / "phase1_logic.pt"
    save_phase1_backbone(model, checkpoint_path)

    backbone_state = extract_phase1_backbone_state(checkpoint_path)

    assert backbone_state
    assert all(not key.startswith("model.") for key in backbone_state)
    assert all(key.startswith(("embed_tokens.", "layers.", "norm.")) for key in backbone_state)


def test_extract_phase1_backbone_state_from_full_training_checkpoint(tmp_path):
    model = _tiny_causal_lm()
    checkpoint_path = tmp_path / "phase1_logic_epoch0.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    backbone_state = extract_phase1_backbone_state(checkpoint_path)

    expected = model.model.state_dict()
    assert backbone_state.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(backbone_state[key], value)


def test_extract_phase1_lm_head_weight_from_full_checkpoint(tmp_path):
    model = _tiny_causal_lm()
    checkpoint_path = tmp_path / "phase1_logic_lm.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    lm_head_weight = extract_phase1_lm_head_weight(checkpoint_path)

    assert torch.equal(lm_head_weight, model.lm_head.weight)


def test_extract_phase1_backbone_state_rejects_unknown_format(tmp_path):
    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {"lm_head.weight": torch.randn(4, 4)}}, checkpoint_path)

    try:
        extract_phase1_backbone_state(checkpoint_path)
    except ValueError as exc:
        assert "not a recognized Phase 1 checkpoint format" in str(exc)
    else:
        raise AssertionError("Expected extract_phase1_backbone_state to reject invalid checkpoint")


def test_extract_phase1_lm_head_weight_rejects_backbone_only_checkpoint(tmp_path):
    model = _tiny_causal_lm()
    checkpoint_path = tmp_path / "phase1_logic.pt"
    save_phase1_backbone(model, checkpoint_path)

    try:
        extract_phase1_lm_head_weight(checkpoint_path)
    except ValueError as exc:
        assert "does not contain a usable Phase 1 LM head" in str(exc)
    else:
        raise AssertionError("Expected extract_phase1_lm_head_weight to reject backbone-only checkpoint")
