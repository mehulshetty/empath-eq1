from functools import partial

import torch
from torch.utils.data import DataLoader

import clsa.training.data as data_module
from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.config.transformer_config import TransformerConfig
from clsa.model import CLSA
from clsa.modules.transformer import TransformerForCausalLM
from clsa.training.data import (
    TokenizedDataset,
    _collate_fn,
    _pack_supervised_token_ids,
    build_combined_dataloader,
)
from clsa.training.trainer import (
    ModuleSpecializationProbe,
    Phase3SpecializationRetainer,
)


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


def test_pack_supervised_token_ids_masks_prompt_and_preserves_target():
    packed = _pack_supervised_token_ids(
        prompt_ids=[10, 11, 12, 13],
        target_ids=[20, 21],
        max_length=5,
        eos_token_id=2,
    )
    assert packed is not None
    input_ids, labels = packed

    assert input_ids.tolist() == [12, 13, 20, 21, 2]
    assert labels.tolist() == [-100, -100, 20, 21, 2]


def test_pack_supervised_token_ids_truncates_prompt_before_target():
    packed = _pack_supervised_token_ids(
        prompt_ids=[1, 2, 3, 4],
        target_ids=[5, 6, 7, 8],
        max_length=4,
        eos_token_id=9,
    )
    assert packed is not None
    input_ids, labels = packed

    # The target is preserved preferentially and the prompt is dropped.
    assert input_ids.tolist() == [5, 6, 7, 9]
    assert labels.tolist() == [5, 6, 7, 9]


def test_phase3_specialization_retainer_returns_probe_loss():
    clsa_model = CLSA(TINY_CLSA_CONFIG)
    phase1_lm = TransformerForCausalLM(TINY_CONFIG)

    dataset = TokenizedDataset(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "labels": torch.tensor([-100, -100, 3, 4], dtype=torch.long),
            }
        ]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(_collate_fn, pad_token_id=0),
    )

    retainer = Phase3SpecializationRetainer(
        probes=[
            ModuleSpecializationProbe(
                module_type=ModuleType.LOGIC,
                lm_head_weight=phase1_lm.lm_head.weight.detach().clone(),
                dataloader=dataloader,
                name="logic",
            ),
            ModuleSpecializationProbe(
                module_type=ModuleType.EQ,
                lm_head_weight=phase1_lm.lm_head.weight.detach().clone(),
                dataloader=dataloader,
                name="eq",
            ),
        ],
        interval=1,
        device="cpu",
    )

    probe_loss = retainer.maybe_compute(clsa_model, step=1)
    assert probe_loss.ndim == 0
    assert torch.isfinite(probe_loss)
    assert probe_loss.item() >= 0.0


def test_phase3_specialization_retainer_respects_interval():
    clsa_model = CLSA(TINY_CLSA_CONFIG)
    phase1_lm = TransformerForCausalLM(TINY_CONFIG)
    dataset = TokenizedDataset(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "labels": torch.tensor([-100, -100, 3, 4], dtype=torch.long),
            }
        ]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(_collate_fn, pad_token_id=0),
    )

    retainer = Phase3SpecializationRetainer(
        probes=[
            ModuleSpecializationProbe(
                module_type=ModuleType.LOGIC,
                lm_head_weight=phase1_lm.lm_head.weight.detach().clone(),
                dataloader=dataloader,
                name="logic",
            )
        ],
        interval=3,
        device="cpu",
    )

    skipped = retainer.maybe_compute(clsa_model, step=1)
    assert skipped.item() == 0.0


class _FakeDataset(list):
    def select(self, indices):
        return _FakeDataset([self[idx] for idx in indices])


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=False,
        return_attention_mask=False,
    ):
        if isinstance(texts, str):
            texts = [texts]

        encoded = []
        for text in texts:
            token_ids = [3 + (ord(ch) % 40) for ch in text]
            if add_special_tokens:
                token_ids = [1] + token_ids
            if truncation and max_length is not None:
                token_ids = token_ids[:max_length]
            encoded.append(token_ids)
        return {"input_ids": encoded}


def test_build_combined_dataloader_uses_supervised_phase1_and_text_phase2(monkeypatch):
    fake_datasets = {
        "logic": _FakeDataset(
            [{"question": "2+2?", "choices": {"label": ["A", "B"], "text": ["3", "4"]}, "answerKey": "B"}]
        ),
        "eq": _FakeDataset(
            [{"Context": "I feel overwhelmed.", "Response": "That sounds really hard. Let's slow it down together."}]
        ),
        "phase2": _FakeDataset(
            [{"Context": "Need advice", "Response": "Take it one step at a time."}]
        ),
        "persuasion": _FakeDataset(
            [
                {"B2": "d1", "Turn": 0, "B4": 0, "Unit": "Would you donate?"},
                {"B2": "d1", "Turn": 1, "B4": 1, "Unit": "Maybe."},
            ]
        ),
    }

    def fake_load_dataset(path, name=None, split=None):
        return fake_datasets[path]

    monkeypatch.setattr(
        data_module,
        "PHASE1_LOGIC_SUPERVISED_DATASETS",
        [{"path": "logic", "name": None, "split": "train", "formatter": "_supervise_arc_example"}],
    )
    monkeypatch.setattr(
        data_module,
        "PHASE1_EQ_SUPERVISED_DATASETS",
        [{"path": "eq", "name": None, "split": "train", "formatter": "_supervise_counseling_example"}],
    )
    monkeypatch.setattr(
        data_module,
        "PHASE2_DATASETS",
        [
            {"path": "phase2", "name": None, "split": "train", "formatter": "_format_counseling_example"},
            {"path": "persuasion", "name": None, "split": "train", "formatter": "_format_persuasion_example"},
        ],
    )
    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    dataloader = build_combined_dataloader(
        tokenizer=_FakeTokenizer(),
        batch_size=8,
        max_length=64,
        num_workers=0,
    )
    batch = next(iter(dataloader))

    assert len(dataloader.dataset) == 4
    # Supervised Phase 1 examples contribute prompt-masked labels.
    assert (batch["labels"] == -100).any()
    # The batch should still contain trainable target tokens.
    assert (batch["labels"] != -100).any()
