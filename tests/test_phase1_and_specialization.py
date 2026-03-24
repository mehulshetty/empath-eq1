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
    _build_persuasion_supervised_examples,
    _collate_fn,
    _pack_supervised_token_ids,
    _supervise_mathdial_example,
    build_phase2_dataloader,
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


def test_supervise_mathdial_example_emits_teacher_turns():
    examples = _supervise_mathdial_example(
        {
            "question": "What is 2 + 2?",
            "student_incorrect_solution": "I think it is 5.",
            "student_profile": "Needs help checking arithmetic carefully.",
            "conversation": (
                "Teacher: (generic)Let's work through it together.|EOM|"
                "Student: I think it is 5.|EOM|"
                "Teacher: (focus)What happens if you count two more from 2?"
            ),
        }
    )

    assert len(examples) == 2
    assert examples[0].target == "Let's work through it together."
    assert "Math problem: What is 2 + 2?" in examples[1].prompt
    assert "Student: I think it is 5." in examples[1].prompt
    assert examples[1].target == "What happens if you count two more from 2?"


def test_build_persuasion_supervised_examples_uses_only_persuader_turns():
    dataset = _FakeDataset(
        [
            {"B2": "d1", "Turn": 0, "B4": 0, "Unit": "Hi, would you consider donating today?"},
            {"B2": "d1", "Turn": 1, "B4": 1, "Unit": "Maybe, but I am not sure."},
            {"B2": "d1", "Turn": 2, "B4": 0, "Unit": "Even a small amount would help children in need."},
        ]
    )

    examples = _build_persuasion_supervised_examples(dataset)

    assert len(examples) == 2
    assert examples[0].target == "Hi, would you consider donating today?"
    assert "Conversation so far:" in examples[1].prompt
    assert "Persuadee: Maybe, but I am not sure." in examples[1].prompt
    assert examples[1].target == "Even a small amount would help children in need."


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


def test_build_combined_dataloader_uses_supervised_phase1_and_phase2(monkeypatch):
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
            {"path": "phase2", "name": None, "split": "train", "formatter": "_supervise_counseling_phase2_example"},
            {"path": "persuasion", "name": None, "split": "train", "builder": "_build_persuasion_supervised_examples"},
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
    # All sources now contribute prompt-masked labels.
    assert (batch["labels"] == -100).any()
    # The batch should still contain trainable target tokens.
    assert (batch["labels"] != -100).any()


def test_build_phase2_dataloader_uses_supervised_phase2_examples(monkeypatch):
    fake_datasets = {
        "phase2_counseling": _FakeDataset(
            [{"Context": "I cannot focus lately.", "Response": "Let's slow down and look at what has changed recently."}]
        ),
        "phase2_math": _FakeDataset(
            [
                {
                    "question": "What is 3 + 4?",
                    "student_incorrect_solution": "I said 10.",
                    "student_profile": "Needs help checking work.",
                    "conversation": (
                        "Teacher: (generic)Let's check it together.|EOM|"
                        "Student: I said 10.|EOM|"
                        "Teacher: (focus)What do you get if you count four more from 3?"
                    ),
                }
            ]
        ),
    }

    def fake_load_dataset(path, name=None, split=None):
        return fake_datasets[path]

    monkeypatch.setattr(
        data_module,
        "PHASE2_DATASETS",
        [
            {
                "path": "phase2_counseling",
                "name": None,
                "split": "train",
                "formatter": "_supervise_counseling_phase2_example",
                "sampling_weight": 1.0,
            },
            {
                "path": "phase2_math",
                "name": None,
                "split": "train",
                "formatter": "_supervise_mathdial_example",
                "sampling_weight": 1.0,
            },
        ],
    )
    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    dataloader = build_phase2_dataloader(
        tokenizer=_FakeTokenizer(),
        batch_size=8,
        max_length=64,
        num_workers=0,
        max_samples=4,
    )
    batch = next(iter(dataloader))

    assert len(dataloader.dataset) == 3
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()
