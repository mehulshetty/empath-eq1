"""Data loading utilities for CLSA training.

Provides dataset builders for each training phase:
  Phase 1: Domain-specific data per module (logic corpora, empathy dialogues)
  Phase 2/3: Multi-faculty data requiring both logic and EQ simultaneously

All builders return standard PyTorch DataLoaders yielding dicts with
"input_ids" and "labels" keys, matching what the trainer classes expect.
"""

import json
import logging
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from clsa.config.clsa_config import ModuleType

logger = logging.getLogger(__name__)

# Logic module datasets. Combined to produce a larger, more diverse
# reasoning corpus (~70K examples total).
LOGIC_DATASETS = [
    {
        "path": "allenai/ai2_arc",
        "name": "ARC-Easy",
        "split": "train",
        "formatter": "_format_arc_example",
    },
    {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "train",
        "formatter": "_format_arc_example",
    },
    {
        "path": "allenai/sciq",
        "name": None,
        "split": "train",
        "formatter": "_format_sciq_example",
    },
    {
        "path": "allenai/openbookqa",
        "name": "main",
        "split": "train",
        "formatter": "_format_openbookqa_example",
    },
    {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "train",
        "formatter": "_format_hellaswag_example",
    },
    {
        "path": "allenai/winogrande",
        "name": "winogrande_debiased",
        "split": "train",
        "formatter": "_format_winogrande_example",
    },
]

# EQ module datasets. Combined to produce a diverse emotional
# intelligence corpus (~180K examples total).
EQ_DATASETS = [
    {
        "path": "allenai/prosocial-dialog",
        "name": "default",
        "split": "train",
        "formatter": "_format_prosocial_example",
    },
    {
        "path": "Anthropic/hh-rlhf",
        "name": None,
        "split": "train",
        "formatter": "_format_hh_rlhf_example",
    },
    {
        "path": "google-research-datasets/go_emotions",
        "name": "simplified",
        "split": "train",
        "formatter": "_format_goemotions_example",
    },
    {
        "path": "dair-ai/emotion",
        "name": "split",
        "split": "train",
        "formatter": "_format_emotion_example",
    },
]

# Phase 2 datasets. Each requires both logical reasoning and emotional
# intelligence -- no single module could produce a quality response alone.
# Combined ~30K examples across 8 diverse domains to teach general-purpose
# inter-module communication (not just counseling).
PHASE2_DATASETS = [
    # Counseling: problem diagnosis + empathetic response (~3.5K)
    {
        "path": "Amod/mental_health_counseling_conversations",
        "name": None,
        "split": "train",
        "formatter": "_format_counseling_example",
    },
    # Patient-doctor dialogue: accurate medical reasoning + sensitivity (~112K,
    # sampled down). Patients describe symptoms, doctors must reason about
    # diagnosis while being reassuring and accessible.
    {
        "path": "lavita/medical-qa-datasets",
        "name": "chatdoctor_healthcaremagic",
        "split": "train",
        "formatter": "_format_medical_dialogue_example",
    },
    # Medical Q&A: factual health info framed for patients (~16K, sampled down).
    # Requires translating clinical knowledge into patient-friendly language.
    {
        "path": "keivalya/MedQuad-MedicalQnADataset",
        "name": None,
        "split": "train",
        "formatter": "_format_medquad_example",
    },
    # Patient health info: accessible explanations of conditions (~5.9K,
    # sampled down). Bridges medical accuracy with plain-language framing.
    {
        "path": "lavita/medical-qa-datasets",
        "name": "medical_meadow_wikidoc_patient_information",
        "split": "train",
        "formatter": "_format_medical_dialogue_example",
    },
    # Emotional support conversations: structured strategies for helping
    # people through crises (~0.9K). Annotated with support strategies
    # like reflection, self-disclosure, providing suggestions.
    {
        "path": "thu-coai/esconv",
        "name": None,
        "split": "train",
        "formatter": "_format_esconv_example",
    },
    # Math tutoring: pedagogical logic + patience with struggling students
    # (~2.2K). Teachers must reason about math while adapting to the
    # student's confusion and emotional state.
    {
        "path": "eth-nlped/mathdial",
        "name": None,
        "split": "train",
        "formatter": "_format_mathdial_example",
    },
    # Negotiation: strategic reasoning + social dynamics and empathy (~1K).
    # Campers negotiate resource allocation, requiring logical trade-off
    # analysis combined with persuasion and rapport.
    {
        "path": "kchawla123/casino",
        "name": None,
        "split": "train",
        "formatter": "_format_casino_example",
    },
    # Persuasion for good: logical argumentation + empathetic appeal (~20.9K
    # utterances, grouped into ~1K dialogues). Persuading someone to donate
    # requires understanding their perspective and building a logical case.
    {
        "path": "spawn99/PersuasionForGood",
        "name": None,
        "split": "FullDialog",
        "formatter": "_format_persuasion_example",
    },
]


class TokenizedDataset(Dataset):
    """Wraps a list of tokenized examples for use with DataLoader."""

    def __init__(self, examples: list[dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def _collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences to the same length.

    Pads input_ids with pad_token_id and labels with -100 (ignored by
    cross-entropy loss). Returns an attention_mask so padding tokens
    are excluded from attention and compute.
    """
    max_len = max(ex["input_ids"].size(0) for ex in batch)

    padded_ids = []
    padded_labels = []
    masks = []

    for ex in batch:
        seq_len = ex["input_ids"].size(0)
        pad_len = max_len - seq_len

        ids = torch.cat([
            ex["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
        ])
        labels = torch.cat([
            ex["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),
        ])
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])

        padded_ids.append(ids)
        padded_labels.append(labels)
        masks.append(mask)

    return {
        "input_ids": torch.stack(padded_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(masks),
    }


def _format_arc_example(example: dict) -> str:
    """Format an ARC (AI2 Reasoning Challenge) example as a text prompt.

    Turns a multiple-choice science question into a completion task:
    "Question: ... Choices: A) ... B) ... Answer: X"
    """
    question = example["question"]
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = example["answerKey"]

    choices_str = " ".join(
        f"{label}) {text}" for label, text in zip(labels, texts)
    )
    return f"Question: {question} Choices: {choices_str} Answer: {answer_key}"


def _format_sciq_example(example: dict) -> str:
    """Format a SciQ example as a text prompt.

    Each example has a question, correct answer, three distractors,
    and a support paragraph explaining the reasoning. Including the
    support teaches the logic module *why* the answer is correct.
    """
    question = example["question"]
    correct = example["correct_answer"]
    support = example.get("support", "")

    distractors = [
        example.get("distractor1", ""),
        example.get("distractor2", ""),
        example.get("distractor3", ""),
    ]
    choices = [correct] + [d for d in distractors if d]

    choices_str = " ".join(
        f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)
    )
    parts = []
    if support:
        parts.append(f"Context: {support}")
    parts.append(f"Question: {question} Choices: {choices_str} Answer: A) {correct}")
    return "\n".join(parts)


def _format_openbookqa_example(example: dict) -> str:
    """Format an OpenBookQA example as a text prompt.

    Science QA with an open-book fact. Each example has a question,
    multiple choices, and a supporting science fact that explains
    the correct answer.
    """
    question = example["question_stem"]
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = example["answerKey"]

    choices_str = " ".join(
        f"{label}) {text}" for label, text in zip(labels, texts)
    )
    fact = example.get("fact1", "")
    parts = []
    if fact:
        parts.append(f"Fact: {fact}")
    parts.append(f"Question: {question} Choices: {choices_str} Answer: {answer_key}")
    return "\n".join(parts)


def _format_hellaswag_example(example: dict) -> str:
    """Format a HellaSwag example.

    Tests commonsense reasoning via sentence completion: given a
    context and activity label, pick the correct continuation.
    """
    ctx = example["ctx"]
    endings = example["endings"]
    label = int(example["label"])

    choices_str = " ".join(
        f"{chr(65 + i)}) {e}" for i, e in enumerate(endings)
    )
    answer = endings[label]
    return f"Context: {ctx}\nChoices: {choices_str}\nAnswer: {answer}"


def _format_winogrande_example(example: dict) -> str:
    """Format a WinoGrande example.

    Tests commonsense coreference resolution: fill in the blank
    with one of two options.
    """
    sentence = example["sentence"]
    option1 = example["option1"]
    option2 = example["option2"]
    answer = example["answer"]  # "1" or "2"

    correct = option1 if answer == "1" else option2
    filled = sentence.replace("_", correct)
    return f"Sentence: {sentence}\nA) {option1} B) {option2}\nAnswer: {filled}"


def _format_hh_rlhf_example(example: dict) -> str:
    """Format an Anthropic HH-RLHF example as a text prompt.

    Each example has a chosen (preferred) and rejected response.
    We train on the chosen response, which demonstrates helpful,
    harmless, and honest behavior -- core EQ competencies.
    """
    return example["chosen"]


def _format_goemotions_example(example: dict) -> str:
    """Format a GoEmotions example as a text prompt.

    Reddit comments labeled with fine-grained emotions (27 categories).
    Teaches the EQ module to recognize nuanced emotional expressions.
    """
    emotion_names = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise",
        "neutral",
    ]
    text = example["text"]
    label_ids = example["labels"]

    labels = [emotion_names[i] for i in label_ids if i < len(emotion_names)]
    labels_str = ", ".join(labels) if labels else "neutral"

    return f"Text: {text}\nEmotions: {labels_str}"


def _format_emotion_example(example: dict) -> str:
    """Format an Emotion dataset example as a text prompt.

    Simple text-to-emotion classification with 6 basic emotions.
    Reinforces core emotion recognition ability.
    """
    emotion_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    text = example["text"]
    label = example["label"]

    emotion = emotion_names[label] if label < len(emotion_names) else "unknown"
    return f"Text: {text}\nEmotion: {emotion}"


def _format_prosocial_example(example: dict) -> str:
    """Format a ProsocialDialog example as a text prompt.

    Each example has a potentially unsafe utterance (context), a
    prosocial response, and rules-of-thumb explaining the social
    reasoning. Including the RoTs teaches the EQ module not just
    what to say but why it is appropriate.

    Format:
        Situation: <context>
        Social rule: <rot>
        Response: <response>
    """
    context = example.get("context", "")
    response = example.get("response", "")
    rots = example.get("rots", [])

    parts = [f"Situation: {context}"]
    if rots:
        # Join multiple rules of thumb into a single line
        rot_text = " ".join(r for r in rots if r)
        if rot_text:
            parts.append(f"Social rule: {rot_text}")
    parts.append(f"Response: {response}")

    return "\n".join(parts)


def _format_counseling_example(example: dict) -> str:
    """Format a mental health counseling conversation.

    These conversations naturally require both logical structure
    (identifying the problem, reasoning about solutions) and
    emotional intelligence (empathy, validation, sensitivity).
    """
    context = example.get("Context", "")
    response = example.get("Response", "")
    return f"Patient: {context}\nCounselor: {response}"


def _format_medical_dialogue_example(example: dict) -> str:
    """Format a lavita medical-qa-datasets example.

    Used for both chatdoctor_healthcaremagic (patient-doctor dialogue)
    and wikidoc_patient_information (patient-facing health info).
    Both require accurate medical reasoning delivered with sensitivity.
    """
    instruction = example.get("instruction", "")
    patient_input = example.get("input", "")
    doctor_output = example.get("output", "")

    parts = []
    if instruction:
        parts.append(f"Task: {instruction}")
    if patient_input:
        parts.append(f"Patient: {patient_input}")
    if doctor_output:
        parts.append(f"Doctor: {doctor_output}")
    return "\n".join(parts)


def _format_medquad_example(example: dict) -> str:
    """Format a MedQuad medical question-answer pair.

    Patient-facing medical Q&A spanning symptoms, treatment, causes,
    prevention, etc. Requires factual medical accuracy communicated
    in accessible, non-alarming language.
    """
    qtype = example.get("qtype", "")
    question = example.get("Question", "")
    answer = example.get("Answer", "")

    parts = []
    if qtype:
        parts.append(f"Topic: {qtype}")
    parts.append(f"Patient question: {question}")
    parts.append(f"Answer: {answer}")
    return "\n".join(parts)


def _format_esconv_example(example: dict) -> str:
    """Format an ESConv emotional support conversation.

    Each dialogue covers a crisis situation (job loss, breakup,
    depression, etc.) with annotated support strategies. The supporter
    must reason about the situation while providing emotional comfort.
    """
    data = json.loads(example["text"])

    situation = data.get("situation", "")
    emotion = data.get("emotion_type", "")
    problem = data.get("problem_type", "")
    dialog = data.get("dialog", [])

    parts = []
    if emotion or problem:
        parts.append(f"Context: {emotion}, {problem}")
    if situation:
        parts.append(f"Situation: {situation}")

    for turn in dialog:
        speaker = "Seeker" if turn.get("speaker") == "usr" else "Supporter"
        text = turn.get("text", "")
        strategy = turn.get("strategy", "")
        if strategy:
            parts.append(f"{speaker} [{strategy}]: {text}")
        else:
            parts.append(f"{speaker}: {text}")

    return "\n".join(parts)


def _format_mathdial_example(example: dict) -> str:
    """Format a MathDial tutoring dialogue.

    A teacher helps a student who got a math problem wrong.
    Requires mathematical reasoning (logic) combined with
    pedagogical patience and encouragement (EQ).
    """
    question = example.get("question", "")
    incorrect = example.get("student_incorrect_solution", "")
    profile = example.get("student_profile", "")
    conversation = example.get("conversation", "")

    parts = [f"Math problem: {question}"]
    if incorrect:
        parts.append(f"Student's incorrect answer: {incorrect}")
    if profile:
        parts.append(f"Student profile: {profile}")

    # Conversation uses |EOM| as turn separator
    if conversation:
        turns = conversation.split("|EOM|")
        for turn in turns:
            turn = turn.strip()
            if turn:
                parts.append(turn)

    return "\n".join(parts)


def _format_casino_example(example: dict) -> str:
    """Format a CaSiNo negotiation dialogue.

    Two people negotiate over camping supplies (food, water, firewood).
    Requires logical trade-off analysis (what to prioritize) combined
    with persuasion, rapport-building, and reading social cues.
    """
    chat_logs = example.get("chat_logs", [])
    if not chat_logs:
        return ""

    parts = ["Negotiation:"]
    for entry in chat_logs:
        speaker = entry.get("id", "unknown")
        text = entry.get("text", "")
        if text:
            parts.append(f"{speaker}: {text}")

    return "\n".join(parts)


def _format_persuasion_example(example: dict) -> str | None:
    """Format a PersuasionForGood utterance.

    Returns None for non-first turns -- the build pipeline groups
    these into full dialogues via _build_persuasion_dialogues().
    Individual utterances are not useful on their own.
    """
    # This formatter is not called directly by _load_and_format_datasets.
    # Instead, _build_persuasion_dialogues handles the grouping.
    # This exists only for the _FORMATTERS registry consistency.
    return None


def _build_persuasion_dialogues(dataset) -> list[str]:
    """Group PersuasionForGood utterances into complete dialogues.

    The raw dataset has one row per utterance. We group by dialogue ID
    (B2 column) and reconstruct full conversations where a persuader
    builds a logical + empathetic case for charitable donation.
    """
    dialogues: dict[str, list[tuple[int, int, str]]] = {}
    for ex in dataset:
        dialog_id = ex.get("B2", "")
        turn = ex.get("Turn", 0)
        role = ex.get("B4", 0)  # 0 = persuader, 1 = persuadee
        text = ex.get("Unit", "")
        if dialog_id and text:
            if dialog_id not in dialogues:
                dialogues[dialog_id] = []
            dialogues[dialog_id].append((turn, role, text))

    texts = []
    for dialog_id, turns in dialogues.items():
        turns.sort(key=lambda t: t[0])
        parts = ["Persuasion dialogue:"]
        for _, role, text in turns:
            speaker = "Persuader" if role == 0 else "Persuadee"
            parts.append(f"{speaker}: {text}")
        texts.append("\n".join(parts))

    return texts


# Formatter lookup so dataset configs can reference them by name.
_FORMATTERS = {
    "_format_arc_example": _format_arc_example,
    "_format_sciq_example": _format_sciq_example,
    "_format_openbookqa_example": _format_openbookqa_example,
    "_format_hellaswag_example": _format_hellaswag_example,
    "_format_winogrande_example": _format_winogrande_example,
    "_format_prosocial_example": _format_prosocial_example,
    "_format_hh_rlhf_example": _format_hh_rlhf_example,
    "_format_goemotions_example": _format_goemotions_example,
    "_format_emotion_example": _format_emotion_example,
    "_format_counseling_example": _format_counseling_example,
    "_format_medical_dialogue_example": _format_medical_dialogue_example,
    "_format_medquad_example": _format_medquad_example,
    "_format_esconv_example": _format_esconv_example,
    "_format_mathdial_example": _format_mathdial_example,
    "_format_casino_example": _format_casino_example,
    "_format_persuasion_example": _format_persuasion_example,
}



def _tokenize_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int = 1024,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize text strings into training examples using batched encoding.

    For causal LM training, labels are the same as input_ids
    (the model learns to predict the next token at each position).
    """
    examples = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        chunk = texts[i : i + batch_size]
        encoded = tokenizer(
            chunk,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )
        for ids_list in encoded["input_ids"]:
            if len(ids_list) < 2:
                continue
            ids = torch.tensor(ids_list, dtype=torch.long)
            examples.append({"input_ids": ids, "labels": ids.clone()})
    return examples


def _load_and_format_datasets(
    dataset_configs: list[dict],
    max_samples: int | None = None,
) -> list[str]:
    """Load multiple HF datasets and format them into text strings.

    If max_samples is set, the cap is applied proportionally across
    datasets so each contributes a fair share to the total.

    Args:
        dataset_configs: list of dicts with path, name, split, formatter keys.
        max_samples: optional total cap on combined example count.

    Returns:
        List of formatted text strings ready for tokenization.
    """
    # First pass: load all datasets to know their sizes
    raw_datasets = []
    for cfg in dataset_configs:
        logger.info("Loading dataset: %s (%s)", cfg["path"], cfg.get("name"))
        ds = load_dataset(cfg["path"], cfg.get("name"), split=cfg["split"])
        raw_datasets.append((ds, cfg["formatter"]))

    # Compute per-dataset caps if max_samples is set
    total_available = sum(len(ds) for ds, _ in raw_datasets)
    texts = []

    for ds, formatter_name in raw_datasets:
        formatter = _FORMATTERS[formatter_name]

        if max_samples is not None:
            # Proportional cap: each dataset gets a share based on its
            # fraction of total available examples
            cap = max(1, int(max_samples * len(ds) / total_available))
            ds = ds.select(range(min(cap, len(ds))))

        logger.info(
            "Formatting %d examples with %s", len(ds), formatter_name
        )
        texts.extend(formatter(ex) for ex in ds)

    return texts


def build_phase1_dataloader(
    module_type: ModuleType,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for Phase 1 module-specific pre-training.

    For LOGIC, combines ARC (Easy + Challenge), SciQ, PIQA, HellaSwag,
    and WinoGrande into a single ~70K example reasoning corpus.
    For EQ, uses ProsocialDialog (~58K examples).

    Args:
        module_type: which module to load data for (LOGIC or EQ).
        tokenizer: the tokenizer to use (SmolLM2).
        batch_size: training batch size.
        max_length: maximum sequence length after tokenization.
        max_samples: optional cap on total number of examples (for debugging).
            Applied proportionally across constituent datasets.
        num_workers: DataLoader worker count.

    Returns:
        DataLoader yielding {"input_ids": ..., "labels": ...} dicts.
    """
    if module_type == ModuleType.LOGIC:
        dataset_configs = LOGIC_DATASETS
    elif module_type == ModuleType.EQ:
        dataset_configs = EQ_DATASETS
    else:
        raise ValueError(f"Phase 1 data not configured for {module_type}")

    texts = _load_and_format_datasets(dataset_configs, max_samples)
    logger.info(
        "Phase 1 [%s]: %d total examples across %d datasets",
        module_type.value, len(texts), len(dataset_configs),
    )

    logger.info("Tokenizing %d examples (max_length=%d)", len(texts), max_length)
    examples = _tokenize_texts(texts, tokenizer, max_length)
    logger.info("Produced %d tokenized examples", len(examples))

    dataset = TokenizedDataset(examples)
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(_collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )


def build_phase2_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for Phase 2/3 multi-faculty training.

    Combines 8 diverse datasets spanning counseling, medical dialogue,
    tutoring, negotiation, persuasion, and emotional support. Every
    dataset requires both logical reasoning and emotional intelligence,
    ensuring the cross-attention layers learn general-purpose
    inter-module communication rather than overfitting to a single domain.

    Total unsampled size: ~160K examples. With proportional sampling
    via max_samples (default ~32K), each domain contributes a balanced
    share relative to its raw size.

    Args:
        tokenizer: the tokenizer to use.
        batch_size: training batch size.
        max_length: maximum sequence length.
        max_samples: optional cap on total examples. Applied
            proportionally across datasets. Defaults to 32000 to
            keep training tractable while covering all domains.
        num_workers: DataLoader worker count.

    Returns:
        DataLoader yielding {"input_ids": ..., "labels": ...} dicts.
    """
    if max_samples is None:
        max_samples = 32000

    # PersuasionForGood needs special handling: its rows are individual
    # utterances that must be grouped into dialogues before formatting.
    # We split it out from the standard pipeline.
    standard_configs = [
        cfg for cfg in PHASE2_DATASETS
        if cfg["formatter"] != "_format_persuasion_example"
    ]
    persuasion_config = next(
        (cfg for cfg in PHASE2_DATASETS
         if cfg["formatter"] == "_format_persuasion_example"),
        None,
    )

    # Load standard datasets (all except PersuasionForGood)
    raw_datasets = []
    for cfg in standard_configs:
        logger.info("Loading dataset: %s (%s)", cfg["path"], cfg.get("name"))
        ds = load_dataset(cfg["path"], cfg.get("name"), split=cfg["split"])
        raw_datasets.append((ds, cfg["formatter"]))

    # Load and group PersuasionForGood dialogues
    persuasion_texts = []
    if persuasion_config is not None:
        logger.info(
            "Loading dataset: %s (grouping utterances into dialogues)",
            persuasion_config["path"],
        )
        persuasion_ds = load_dataset(
            persuasion_config["path"],
            persuasion_config.get("name"),
            split=persuasion_config["split"],
        )
        persuasion_texts = _build_persuasion_dialogues(persuasion_ds)
        logger.info(
            "Built %d persuasion dialogues from %d utterances",
            len(persuasion_texts), len(persuasion_ds),
        )

    # Compute proportional caps across all sources
    standard_total = sum(len(ds) for ds, _ in raw_datasets)
    persuasion_total = len(persuasion_texts)
    grand_total = standard_total + persuasion_total

    texts = []

    # Format standard datasets with proportional sampling
    for ds, formatter_name in raw_datasets:
        formatter = _FORMATTERS[formatter_name]
        cap = max(1, int(max_samples * len(ds) / grand_total))
        cap = min(cap, len(ds))
        ds_sampled = ds.select(range(cap))

        logger.info(
            "Formatting %d/%d examples with %s",
            cap, len(ds), formatter_name,
        )
        formatted = [formatter(ex) for ex in ds_sampled]
        texts.extend(t for t in formatted if t)

    # Add proportionally sampled persuasion dialogues
    if persuasion_texts:
        cap = max(1, int(max_samples * persuasion_total / grand_total))
        cap = min(cap, len(persuasion_texts))
        logger.info(
            "Adding %d/%d persuasion dialogues",
            cap, len(persuasion_texts),
        )
        texts.extend(persuasion_texts[:cap])

    logger.info(
        "Phase 2: %d total examples across %d datasets",
        len(texts), len(PHASE2_DATASETS),
    )

    logger.info("Tokenizing %d examples (max_length=%d)", len(texts), max_length)
    examples = _tokenize_texts(texts, tokenizer, max_length)
    logger.info("Produced %d tokenized examples", len(examples))

    dataset = TokenizedDataset(examples)
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(_collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )
