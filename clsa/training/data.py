"""Data loading utilities for CLSA training.

Provides dataset builders for each training phase:
  Phase 1: Domain-specific data per module (logic corpora, empathy dialogues)
  Phase 2/3: Multi-faculty data requiring both logic and EQ simultaneously

All builders return standard PyTorch DataLoaders yielding dicts with
"input_ids" and "labels" keys, matching what the trainer classes expect.
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from clsa.config.clsa_config import ModuleType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupervisedTextExample:
    """One prompt/target example for supervised causal-LM training."""

    prompt: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return bool(self.prompt.strip() and self.target.strip())

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

# Structured Phase 1 supervision uses prompt/target examples with target-only
# loss masking so modules learn to answer a task rather than merely continue a
# preformatted record.
PHASE1_LOGIC_SUPERVISED_DATASETS = [
    {
        "path": "allenai/ai2_arc",
        "name": "ARC-Easy",
        "split": "train",
        "formatter": "_supervise_arc_example",
    },
    {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "train",
        "formatter": "_supervise_arc_example",
    },
    {
        "path": "allenai/sciq",
        "name": None,
        "split": "train",
        "formatter": "_supervise_sciq_example",
    },
    {
        "path": "allenai/openbookqa",
        "name": "main",
        "split": "train",
        "formatter": "_supervise_openbookqa_example",
    },
    {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "train",
        "formatter": "_supervise_hellaswag_example",
    },
    {
        "path": "allenai/winogrande",
        "name": "winogrande_debiased",
        "split": "train",
        "formatter": "_supervise_winogrande_example",
    },
]

PHASE1_EQ_SUPERVISED_DATASETS = [
    {
        "path": "allenai/prosocial-dialog",
        "name": "default",
        "split": "train",
        "formatter": "_supervise_prosocial_example",
    },
    {
        "path": "Amod/mental_health_counseling_conversations",
        "name": None,
        "split": "train",
        "formatter": "_supervise_counseling_example",
    },
    {
        "path": "thu-coai/esconv",
        "name": None,
        "split": "train",
        "formatter": "_supervise_esconv_example",
    },
    {
        "path": "Anthropic/hh-rlhf",
        "name": None,
        "split": "train",
        "formatter": "_supervise_hh_rlhf_example",
    },
    {
        "path": "google-research-datasets/go_emotions",
        "name": "simplified",
        "split": "train",
        "formatter": "_supervise_goemotions_example",
    },
    {
        "path": "dair-ai/emotion",
        "name": "split",
        "split": "train",
        "formatter": "_supervise_emotion_example",
    },
]

# Phase 2 datasets. These are formatted as prompt/target supervision so CLSA
# learns to answer from a prompt rather than continue a pre-rendered transcript.
# The mix is balanced by source, not by raw row count, so the large medical
# corpora do not overwhelm smaller dialogue-heavy domains.
PHASE2_DATASETS = [
    # Counseling: problem diagnosis + empathetic response (~3.5K)
    {
        "path": "Amod/mental_health_counseling_conversations",
        "name": None,
        "split": "train",
        "formatter": "_supervise_counseling_phase2_example",
        "sampling_weight": 1.0,
    },
    # Patient-doctor dialogue: accurate medical reasoning + sensitivity (~112K,
    # sampled down). Patients describe symptoms, doctors must reason about
    # diagnosis while being reassuring and accessible.
    {
        "path": "lavita/medical-qa-datasets",
        "name": "chatdoctor_healthcaremagic",
        "split": "train",
        "formatter": "_supervise_medical_dialogue_example",
        "sampling_weight": 1.0,
    },
    # Medical Q&A: factual health info framed for patients (~16K, sampled down).
    # Requires translating clinical knowledge into patient-friendly language.
    {
        "path": "keivalya/MedQuad-MedicalQnADataset",
        "name": None,
        "split": "train",
        "formatter": "_supervise_medquad_example",
        "sampling_weight": 1.0,
    },
    # Patient health info: accessible explanations of conditions (~5.9K,
    # sampled down). Bridges medical accuracy with plain-language framing.
    {
        "path": "lavita/medical-qa-datasets",
        "name": "medical_meadow_wikidoc_patient_information",
        "split": "train",
        "formatter": "_supervise_medical_dialogue_example",
        "sampling_weight": 1.0,
    },
    # Emotional support conversations: structured strategies for helping
    # people through crises (~0.9K raw dialogues, expanded to turn-level
    # supporter responses).
    {
        "path": "thu-coai/esconv",
        "name": None,
        "split": "train",
        "formatter": "_supervise_esconv_phase2_example",
        "sampling_weight": 1.0,
    },
    # Math tutoring: pedagogical logic + patience with struggling students
    # (~2.2K raw dialogues, expanded to teacher turns). Tutors must reason
    # correctly while responding to confusion and frustration.
    {
        "path": "eth-nlped/mathdial",
        "name": None,
        "split": "train",
        "formatter": "_supervise_mathdial_example",
        "sampling_weight": 1.0,
    },
    # Persuasion for good: logical argumentation + empathetic appeal (~20.9K
    # utterances, grouped into turn-level persuader responses). Persuading
    # someone to donate requires understanding their perspective while
    # building a reasoned, socially aware case.
    {
        "path": "spawn99/PersuasionForGood",
        "name": None,
        "split": "FullDialog",
        "builder": "_build_persuasion_supervised_examples",
        "sampling_weight": 1.0,
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


def _supervised_answer_target(answer: str) -> str:
    answer = str(answer).strip()
    return answer if answer.endswith((".", "!", "?")) else f"{answer}."


def _emotion_assessment_target(labels: list[str]) -> str:
    cleaned = [label.strip() for label in labels if label.strip()]
    if not cleaned:
        return "The message seems emotionally neutral."
    if len(cleaned) == 1:
        return f"The message primarily conveys {cleaned[0]}."
    joined = ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"
    return f"The message conveys {joined}."


def _split_hh_rlhf_transcript(transcript: str) -> SupervisedTextExample | None:
    text = transcript.strip()
    marker_positions = [match.start() for match in re.finditer(r"(?:^|\n\n)Assistant:", text)]
    if not marker_positions:
        return None

    marker_start = marker_positions[-1]
    assistant_offset = text.find("Assistant:", marker_start)
    prompt = text[: assistant_offset + len("Assistant:")].strip()
    target = text[assistant_offset + len("Assistant:") :].strip()
    if not prompt or not target:
        return None
    return SupervisedTextExample(prompt=prompt, target=target)


def _supervise_arc_example(example: dict) -> SupervisedTextExample:
    question = str(example["question"]).strip()
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = str(example["answerKey"]).strip()

    choice_lines = [f"{label}) {text}" for label, text in zip(labels, texts)]
    answer_text = next(
        (str(text).strip() for label, text in zip(labels, texts) if str(label).strip() == answer_key),
        answer_key,
    )
    prompt = (
        "You are the logic specialist. Solve the multiple-choice problem and answer concisely.\n\n"
        f"Question: {question}\n"
        "Choices:\n"
        + "\n".join(choice_lines)
        + "\n\nAnswer:"
    )
    return SupervisedTextExample(prompt=prompt, target=_supervised_answer_target(answer_text))


def _supervise_sciq_example(example: dict) -> SupervisedTextExample:
    question = str(example["question"]).strip()
    support = str(example.get("support", "")).strip()
    correct = str(example["correct_answer"]).strip()
    distractors = [
        str(example.get("distractor1", "")).strip(),
        str(example.get("distractor2", "")).strip(),
        str(example.get("distractor3", "")).strip(),
    ]
    choices = [correct] + [choice for choice in distractors if choice]
    choice_lines = [f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)]

    parts = ["You are the logic specialist. Answer the science question correctly."]
    if support:
        parts.append(f"Context: {support}")
    parts.append(f"Question: {question}")
    parts.append("Choices:\n" + "\n".join(choice_lines))
    parts.append("Answer:")
    prompt = "\n\n".join(parts)
    return SupervisedTextExample(prompt=prompt, target=_supervised_answer_target(correct))


def _supervise_openbookqa_example(example: dict) -> SupervisedTextExample:
    question = str(example["question_stem"]).strip()
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = str(example["answerKey"]).strip()
    fact = str(example.get("fact1", "")).strip()

    choice_lines = [f"{label}) {text}" for label, text in zip(labels, texts)]
    answer_text = next(
        (str(text).strip() for label, text in zip(labels, texts) if str(label).strip() == answer_key),
        answer_key,
    )

    parts = ["You are the logic specialist. Use the provided fact if it helps and answer concisely."]
    if fact:
        parts.append(f"Fact: {fact}")
    parts.append(f"Question: {question}")
    parts.append("Choices:\n" + "\n".join(choice_lines))
    parts.append("Answer:")
    prompt = "\n\n".join(parts)
    return SupervisedTextExample(prompt=prompt, target=_supervised_answer_target(answer_text))


def _supervise_hellaswag_example(example: dict) -> SupervisedTextExample:
    ctx = str(example["ctx"]).strip()
    endings = [str(ending).strip() for ending in example["endings"]]
    label = int(example["label"])

    choice_lines = [f"{chr(65 + i)}) {ending}" for i, ending in enumerate(endings)]
    answer = endings[label]
    prompt = (
        "You are the logic specialist. Choose the best continuation and answer naturally.\n\n"
        f"Context: {ctx}\n"
        "Choices:\n"
        + "\n".join(choice_lines)
        + "\n\nBest continuation:"
    )
    return SupervisedTextExample(prompt=prompt, target=_supervised_answer_target(answer))


def _supervise_winogrande_example(example: dict) -> SupervisedTextExample:
    sentence = str(example["sentence"]).strip()
    option1 = str(example["option1"]).strip()
    option2 = str(example["option2"]).strip()
    answer = str(example["answer"]).strip()

    correct = option1 if answer == "1" else option2
    prompt = (
        "You are the logic specialist. Resolve the blank correctly.\n\n"
        f"Sentence: {sentence}\n"
        f"Options:\nA) {option1}\nB) {option2}\n\n"
        "Completed sentence:"
    )
    return SupervisedTextExample(
        prompt=prompt,
        target=_supervised_answer_target(sentence.replace("_", correct)),
    )


def _supervise_hh_rlhf_example(example: dict) -> SupervisedTextExample | None:
    return _split_hh_rlhf_transcript(str(example["chosen"]))


def _supervise_goemotions_example(example: dict) -> SupervisedTextExample:
    emotion_names = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise",
        "neutral",
    ]
    text = str(example["text"]).strip()
    label_ids = example["labels"]
    labels = [emotion_names[i] for i in label_ids if i < len(emotion_names)]
    prompt = (
        "You are the EQ specialist. Identify the emotions expressed in the message so a"
        " supportive response can be calibrated well.\n\n"
        f"Message: {text}\n\n"
        "Emotion assessment:"
    )
    return SupervisedTextExample(prompt=prompt, target=_emotion_assessment_target(labels))


def _supervise_emotion_example(example: dict) -> SupervisedTextExample:
    emotion_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    text = str(example["text"]).strip()
    label = int(example["label"])
    emotion = emotion_names[label] if label < len(emotion_names) else "unknown"
    prompt = (
        "You are the EQ specialist. Identify the primary emotion in the message.\n\n"
        f"Message: {text}\n\n"
        "Emotion assessment:"
    )
    return SupervisedTextExample(prompt=prompt, target=_emotion_assessment_target([emotion]))


def _supervise_prosocial_example(example: dict) -> SupervisedTextExample:
    context = str(example.get("context", "")).strip()
    response = str(example.get("response", "")).strip()
    rots = [str(rot).strip() for rot in example.get("rots", []) if str(rot).strip()]

    parts = [
        "You are the EQ specialist. Write a prosocial, emotionally attuned response.",
        f"Message: {context}",
    ]
    if rots:
        parts.append("Considerations: " + " ".join(rots))
    parts.append("Response:")
    prompt = "\n\n".join(parts)
    return SupervisedTextExample(prompt=prompt, target=response)


def _supervise_counseling_example(example: dict) -> SupervisedTextExample:
    context = str(example.get("Context", "")).strip()
    response = str(example.get("Response", "")).strip()
    prompt = (
        "You are the EQ specialist. Write a supportive counselor response.\n\n"
        f"Client: {context}\n\n"
        "Counselor:"
    )
    return SupervisedTextExample(prompt=prompt, target=response)


def _supervise_counseling_phase2_example(example: dict) -> SupervisedTextExample:
    context = str(example.get("Context", "")).strip()
    response = str(example.get("Response", "")).strip()
    prompt = (
        "Write the counselor's response. Keep it emotionally supportive and"
        " practically helpful.\n\n"
        f"Client: {context}\n\n"
        "Counselor:"
    )
    return SupervisedTextExample(prompt=prompt, target=response)


def _supervise_medical_dialogue_example(example: dict) -> SupervisedTextExample:
    instruction = str(example.get("instruction", "")).strip()
    patient_input = str(example.get("input", "")).strip()
    doctor_output = str(example.get("output", "")).strip()

    prompt_parts = [
        "Write the clinician's response. Keep it medically grounded, clear, and appropriately empathetic."
    ]
    if instruction:
        prompt_parts.append(f"Task: {instruction}")
    if patient_input:
        prompt_parts.append(f"Patient: {patient_input}")
    prompt_parts.append("Clinician:")
    return SupervisedTextExample(prompt="\n\n".join(prompt_parts), target=doctor_output)


def _supervise_medquad_example(example: dict) -> SupervisedTextExample:
    qtype = str(example.get("qtype", "")).strip()
    question = str(example.get("Question", "")).strip()
    answer = str(example.get("Answer", "")).strip()

    prompt_parts = [
        "Answer the patient's question in plain language. Be accurate, calm, and easy to follow."
    ]
    if qtype:
        prompt_parts.append(f"Topic: {qtype}")
    prompt_parts.append(f"Patient question: {question}")
    prompt_parts.append("Answer:")
    return SupervisedTextExample(prompt="\n\n".join(prompt_parts), target=answer)


def _strip_parenthetical_tag(text: str) -> str:
    return re.sub(r"^\([^)]+\)\s*", "", text.strip())


def _split_dialogue_turn(turn: str) -> tuple[str, str]:
    if ":" not in turn:
        return "", turn.strip()
    speaker, text = turn.split(":", 1)
    return speaker.strip(), text.strip()


def _supervise_mathdial_example(example: dict) -> list[SupervisedTextExample]:
    question = str(example.get("question", "")).strip()
    incorrect = str(example.get("student_incorrect_solution", "")).strip()
    profile = str(example.get("student_profile", "")).strip()
    conversation = str(example.get("conversation", "")).strip()

    rendered_turns: list[str] = []
    supervised_examples: list[SupervisedTextExample] = []

    for raw_turn in conversation.split("|EOM|"):
        raw_turn = raw_turn.strip()
        if not raw_turn:
            continue
        speaker, text = _split_dialogue_turn(raw_turn)
        if not text:
            continue

        clean_text = _strip_parenthetical_tag(text)
        speaker_lower = speaker.lower()
        rendered_speaker = "Tutor" if speaker_lower == "teacher" else "Student"

        if speaker_lower == "teacher":
            prompt_parts = [
                "Write the tutor's next response. Be mathematically correct, encouraging, and responsive to the student's confusion."
            ]
            if question:
                prompt_parts.append(f"Math problem: {question}")
            if incorrect:
                prompt_parts.append(f"Student's current attempt: {incorrect}")
            if profile:
                prompt_parts.append(f"Student profile: {profile}")
            if rendered_turns:
                prompt_parts.append("Conversation so far:\n" + "\n".join(rendered_turns))
            prompt_parts.append("Tutor:")
            supervised_examples.append(
                SupervisedTextExample(
                    prompt="\n\n".join(prompt_parts),
                    target=clean_text,
                )
            )

        rendered_turns.append(f"{rendered_speaker}: {clean_text}")

    return supervised_examples


def _supervise_esconv_example(example: dict) -> list[SupervisedTextExample]:
    data = json.loads(example["text"])
    situation = str(data.get("situation", "")).strip()
    emotion = str(data.get("emotion_type", "")).strip()
    problem = str(data.get("problem_type", "")).strip()
    dialog = data.get("dialog", [])

    rendered_turns: list[str] = []
    supervised_examples: list[SupervisedTextExample] = []
    for turn in dialog:
        speaker = "Seeker" if turn.get("speaker") == "usr" else "Supporter"
        text = str(turn.get("text", "")).strip()
        strategy = str(turn.get("strategy", "")).strip()
        if not text:
            continue

        if speaker == "Supporter":
            prompt_parts = [
                "You are the EQ specialist. Continue the conversation with a supportive response."
            ]
            if emotion or problem:
                prompt_parts.append(
                    "Context: "
                    + ", ".join(part for part in [emotion, problem] if part)
                )
            if situation:
                prompt_parts.append(f"Situation: {situation}")
            if rendered_turns:
                prompt_parts.append("Conversation so far:\n" + "\n".join(rendered_turns))
            if strategy:
                prompt_parts.append(f"Support strategy: {strategy}")
            prompt_parts.append("Supporter:")
            supervised_examples.append(
                SupervisedTextExample(
                    prompt="\n\n".join(prompt_parts),
                    target=text,
                    metadata={"strategy": strategy},
                )
            )

        rendered_turns.append(f"{speaker}: {text}")

    return supervised_examples


def _supervise_esconv_phase2_example(example: dict) -> list[SupervisedTextExample]:
    data = json.loads(example["text"])
    situation = str(data.get("situation", "")).strip()
    emotion = str(data.get("emotion_type", "")).strip()
    problem = str(data.get("problem_type", "")).strip()
    dialog = data.get("dialog", [])

    rendered_turns: list[str] = []
    supervised_examples: list[SupervisedTextExample] = []
    for turn in dialog:
        speaker = "Seeker" if turn.get("speaker") == "usr" else "Supporter"
        text = str(turn.get("text", "")).strip()
        strategy = str(turn.get("strategy", "")).strip()
        if not text:
            continue

        if speaker == "Supporter":
            prompt_parts = [
                "Write the supporter's next response. Be emotionally supportive, practical, and tactful."
            ]
            if emotion or problem:
                prompt_parts.append(
                    "Context: " + ", ".join(part for part in [emotion, problem] if part)
                )
            if situation:
                prompt_parts.append(f"Situation: {situation}")
            if rendered_turns:
                prompt_parts.append("Conversation so far:\n" + "\n".join(rendered_turns))
            if strategy:
                prompt_parts.append(f"Support strategy: {strategy}")
            prompt_parts.append("Supporter:")
            supervised_examples.append(
                SupervisedTextExample(
                    prompt="\n\n".join(prompt_parts),
                    target=text,
                    metadata={"strategy": strategy},
                )
            )

        rendered_turns.append(f"{speaker}: {text}")

    return supervised_examples


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


def _build_persuasion_supervised_examples(dataset) -> list[SupervisedTextExample]:
    """Build turn-level persuader response examples from PersuasionForGood.

    The raw split stores one utterance per row. We reconstruct each dialogue and
    emit only the persuader turns so the task stays assistant-like: generate a
    persuasive but socially aware next response.
    """
    dialogues: dict[str, list[tuple[int, int, str]]] = {}
    for ex in dataset:
        dialog_id = str(ex.get("B2", "")).strip()
        turn = int(ex.get("Turn", 0))
        role = int(ex.get("B4", 0))  # 0 = persuader, 1 = persuadee
        text = str(ex.get("Unit", "")).strip()
        if dialog_id and text:
            dialogues.setdefault(dialog_id, []).append((turn, role, text))

    supervised_examples: list[SupervisedTextExample] = []
    for turns in dialogues.values():
        turns.sort(key=lambda row: row[0])
        rendered_turns: list[str] = []
        for _, role, text in turns:
            speaker = "Persuader" if role == 0 else "Persuadee"
            if role == 0:
                prompt_parts = [
                    "Continue the fundraising conversation as the persuader. Be respectful, socially aware, and make a reasoned case."
                ]
                if rendered_turns:
                    prompt_parts.append("Conversation so far:\n" + "\n".join(rendered_turns))
                prompt_parts.append("Persuader:")
                supervised_examples.append(
                    SupervisedTextExample(
                        prompt="\n\n".join(prompt_parts),
                        target=text,
                    )
                )
            rendered_turns.append(f"{speaker}: {text}")

    return supervised_examples


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

_SUPERVISED_FORMATTERS = {
    "_supervise_arc_example": _supervise_arc_example,
    "_supervise_sciq_example": _supervise_sciq_example,
    "_supervise_openbookqa_example": _supervise_openbookqa_example,
    "_supervise_hellaswag_example": _supervise_hellaswag_example,
    "_supervise_winogrande_example": _supervise_winogrande_example,
    "_supervise_hh_rlhf_example": _supervise_hh_rlhf_example,
    "_supervise_goemotions_example": _supervise_goemotions_example,
    "_supervise_emotion_example": _supervise_emotion_example,
    "_supervise_prosocial_example": _supervise_prosocial_example,
    "_supervise_counseling_example": _supervise_counseling_example,
    "_supervise_counseling_phase2_example": _supervise_counseling_phase2_example,
    "_supervise_medical_dialogue_example": _supervise_medical_dialogue_example,
    "_supervise_medquad_example": _supervise_medquad_example,
    "_supervise_esconv_example": _supervise_esconv_example,
    "_supervise_esconv_phase2_example": _supervise_esconv_phase2_example,
    "_supervise_mathdial_example": _supervise_mathdial_example,
}

_SUPERVISED_DATASET_BUILDERS = {
    "_build_persuasion_supervised_examples": _build_persuasion_supervised_examples,
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


def _pack_supervised_token_ids(
    prompt_ids: list[int],
    target_ids: list[int],
    *,
    max_length: int,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pack prompt/target tokens into a masked causal-LM example.

    Prompt tokens are included in `input_ids` but masked out in `labels`.
    When truncation is needed, the target is preserved preferentially and the
    prompt is cropped from the left to keep the most relevant trailing context.
    """
    if not prompt_ids or not target_ids:
        return None

    extra_tokens = 1 if eos_token_id is not None else 0
    target_budget = max_length - extra_tokens
    if target_budget <= 0:
        return None

    trimmed_target_ids = target_ids[:target_budget]
    if not trimmed_target_ids:
        return None

    prompt_budget = max_length - len(trimmed_target_ids) - extra_tokens
    trimmed_prompt_ids = prompt_ids[-max(prompt_budget, 0):] if prompt_budget > 0 else []

    input_ids = list(trimmed_prompt_ids) + list(trimmed_target_ids)
    labels = ([-100] * len(trimmed_prompt_ids)) + list(trimmed_target_ids)
    if eos_token_id is not None and len(input_ids) < max_length:
        input_ids.append(eos_token_id)
        labels.append(eos_token_id)

    if not any(label != -100 for label in labels):
        return None

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def _tokenize_supervised_texts(
    examples: list[SupervisedTextExample],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int = 512,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize supervised prompt/target examples with target-only loss."""
    tokenized_examples = []
    eos_token_id = tokenizer.eos_token_id

    for i in tqdm(range(0, len(examples), batch_size), desc="Tokenizing"):
        chunk = examples[i : i + batch_size]
        prompts = [example.prompt for example in chunk]
        targets = [example.target for example in chunk]

        prompt_batch = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        target_batch = tokenizer(
            targets,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )

        for prompt_ids, target_ids in zip(prompt_batch["input_ids"], target_batch["input_ids"]):
            packed = _pack_supervised_token_ids(
                prompt_ids,
                target_ids,
                max_length=max_length,
                eos_token_id=eos_token_id,
            )
            if packed is None:
                continue
            input_ids, labels = packed
            tokenized_examples.append({"input_ids": input_ids, "labels": labels})

    return tokenized_examples


def _extend_supervised_examples(
    sink: list[SupervisedTextExample],
    formatted: SupervisedTextExample | list[SupervisedTextExample] | None,
) -> None:
    if formatted is None:
        return
    if isinstance(formatted, list):
        sink.extend(example for example in formatted if example.is_valid())
        return
    if formatted.is_valid():
        sink.append(formatted)


def _sample_examples_deterministically(
    examples: list[SupervisedTextExample],
    cap: int,
    *,
    seed: int,
) -> list[SupervisedTextExample]:
    if cap >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    return [examples[idx] for idx in indices[:cap]]


def _allocate_balanced_caps(
    available_counts: list[int],
    weights: list[float],
    max_total: int,
) -> list[int]:
    """Allocate a capped total budget across sources using source weights.

    This is intentionally source-balanced rather than raw-row-proportional so a
    single huge corpus does not dominate the mixture.
    """
    if max_total <= 0:
        return [0 for _ in available_counts]

    caps = [0 for _ in available_counts]
    remaining_budget = max_total
    remaining = {
        idx for idx, available in enumerate(available_counts) if available > 0
    }

    while remaining and remaining_budget > 0:
        total_weight = sum(weights[idx] for idx in remaining)
        if total_weight <= 0:
            break

        allocated_this_round = 0
        for idx in list(remaining):
            remaining_capacity = available_counts[idx] - caps[idx]
            if remaining_capacity <= 0:
                remaining.discard(idx)
                continue

            share = int(remaining_budget * (weights[idx] / total_weight))
            if share <= 0:
                continue

            share = min(share, remaining_capacity)
            caps[idx] += share
            allocated_this_round += share
            if caps[idx] >= available_counts[idx]:
                remaining.discard(idx)

        remaining_budget -= allocated_this_round
        if remaining_budget <= 0 or not remaining:
            break

        if allocated_this_round == 0:
            for idx in sorted(remaining, key=lambda i: weights[i], reverse=True):
                if remaining_budget <= 0:
                    break
                caps[idx] += 1
                remaining_budget -= 1
                if caps[idx] >= available_counts[idx]:
                    remaining.discard(idx)

    return caps


def _load_weighted_supervised_sources(
    dataset_configs: list[dict],
    max_samples: int | None = None,
    *,
    sample_seed: int = 0,
) -> list[SupervisedTextExample]:
    """Load prompt/target sources and optionally sample them with source balance."""
    per_source_examples: list[tuple[dict, list[SupervisedTextExample]]] = []

    for cfg in dataset_configs:
        logger.info("Loading dataset: %s (%s)", cfg["path"], cfg.get("name"))
        ds = load_dataset(cfg["path"], cfg.get("name"), split=cfg["split"])

        examples: list[SupervisedTextExample] = []
        if "builder" in cfg:
            builder = _SUPERVISED_DATASET_BUILDERS[cfg["builder"]]
            examples = builder(ds)
        else:
            formatter = _SUPERVISED_FORMATTERS[cfg["formatter"]]
            for record in ds:
                _extend_supervised_examples(examples, formatter(record))

        logger.info(
            "Built %d supervised examples from %s",
            len(examples),
            cfg.get("builder") or cfg["formatter"],
        )
        per_source_examples.append((cfg, examples))

    if max_samples is None:
        return [
            example
            for _, examples in per_source_examples
            for example in examples
        ]

    available_counts = [len(examples) for _, examples in per_source_examples]
    weights = [float(cfg.get("sampling_weight", 1.0)) for cfg, _ in per_source_examples]
    caps = _allocate_balanced_caps(available_counts, weights, max_samples)

    sampled_examples: list[SupervisedTextExample] = []
    for idx, ((cfg, examples), cap) in enumerate(zip(per_source_examples, caps)):
        source_examples = _sample_examples_deterministically(
            examples,
            cap,
            seed=sample_seed + idx,
        )
        logger.info(
            "Sampling %d/%d supervised examples from %s",
            len(source_examples),
            len(examples),
            cfg["path"],
        )
        sampled_examples.extend(source_examples)

    return sampled_examples


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


def _load_and_format_supervised_datasets(
    dataset_configs: list[dict],
    max_samples: int | None = None,
) -> list[SupervisedTextExample]:
    """Load multiple datasets and convert them into prompt/target examples."""
    raw_datasets = []
    for cfg in dataset_configs:
        logger.info("Loading dataset: %s (%s)", cfg["path"], cfg.get("name"))
        ds = load_dataset(cfg["path"], cfg.get("name"), split=cfg["split"])
        raw_datasets.append((ds, cfg["formatter"]))

    total_available = sum(len(ds) for ds, _ in raw_datasets)
    supervised_examples: list[SupervisedTextExample] = []

    for ds, formatter_name in raw_datasets:
        formatter = _SUPERVISED_FORMATTERS[formatter_name]

        if max_samples is not None:
            cap = max(1, int(max_samples * len(ds) / total_available))
            ds = ds.select(range(min(cap, len(ds))))

        logger.info("Formatting %d examples with %s", len(ds), formatter_name)
        for record in ds:
            formatted = formatter(record)
            _extend_supervised_examples(supervised_examples, formatted)

    return supervised_examples


def build_phase1_dataloader(
    module_type: ModuleType,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for Phase 1 module-specific pre-training.

    Phase 1 now uses structured prompt/target supervision with target-only
    loss masking. This teaches each module to answer a task from a prompt
    rather than simply continue an already-completed record.

    LOGIC uses reasoning QA / completion datasets.
    EQ uses empathetic response datasets plus emotion-recognition tasks
    framed as prompt/target supervision.

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
        dataset_configs = PHASE1_LOGIC_SUPERVISED_DATASETS
    elif module_type == ModuleType.EQ:
        dataset_configs = PHASE1_EQ_SUPERVISED_DATASETS
    else:
        raise ValueError(f"Phase 1 data not configured for {module_type}")

    supervised_examples = _load_and_format_supervised_datasets(dataset_configs, max_samples)
    logger.info(
        "Phase 1 [%s]: %d prompt/target examples across %d datasets",
        module_type.value, len(supervised_examples), len(dataset_configs),
    )

    logger.info("Tokenizing %d examples (max_length=%d)", len(supervised_examples), max_length)
    examples = _tokenize_supervised_texts(supervised_examples, tokenizer, max_length)
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

    Uses structured prompt/target supervision across diverse domains so the
    decoder learns to answer from a prompt rather than continue a pre-rendered
    transcript. Sampling is source-balanced after formatting, which prevents the
    very large medical corpora from overwhelming smaller dialogue-heavy domains.

    Args:
        tokenizer: the tokenizer to use.
        batch_size: training batch size.
        max_length: maximum sequence length.
        max_samples: optional cap on total prompt/target examples.
            Defaults to 32000 to keep training tractable while preserving a
            balanced source mixture.
        num_workers: DataLoader worker count.

    Returns:
        DataLoader yielding {"input_ids": ..., "labels": ...} dicts.
    """
    if max_samples is None:
        max_samples = 32000
    supervised_examples = _load_weighted_supervised_sources(
        PHASE2_DATASETS,
        max_samples=max_samples,
        sample_seed=17,
    )

    logger.info(
        "Phase 2: %d prompt/target examples across %d datasets",
        len(supervised_examples), len(PHASE2_DATASETS),
    )

    logger.info("Tokenizing %d examples (max_length=%d)", len(supervised_examples), max_length)
    examples = _tokenize_supervised_texts(supervised_examples, tokenizer, max_length)
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


def build_combined_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader combining ALL training data for baseline fine-tuning.

    Merges the current CLSA training corpora into a single monolithic baseline
    dataloader, using prompt/target supervision throughout:
    - Phase 1 logic supervision
    - Phase 1 EQ supervision
    - Phase 2/3 multi-faculty response supervision

    This keeps the baseline aligned with the corrected CLSA data design
    without forcing the baseline into the same phased optimization schedule.

    Args:
        tokenizer: the tokenizer to use.
        batch_size: training batch size.
        max_length: maximum sequence length.
        max_samples: optional cap on total prompt/target examples.
        num_workers: DataLoader worker count.

    Returns:
        DataLoader yielding {"input_ids": ..., "labels": ...} dicts.
    """
    all_configs = (
        PHASE1_LOGIC_SUPERVISED_DATASETS
        + PHASE1_EQ_SUPERVISED_DATASETS
        + PHASE2_DATASETS
    )
    supervised_examples = _load_weighted_supervised_sources(
        all_configs,
        max_samples=max_samples,
        sample_seed=29,
    )

    logger.info("Combined baseline: %d supervised examples", len(supervised_examples))

    examples = _tokenize_supervised_texts(supervised_examples, tokenizer, max_length)
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
