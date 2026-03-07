"""Utilities for loading HuggingFace SmolLM2 weights into our custom transformer.

The HuggingFace LlamaForCausalLM state dict uses a specific naming convention.
This module maps those names to our Transformer and TransformerForCausalLM
parameter names.
"""

import logging

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from clsa.config.transformer_config import TransformerConfig
from clsa.modules.transformer import Transformer, TransformerForCausalLM

logger = logging.getLogger(__name__)

# Mapping from HuggingFace LlamaForCausalLM keys to our keys.
# "model.layers.{i}" maps to "model.layers.{i}" in both, so most
# layer-internal names match directly. The main differences are
# at the top level.
HF_TO_OURS_CAUSAL_LM = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# Within each transformer block, HF and our naming match exactly:
# self_attn.{q,k,v,o}_proj.weight
# mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
# input_layernorm.weight, post_attention_layernorm.weight


def map_hf_key_to_ours(hf_key: str, for_causal_lm: bool = True) -> str:
    """Convert a HuggingFace state dict key to our naming convention.

    For TransformerForCausalLM, keys start with "model." for the backbone.
    For bare Transformer, we strip the "model." prefix.
    """
    if for_causal_lm:
        # Top-level keys that differ
        if hf_key in HF_TO_OURS_CAUSAL_LM:
            return HF_TO_OURS_CAUSAL_LM[hf_key]
        # Layer keys match as-is (model.layers.{i}.*)
        return hf_key
    else:
        # For bare Transformer, strip "model." prefix
        if hf_key.startswith("model."):
            return hf_key[len("model."):]
        # lm_head is not part of bare Transformer
        return None


def download_smollm2_weights(
    model_id: str = "HuggingFaceTB/SmolLM2-135M",
    cache_dir: str | None = None,
) -> str:
    """Download SmolLM2 weights from HuggingFace Hub.

    Returns the local directory path containing the model files.
    """
    logger.info("Downloading %s weights...", model_id)
    local_dir = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*"],
    )
    logger.info("Weights downloaded to %s", local_dir)
    return local_dir


def load_safetensors_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    """Load all safetensors files from a model directory into a single state dict."""
    import glob
    import os

    pattern = os.path.join(model_dir, "*.safetensors")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    state_dict = {}
    for f in sorted(files):
        state_dict.update(load_file(f))
    return state_dict


def load_smollm2_into_causal_lm(
    model: TransformerForCausalLM,
    model_dir: str | None = None,
    model_id: str = "HuggingFaceTB/SmolLM2-135M",
) -> TransformerForCausalLM:
    """Load SmolLM2 pretrained weights into a TransformerForCausalLM instance.

    If model_dir is not provided, downloads the weights first.
    """
    if model_dir is None:
        model_dir = download_smollm2_weights(model_id)

    hf_state_dict = load_safetensors_state_dict(model_dir)
    our_state_dict = {}

    for hf_key, tensor in hf_state_dict.items():
        our_key = map_hf_key_to_ours(hf_key, for_causal_lm=True)
        if our_key is not None:
            our_state_dict[our_key] = tensor

    # Load with strict=False to get informative error messages
    missing, unexpected = model.load_state_dict(our_state_dict, strict=False)

    # With tied embeddings, lm_head.weight will show as "missing" since
    # it shares memory with embed_tokens.weight. That is expected.
    tied_missing = {"lm_head.weight"}
    real_missing = set(missing) - tied_missing

    if real_missing:
        logger.warning("Missing keys: %s", real_missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    logger.info(
        "Loaded %d parameters from SmolLM2 into TransformerForCausalLM",
        len(our_state_dict),
    )
    return model


def load_smollm2_into_transformer(
    model: Transformer,
    model_dir: str | None = None,
    model_id: str = "HuggingFaceTB/SmolLM2-135M",
) -> Transformer:
    """Load SmolLM2 pretrained weights into a bare Transformer (no LM head).

    This is used for initializing CLSA cognitive modules, which use the
    transformer backbone but replace the LM head with probabilistic
    output projections.
    """
    if model_dir is None:
        model_dir = download_smollm2_weights(model_id)

    hf_state_dict = load_safetensors_state_dict(model_dir)
    our_state_dict = {}

    for hf_key, tensor in hf_state_dict.items():
        our_key = map_hf_key_to_ours(hf_key, for_causal_lm=False)
        if our_key is not None:
            our_state_dict[our_key] = tensor

    missing, unexpected = model.load_state_dict(our_state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    logger.info(
        "Loaded %d parameters from SmolLM2 into Transformer",
        len(our_state_dict),
    )
    return model
