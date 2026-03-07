# CLSA: Collaborative Latent Superposition Architecture

A multi-module neural architecture for structured latent-space deliberation with probabilistic superposition and guaranteed emotional intelligence modulation.

## The Problem: The See-Saw Effect

Monolithic LLMs suffer from a zero-sum trade-off: prompting for more emotional sensitivity degrades logical accuracy, and vice versa. This happens because all cognitive faculties compete for the same representational capacity in a single parameter space.

## The Solution

CLSA replaces the single monolithic model with **structurally independent cognitive modules** that deliberate simultaneously in a shared latent space. Each module outputs a probability distribution, and these are combined via precision-weighted product of Gaussians. Module competence is structurally independent of module influence, eliminating the see-saw effect.

### Key Innovations

- **Probabilistic multi-module deliberation** in continuous latent space (each module outputs a Gaussian, not a vector)
- **Soft-orthogonal module identity** via regularization (modules stay specialized without hard dimensional partitioning)
- **Structured superposition** through distributional interference (combined state is a product of module distributions)
- **Precision-weighted mixing board** for steerability without prompt engineering

## Architecture

```text
Input Tokens
     |
     v
+----+-----+-----+
| Logic    | EQ   |  ... (extensible to Creativity, Retrieval)
| Module   | Module|
| (SmolLM2)| (SmolLM2)
+----+-----+-----+
     |   cross-attention
     v
  Deliberation Loop
  (product of Gaussians,
   EQ reweighting,
   convergence check)
     |
     v
  Projection Decoder --> Output Tokens
```

### Modules (MVP: 2 of 4)

| Module | Function | Status |
| -------- | ---------- | -------- |
| Logic & Reasoning | Formal reasoning, causal inference, logical consistency | Active |
| Emotional Intelligence (EQ) | Empathetic framing, social appropriateness, conflict sensitivity | Active |
| Creativity | Divergent thinking, analogical reasoning, metaphor generation | Planned |
| Retrieval & Grounding | External knowledge via MCP | Planned |

## Project Structure

```text
clsa/
  config/           # TransformerConfig, CLSAConfig
  modules/          # Custom transformer, cognitive modules, weight loading
  deliberation/     # Cross-attention, product of Gaussians, deliberation loop
  decoder/          # Projection interface (latent states -> tokens)
  training/         # Phase 1/2/3 trainers, loss functions
  evaluation/       # See-saw hypothesis test
  model.py          # Top-level CLSA model
tests/
  test_smoke.py     # Structural smoke tests
```

## Setup

```bash
# Requires Python 3.12+
uv sync
uv sync --group dev  # for pytest and ruff
```

## Quick Start

```python
from clsa.config.clsa_config import CLSAConfig, ModuleType
from clsa.model import CLSA

config = CLSAConfig()
model = CLSA(config)

# The "mixing board": adjust module influence without retraining
output = model(
    input_ids,
    labels=labels,
    precision_overrides={ModuleType.EQ: 2.0, ModuleType.LOGIC: 1.0},
)
```

## Training (Three Phases)

1. **Phase 1**: Module-specific pre-training on domain data (modules trained independently)
2. **Phase 2**: Communication training (cross-attention and decoder; module backbones frozen)
3. **Phase 3**: End-to-end fine-tuning with orthogonality, specialization, and diversity guardrails

## Tests

```bash
uv run python -m pytest tests/ -v
```

## Base Model

Cognitive modules are initialized from [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) weights using a custom transformer implementation (LLaMA-family: RMSNorm, SiLU/SwiGLU, Rotary Position Embeddings, Grouped Query Attention).
