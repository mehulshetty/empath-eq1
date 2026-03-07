"""CLSA auxiliary loss functions for maintaining module identity during training.

Section 5.4: Total Loss = Task Loss + alpha * L_orth + beta * L_spec + gamma * L_div

These losses prevent module identity collapse during end-to-end fine-tuning
by enforcing structural constraints on module representations.
"""

import torch
import torch.nn.functional as F

from clsa.config.clsa_config import ModuleType


def orthogonality_loss(
    hidden_states: dict[ModuleType, torch.Tensor],
) -> torch.Tensor:
    """Penalize cosine similarity between module representations.

    Section 3.3: L_orth = sum_{i != j} |cos(h_i, h_j)|

    Encourages modules to maintain distinct representations in the
    shared latent space. Uses absolute cosine similarity so both
    aligned and anti-aligned representations are penalized.

    Args:
        hidden_states: dict mapping module type to its hidden states
            tensor of shape (batch, seq_len, hidden_size).

    Returns:
        Scalar loss value.
    """
    types = list(hidden_states.keys())
    if len(types) < 2:
        return torch.tensor(0.0, device=next(iter(hidden_states.values())).device)

    total = torch.tensor(0.0, device=hidden_states[types[0]].device)
    count = 0

    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            h_i = hidden_states[types[i]]
            h_j = hidden_states[types[j]]

            # Flatten to (batch * seq_len, hidden_size) for cosine similarity
            h_i_flat = h_i.reshape(-1, h_i.shape[-1])
            h_j_flat = h_j.reshape(-1, h_j.shape[-1])

            cos_sim = F.cosine_similarity(h_i_flat, h_j_flat, dim=-1)
            total = total + cos_sim.abs().mean()
            count += 1

    return total / count


def diversity_loss(logvar: torch.Tensor) -> torch.Tensor:
    """Encourage the combined posterior to maintain meaningful entropy.

    Section 5.4: Prevents premature collapse to a single module's
    perspective by penalizing very low entropy (very high precision)
    in the combined distribution.

    We want the combined distribution to be confident but not
    degenerate. This loss increases when variance drops too low,
    acting as a soft floor on uncertainty.

    Args:
        logvar: (batch, seq_len, latent_dim) log-variance of the
            combined posterior distribution.

    Returns:
        Scalar loss value. Higher when variance is too low.
    """
    # Mean log-variance across all dimensions and positions.
    # Penalize when this goes below a threshold (negative logvar = low variance).
    # Using softplus to create a smooth penalty that activates as logvar
    # goes below -2.0 (variance < ~0.14).
    mean_logvar = logvar.mean()
    threshold = -2.0
    return F.softplus(threshold - mean_logvar)


def specialization_loss(
    module_type: ModuleType,
    current_logits: torch.Tensor,
    labels: torch.Tensor,
    baseline_accuracy: float,
) -> torch.Tensor:
    """Penalize degradation from Phase 1 domain-specific performance.

    Section 5.4: Periodically probes each module on domain-specific
    benchmarks and penalizes if accuracy drops below the Phase 1
    baseline.

    This is a simplified version that compares current per-token
    accuracy against a stored baseline. The full implementation would
    use dedicated benchmark datasets per module.

    Args:
        module_type: which module is being evaluated.
        current_logits: (batch, seq_len, vocab_size) current predictions.
        labels: (batch, seq_len) ground truth tokens.
        baseline_accuracy: Phase 1 accuracy on this module's benchmark.

    Returns:
        Scalar loss. Zero if current accuracy >= baseline, positive
        penalty proportional to the degradation otherwise.
    """
    # Compute current accuracy (ignoring padding at -100)
    preds = current_logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=current_logits.device)

    correct = (preds == labels) & mask
    current_accuracy = correct.sum().float() / mask.sum().float()

    # Penalty: how much accuracy has dropped below baseline
    degradation = baseline_accuracy - current_accuracy
    return F.relu(degradation)


def total_clsa_loss(
    task_loss: torch.Tensor,
    module_hidden_states: dict[ModuleType, torch.Tensor],
    combined_logvar: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 0.05,
    gamma: float = 0.01,
    specialization_terms: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the full CLSA training loss.

    Total = task_loss + alpha * L_orth + beta * L_spec + gamma * L_div

    Args:
        task_loss: primary task loss (e.g. cross-entropy for language modeling).
        module_hidden_states: dict of hidden states per module for orthogonality.
        combined_logvar: log-variance of the combined posterior for diversity.
        alpha: orthogonality loss coefficient.
        beta: specialization loss coefficient.
        gamma: diversity loss coefficient.
        specialization_terms: optional list of per-module specialization losses.

    Returns:
        Dict with 'total', 'task', 'orthogonality', 'specialization', 'diversity'.
    """
    l_orth = orthogonality_loss(module_hidden_states)
    l_div = diversity_loss(combined_logvar)

    l_spec = torch.tensor(0.0, device=task_loss.device)
    if specialization_terms:
        l_spec = torch.stack(specialization_terms).mean()

    total = task_loss + alpha * l_orth + beta * l_spec + gamma * l_div

    return {
        "total": total,
        "task": task_loss,
        "orthogonality": l_orth,
        "specialization": l_spec,
        "diversity": l_div,
    }
