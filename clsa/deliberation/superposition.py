"""Structured superposition via product of Gaussian distributions.

Section 3.4-3.5: Module distributions are combined via precision-weighted
product of Gaussians to produce the combined posterior. This is the
mathematical core of CLSA's "superposition" mechanism.

For diagonal Gaussians, the product is another Gaussian with:
    precision_combined = sum(precision_i)
    mu_combined = (sum(precision_i * mu_i)) / precision_combined

where precision_i = 1 / sigma^2_i (inverse variance).
"""

import torch


def product_of_gaussians(
    mus: list[torch.Tensor],
    logvars: list[torch.Tensor],
    precisions: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine multiple Gaussian distributions via precision-weighted product.

    Each module contributes a diagonal Gaussian N(mu_i, sigma^2_i). The
    combined distribution is also Gaussian, computed in closed form.

    All arithmetic stays in log-space to avoid exp/log roundtrips that
    overflow in reduced precision. The combined mean is computed as a
    softmax-weighted average over module means, where the weights are
    the normalized effective precisions.

    Args:
        mus: List of mean tensors, each (batch, seq_len, latent_dim).
        logvars: List of log-variance tensors, same shape.
        precisions: List of scalar precision weights (pi_i), one per module.
            These scale each module's natural precision.

    Returns:
        combined_mu: (batch, seq_len, latent_dim) mean of combined distribution.
        combined_logvar: (batch, seq_len, latent_dim) log-variance of combined distribution.
    """
    # Log effective precision per module: log(pi_i) - logvar_i
    log_eff_precs = []
    for logvar, pi in zip(logvars, precisions):
        log_eff_precs.append(pi.log() - logvar)

    # (num_modules, batch, seq_len, latent_dim)
    stacked_log_precs = torch.stack(log_eff_precs, dim=0)

    # Combined log-precision via logsumexp (numerically stable)
    combined_log_precision = torch.logsumexp(stacked_log_precs, dim=0)
    combined_logvar = -combined_log_precision

    # Normalized weights = softmax over log-precisions along module dim.
    # Each weight is in [0, 1] and they sum to 1 — no overflow possible.
    log_weights = stacked_log_precs - combined_log_precision.unsqueeze(0)
    weights = log_weights.exp()

    # Combined mean is the precision-weighted average
    stacked_mus = torch.stack(mus, dim=0)
    combined_mu = (weights * stacked_mus).sum(dim=0)

    return combined_mu, combined_logvar


def sample_from_gaussian(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample from a diagonal Gaussian using the reparameterization trick.

    Args:
        mu: Mean tensor of any shape.
        logvar: Log-variance tensor, same shape as mu.
        temperature: Scaling factor for the noise. At 0.0, returns the mean
            (deterministic). At 1.0, standard sampling.

    Returns:
        Sample tensor, same shape as mu.
    """
    if temperature == 0.0:
        return mu
    std = (0.5 * logvar).exp() * temperature
    eps = torch.randn_like(std)
    return mu + eps * std


def gaussian_entropy(logvar: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of a diagonal Gaussian.

    H = 0.5 * sum(1 + log(2*pi) + logvar) per element.
    We return the mean over the latent dimensions to get a
    per-position scalar.

    Args:
        logvar: (batch, seq_len, latent_dim)

    Returns:
        Entropy tensor of shape (batch, seq_len).
    """
    import math

    return 0.5 * (1.0 + math.log(2 * math.pi) + logvar).mean(dim=-1)
