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

    Args:
        mus: List of mean tensors, each (batch, seq_len, latent_dim).
        logvars: List of log-variance tensors, same shape.
        precisions: List of scalar precision weights (pi_i), one per module.
            These scale each module's natural precision.

    Returns:
        combined_mu: (batch, seq_len, latent_dim) mean of combined distribution.
        combined_logvar: (batch, seq_len, latent_dim) log-variance of combined distribution.
    """
    # Compute effective precision for each module:
    # effective_precision_i = pi_i / sigma^2_i = pi_i * exp(-logvar_i)
    effective_precisions = []
    weighted_means = []

    for mu, logvar, pi in zip(mus, logvars, precisions):
        # pi is a scalar (per-module weight), broadcast over all dimensions
        eff_prec = pi * torch.exp(-logvar)
        effective_precisions.append(eff_prec)
        weighted_means.append(eff_prec * mu)

    # Combined precision is the sum of individual precisions
    combined_precision = torch.stack(effective_precisions, dim=0).sum(dim=0)

    # Combined mean is the precision-weighted average
    combined_mu = torch.stack(weighted_means, dim=0).sum(dim=0) / combined_precision

    # Combined variance is the inverse of combined precision
    combined_logvar = -torch.log(combined_precision)

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
