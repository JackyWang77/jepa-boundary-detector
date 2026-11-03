"""Free-energy objective and the patch baseline.

E2-JEPA principle:  minimize F = U - T*S
  U: prediction energy (squared latent error, per-dim)
  S: entropy of the representation (per-dim Gaussian differential entropy,
     estimated from the batch covariance: S = 0.5*logdet(Cov + eps*I)/d)
  T: temperature (annealed) == Lagrange multiplier == IB beta.
Collapse faces a sizeable barrier: at full collapse the eps floor pins S
at 0.5*log(eps) ~ -4.6 nats per dim, so collapsing costs +4.6*T in free
energy — a barrier that vanishes only at T=0, where the energy_only
control indeed collapses (measured dim std 0.003). Finite but sufficient.

Patch baseline (VICReg-style): variance hinge + covariance whitening —
a diagonal, second-order surrogate for the same entropy requirement.
"""
import math

import torch


def gaussian_entropy(z, eps=1e-4):
    """Per-dim differential entropy of a batch Gaussian fit. z: (B,d)."""
    d = z.shape[1]
    mu = z.mean(dim=0, keepdim=True)
    zc = z - mu
    cov = (zc.t() @ zc) / max(z.shape[0] - 1, 1)
    cov = cov + eps * torch.eye(d, device=z.device, dtype=z.dtype)
    _, logdet = torch.linalg.slogdet(cov)
    return 0.5 * logdet / d  # per-dim nats


def vicreg_terms(z, sigma_target=1.0, lambda_var=10.0, lambda_cov=1.0):
    """The patch baseline: var hinge (anti scalar-collapse) + off-diag cov
    (anti dim-collapse). Together: a crude second-order entropy surrogate."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = lambda_var * torch.relu(sigma_target - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    B = z.shape[0]
    cov = (zc.t() @ zc) / max(B - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).mean() * lambda_cov
    return var_loss + cov_loss


def temperature(step, total_steps, t_max=1.0, t_min=0.05):
    """Cosine annealing: high T (explore) -> low T (crystallize)."""
    frac = min(step / max(total_steps - 1, 1), 1.0)
    cos = 0.5 * (1 + math.cos(frac * math.pi))
    return t_max - (t_max - t_min) * (1 - cos)
