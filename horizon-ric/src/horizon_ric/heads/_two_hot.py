"""Two-hot symlog regression utilities (DreamerV3 trick, paradigm M13).

Wireless KPIs span many orders of magnitude (latency µs–s, throughput
kbps–Gbps). Plain MSE on raw scalars is unstable; two-hot regression
over a transformed bin grid is robust.

Bins are placed in *bin-space*. The caller decides whether bin-space is
raw, symlog, or logit:

    apply_symlog=True   bin grid covers symlog(target). Use for raw KPIs
                        whose range spans many decades. Round-trip:
                            encode(t) = two_hot(symlog(t))
                            decode(p) = symexp(sum(p * bins))
    apply_symlog=False  bin grid IS the target space. Use for logits,
                        bounded values, or any case where you don't want
                        the warp. Round-trip:
                            encode(t) = two_hot(t)
                            decode(p) = sum(p * bins)

Both modes support arbitrary leading dimensions, so a (B, T) sequence
input encodes to (B, T, num_bins) and decodes back to (B, T).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Reversible symmetric-log transform: sign(x) * log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def make_bins(num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """Bin centers spanning [low, high] in bin-space (caller chooses meaning)."""
    if num_bins < 3:
        raise ValueError(f"num_bins must be ≥ 3, got {num_bins}")
    if not (high > low):
        raise ValueError(f"high ({high}) must be > low ({low})")
    return torch.linspace(low, high, num_bins)


def two_hot(
    target: torch.Tensor,
    bins: torch.Tensor,
    apply_symlog: bool = True,
) -> torch.Tensor:
    """Two-hot encode targets of shape (...) into (..., num_bins).

    Args:
        target: tensor of any shape (...).
        bins: monotonically increasing 1-D tensor of length B.
        apply_symlog: if True, target is symlog-transformed before bucketization.

    Returns:
        Distribution of shape (..., len(bins)) summing to 1 along the last axis.
    """
    if bins.ndim != 1:
        raise ValueError(f"bins must be 1-D, got shape {tuple(bins.shape)}")

    target_s = symlog(target) if apply_symlog else target
    target_s = target_s.clamp(bins[0].item(), bins[-1].item())

    # bucketize with right=False: returns insertion index in [1, len(bins)] for
    # values inside the range. Subtracting 1 gives left-bin index in [0, len-2].
    idx = torch.bucketize(target_s, bins) - 1
    idx = idx.clamp(0, len(bins) - 2)

    left = bins[idx]
    right = bins[idx + 1]
    width = (right - left).clamp_min(1e-12)
    w_right = ((target_s - left) / width).clamp(0.0, 1.0)
    w_left = 1.0 - w_right

    out = torch.zeros(*target.shape, len(bins), device=target.device, dtype=target.dtype)
    out.scatter_(-1, idx.unsqueeze(-1), w_left.unsqueeze(-1))
    out.scatter_(-1, (idx + 1).unsqueeze(-1), w_right.unsqueeze(-1))
    return out


def two_hot_decode(
    logits: torch.Tensor,
    bins: torch.Tensor,
    apply_symlog: bool = True,
) -> torch.Tensor:
    """Decode (..., num_bins) logits to (...) scalars.

    Args:
        logits: tensor of shape (..., num_bins).
        bins: monotonically increasing 1-D tensor of length num_bins.
        apply_symlog: if True, decode result is passed through symexp (inverse
            of the symlog applied at encode time). Set to False when bins live
            directly in target space (e.g. logits).
    """
    probs = F.softmax(logits, dim=-1)
    val = (probs * bins).sum(dim=-1)
    return symexp(val) if apply_symlog else val


def two_hot_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bins: torch.Tensor,
    apply_symlog: bool = True,
) -> torch.Tensor:
    """Cross-entropy between predicted distribution and two-hot target.

    Args:
        logits: (..., num_bins) predicted unnormalised log-probabilities.
        target: (...) raw scalar targets (in raw or bin space; see apply_symlog).
        bins: 1-D bin grid.
        apply_symlog: encode-side symlog flag; must match decode-side.

    Returns:
        Mean cross-entropy across all leading dims.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    target_dist = two_hot(target, bins, apply_symlog=apply_symlog)
    return -(target_dist * log_probs).sum(dim=-1).mean()
