"""Bandwidth-saving compression for FL updates.

Three primitives, each independently composable:

    top_k_sparsify    keep the largest |Δw| coordinates, zero the rest.
    sign_sgd_compress map each coordinate to its sign × magnitude mean.
    quantize_int8     symmetric per-tensor int8 quantization.

These reduce the per-round payload from O(50M·4B = 200MB) to O(few MB),
which is what the AI-RAN compute-physics brief identified as the wall on
real-world cellular backhaul.

References:
    Lin et al. *Deep Gradient Compression*, ICLR 2018 (top-k).
    Bernstein et al. *signSGD*, ICML 2018.
    Konečný et al. *Federated Learning: Strategies for Improving
        Communication Efficiency*, NeurIPS 2016 Workshop.
"""

from __future__ import annotations

import torch


def top_k_sparsify(
    delta: dict[str, torch.Tensor],
    sparsity: float = 0.99,
) -> dict[str, torch.Tensor]:
    """Zero all but the top |delta| coordinates.

    Args:
        delta: dict of weight-delta tensors (w_local − w_global).
        sparsity: fraction of coordinates to ZERO (default 0.99 → keep 1%).

    Returns:
        New dict; original is not modified.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    out: dict[str, torch.Tensor] = {}
    for k, t in delta.items():
        flat = t.flatten()
        n = flat.numel()
        keep = max(int(round(n * (1.0 - sparsity))), 1)
        if keep >= n:
            out[k] = t.clone()
            continue
        # threshold = magnitude of the keep-th largest element
        threshold = torch.kthvalue(flat.abs(), n - keep + 1).values
        mask = (t.abs() >= threshold).to(t.dtype)
        out[k] = t * mask
    return out


def sign_sgd_compress(
    delta: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """signSGD compression: each coordinate becomes sign × per-tensor mean |Δ|.

    1-bit-per-coordinate plus one float scalar per tensor.
    """
    out: dict[str, torch.Tensor] = {}
    for k, t in delta.items():
        magnitude = t.abs().mean()
        out[k] = magnitude * torch.sign(t)
    return out


def quantize_int8(
    delta: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Symmetric per-tensor int8 quantization.

    Returns (q_dict, scales) — caller transmits both. Reconstruction:
        delta_k_recon = q_dict[k].to(float) * scales[k]
    """
    q_out: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    for k, t in delta.items():
        absmax = t.abs().max().item()
        scale = absmax / 127.0 if absmax > 0 else 1.0
        q_out[k] = (t / scale).round().clamp(-127, 127).to(torch.int8)
        scales[k] = scale
    return q_out, scales


__all__ = ["quantize_int8", "sign_sgd_compress", "top_k_sparsify"]
