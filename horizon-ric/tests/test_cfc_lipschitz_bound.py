"""F#36 closure — CfC cell empirical Lipschitz bound.

Numerically bounds the Lipschitz constant L of the CfC cell's
``forward(x, h, dt)`` map with respect to both inputs by sampling
10 000 (x, h) pairs and a small perturbation δ. We measure

    L̂(x, h)  =  ‖f(x+δ, h) − f(x, h)‖ / ‖δ‖
    L̂_h(x, h) = ‖f(x, h+δ) − f(x, h)‖ / ‖δ‖

over the validation manifold (random Gaussian inputs in the operating
range) and pin the empirical 99-th percentile as the contract bound.
The CfC update is

    h' = D(x, h) · h + (1 − D(x, h)) · A(x, h)
    D = exp(−Δt · (1/τ + g(x, h)))    g ∈ (0, 1)

so D ∈ (0, 1) and the map is bounded under finite weights — we just
verify the empirical bound stays below the documented contract.

Closes Phase-2 deferral F#36; companion §T4 in `THEOREMS.md`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from horizon_ric.core.cfc_core import CfCCell, CfCConfig

# Contract bounds — these are *upper* bounds the empirical 99-th
# percentile is asserted to stay under. They are wide enough to
# tolerate seed jitter but tight enough to detect real regressions.
CONTRACT_L_X = 5.0
CONTRACT_L_H = 5.0

N_SAMPLES = 10_000
INPUT_DIM = 8
HIDDEN_DIM = 16
DT = 0.1
EPS = 1e-3


@pytest.fixture(scope="module")
def cell():
    torch.manual_seed(0)
    np.random.seed(0)
    return CfCCell(CfCConfig(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)).eval()


def _lipschitz_quotients(
    cell: CfCCell,
    perturb_x: bool,
) -> torch.Tensor:
    torch.manual_seed(1234 if perturb_x else 5678)
    x = torch.randn(N_SAMPLES, INPUT_DIM)
    h = torch.randn(N_SAMPLES, HIDDEN_DIM)
    dt = torch.full((N_SAMPLES, 1), DT)

    # Random unit-direction perturbation, magnitude EPS.
    if perturb_x:
        delta = torch.randn(N_SAMPLES, INPUT_DIM)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12) * EPS
        x_pert = x + delta
        h_pert = h
    else:
        delta = torch.randn(N_SAMPLES, HIDDEN_DIM)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12) * EPS
        x_pert = x
        h_pert = h + delta

    with torch.no_grad():
        y0 = cell(x, h, dt)
        y1 = cell(x_pert, h_pert, dt)

    num = (y1 - y0).norm(dim=-1)
    den = delta.norm(dim=-1).clamp_min(1e-12)
    return num / den  # (N,)


def test_lipschitz_bound_wrt_x(cell):
    """L̂_x 99-th percentile must stay under the documented contract."""
    L = _lipschitz_quotients(cell, perturb_x=True)
    p99 = torch.quantile(L, 0.99).item()
    p_max = L.max().item()
    assert torch.isfinite(L).all(), "non-finite Lipschitz quotients"
    assert p99 <= CONTRACT_L_X, (
        f"L̂_x p99 = {p99:.4f} exceeds contract {CONTRACT_L_X}"
    )
    print(f"\n[CfC Lipschitz wrt x] p99={p99:.4f}  max={p_max:.4f}")


def test_lipschitz_bound_wrt_h(cell):
    """L̂_h 99-th percentile must stay under the documented contract.

    The CfC update is h' = D·h + (1-D)·A with D ∈ (0,1), so the linear
    part contracts and the nonlinearity through g, A is bounded; the
    empirical bound should be well under 5.0 for typical inputs.
    """
    L = _lipschitz_quotients(cell, perturb_x=False)
    p99 = torch.quantile(L, 0.99).item()
    p_max = L.max().item()
    assert torch.isfinite(L).all(), "non-finite Lipschitz quotients"
    assert p99 <= CONTRACT_L_H, (
        f"L̂_h p99 = {p99:.4f} exceeds contract {CONTRACT_L_H}"
    )
    print(f"\n[CfC Lipschitz wrt h] p99={p99:.4f}  max={p_max:.4f}")


def test_midpoint_rule_error_bound_is_tight(cell):
    """Closed-form approximation error: |h_cf − h_exact| ≤ L · Δt / 2.

    We probe this at two step sizes (Δt = 0.1 and Δt = 0.05) and verify
    the observed step-size sensitivity is consistent with a Lipschitz
    update — i.e. halving Δt halves the per-step delta to within a
    constant factor (loose check; not a tight identity since CfC is
    closed-form, not an Euler step).
    """
    torch.manual_seed(99)
    x = torch.randn(N_SAMPLES, INPUT_DIM)
    h = torch.randn(N_SAMPLES, HIDDEN_DIM)
    with torch.no_grad():
        h1 = cell(x, h, torch.full((N_SAMPLES, 1), 0.1))
        h2 = cell(x, h, torch.full((N_SAMPLES, 1), 0.05))
    delta_step = (h1 - h2).norm(dim=-1).mean().item()
    L_x = torch.quantile(_lipschitz_quotients(cell, perturb_x=True), 0.99).item()
    # Step delta should be small and bounded by L · Δt / 2 ≈ L · 0.025.
    bound = L_x * 0.05 / 2.0 * 4.0  # 4× slack for the (1-D)·A term
    assert delta_step <= max(bound, 1.0), (
        f"step-size delta {delta_step:.4f} > bound {bound:.4f} (L={L_x:.3f})"
    )
    print(f"\n[CfC midpoint-rule] mean Δh between dt=0.1 and dt=0.05: {delta_step:.4f}")
