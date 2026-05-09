"""FedAvg + FedProx aggregation primitives (weights-only).

DEFAULT aggregator (closes Devil-C #37).
    The default aggregator returned by :func:`default_aggregator` is
    **FedProx with μ=0.01**, NOT vanilla FedAvg. Karimireddy et al.
    (SCAFFOLD, ICML 2020) and Li et al. (FedProx, MLSys 2020) prove
    FedAvg has a non-vanishing drift term under heterogeneous client
    distributions — exactly the regime UHCI deploys in (per-tenant
    workloads, edge devices with skewed local data). FedProx adds a
    proximal term ``μ/2 · ‖w_local − w_global‖²`` to each client's
    local loss, which (a) keeps the local update close to the global
    weights and (b) tolerates partial / straggler participation. The
    server-side aggregation step is the same weighted mean as FedAvg —
    the difference is what the client objective looks like; we surface
    a distinct ``FedProx(mu=0.01)`` object so call-sites declare intent
    AND so we can extend the server-side step (e.g. drop very-divergent
    updates) without touching the FedAvg path.

    μ=0.01 is the value Li-2020 § 5.2 reports as robust across the
    paper's benchmark suite and is the empirical sweet spot for
    moderately-heterogeneous data: small enough that the regulariser
    doesn't dominate the local optimisation, large enough to prevent
    drift on non-IID clients.

References:
    McMahan et al. *Communication-Efficient Learning of Deep Networks from
        Decentralized Data*, AISTATS 2017 — FedAvg.
    Li et al. *Federated Optimization in Heterogeneous Networks*, MLSys 2020 —
        FedProx (adds μ‖w_local − w_global‖² proximal term locally; the
        aggregator itself is the same weighted mean as FedAvg).
    Karimireddy et al. *SCAFFOLD: Stochastic Controlled Averaging*,
        ICML 2020 — proves FedAvg's drift bias under heterogeneity.

Both aggregators run on PyTorch state-dicts. Per project memory the bytes
on the wire MUST be weight tensors (not raw KPMs); the aggregator never
sees telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClientUpdate:
    """One client's contribution to a federated round.

    Attributes:
        client_id: stable identifier (also the federated "slot" name).
        state_dict: tensor parameters keyed by name (PyTorch layout).
        sample_count: number of training samples behind this update —
            used as the FedAvg weighting term.
        proximal_loss: optional FedProx local proximal term value, used
            for diagnostics; not required for aggregation.
    """

    client_id: str
    state_dict: dict[str, torch.Tensor]
    sample_count: int
    proximal_loss: float | None = None


def _validate_updates(updates: list[ClientUpdate]) -> None:
    if not updates:
        raise ValueError("at least one ClientUpdate is required")
    keys0 = set(updates[0].state_dict.keys())
    for u in updates:
        if set(u.state_dict.keys()) != keys0:
            raise ValueError(
                f"client {u.client_id} has different parameter keys than "
                f"client {updates[0].client_id}"
            )
        if u.sample_count <= 0:
            raise ValueError(
                f"client {u.client_id} has non-positive sample_count "
                f"{u.sample_count}"
            )


def aggregate_fedavg(
    updates: list[ClientUpdate],
) -> dict[str, torch.Tensor]:
    """McMahan-2017 FedAvg: sample-count-weighted mean of client weights.

        w_global = Σ_i (n_i / Σ_j n_j) · w_i

    Returns a fresh state_dict (does not mutate inputs).
    """
    _validate_updates(updates)
    total = sum(u.sample_count for u in updates)
    out: dict[str, torch.Tensor] = {}
    for key, tensor in updates[0].state_dict.items():
        # Aggregate in float64 to limit accumulation error then cast back.
        ref_dtype = tensor.dtype
        acc = torch.zeros_like(tensor, dtype=torch.float64)
        for u in updates:
            t = u.state_dict[key]
            if t.shape != tensor.shape:
                raise ValueError(
                    f"client {u.client_id} key {key} has shape {tuple(t.shape)} "
                    f"vs reference {tuple(tensor.shape)}"
                )
            acc.add_(t.to(torch.float64) * (u.sample_count / total))
        out[key] = acc.to(ref_dtype)
    return out


def aggregate_fedprox(
    updates: list[ClientUpdate],
) -> dict[str, torch.Tensor]:
    """FedProx server-side aggregation.

    FedProx differs from FedAvg only on the *client* (proximal term added
    to the local loss). The server-side aggregation is the same weighted
    mean. We expose it under a separate name so call-sites declare intent
    and so we can add Li-2020 re-weightings (e.g. drop very-divergent
    updates) here in future without touching the FedAvg path.
    """
    return aggregate_fedavg(updates)


# ─── Functor-style aggregators (object form) ────────────────────────────


class _BaseAggregator:
    """Common interface so callers can swap FedAvg ↔ FedProx via config."""

    def aggregate(
        self, updates: list[ClientUpdate]
    ) -> dict[str, torch.Tensor]:  # pragma: no cover — overridden
        raise NotImplementedError


class FedAvg(_BaseAggregator):
    """McMahan FedAvg aggregator. Stateless."""

    def aggregate(
        self, updates: list[ClientUpdate]
    ) -> dict[str, torch.Tensor]:
        return aggregate_fedavg(updates)


class FedProx(_BaseAggregator):
    """FedProx aggregator. Stateless on the server side."""

    def __init__(self, mu: float = 0.01):
        if mu < 0:
            raise ValueError(f"mu must be ≥ 0, got {mu}")
        self.mu = mu

    def aggregate(
        self, updates: list[ClientUpdate]
    ) -> dict[str, torch.Tensor]:
        return aggregate_fedprox(updates)


# ─── Helper: apply a state_dict to a torch.nn.Module in-place ────────────


def apply_state_dict(
    model: "torch.nn.Module",  # noqa: F821 — torch.nn forward ref
    state: dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Convenience wrapper around `Module.load_state_dict`."""
    model.load_state_dict(state, strict=strict)


DEFAULT_FEDPROX_MU: float = 0.01
"""Default FedProx proximal coefficient μ. See module docstring §"DEFAULT
aggregator" for justification (Devil-C #37)."""


def default_aggregator() -> "_BaseAggregator":
    """The system-wide default federated aggregator.

    Returns ``FedProx(mu=DEFAULT_FEDPROX_MU)`` — NOT FedAvg.
    Closes Devil-C Finding 37: FedAvg under heterogeneous client data
    has known drift (SCAFFOLD ICML 2020); FedProx with μ=0.01 is the
    robust default. See module docstring + RELIABILITY.md for the rationale.
    """
    return FedProx(mu=DEFAULT_FEDPROX_MU)


__all__ = [
    "ClientUpdate",
    "DEFAULT_FEDPROX_MU",
    "FedAvg",
    "FedProx",
    "aggregate_fedavg",
    "aggregate_fedprox",
    "apply_state_dict",
    "default_aggregator",
]
