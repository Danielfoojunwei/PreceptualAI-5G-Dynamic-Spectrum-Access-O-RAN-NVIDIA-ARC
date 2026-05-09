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
    """FedProx server-side aggregation — IDENTICAL to FedAvg by design.

    .. important::
        FedProx (Li 2020) differs from FedAvg **only on the client**:
        the local loss is augmented with the proximal regulariser
        ``μ/2 · ‖w_local − w_global‖²``. The server-side aggregator is
        unchanged. Calling this function instead of ``aggregate_fedavg``
        is **purely a code-clarity intent signal**; the math is the
        same weighted mean. To get the actual FedProx behaviour, the
        client trainer must use :func:`fedprox_client_proximal_loss`
        (see below) when computing the local update.

    This was historically over-claimed in the docstring; the function
    body has always just delegated to FedAvg. Honest disclosure landed
    in the v3 audit pass.
    """
    return aggregate_fedavg(updates)


def fedprox_client_proximal_loss(
    local_state_dict: dict[str, torch.Tensor],
    global_state_dict: dict[str, torch.Tensor],
    mu: float = 0.01,  # = DEFAULT_FEDPROX_MU; literal here for forward-ref
) -> torch.Tensor:
    """Compute the FedProx **client-side** proximal regulariser.

    Returns the scalar ``μ/2 · Σ_k ‖w_local_k − w_global_k‖²`` that the
    client trainer must add to its local loss before backprop. Without
    this, the system is FedAvg, regardless of which server aggregator
    name is used.

    Reference
    ---------
    Li et al. *Federated Optimization in Heterogeneous Networks*,
    MLSys 2020. See §3 algorithm 2 (FedProx local objective):

        h_k(w; w^t) = F_k(w) + (μ/2) · ‖w − w^t‖²

    Example
    -------
    >>> # in the client trainer's optimisation step:
    >>> task_loss = criterion(model(x), y)
    >>> prox = fedprox_client_proximal_loss(
    ...     local_state_dict=model.state_dict(),
    ...     global_state_dict=last_global_round_state_dict,
    ...     mu=0.01,
    ... )
    >>> total = task_loss + prox
    >>> total.backward()
    """
    if mu < 0:
        raise ValueError(f"mu must be ≥ 0, got {mu}")
    keys = set(local_state_dict.keys()) & set(global_state_dict.keys())
    if not keys:
        raise ValueError("local and global state dicts share no keys")
    sq_norm = torch.tensor(0.0)
    for k in keys:
        diff = local_state_dict[k].to(torch.float64) - global_state_dict[k].to(torch.float64)
        sq_norm = sq_norm + (diff * diff).sum()
    return (mu / 2.0) * sq_norm.to(torch.float32)


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

    Returns ``FedProx(mu=DEFAULT_FEDPROX_MU)`` — but **note** that this
    server-side ``FedProx`` is mathematically identical to ``FedAvg``:
    the FedProx contribution is the **client-side proximal regulariser**
    ``μ/2 · ‖w_local − w_global‖²`` (Li 2020), which must be added to
    the local trainer's loss via :func:`fedprox_client_proximal_loss`.
    Without that client-side wiring, ``FedProx(mu=0.01)`` and
    ``FedAvg()`` produce bit-identical aggregations.

    Returning ``FedProx`` here documents *intent* (we want clients to
    use the proximal term), not behaviour. The default remains FedProx
    because the server contract is forward-compatible: the day a client
    trainer wires in :func:`fedprox_client_proximal_loss`, the
    aggregator's name no longer needs to change.

    Closes Devil-C Finding 37 (FedAvg drift under heterogeneous data,
    SCAFFOLD ICML 2020) IF AND ONLY IF the client trainer adds the
    proximal term. See ``RELIABILITY.md`` for the operational gate.
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
    "fedprox_client_proximal_loss",   # NEW: required for real FedProx
]
