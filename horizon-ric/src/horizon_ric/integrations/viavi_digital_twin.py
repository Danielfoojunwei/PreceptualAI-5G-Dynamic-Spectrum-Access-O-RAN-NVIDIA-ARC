"""VIAVI Pipeline 2 digital-twin driver for counterfactual rollouts.

This module ships M3 part 2 of the AI-RAN Alliance integration plan:
a counterfactual-rollout driver that runs against the VIAVI Pipeline 2
digital twin. Given a current ``state``, a chosen action and an
alternative action, it asks the twin "what would have happened if
we had emitted action B instead?" and returns the predicted KPIs for
both actions, the divergence from the actual observed KPIs (when
available) and a confidence score.

Two backends are supported:

  * **Real VIAVI Pipeline 2** — any object implementing the
    :class:`DigitalTwinBackend` Protocol; an ``async evaluate(state,
    action, n_steps)`` coroutine returning a KPI dict.
  * **Synthetic fallback** — a deterministic closed-form scenario used
    in dev / CI when the VIAVI bench is not available. The synthetic
    backend is *not* a substitute for the real twin; it exists so the
    counterfactual emit path stays exercised in unit tests and on
    laptops where the lab is not reachable.

Synthetic-backend formulae
--------------------------

Given ``state`` and ``action`` dicts (any keys missing default to 1.0),
the synthetic backend computes::

    prb       = float(action.get("prb_count", 1.0))
    snr_db    = float(state.get("snr_db", 1.0))
    tx_dbm    = float(action.get("tx_power_dbm", state.get("tx_power_dbm", 20.0)))
    n_ues     = float(state.get("n_ues", 1.0))

    throughput_mbps = prb * snr_db * 0.1 * (1.0 + 0.01 * n_steps)
    latency_ms      = max(1.0, 50.0 - prb * 0.5 + (tx_dbm - 20.0) * 0.1)
    energy_w        = 5.0 + max(0.0, tx_dbm) * 0.05 * prb / max(n_ues, 1.0)
    sla_risk        = 1.0 / (1.0 + throughput_mbps)

These rules are **not** physically calibrated — they only have to be
deterministic and monotonic enough that tests can assert "more PRBs ⇒
higher throughput" without being flaky.

All public surface is asyncio-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

__all__ = [
    "DigitalTwinBackend",
    "CounterfactualRollout",
    "ViaviDigitalTwin",
]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DigitalTwinBackend(Protocol):
    """Anything that takes ``(state, action)`` and returns predicted KPIs.

    The real implementation will be a thin async client around the
    VIAVI Pipeline 2 REST/gRPC surface; in tests we drop in a mock.

    Implementations MUST be coroutines (``async def evaluate``). The
    returned mapping is treated opaquely except that values must be
    finite floats so divergence arithmetic is well-defined.
    """

    async def evaluate(
        self, state: dict, action: dict, n_steps: int,
    ) -> dict: ...


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualRollout:
    """One ``(chosen, alternative)`` rollout pair through the digital twin."""

    state: dict
    action_chosen: dict
    action_alternative: dict
    predicted_kpis_chosen: dict
    predicted_kpis_alternative: dict
    divergence_from_actual: Optional[float]
    confidence: float
    n_steps: int
    backend_name: str


# ---------------------------------------------------------------------------
# Synthetic fallback backend
# ---------------------------------------------------------------------------


class _SyntheticBackend:
    """Deterministic in-process fallback used when no VIAVI bench is wired.

    See module docstring for the closed-form rules.
    """

    name = "synthetic"

    @staticmethod
    def _f(d: dict, key: str, default: float) -> float:
        v = d.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    async def evaluate(
        self, state: dict, action: dict, n_steps: int,
    ) -> dict:
        prb = self._f(action, "prb_count", 1.0)
        snr_db = self._f(state, "snr_db", 1.0)
        tx_dbm = self._f(
            action, "tx_power_dbm",
            self._f(state, "tx_power_dbm", 20.0),
        )
        n_ues = max(self._f(state, "n_ues", 1.0), 1.0)
        steps = max(int(n_steps), 1)

        throughput_mbps = prb * snr_db * 0.1 * (1.0 + 0.01 * steps)
        latency_ms = max(1.0, 50.0 - prb * 0.5 + (tx_dbm - 20.0) * 0.1)
        energy_w = 5.0 + max(0.0, tx_dbm) * 0.05 * prb / n_ues
        sla_risk = 1.0 / (1.0 + max(throughput_mbps, 0.0))
        return {
            "throughput_mbps": throughput_mbps,
            "latency_ms": latency_ms,
            "energy_w": energy_w,
            "sla_risk": sla_risk,
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class ViaviDigitalTwin:
    """Drive the VIAVI Pipeline 2 digital twin for counterfactual rollouts.

    The driver is intentionally thin: it normalises inputs, calls the
    backend twice (chosen + alternative), assembles a
    :class:`CounterfactualRollout`, and computes confidence + divergence.
    All scenario knowledge lives either in the backend or in the
    synthetic-formula module docstring.

    Parameters
    ----------
    backend
        A :class:`DigitalTwinBackend`. When ``None``, an internal
        deterministic synthetic backend is used (dev / CI mode).
    confidence_floor
        Minimum confidence even at very small ``n_steps``. Confidence
        grows monotonically with ``n_steps`` and is capped at 0.95
        (we never claim a twin rollout is ground truth).
    """

    def __init__(
        self,
        backend: Optional[DigitalTwinBackend] = None,
        confidence_floor: float = 0.10,
    ) -> None:
        if not (0.0 <= confidence_floor < 0.95):
            raise ValueError(
                "confidence_floor must lie in [0.0, 0.95); "
                f"got {confidence_floor!r}",
            )
        self._backend: DigitalTwinBackend
        self._backend_name: str
        if backend is None:
            self._backend = _SyntheticBackend()
            self._backend_name = "synthetic"
        else:
            self._backend = backend
            self._backend_name = "viavi_pipeline_2"
        self._confidence_floor = float(confidence_floor)

    # --- public api ----------------------------------------------------

    @property
    def backend_name(self) -> str:
        return self._backend_name

    async def rollout(
        self,
        state: dict,
        action_chosen: dict,
        action_alternative: dict,
        *,
        n_steps: int = 100,
        actual_kpis: Optional[dict] = None,
    ) -> CounterfactualRollout:
        """Replay ``(state, action_chosen)`` and ``(state, action_alt)``.

        Both rollouts run against the same ``state`` for the same number
        of ``n_steps``. The backend is responsible for any internal
        stochasticity; this driver itself is deterministic given a
        deterministic backend.
        """
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError(
                f"n_steps must be a positive int; got {n_steps!r}",
            )
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")
        if not isinstance(action_chosen, dict):
            raise TypeError("action_chosen must be a dict")
        if not isinstance(action_alternative, dict):
            raise TypeError("action_alternative must be a dict")

        predicted_chosen = await self._backend.evaluate(
            state, action_chosen, n_steps,
        )
        predicted_alt = await self._backend.evaluate(
            state, action_alternative, n_steps,
        )

        divergence: Optional[float]
        if actual_kpis is None:
            divergence = None
        else:
            divergence = self._divergence(predicted_chosen, actual_kpis)

        confidence = self._confidence(n_steps)

        return CounterfactualRollout(
            state=dict(state),
            action_chosen=dict(action_chosen),
            action_alternative=dict(action_alternative),
            predicted_kpis_chosen=dict(predicted_chosen),
            predicted_kpis_alternative=dict(predicted_alt),
            divergence_from_actual=divergence,
            confidence=confidence,
            n_steps=int(n_steps),
            backend_name=self._backend_name,
        )

    # --- helpers -------------------------------------------------------

    @staticmethod
    def _divergence(predicted: dict, actual: dict) -> float:
        """L1 divergence on the keys of ``actual``."""
        total = 0.0
        for k, actual_v in actual.items():
            pred_v = predicted.get(k)
            if pred_v is None:
                # Predicted didn't cover this KPI; treat as full-actual gap.
                total += abs(_safe_float(actual_v))
                continue
            total += abs(_safe_float(pred_v) - _safe_float(actual_v))
        return float(total)

    def _confidence(self, n_steps: int) -> float:
        """Monotone-increasing in n_steps, floored, capped at 0.95.

        Concretely: ``c = 1 − exp(−n_steps / 100)``, then clamped to
        ``[confidence_floor, 0.95]``. n_steps=100 → ≈0.63;
        n_steps=300 → ≈0.95 (cap).
        """
        import math

        raw = 1.0 - math.exp(-float(n_steps) / 100.0)
        return float(min(0.95, max(self._confidence_floor, raw)))


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0
