"""DPoD activation **policy** — deterministic rule-based gate (NOT itself AI).

> **Honest scope.** This module contains **zero neural network weights**
> and zero learned parameters. It is a deterministic if/else state
> machine that decides *when to enable* the AI-RAN Alliance HybridDeepRx
> digital post-distortion (DPoD) backend at the gNB receiver. The
> backend is the AI; this file is the regulator-readable arbiter that
> routes traffic to it. Class is named `DPoDActivation` for backward
> compatibility but is more accurately characterised as
> ``DPoDActivationPolicy``. The audit-chain envelope it emits is a
> regulator-replayable trace of the rule firing — not a model
> prediction.

Decides whether to enable the AI-RAN Alliance HybridDeepRx digital
post-distortion module on the gNB receiver, based on PA backoff, EVM
target, and the SLA risk head's 30-second prediction.

Decision rule (deterministic, regulator-readable)
-------------------------------------------------
Enable DPoD when ALL of:
    pa_backoff_db ≤ 5.5 dB         (high-efficiency PA regime where
                                    DPoD's nonlinear mitigation pays off)
    AND sla_risk_30s_p99 < 0.10    (no in-flight SLA breach risk)
    AND evm_target ≥ 7.5 %         (operating EVM gives DPoD enough
                                    headroom to converge)

Otherwise keep DPoD OFF.

Each ON↔OFF transition is logged via a sink callable so the audit chain
can persist it; the in-memory state machine ensures we never emit
duplicate transitions.

Reference
---------
~/.claude/plans/preceptualai-airan-alliance-integration.md §3 M1.
AI-RAN Alliance / Nokia Bell Labs / R&S — HybridDeepRx, MWC 2026.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal, Optional

DPoDState = Literal["off", "on"]


@dataclass(frozen=True)
class DPoDInputs:
    """Inputs to the DPoD activation rule."""

    pa_backoff_db: float
    sla_risk_30s_p99: float  # ∈ [0, 1]
    evm_target_pct: float


@dataclass(frozen=True)
class DPoDTransition:
    """One audited DPoD ON↔OFF transition."""

    timestamp: datetime
    from_state: DPoDState
    to_state: DPoDState
    pinned_seed: int
    inputs: dict
    reason_human: str
    reason_machine: dict
    predicted_evm_gain_pp: float       # percentage points
    predicted_pa_efficiency_lift_pct: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "from": self.from_state,
            "to": self.to_state,
            "pinned_seed": self.pinned_seed,
            "inputs": self.inputs,
            "reason_human": self.reason_human,
            "reason_machine": self.reason_machine,
            "predicted_evm_gain_pp": self.predicted_evm_gain_pp,
            "predicted_pa_efficiency_lift_pct": self.predicted_pa_efficiency_lift_pct,
        }

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def _predicted_gain(inputs: DPoDInputs) -> tuple[float, float]:
    """Predicted EVM gain (percentage points) + PA efficiency lift (%).

    Numbers are based on the AI-RAN Alliance HybridDeepRx published
    operating point: EVM 8.3 % achievable at 5.5 dB PA backoff.
    """
    if inputs.pa_backoff_db <= 5.5:
        evm_gain_pp = max(0.0, inputs.evm_target_pct - 8.3)
        pa_eff_pct = max(0.0, (5.5 - inputs.pa_backoff_db) * 5.0)  # ~5% per dB
    else:
        evm_gain_pp = 0.0
        pa_eff_pct = 0.0
    return evm_gain_pp, pa_eff_pct


class DPoDActivation:
    """Stateful DPoD on/off decider with audit-chain transition log.

    Thread-safe; the state machine guards against duplicate transitions
    so the audit chain only carries genuine ON↔OFF events, not every
    decision tick that happens to keep the same state.
    """

    PA_BACKOFF_THRESHOLD_DB = 5.5
    SLA_RISK_THRESHOLD = 0.10
    EVM_TARGET_THRESHOLD_PCT = 7.5

    def __init__(
        self,
        *,
        initial_state: DPoDState = "off",
        sink: Optional[Callable[[DPoDTransition], None]] = None,
    ):
        self._state: DPoDState = initial_state
        self._sink = sink
        self._lock = threading.RLock()
        self._history: list[DPoDTransition] = []

    def state(self) -> DPoDState:
        with self._lock:
            return self._state

    def history(self) -> list[DPoDTransition]:
        with self._lock:
            return list(self._history)

    def evaluate(self, inputs: DPoDInputs, *, seed: int = 0) -> DPoDState:
        """Compute desired state from inputs alone (no transition emitted)."""
        if (
            inputs.pa_backoff_db <= self.PA_BACKOFF_THRESHOLD_DB
            and inputs.sla_risk_30s_p99 < self.SLA_RISK_THRESHOLD
            and inputs.evm_target_pct >= self.EVM_TARGET_THRESHOLD_PCT
        ):
            return "on"
        return "off"

    def step(
        self,
        inputs: DPoDInputs,
        *,
        seed: int = 0,
    ) -> Optional[DPoDTransition]:
        """Evaluate inputs; emit a transition if state changed.

        Returns the emitted ``DPoDTransition`` or ``None`` if no change.
        """
        desired = self.evaluate(inputs, seed=seed)
        with self._lock:
            if desired == self._state:
                return None
            evm_gain_pp, pa_eff_pct = _predicted_gain(inputs)
            t = DPoDTransition(
                timestamp=datetime.now(timezone.utc),
                from_state=self._state,
                to_state=desired,
                pinned_seed=int(seed),
                inputs={
                    "pa_backoff_db": inputs.pa_backoff_db,
                    "sla_risk_30s_p99": inputs.sla_risk_30s_p99,
                    "evm_target_pct": inputs.evm_target_pct,
                },
                reason_human=(
                    f"DPoD {self._state}→{desired}: "
                    f"PA backoff {inputs.pa_backoff_db:.1f} dB "
                    f"({'≤' if inputs.pa_backoff_db <= self.PA_BACKOFF_THRESHOLD_DB else '>'} "
                    f"{self.PA_BACKOFF_THRESHOLD_DB} dB), "
                    f"SLA risk {inputs.sla_risk_30s_p99:.3f} "
                    f"({'<' if inputs.sla_risk_30s_p99 < self.SLA_RISK_THRESHOLD else '≥'} "
                    f"{self.SLA_RISK_THRESHOLD}), "
                    f"EVM target {inputs.evm_target_pct:.1f}%."
                ),
                reason_machine={
                    "primary_cause": "dpod_envelope_check",
                    "pa_backoff_threshold_db": self.PA_BACKOFF_THRESHOLD_DB,
                    "sla_risk_threshold": self.SLA_RISK_THRESHOLD,
                    "evm_target_threshold_pct": self.EVM_TARGET_THRESHOLD_PCT,
                },
                predicted_evm_gain_pp=evm_gain_pp,
                predicted_pa_efficiency_lift_pct=pa_eff_pct,
            )
            # Emit to sink BEFORE flipping state — sink failure rolls back.
            if self._sink is not None:
                self._sink(t)
            self._history.append(t)
            self._state = desired
            return t


# Honest-name alias. `DPoDActivationPolicy` is the canonical name going
# forward; `DPoDActivation` is preserved for backwards compatibility
# (test surface, M1 audit chain).
DPoDActivationPolicy = DPoDActivation


__all__ = [
    "DPoDActivation",            # legacy
    "DPoDActivationPolicy",      # canonical (rule-based, not AI)
    "DPoDInputs",
    "DPoDState",
    "DPoDTransition",
]
