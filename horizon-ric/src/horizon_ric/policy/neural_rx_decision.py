"""Neural-RX vs LMMSE **arbiter policy** — deterministic rule (NOT itself AI).

> **Honest scope.** This module contains **zero neural network weights**
> and **zero learned parameters**. It is a deterministic if/else gate
> that decides *whether to route a UE's PUSCH* through the AI-RAN
> Alliance HybridDeepRx neural-receiver backend or the classical LMMSE
> path. The neural receiver itself lives downstream (in Aerial / cuBB /
> the AI-RAN Alliance reference implementation); this file is the
> regulator-readable arbiter that fires the routing decision and emits
> the audit envelope. The TBLER curve used in the envelope is a 2-point
> linear interpolation calibrated to the AI-RAN Alliance HybridDeepRx
> published operating point (Nokia + R&S, MWC 2026) — it is NOT a
> learned predictor. Class is canonically named ``NeuralRxArbiterPolicy``;
> ``NeuralRxDecision`` is the backwards-compatible alias.

Per the PreceptualAI AI-RAN integration plan §3 M1, this module emits a
deterministic, regulator-readable choice between neural-RX (learned
demapper, ~0.3-1.0 dB BLER gain in low-mobility / mid-SNR regimes) and
classical LMMSE+turbo decoding. The decision rule is encoded as a
visible if/else so a regulator can replay it without weights.

Decision rule
-------------
Use neural-RX when ALL of:
    * PA backoff <= 6 dB        (transmit IQ is clean enough that the
                                 learned demapper's distribution match
                                 helps rather than fits non-linear noise)
    * SNR        >= 8 dB        (below this, classical LMMSE wins)
    * mobility != "high"        (high-Doppler regimes have not been
                                 covered by the training distribution
                                 and trigger fallback per O-RAN
                                 WG2 AI/ML lifecycle)

Otherwise fall back to LMMSE.

Each decision emits a counterfactual envelope with the standard 5
fields (`chosen`, `rejected_alternatives`, `reason_machine`,
`reason_human`, `pinned_seed`) and audit-chains back to
`evidence/store.py` via the existing `DecisionRecord` schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from horizon_ric.evidence.explanation import generate_human_explanation
from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
    RejectedAlternative,
    RejectionReasonMachine,
)

MobilityClass = Literal["low", "med", "high"]
RxKind = Literal["neural", "lmmse"]

# Card ref for 3GPP TS 28.105 (AI/ML Management) provenance — the
# regulator pulls the full card from the SMO catalogue keyed on this id.
NEURAL_RX_MODEL_CARD = "preceptualai.neural_rx.v1"

# Predicted TBLER tables (illustrative, calibrated against 3GPP TR 38.901
# UMa channels). Real production values come from the model card; these
# are the slide-claimed envelope values pinned for replay.
_NEURAL_RX_TBLER_GAIN_DB = 0.7  # neural-RX is ~0.7 dB better when in-domain.
_LMMSE_BASELINE_TBLER_AT_SNR8 = 0.10
_LMMSE_BASELINE_TBLER_AT_SNR15 = 0.005


@dataclass(frozen=True)
class NeuralRxState:
    """Inputs to the neural-RX vs LMMSE decision.

    pa_backoff_db: PA output back-off in dB (0 = saturating; high = clean).
    snr_db:        Per-RE post-equaliser SNR in dB.
    mobility:      "low" / "med" / "high" — UE Doppler-bin class.
    mcs:           3GPP TS 38.214 MCS index (0–28).
    """

    pa_backoff_db: float
    snr_db: float
    mobility: MobilityClass
    mcs: int


class NeuralRxEnvelope(BaseModel):
    """Counterfactual envelope for one neural-RX/LMMSE decision.

    Carries the 5-field counterfactual contract:
        chosen, rejected_alternatives, reason_machine, reason_human,
        pinned_seed.

    Plus AI-PHY-specific provenance: predicted TBLER for both branches
    and the model card pointer (TS 28.105 §6.2.2.2).
    """

    chosen: RxKind
    rejected_alternatives: list[dict[str, Any]] = Field(default_factory=list)
    reason_machine: dict[str, Any]
    reason_human: str
    pinned_seed: int
    predicted_tbler_neural: float
    predicted_tbler_lmmse: float
    model_card_ref: str
    state_hash: str

    def envelope_sha256(self) -> str:
        """Stable hash over the envelope's content (used for replay tests)."""
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_hash(state: NeuralRxState) -> str:
    payload = json.dumps(
        {
            "pa_backoff_db": round(state.pa_backoff_db, 6),
            "snr_db": round(state.snr_db, 6),
            "mobility": state.mobility,
            "mcs": int(state.mcs),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _predict_lmmse_tbler(snr_db: float, mcs: int) -> float:
    """Linear-interp TBLER curve in [0, 1] keyed off (snr_db, mcs).

    Deliberately simple and deterministic — regulator replay only needs
    the same number out for the same number in.
    """
    # Interpolate between the two pinned points; clamp to [1e-4, 0.5].
    if snr_db <= 8.0:
        base = _LMMSE_BASELINE_TBLER_AT_SNR8
    elif snr_db >= 15.0:
        base = _LMMSE_BASELINE_TBLER_AT_SNR15
    else:
        t = (snr_db - 8.0) / 7.0
        base = (
            _LMMSE_BASELINE_TBLER_AT_SNR8
            + t * (_LMMSE_BASELINE_TBLER_AT_SNR15 - _LMMSE_BASELINE_TBLER_AT_SNR8)
        )
    # Higher MCS → higher TBLER (steeper curve).
    mcs_penalty = max(0.0, (mcs - 14) * 0.005)
    return float(min(0.5, max(1e-4, base + mcs_penalty)))


def _predict_neural_tbler(snr_db: float, mcs: int, mobility: MobilityClass) -> float:
    """Neural-RX is `_NEURAL_RX_TBLER_GAIN_DB` dB better when in-domain.

    Out-of-domain (high mobility) we conservatively report parity with
    LMMSE so the regulator sees no false claim of gain.
    """
    base = _predict_lmmse_tbler(snr_db + _NEURAL_RX_TBLER_GAIN_DB, mcs)
    if mobility == "high":
        return _predict_lmmse_tbler(snr_db, mcs)
    return base


class NeuralRxDecision:
    """Deterministic neural-RX vs LMMSE decision with audit envelope.

    The decision rule is fully deterministic: same `NeuralRxState` →
    same `(chosen_rx, envelope)`. The pinned seed is carried for
    regulator replay (Devil-A/C #9, #26) even though the rule is
    weight-free — downstream RX-stage RNG (e.g. dithering) does
    consume it.
    """

    def __init__(
        self,
        rapp_instance_id: str = "preceptualai-airan-m1",
        random_seed: int = 0,
        model_card_ref: str = NEURAL_RX_MODEL_CARD,
    ):
        self._rapp_instance_id = rapp_instance_id
        self._random_seed = int(random_seed)
        self._model_card_ref = model_card_ref

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def decide(self, state: NeuralRxState) -> tuple[RxKind, dict[str, Any]]:
        in_domain = (
            state.pa_backoff_db <= 6.0
            and state.snr_db >= 8.0
            and state.mobility != "high"
        )
        chosen: RxKind = "neural" if in_domain else "lmmse"

        tbler_neural = _predict_neural_tbler(state.snr_db, state.mcs, state.mobility)
        tbler_lmmse = _predict_lmmse_tbler(state.snr_db, state.mcs)

        # The rejected alternative + the cause that disqualified it.
        if chosen == "neural":
            rejected_kind: RxKind = "lmmse"
            cause = "energy_cost"
            reason_metric = "tbler_residual"
            predicted_value = tbler_lmmse
            threshold = tbler_neural
            reason_machine = RejectionReasonMachine(
                primary_cause=cause,
                primary_metric=reason_metric,
                predicted_value=float(predicted_value),
                threshold=float(threshold),
                horizon="30s",
            )
        else:
            rejected_kind = "neural"
            # Pick the most explanatory cause for *why* neural-RX was rejected.
            if state.mobility == "high":
                cause = "policy_oscillation"
                reason_metric = "doppler_out_of_domain"
                predicted_value = 1.0
                threshold = 0.0
            elif state.snr_db < 8.0:
                cause = "sla_breach_predicted"
                reason_metric = "snr_db"
                predicted_value = float(state.snr_db)
                threshold = 8.0
            else:  # pa_backoff_db > 6.0
                cause = "constraint_violation_soft"
                reason_metric = "pa_backoff_db"
                predicted_value = float(state.pa_backoff_db)
                threshold = 6.0
            reason_machine = RejectionReasonMachine(
                primary_cause=cause,
                primary_metric=reason_metric,
                predicted_value=float(predicted_value),
                threshold=float(threshold),
                horizon="30s",
            )

        rejected_alt = RejectedAlternative(
            rank=1,
            action={"rx": rejected_kind},
            predicted_outcome=PredictedOutcome(
                sla_risk_30s=float(
                    tbler_lmmse if rejected_kind == "lmmse" else tbler_neural
                ),
                sla_risk_1min=0.0,
                sla_risk_5min=0.0,
            ),
            rejection_reason_machine=reason_machine,
            rejection_reason_human=generate_human_explanation(reason_machine),
            random_seed=self._random_seed,
        )

        envelope = NeuralRxEnvelope(
            chosen=chosen,
            rejected_alternatives=[rejected_alt.model_dump(mode="json")],
            reason_machine=reason_machine.model_dump(mode="json"),
            reason_human=generate_human_explanation(reason_machine),
            pinned_seed=self._random_seed,
            predicted_tbler_neural=float(tbler_neural),
            predicted_tbler_lmmse=float(tbler_lmmse),
            model_card_ref=self._model_card_ref,
            state_hash=_state_hash(state),
        )
        return chosen, envelope.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Audit-chain emission (wires into evidence/store.py)
    # ------------------------------------------------------------------
    def to_decision_record(
        self,
        decision_id: str,
        state: NeuralRxState,
        model_versions: ModelVersions,
    ) -> DecisionRecord:
        """Build a `DecisionRecord` for this state, ready for `store.append`."""
        chosen, env = self.decide(state)
        rejected = [
            RejectedAlternative.model_validate(alt)
            for alt in env["rejected_alternatives"]
        ]
        return DecisionRecord(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            rapp_instance_id=self._rapp_instance_id,
            state_hash=env["state_hash"],
            chosen_action={
                "rx": chosen,
                "model_card_ref": self._model_card_ref,
                "predicted_tbler_neural": env["predicted_tbler_neural"],
                "predicted_tbler_lmmse": env["predicted_tbler_lmmse"],
            },
            predicted_outcome_chosen=PredictedOutcome(
                sla_risk_30s=float(
                    env["predicted_tbler_neural"]
                    if chosen == "neural"
                    else env["predicted_tbler_lmmse"]
                ),
                sla_risk_1min=0.0,
                sla_risk_5min=0.0,
            ),
            rejected_alternatives=rejected,
            model_versions=model_versions,
            random_seed=self._random_seed,
        )


# Honest-name alias.
NeuralRxArbiterPolicy = NeuralRxDecision


__all__ = [
    "MobilityClass",
    "NEURAL_RX_MODEL_CARD",
    "NeuralRxArbiterPolicy",     # canonical (rule-based, not AI)
    "NeuralRxDecision",          # legacy
    "NeuralRxEnvelope",
    "NeuralRxState",
    "RxKind",
]
