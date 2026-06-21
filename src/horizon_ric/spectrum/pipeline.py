"""End-to-end deployable trust loop for federated DSA on licensed spectrum.

One decision, wired through every layer the rApp owns:

    federated DSA agent  ──pick (channel, tx_power)──▶  physical RF action
              │
              ▼
    horizon_ric.shield   ──project onto spectral-mask / EIRP (/PFD) safe set──▶  safe action
              │
              ▼
    horizon_ric.evidence ──append a hash-chained DecisionRecord + SafetyCertificate──▶  audit

This is the join the reviewers asked for: the robust/secure aggregators in
:mod:`horizon_ric.spectrum.federated_q` actually drive an RL spectrum-decision
agent, whose every emitted ``(channel, power)`` is then forced legal by the
Shield and recorded in the tamper-evident evidence chain — regardless of whether
the federated policy was poisoned. A poisoned policy can lose *throughput*; it
can never emit an *illegal* RF action.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from horizon_ric.evidence.schema import (
    ConstraintCorrection,
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import EvidenceStore
from horizon_ric.shield.certificate import SafetyCertificate
from horizon_ric.shield.shield import Shield, default_terrestrial_shield
from horizon_ric.spectrum.federated_q import _greedy_action


@dataclass
class ChannelPlan:
    """Maps a DSA channel index to a real RF carrier on a licensed band.

    The K DSA channels are equal sub-bands tiling ``[band_lo_hz, band_hi_hz]``;
    channel ``k`` is centred on its sub-band with ``carrier_bw_hz`` occupied
    bandwidth. This is what turns the abstract DSA action into a concrete
    ``(frequency_hz, bandwidth_hz)`` the Shield can grade.
    """

    band_lo_hz: float
    band_hi_hz: float
    n_channels: int
    carrier_bw_hz: float

    def channel_center_hz(self, channel: int) -> float:
        sub_w = (self.band_hi_hz - self.band_lo_hz) / self.n_channels
        return self.band_lo_hz + sub_w * (channel + 0.5)


@dataclass
class DSADecisionConfig:
    band_lo_hz: float = 3.40e9
    band_hi_hz: float = 3.50e9
    carrier_bw_hz: float = 20e6
    max_eirp_dBm: float = 33.0
    antenna_gain_dBi: float = 6.0
    rapp_instance_id: str = "dsa-fed-rapp-1"


def _model_versions(method: str) -> ModelVersions:
    return ModelVersions(
        encoder="dsa-tabular-q-1",
        risk_heads="n/a",
        dyna="dsa-gilbert-elliott-1",
        policy=f"fed-dsa-{method}",
        constraint_layer="shield-terrestrial-1",
        rapp="horizon-dsa-1",
    )


def action_from_policy(
    q: np.ndarray,
    obs: int,
    *,
    plan: ChannelPlan,
    cfg: DSADecisionConfig,
    requested_tx_power_dBm: float,
) -> dict[str, Any]:
    """Turn a greedy DSA policy choice into a concrete RF action dict.

    If the policy chooses "idle" (action == n_channels) we emit a sense-only
    action with ``emit=False``; otherwise we map the chosen channel to a carrier.
    """
    a = _greedy_action(q, int(obs))
    if a >= plan.n_channels:
        return {
            "block": "fed_dsa_policy",
            "dsa_channel": int(a),
            "emit": False,
            "frequency_hz": plan.band_lo_hz + (plan.band_hi_hz - plan.band_lo_hz) / 2,
            "bandwidth_hz": plan.carrier_bw_hz,
            "tx_power_dBm": float(requested_tx_power_dBm),
            "antenna_gain_dBi": cfg.antenna_gain_dBi,
        }
    return {
        "block": "fed_dsa_policy",
        "dsa_channel": int(a),
        "emit": True,
        "frequency_hz": plan.channel_center_hz(a),
        "bandwidth_hz": plan.carrier_bw_hz,
        "tx_power_dBm": float(requested_tx_power_dBm),
        "antenna_gain_dBi": cfg.antenna_gain_dBi,
    }


@dataclass
class DSADecisionResult:
    safe_action: dict[str, Any]
    certificate: SafetyCertificate
    record: DecisionRecord
    chain_hash: Optional[str]


def decide_and_record(
    q: np.ndarray,
    obs: int,
    *,
    cfg: Optional[DSADecisionConfig] = None,
    shield: Optional[Shield] = None,
    evidence: Optional[EvidenceStore] = None,
    method: str = "krum",
    requested_tx_power_dBm: float = 30.0,
    decision_id: str = "dsa-0",
    rng_seed: int = 0,
    model_provenance: Optional[dict[str, Any]] = None,
) -> DSADecisionResult:
    """Run one full DSA decision: policy → Shield → evidence chain.

    Returns the safe action, its certificate, the DecisionRecord, and (if an
    evidence store was supplied) the chain hash of the appended record.
    """
    cfg = cfg or DSADecisionConfig()
    plan = ChannelPlan(
        band_lo_hz=cfg.band_lo_hz,
        band_hi_hz=cfg.band_hi_hz,
        n_channels=q.shape[1] - 1,  # actions = channels + idle
        carrier_bw_hz=cfg.carrier_bw_hz,
    )
    if shield is None:
        shield = default_terrestrial_shield(
            band_lo_hz=cfg.band_lo_hz,
            band_hi_hz=cfg.band_hi_hz,
            max_eirp_dBm=cfg.max_eirp_dBm,
        )

    proposed = action_from_policy(
        q, obs, plan=plan, cfg=cfg, requested_tx_power_dBm=requested_tx_power_dBm
    )

    disp = shield.dispose(
        proposed,
        decision_id=decision_id,
        rng_seed=rng_seed,
        loop_tier="near_rt",
        model_provenance=model_provenance,
    )
    cert = disp.certificate

    # Build the hash-chained evidence record carrying the SafetyCertificate.
    corrections = [
        ConstraintCorrection(
            constraint_id=c.constraint_id,
            severity=c.severity if c.severity in ("hard", "soft") else "hard",
            margin_dB=c.margin_dB,
            message=c.message,
        )
        for c in cert.corrections
    ]
    state_hash = _state_hash(obs, proposed)
    record = DecisionRecord.new(
        decision_id=decision_id,
        rapp_instance_id=cfg.rapp_instance_id,
        state_hash=state_hash,
        chosen_action={
            **disp.safe_action,
            "safety_certificate": cert.to_dict(),
        },
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.0,
            sla_risk_1min=0.0,
            sla_risk_5min=0.0,
            constraint_violations=list(cert.violated_ids),
        ),
        rejected_alternatives=[],
        constraint_corrections=corrections,
        model_versions=_model_versions(method),
        random_seed=rng_seed,
    )

    chain_hash = None
    if evidence is not None:
        chain_hash = evidence.append(record)

    return DSADecisionResult(
        safe_action=disp.safe_action,
        certificate=cert,
        record=record,
        chain_hash=chain_hash,
    )


def _state_hash(obs: int, action: dict[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        {"obs": int(obs), "channel": action.get("dsa_channel")},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ChannelPlan",
    "DSADecisionConfig",
    "DSADecisionResult",
    "action_from_policy",
    "decide_and_record",
]
