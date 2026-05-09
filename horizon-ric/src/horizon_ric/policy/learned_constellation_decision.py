"""Learned-constellation vs classical-QAM decision.

Decides whether to switch this UE's modulation from classical 16-QAM /
64-QAM to an AI-RAN-Alliance-learned constellation that doubles as a
pilot signal (pilotless communication).

Per the slide-published Nokia + R&S benchmarks, the learned constellation
gains:
    +31 % throughput at low mobility (e.g. pedestrian)
    +16 % throughput at medium mobility (e.g. car)
    +28 % throughput at high mobility (e.g. train)

The middle band (16 %) is the weakest gain; for medium mobility we keep
classical QAM unless the gain exceeds a configurable threshold (default
25 %).

Decision rule (deterministic, regulator-readable)
-------------------------------------------------
Use learned-constellation when:
    mobility ∈ {low, high}
    AND mcs_index ≤ 22                  (skip top-MCS where DMRS overhead
                                         is already amortised)
    AND prb_count ≥ 4                   (avoid cliff regimes with too
                                         few PRBs to learn over)
Otherwise use classical 16-QAM / 64-QAM.

Reference
---------
~/.claude/plans/preceptualai-airan-alliance-integration.md §3 M1.
AI-RAN Alliance learned-constellation slide (Nokia / R&S, 2026).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

MobilityClass = Literal["low", "medium", "high"]
ConstellationKind = Literal["classical_qam", "learned"]


# Slide-published throughput-gain table.
_GAIN_PCT: dict[MobilityClass, float] = {
    "low": 31.0,
    "medium": 16.0,
    "high": 28.0,
}


@dataclass(frozen=True)
class LearnedConstellationState:
    mobility: MobilityClass
    mcs_index: int
    prb_count: int


@dataclass(frozen=True)
class LearnedConstellationEnvelope:
    timestamp: datetime
    chosen: ConstellationKind
    rejected_alternatives: list[dict]
    reason_machine: dict
    reason_human: str
    pinned_seed: int
    state: dict
    predicted_throughput_gain_pct: float
    envelope_kind: str = "learned_constellation_decision"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "envelope_kind": self.envelope_kind,
            "chosen": self.chosen,
            "rejected_alternatives": self.rejected_alternatives,
            "reason_machine": self.reason_machine,
            "reason_human": self.reason_human,
            "pinned_seed": self.pinned_seed,
            "state": self.state,
            "predicted_throughput_gain_pct": self.predicted_throughput_gain_pct,
        }

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


class LearnedConstellationDecision:
    """Stateless decider with deterministic rule + slide-cited gain.

    The default ``min_gain_pct`` threshold (25 %) excludes medium mobility
    by design: 16 % gain is below the bar where the additional model
    surface area is justified.
    """

    MIN_GAIN_PCT_DEFAULT = 25.0
    MAX_MCS_DEFAULT = 22
    MIN_PRB_DEFAULT = 4

    def __init__(
        self,
        *,
        min_gain_pct: float = MIN_GAIN_PCT_DEFAULT,
        max_mcs: int = MAX_MCS_DEFAULT,
        min_prb: int = MIN_PRB_DEFAULT,
    ):
        self._min_gain_pct = min_gain_pct
        self._max_mcs = max_mcs
        self._min_prb = min_prb

    def decide(
        self,
        state: LearnedConstellationState,
        *,
        seed: int = 0,
    ) -> tuple[ConstellationKind, LearnedConstellationEnvelope]:
        gain = _GAIN_PCT[state.mobility]
        gate_gain = gain >= self._min_gain_pct
        gate_mcs = state.mcs_index <= self._max_mcs
        gate_prb = state.prb_count >= self._min_prb

        if gate_gain and gate_mcs and gate_prb:
            chosen: ConstellationKind = "learned"
            rejected: ConstellationKind = "classical_qam"
            reason_human = (
                f"Learned constellation selected: mobility={state.mobility} "
                f"(slide-published gain {gain:.0f}% ≥ threshold "
                f"{self._min_gain_pct:.0f}%), "
                f"MCS {state.mcs_index} ≤ {self._max_mcs}, "
                f"PRBs {state.prb_count} ≥ {self._min_prb}."
            )
            reason_machine = {
                "primary_cause": "learned_constellation_advantage",
                "primary_metric": "throughput_gain_pct",
                "predicted_value": float(gain),
                "threshold": float(self._min_gain_pct),
            }
        else:
            chosen = "classical_qam"
            rejected = "learned"
            why = []
            if not gate_gain:
                why.append(
                    f"mobility={state.mobility} gain {gain:.0f}% < "
                    f"threshold {self._min_gain_pct:.0f}%"
                )
            if not gate_mcs:
                why.append(f"MCS {state.mcs_index} > {self._max_mcs}")
            if not gate_prb:
                why.append(f"PRBs {state.prb_count} < {self._min_prb}")
            reason_human = "Classical QAM selected: " + "; ".join(why) + "."
            reason_machine = {
                "primary_cause": "learned_constellation_below_threshold",
                "primary_metric": (
                    "throughput_gain_pct" if not gate_gain
                    else "mcs_index" if not gate_mcs
                    else "prb_count"
                ),
                "predicted_value": float(
                    gain if not gate_gain
                    else state.mcs_index if not gate_mcs
                    else state.prb_count
                ),
                "threshold": float(
                    self._min_gain_pct if not gate_gain
                    else self._max_mcs if not gate_mcs
                    else self._min_prb
                ),
            }

        env = LearnedConstellationEnvelope(
            timestamp=datetime.now(timezone.utc),
            chosen=chosen,
            rejected_alternatives=[
                {
                    "action": rejected,
                    "reason_machine": {
                        "primary_cause": "rejected_alternative",
                        "primary_metric": "throughput_gain_pct",
                        "predicted_value": float(gain),
                        "threshold": float(self._min_gain_pct),
                    },
                    "reason_human": (
                        f"Rejected {rejected}; "
                        f"chosen {chosen} better fits mobility={state.mobility}."
                    ),
                }
            ],
            reason_machine=reason_machine,
            reason_human=reason_human,
            pinned_seed=int(seed),
            state={
                "mobility": state.mobility,
                "mcs_index": state.mcs_index,
                "prb_count": state.prb_count,
            },
            predicted_throughput_gain_pct=float(gain),
        )
        return chosen, env


__all__ = [
    "ConstellationKind",
    "LearnedConstellationDecision",
    "LearnedConstellationEnvelope",
    "LearnedConstellationState",
]
