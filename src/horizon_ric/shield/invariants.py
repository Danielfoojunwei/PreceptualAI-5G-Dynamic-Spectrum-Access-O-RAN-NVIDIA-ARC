"""Invariants enforced by the Decision Safety Shield.

An *invariant* is a property the RAN must hold regardless of what the AI
proposes. Two families:

* **RAN-physics / spectrum-regulatory** — true of any terrestrial RAN:
  the 3GPP TS 38.104 spectral emission mask and a maximum EIRP / Tx-power
  ceiling. A poisoned or adversarial model cannot talk the cell out of these.
* **AI-PHY** — invariants that exist *because* the PHY is now a neural block:
  the neural receiver's predicted error rate must not regress far past the
  classical LMMSE baseline, and a learned constellation must be a legal order
  with bounded PAPR. When an AI-PHY invariant fails, the safe move is to fall
  back to the certified classical baseline.

Each invariant exposes:
    id                                  stable string id
    evaluate(action, ctx) -> InvariantCheck     (single source of truth)
    project(action, ctx)  -> (action, [ConstraintViolation])

The Shield derives violations from ``evaluate`` and applies ``project`` to move
an infeasible action toward the safe set (or to a classical fallback).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from horizon_ric.shield.certificate import ConstraintViolation, InvariantCheck

Action = dict[str, Any]
Context = dict[str, Any]

# Sentinel the AI-PHY invariants set on the action when they force a fallback.
FALLBACK_KEY = "shield_fallback_to"


@runtime_checkable
class Invariant(Protocol):
    id: str

    def evaluate(self, action: Action, context: Context) -> InvariantCheck: ...

    def project(
        self, action: Action, context: Context
    ) -> tuple[Action, list[ConstraintViolation]]: ...


# ---------------------------------------------------------------------------
# RAN-physics / spectrum-regulatory invariants (terrestrial)
# ---------------------------------------------------------------------------
@dataclass
class SpectralMaskInvariant:
    """3GPP TS 38.104 §6.6 — the emitted carrier must sit inside the licensed
    channel. Simplified to an occupied-bandwidth-within-band check: the carrier
    centred at ``frequency_hz`` with ``bandwidth_hz`` must not spill past the
    permitted band edges.
    """

    band_lo_hz: float
    band_hi_hz: float
    id: str = "spectral_mask_ts38104"

    def _edges(self, action: Action) -> tuple[float, float]:
        f = float(action.get("frequency_hz", 0.0))
        bw = float(action.get("bandwidth_hz", 0.0))
        return f - bw / 2.0, f + bw / 2.0

    def evaluate(self, action: Action, context: Context) -> InvariantCheck:
        lo, hi = self._edges(action)
        margin_hz = min(lo - self.band_lo_hz, self.band_hi_hz - hi)
        ok = margin_hz >= 0.0
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=ok,
            margin=margin_hz,
            unit="Hz",
            detail=(
                f"occupied [{lo / 1e6:.3f},{hi / 1e6:.3f}] MHz vs band "
                f"[{self.band_lo_hz / 1e6:.3f},{self.band_hi_hz / 1e6:.3f}] MHz"
            ),
        )

    def project(
        self, action: Action, context: Context
    ) -> tuple[Action, list[ConstraintViolation]]:
        out = dict(action)
        corr: list[ConstraintViolation] = []
        bw = float(out.get("bandwidth_hz", 0.0))
        band_w = self.band_hi_hz - self.band_lo_hz
        if bw > band_w:
            # Carrier wider than the channel — cannot place it. Refuse (off).
            out["emit_blocked"] = True
            corr.append(
                ConstraintViolation(
                    self.id, "hard", None,
                    f"carrier bandwidth {bw / 1e6:.3f} MHz exceeds licensed "
                    f"channel {band_w / 1e6:.3f} MHz; cannot place — emit blocked.",
                )
            )
            return out, corr
        lo_allowed = self.band_lo_hz + bw / 2.0
        hi_allowed = self.band_hi_hz - bw / 2.0
        f = float(out.get("frequency_hz", 0.0))
        clipped = min(max(f, lo_allowed), hi_allowed)
        if clipped != f:
            out["frequency_hz"] = clipped
            corr.append(
                ConstraintViolation(
                    self.id, "hard", None,
                    f"centre frequency clipped {f / 1e6:.3f}→{clipped / 1e6:.3f} MHz "
                    "to keep the carrier inside the licensed channel.",
                )
            )
        return out, corr


@dataclass
class MaxEirpInvariant:
    """Regulatory EIRP / conducted-power ceiling.

    EIRP_dBm = tx_power_dBm + antenna_gain_dBi must not exceed ``max_eirp_dBm``
    (e.g. the band's block-edge mask / local licence limit).
    """

    max_eirp_dBm: float = 33.0
    id: str = "max_eirp"

    def _eirp(self, action: Action) -> float:
        return float(action.get("tx_power_dBm", 0.0)) + float(
            action.get("antenna_gain_dBi", 0.0)
        )

    def evaluate(self, action: Action, context: Context) -> InvariantCheck:
        eirp = self._eirp(action)
        margin = self.max_eirp_dBm - eirp
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=margin >= 0.0,
            margin=margin,
            unit="dB",
            detail=f"EIRP {eirp:.2f} dBm vs ceiling {self.max_eirp_dBm:.2f} dBm",
        )

    def project(
        self, action: Action, context: Context
    ) -> tuple[Action, list[ConstraintViolation]]:
        out = dict(action)
        corr: list[ConstraintViolation] = []
        eirp = self._eirp(out)
        overage = eirp - self.max_eirp_dBm
        if overage > 0:
            out["tx_power_dBm"] = float(out.get("tx_power_dBm", 0.0)) - overage
            corr.append(
                ConstraintViolation(
                    self.id, "hard", -overage,
                    f"Tx power reduced by {overage:.2f} dB to meet EIRP ceiling "
                    f"{self.max_eirp_dBm:.2f} dBm.",
                )
            )
        return out, corr


# ---------------------------------------------------------------------------
# AI-PHY invariants (exist because the PHY is a neural block)
# ---------------------------------------------------------------------------
@dataclass
class NeuralRxEnvelopeInvariant:
    """The neural receiver must not regress far past the classical baseline.

    ``predicted_tbler`` is the neural-RX self-reported / monitored block error
    rate; ``baseline_tbler`` is the classical LMMSE receiver's predicted TBLER
    for the same slot. If the neural block claims an error rate more than
    ``tolerance_dB`` worse than the baseline, or its ``demap_confidence`` falls
    below ``min_confidence``, the block is misbehaving (drift, adversarial input,
    or poisoning) and the safe move is to fall back to the classical receiver.
    """

    tolerance_dB: float = 1.0
    min_confidence: float = 0.2
    fallback_block: str = "classical_lmmse"
    id: str = "neural_rx_envelope"

    def evaluate(self, action: Action, context: Context) -> InvariantCheck:
        if not str(action.get("block", "")).startswith("neural_rx"):
            return InvariantCheck(self.id, True, None, "n/a", "not a neural-RX decision")
        pred = float(action.get("predicted_tbler", 0.0))
        base = float(action.get("baseline_tbler", pred))
        conf = float(action.get("demap_confidence", 1.0))
        # Ratio margin in dB: positive ⇒ neural is within tolerance of baseline.
        eps = 1e-9
        ratio_dB = 10.0 * math.log10((base + eps) / (pred + eps))
        margin_dB = ratio_dB + self.tolerance_dB
        ok = (margin_dB >= 0.0) and (conf >= self.min_confidence)
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=ok,
            margin=margin_dB,
            unit="dB",
            detail=(
                f"neural TBLER {pred:.3g} vs baseline {base:.3g} "
                f"(conf {conf:.2f} ≥ {self.min_confidence})"
            ),
        )

    def project(
        self, action: Action, context: Context
    ) -> tuple[Action, list[ConstraintViolation]]:
        check = self.evaluate(action, context)
        if check.satisfied:
            return dict(action), []
        out = dict(action)
        out["block"] = self.fallback_block
        out[FALLBACK_KEY] = self.fallback_block
        # The classical baseline's error rate becomes the operative prediction.
        out["predicted_tbler"] = float(action.get("baseline_tbler", action.get("predicted_tbler", 0.0)))
        return out, [
            ConstraintViolation(
                self.id, "hard", check.margin,
                f"neural-RX outside its envelope; fell back to {self.fallback_block} "
                "(certified classical receiver).",
            )
        ]


# Legal constellation orders for terrestrial NR (M-QAM).
_LEGAL_QAM_ORDERS = (4, 16, 64, 256)


@dataclass
class ConstellationLegalityInvariant:
    """A learned constellation must be a legal M-QAM order with bounded PAPR.

    Guards against a poisoned learned-constellation block emitting a non-standard
    order or a high-PAPR shape that a UE cannot demodulate / that breaches the
    power amplifier envelope. On failure: snap to the nearest legal order ≤ the
    requested order and fall back to classical QAM at that order.
    """

    max_papr_dB: float = 8.5
    allowed_orders: tuple[int, ...] = _LEGAL_QAM_ORDERS
    fallback_block: str = "classical_qam"
    id: str = "constellation_legality"

    def _applies(self, action: Action) -> bool:
        return "constellation_order" in action or "learned_constellation" in str(
            action.get("block", "")
        )

    def evaluate(self, action: Action, context: Context) -> InvariantCheck:
        if not self._applies(action):
            return InvariantCheck(self.id, True, None, "n/a", "no constellation in decision")
        order = int(action.get("constellation_order", 0))
        papr = float(action.get("papr_dB", 0.0))
        order_ok = order in self.allowed_orders
        papr_margin = self.max_papr_dB - papr
        ok = order_ok and papr_margin >= 0.0
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=ok,
            margin=papr_margin,
            unit="dB",
            detail=(
                f"order {order} ({'legal' if order_ok else 'ILLEGAL'}), "
                f"PAPR {papr:.2f} dB vs ceiling {self.max_papr_dB:.2f} dB"
            ),
        )

    def project(
        self, action: Action, context: Context
    ) -> tuple[Action, list[ConstraintViolation]]:
        if not self._applies(action):
            return dict(action), []
        out = dict(action)
        corr: list[ConstraintViolation] = []
        order = int(out.get("constellation_order", 0))
        if order not in self.allowed_orders:
            legal_le = [o for o in self.allowed_orders if o <= order] or [self.allowed_orders[0]]
            snapped = max(legal_le)
            out["constellation_order"] = snapped
            out["block"] = self.fallback_block
            out[FALLBACK_KEY] = self.fallback_block
            corr.append(
                ConstraintViolation(
                    self.id, "hard", None,
                    f"illegal constellation order {order} snapped to {snapped}-QAM "
                    f"({self.fallback_block}).",
                )
            )
        papr = float(out.get("papr_dB", 0.0))
        if papr > self.max_papr_dB:
            out["papr_dB"] = self.max_papr_dB
            out["block"] = self.fallback_block
            out[FALLBACK_KEY] = self.fallback_block
            corr.append(
                ConstraintViolation(
                    self.id, "hard", self.max_papr_dB - papr,
                    f"PAPR {papr:.2f} dB clipped to {self.max_papr_dB:.2f} dB via "
                    f"classical QAM fallback.",
                )
            )
        return out, corr


# ---------------------------------------------------------------------------
# Lawful-intercept adapter (wraps the existing fail-closed LIConstraint)
# ---------------------------------------------------------------------------
@dataclass
class LawfulInterceptInvariant:
    """Adapts :class:`horizon_ric.policy.li_constraint.LIConstraint` (3GPP TS
    33.127) to the Invariant protocol. Fail-closed by default: with no LIMF rule
    catalogue every action is refused."""

    li: Any  # LIConstraint instance
    id: str = "lawful_intercept"

    def evaluate(self, action: Action, context: Context) -> InvariantCheck:
        violations = self.li.check_feasibility(action, context)
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=not violations,
            margin=None,
            unit="bool",
            detail=(violations[0].message if violations else "no LI conflict"),
        )

    def project(
        self, action: Action, context: Context
    ) -> tuple[Action, list[ConstraintViolation]]:
        return self.li.project(action, context)


__all__ = [
    "Action",
    "Context",
    "FALLBACK_KEY",
    "Invariant",
    "SpectralMaskInvariant",
    "MaxEirpInvariant",
    "NeuralRxEnvelopeInvariant",
    "ConstellationLegalityInvariant",
    "LawfulInterceptInvariant",
]
