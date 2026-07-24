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
from dataclasses import dataclass
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
class NumericSanityInvariant:
    """Fail closed on malformed or non-physical numeric domains.

    Physics and regulatory checks are only meaningful for finite values in
    their declared domains. In particular, NaN comparisons are false in
    surprising ways and a negative bandwidth can appear to fit inside a band.
    This invariant is therefore first in each default chain.
    """

    id: str = "numeric_domain_sanity"

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    def _problems(self, action: Action) -> list[str]:
        problems: list[str] = []

        frequency = self._number(action.get("frequency_hz"))
        if frequency is None or frequency <= 0.0:
            problems.append("frequency_hz must be finite and > 0")

        bandwidth = self._number(action.get("bandwidth_hz"))
        if bandwidth is None or bandwidth <= 0.0:
            problems.append("bandwidth_hz must be finite and > 0")

        tx_power = self._number(action.get("tx_power_dBm"))
        if tx_power is None:
            problems.append("tx_power_dBm must be finite")

        for field_name in ("antenna_gain_dBi", "sat_antenna_gain_dBi"):
            if field_name in action and self._number(action[field_name]) is None:
                problems.append(f"{field_name} must be finite")

        for field_name in ("predicted_tbler", "baseline_tbler", "demap_confidence"):
            if field_name not in action:
                continue
            value = self._number(action[field_name])
            if value is None or not 0.0 <= value <= 1.0:
                problems.append(f"{field_name} must be finite and in [0, 1]")

        if "papr_dB" in action:
            papr = self._number(action["papr_dB"])
            if papr is None or papr < 0.0:
                problems.append("papr_dB must be finite and >= 0")

        if "constellation_order" in action:
            order = self._number(action["constellation_order"])
            if order is None or order <= 0.0 or not order.is_integer():
                problems.append("constellation_order must be a positive integer")

        if bool(action.get("ntn", False)):
            slant_range = self._number(action.get("slant_range_m"))
            if slant_range is None or slant_range <= 0.0:
                problems.append("slant_range_m must be finite and > 0 for NTN")

        return problems

    def evaluate(self, action: Action, context: Context) -> InvariantCheck:
        problems = self._problems(action)
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=not problems,
            margin=None,
            unit="domain",
            detail="numeric domains valid" if not problems else "; ".join(problems),
        )

    def project(
        self, action: Action, context: Context
    ) -> tuple[Action, list[ConstraintViolation]]:
        problems = self._problems(action)
        if not problems:
            return dict(action), []
        out = dict(action)
        out["emit_blocked"] = True
        return out, [
            ConstraintViolation(
                self.id,
                "hard",
                None,
                "malformed/non-physical action blocked: " + "; ".join(problems),
            )
        ]


@dataclass
class SpectralMaskInvariant:
    """3GPP TS 38.104 §6.6 — the emitted carrier must sit inside the licensed
    channel. Occupied-bandwidth-within-band check: the carrier centred at
    ``frequency_hz`` with ``bandwidth_hz`` must not spill past the permitted band
    edges.

    ``guard_band_hz`` (additive; default 0 preserves the original behaviour)
    keeps the *occupied* bandwidth that many Hz inside each edge. This closes an
    honest gap the independent ACLR benchmark surfaced: clipping an out-of-band
    carrier to sit *flush* against the band edge leaves its adjacent-channel
    leakage (transition skirt + spectral regrowth) spilling into the neighbour
    band — band-edge-legal but ACLR-illegal. A small guard band restores
    adjacent-channel-leakage-ratio compliance.
    """

    band_lo_hz: float
    band_hi_hz: float
    guard_band_hz: float = 0.0
    id: str = "spectral_mask_ts38104"

    def _edges(self, action: Action) -> tuple[float, float]:
        f = float(action.get("frequency_hz", 0.0))
        bw = float(action.get("bandwidth_hz", 0.0))
        return f - bw / 2.0, f + bw / 2.0

    def evaluate(self, action: Action, context: Context) -> InvariantCheck:
        lo, hi = self._edges(action)
        margin_hz = min(
            lo - (self.band_lo_hz + self.guard_band_hz),
            (self.band_hi_hz - self.guard_band_hz) - hi,
        )
        ok = margin_hz >= 0.0
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=ok,
            margin=margin_hz,
            unit="Hz",
            detail=(
                f"occupied [{lo / 1e6:.3f},{hi / 1e6:.3f}] MHz vs band "
                f"[{self.band_lo_hz / 1e6:.3f},{self.band_hi_hz / 1e6:.3f}] MHz "
                f"(guard {self.guard_band_hz / 1e6:.3f} MHz)"
            ),
        )

    def project(
        self, action: Action, context: Context
    ) -> tuple[Action, list[ConstraintViolation]]:
        out = dict(action)
        corr: list[ConstraintViolation] = []
        bw = float(out.get("bandwidth_hz", 0.0))
        usable_lo = self.band_lo_hz + self.guard_band_hz
        usable_hi = self.band_hi_hz - self.guard_band_hz
        usable_w = usable_hi - usable_lo
        if bw > usable_w:
            # Carrier wider than the usable channel — cannot place it. Refuse.
            out["emit_blocked"] = True
            corr.append(
                ConstraintViolation(
                    self.id, "hard", None,
                    f"carrier bandwidth {bw / 1e6:.3f} MHz exceeds usable channel "
                    f"{usable_w / 1e6:.3f} MHz (after {self.guard_band_hz / 1e6:.3f} "
                    "MHz guard band); cannot place — emit blocked.",
                )
            )
            return out, corr
        lo_allowed = usable_lo + bw / 2.0
        hi_allowed = usable_hi - bw / 2.0
        f = float(out.get("frequency_hz", 0.0))
        clipped = min(max(f, lo_allowed), hi_allowed)
        if clipped != f:
            out["frequency_hz"] = clipped
            corr.append(
                ConstraintViolation(
                    self.id, "hard", None,
                    f"centre frequency clipped {f / 1e6:.3f}→{clipped / 1e6:.3f} MHz "
                    "to keep the carrier inside the licensed channel (with guard band).",
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


@dataclass
class PfdCeilingInvariant:
    """ITU-R-style downlink power-flux-density (PFD) ceiling for NTN coexistence.

    Funding-relevant (the team's LEO-satellite power-control problem). A
    non-geostationary (LEO) satellite transmitting in a band shared with
    terrestrial services must not exceed a regulatory **power-flux-density**
    limit at the Earth's surface (ITU-R Radio Regulations Article 21 / RR
    Table 21-4 style PFD masks, e.g. for FSS downlinks in C/Ku bands). PFD is
    the EIRP spread over the spherical surface at the satellite's slant range:

        PFD(dBW/m^2/MHz) = EIRP_dBW - 10*log10(4*pi*d^2) - 10*log10(BW_MHz)

    where ``d`` is the slant range in metres. A poisoned or mis-trained LEO
    power-control agent that commands too much downlink power violates this
    even if the *terrestrial* EIRP ceiling is satisfied — so this is a distinct
    invariant. On violation we reduce the satellite Tx power by exactly the
    overage so the PFD lands on the ceiling.

    Action fields (all optional with sane defaults so terrestrial-only actions
    are a no-op):
        ntn (bool)                  — only applies when truthy.
        tx_power_dBm                — satellite downlink conducted power.
        sat_antenna_gain_dBi        — satellite Tx antenna gain.
        slant_range_m               — satellite-to-ground slant range (m).
        bandwidth_hz                — emission bandwidth.
    """

    max_pfd_dBW_m2_MHz: float = -146.0  # representative ITU-R RR Art.21 PFD limit
    id: str = "pfd_ceiling_ntn"

    def _applies(self, action: Action) -> bool:
        return bool(action.get("ntn", False))

    @staticmethod
    def compute_pfd(action: Action) -> float:
        """PFD in dBW/m^2/MHz at the surface for a satellite downlink action."""
        eirp_dBm = float(action.get("tx_power_dBm", 0.0)) + float(
            action.get("sat_antenna_gain_dBi", 0.0)
        )
        eirp_dBW = eirp_dBm - 30.0
        d = max(float(action.get("slant_range_m", 5.5e5)), 1.0)  # default ~550 km LEO
        spread_dB = 10.0 * math.log10(4.0 * math.pi * d * d)
        bw_mhz = max(float(action.get("bandwidth_hz", 1e6)) / 1e6, 1e-6)
        bw_dB = 10.0 * math.log10(bw_mhz)
        return eirp_dBW - spread_dB - bw_dB

    def evaluate(self, action: Action, context: Context) -> InvariantCheck:
        if not self._applies(action):
            return InvariantCheck(self.id, True, None, "n/a", "not an NTN downlink decision")
        pfd = self.compute_pfd(action)
        margin = self.max_pfd_dBW_m2_MHz - pfd
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=margin >= 0.0,
            margin=margin,
            unit="dB",
            detail=(
                f"PFD {pfd:.2f} dBW/m^2/MHz vs ceiling "
                f"{self.max_pfd_dBW_m2_MHz:.2f} dBW/m^2/MHz"
            ),
        )

    def project(
        self, action: Action, context: Context
    ) -> tuple[Action, list[ConstraintViolation]]:
        if not self._applies(action):
            return dict(action), []
        out = dict(action)
        pfd = self.compute_pfd(out)
        overage = pfd - self.max_pfd_dBW_m2_MHz
        corr: list[ConstraintViolation] = []
        if overage > 0:
            # PFD is linear in Tx power (dB-for-dB) — cut power by the overage.
            out["tx_power_dBm"] = float(out.get("tx_power_dBm", 0.0)) - overage
            corr.append(
                ConstraintViolation(
                    self.id, "hard", -overage,
                    f"LEO downlink power reduced by {overage:.2f} dB to meet the "
                    f"ITU-R PFD ceiling {self.max_pfd_dBW_m2_MHz:.2f} dBW/m^2/MHz.",
                )
            )
        return out, corr


# ---------------------------------------------------------------------------
# AI-PHY invariants (exist because the PHY is a neural block)
# ---------------------------------------------------------------------------
@dataclass
class NeuralRxEnvelopeInvariant:
    """The neural receiver must not regress far past the classical baseline.

    **Security change (closes the "safe by construction" bypass).** This
    invariant previously graded the neural receiver on its *self-reported*
    ``predicted_tbler`` / ``demap_confidence``. That is exactly the field a
    poisoned model controls — a backdoored neural-RX simply reports a great
    TBLER and a high confidence and walks straight through the Shield while
    actually corrupting the slot. Self-attestation is not assurance.

    The fix: this invariant now grades the neural-RX against an
    **independently-measured** block error rate that the model does NOT produce,
    supplied via ``context["measured_tbler"]`` (in production: the real CRC pass/
    fail and HARQ ACK/NACK telemetry from the MAC, which the receiver cannot
    forge). The model's self-reported ``predicted_tbler`` is recorded for the
    audit but is NOT trusted for the safety decision.

    Verification policy:

    * If ``context["measured_tbler"]`` is present, compare it to the classical
      LMMSE ``baseline_tbler``. If the *measured* neural error rate is more than
      ``tolerance_dB`` worse than the baseline, fall back.
    * If ``context["measured_tbler"]`` is absent, the decision is **unverified**.
      We are conservative and fall back to the classical receiver rather than
      trust the model's self-report. (Set ``require_measurement=False`` only for
      lab/bring-up where no CRC/HARQ telemetry exists yet; this re-opens the
      bypass and is logged as such in the detail string.)

    ``demap_confidence`` is still read, but only as a *secondary* soft signal and
    only when a measurement is present — it can never on its own admit a model
    that the independent measurement says is bad.
    """

    tolerance_dB: float = 1.0
    min_confidence: float = 0.2
    require_measurement: bool = True
    fallback_block: str = "classical_lmmse"
    id: str = "neural_rx_envelope"

    def evaluate(self, action: Action, context: Context) -> InvariantCheck:
        if not str(action.get("block", "")).startswith("neural_rx"):
            return InvariantCheck(self.id, True, None, "n/a", "not a neural-RX decision")

        base = float(action.get("baseline_tbler", 0.0))
        self_reported = float(action.get("predicted_tbler", 0.0))
        measured = context.get("measured_tbler", None)

        # Unverified path: no independent measurement supplied.
        if measured is None:
            if self.require_measurement:
                # Conservative: do NOT trust the model's self-report. Fail closed
                # to the classical fallback.
                return InvariantCheck(
                    invariant_id=self.id,
                    satisfied=False,
                    margin=None,
                    unit="dB",
                    detail=(
                        "UNVERIFIED neural-RX: no context['measured_tbler'] "
                        "(CRC/HARQ telemetry) supplied — self-reported TBLER "
                        f"{self_reported:.3g} is NOT trusted; falling back."
                    ),
                )
            # Lab override: explicitly trust the self-report (re-opens the bypass).
            measured_val = self_reported
            verified = False
        else:
            measured_val = float(measured)
            verified = True

        eps = 1e-9
        ratio_dB = 10.0 * math.log10((base + eps) / (measured_val + eps))
        margin_dB = ratio_dB + self.tolerance_dB
        conf = float(action.get("demap_confidence", 1.0))
        ok = (margin_dB >= 0.0) and (conf >= self.min_confidence)
        src = "measured CRC/HARQ" if verified else "self-reported (LAB, untrusted)"
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=ok,
            margin=margin_dB,
            unit="dB",
            detail=(
                f"neural TBLER {measured_val:.3g} [{src}] vs baseline {base:.3g} "
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
        unverified = context.get("measured_tbler", None) is None and self.require_measurement
        msg = (
            "neural-RX UNVERIFIED (no independent CRC/HARQ measurement); fell back "
            f"to {self.fallback_block}."
            if unverified
            else f"neural-RX outside its measured envelope; fell back to "
            f"{self.fallback_block} (certified classical receiver)."
        )
        return out, [ConstraintViolation(self.id, "hard", check.margin, msg)]


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
    "NumericSanityInvariant",
    "SpectralMaskInvariant",
    "MaxEirpInvariant",
    "PfdCeilingInvariant",
    "NeuralRxEnvelopeInvariant",
    "ConstellationLegalityInvariant",
    "LawfulInterceptInvariant",
]
