#!/usr/bin/env python3
"""Which invariants can REFUSE, and which can only CORRECT.

Why this exists
---------------
``audit/check_unattended.py`` disclosed a gap: §VII of the proposal promises

    "a Shield-corrected over-power proposal refused rather than emitted"

and the committed refusal proof exercises ``numeric_domain_sanity`` (negative
bandwidth), not ``max_eirp``. That was recorded as *the proof is missing*.

It is not missing. It is unobtainable, and for a good reason.
``MaxEirpInvariant.project`` subtracts the overage from ``tx_power_dBm``
unconditionally — there is no over-power action it cannot repair, so there is
no over-power action it refuses. Refusal in this Shield is reserved for
actions no projection can fix. §VII therefore describes behaviour the chain
deliberately does not have, and no amount of test-writing will produce it.

What this module does
---------------------
Turns that from an assertion into a characterisation, by asking each
invariant the question **directly**: given an action *it* says is violated,
what does *its own* ``project`` return?

  CORRECTING — its projection repairs the violation
  REFUSING   — its projection sets ``emit_blocked``
  DEFERRING  — it neither repairs nor refuses, leaving the violation for the
               Shield's outer ``or bool(violated_ids)`` check to catch
  INERT      — this sweep never got it to fail (reported, never assumed safe)

Probing each invariant in isolation is the point. An earlier version of this
file read the *chain's* certificate and attributed a refusal to every
invariant listed in ``violated_ids``. That made ``max_eirp`` look REFUSING on
the strength of a single ``tx_power_dBm: inf`` action — which
``numeric_domain_sanity`` blocks first, before ``max_eirp`` ever gets to
project, leaving ``inf`` in place and ``max_eirp`` trivially still violated.
"Was unsatisfied when something else refused" is not "refuses".

Additive: reads the shipping Shield, writes nothing to it.
"""

from __future__ import annotations

import itertools
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from horizon_ric.shield.invariants import (  # noqa: E402
    ConstellationLegalityInvariant,
    MaxEirpInvariant,
    NeuralRxEnvelopeInvariant,
    NumericSanityInvariant,
    SpectralMaskInvariant,
)
from horizon_ric.shield.shield import Shield, default_terrestrial_shield  # noqa: E402

__all__ = [
    "BAND_HI_HZ",
    "BAND_LO_HZ",
    "CORRECTING",
    "DEFERRING",
    "INERT",
    "MAX_EIRP_DBM",
    "REFUSING",
    "Classification",
    "characterise",
    "over_power_actions",
]

CORRECTING = "CORRECTING"
REFUSING = "REFUSING"
DEFERRING = "DEFERRING"
INERT = "INERT"

BAND_LO_HZ = 3.4e9
BAND_HI_HZ = 3.5e9
MAX_EIRP_DBM = 33.0
MAX_PAPR_DB = 8.5


def build_shield() -> Shield:
    return default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ, max_eirp_dBm=MAX_EIRP_DBM
    )


def _invariants() -> dict[str, Any]:
    """The same instances ``default_terrestrial_shield`` builds, by id.

    Constructed from the public classes with the same arguments rather than
    reached out of the Shield's private list, and cross-checked against
    ``Shield.invariant_ids`` by the gate so a chain change is not silently
    missed here.
    """
    return {
        inv.id: inv
        for inv in (
            NumericSanityInvariant(),
            SpectralMaskInvariant(band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ),
            MaxEirpInvariant(max_eirp_dBm=MAX_EIRP_DBM),
            NeuralRxEnvelopeInvariant(tolerance_dB=1.0),
            ConstellationLegalityInvariant(max_papr_dB=MAX_PAPR_DB),
        )
    }


def _base() -> dict[str, Any]:
    """A legal action at the centre of the band, comfortably under the ceiling."""
    return {
        "block": "spectrum",
        "frequency_hz": (BAND_LO_HZ + BAND_HI_HZ) / 2.0,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": 0.0,
    }


def over_power_actions() -> Iterator[dict[str, Any]]:
    """Every over-power action this characterisation puts to ``max_eirp``.

    Crossed with the fields the *other* invariants read, because the only way
    an over-power proposal could fail to be repaired is if some interaction
    stopped the clamp. If that combination exists, it is in here.
    """
    powers = (33.1, 35.0, 40.0, 50.0, 80.0, 200.0, 1e6)
    gains = (0.0, 6.0, 20.0, 60.0)
    orders = (None, 4, 64, 256, 1024, 4096)
    paprs = (None, 8.0, 14.0, 30.0)
    tblers = ((None, None), (0.001, 0.1), (0.5, 0.01), (0.9, 0.05))
    for tx, gain, order, papr, (ptb, btb) in itertools.product(
        powers, gains, orders, paprs, tblers
    ):
        action = _base()
        action["tx_power_dBm"] = tx
        action["antenna_gain_dBi"] = gain
        if order is not None:
            action["constellation_order"] = order
        if papr is not None:
            action["papr_dB"] = papr
        if ptb is not None:
            action["predicted_tbler"] = ptb
            action["baseline_tbler"] = btb
        yield action


def _actions_for(inv_id: str) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """``(action, context)`` pairs built to violate ``inv_id`` specifically."""
    if inv_id == "numeric_domain_sanity":
        for bad in (
            {"bandwidth_hz": -1.0},
            {"frequency_hz": float("nan")},
            {"tx_power_dBm": float("inf")},
            {"papr_dB": -3.0},
            {"constellation_order": 5.5},
            {"ntn": True, "slant_range_m": -1.0},
        ):
            a = _base()
            a.update(bad)
            yield a, {}

    elif inv_id == "spectral_mask_ts38104":
        for freq, bw in (
            (3.30e9, 20e6),
            (3.60e9, 20e6),
            (3.39e9, 40e6),
            (3.45e9, 200e6),
            (3.45e9, 100e6 + 1.0),
        ):
            a = _base()
            a["frequency_hz"], a["bandwidth_hz"] = freq, bw
            yield a, {}

    elif inv_id == "max_eirp":
        for a in over_power_actions():
            yield a, {}

    elif inv_id == "neural_rx_envelope":
        # This invariant is a no-op unless `block` names a neural receiver —
        # the reason an earlier sweep reported it INERT.
        for measured, base, conf in (
            (None, 0.1, 0.9),  # unverified: no measured_tbler in context
            (0.5, 0.01, 0.9),  # measured far worse than baseline
            (0.02, 0.01, 0.05),  # fine TBLER, confidence below the floor
        ):
            a = _base()
            a["block"] = "neural_rx"
            a["baseline_tbler"] = base
            a["predicted_tbler"] = 0.001
            a["demap_confidence"] = conf
            ctx = {} if measured is None else {"measured_tbler": measured}
            yield a, ctx

    elif inv_id == "constellation_legality":
        for order, papr in ((1024, 12.0), (4096, 14.0), (65536, 30.0), (256, 20.0)):
            a = _base()
            a["constellation_order"] = order
            a["papr_dB"] = papr
            yield a, {}


@dataclass
class Classification:
    """What one invariant's own projection did when handed its own violations."""

    invariant_id: str
    verdict: str = INERT
    violations_probed: int = 0
    repaired: int = 0
    refused: int = 0
    deferred: int = 0
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "verdict": self.verdict,
            "violations_probed": self.violations_probed,
            "repaired": self.repaired,
            "refused": self.refused,
            "deferred": self.deferred,
            "examples": self.examples[:3],
        }


def characterise() -> tuple[dict[str, Classification], dict[str, Any]]:
    """Probe every invariant's own projection. Returns ``(by_id, summary)``."""
    logging.disable(logging.CRITICAL)
    try:
        invariants = _invariants()
        result = {i: Classification(i) for i in invariants}

        over_power_probed = 0
        over_power_refused = 0
        over_power_unrepaired = 0
        largest_clamp_dB = 0.0

        for inv_id, inv in invariants.items():
            c = result[inv_id]
            for action, ctx in _actions_for(inv_id):
                if inv.evaluate(action, ctx).satisfied:
                    continue  # not a violation of THIS invariant; nothing to probe
                c.violations_probed += 1
                projected, _corrections = inv.project(dict(action), ctx)

                if projected.get("emit_blocked"):
                    c.refused += 1
                    outcome = "refused"
                elif inv.evaluate(projected, ctx).satisfied:
                    c.repaired += 1
                    outcome = "repaired"
                else:
                    c.deferred += 1
                    outcome = "deferred"
                if len(c.examples) < 3:
                    c.examples.append(f"{outcome}: {_brief(action)}")

                if inv_id == "max_eirp":
                    over_power_probed += 1
                    if projected.get("emit_blocked"):
                        over_power_refused += 1
                    if not inv.evaluate(projected, ctx).satisfied:
                        over_power_unrepaired += 1
                    largest_clamp_dB = max(
                        largest_clamp_dB,
                        float(action["tx_power_dBm"])
                        - float(projected.get("tx_power_dBm", action["tx_power_dBm"])),
                    )

            if c.violations_probed == 0:
                c.verdict = INERT
            elif c.refused:
                c.verdict = REFUSING
            elif c.deferred:
                c.verdict = DEFERRING
            else:
                c.verdict = CORRECTING

        summary = {
            "over_power_probed": over_power_probed,
            "over_power_refused": over_power_refused,
            "over_power_unrepaired": over_power_unrepaired,
            "largest_power_clamp_dB": round(largest_clamp_dB, 2),
            "refusing": sorted(i for i, c in result.items() if c.verdict == REFUSING),
            "correcting": sorted(i for i, c in result.items() if c.verdict == CORRECTING),
            "deferring": sorted(i for i, c in result.items() if c.verdict == DEFERRING),
            "inert": sorted(i for i, c in result.items() if c.verdict == INERT),
        }
        return result, summary
    finally:
        logging.disable(logging.NOTSET)


def _brief(action: dict[str, Any]) -> str:
    keep = ("block", "frequency_hz", "bandwidth_hz", "tx_power_dBm", "antenna_gain_dBi")
    bits = []
    for k in keep:
        if k not in action:
            continue
        v = action[k]
        if k.endswith("_hz") and isinstance(v, float):
            bits.append(f"{k[:-3]}={v / 1e6:.2f}MHz")
        else:
            bits.append(f"{k}={v}")
    return " ".join(bits)


if __name__ == "__main__":  # pragma: no cover — the gate is the entry point
    import json

    by_id, summary = characterise()
    print(
        json.dumps(
            {"summary": summary, "by_invariant": {k: v.to_dict() for k, v in by_id.items()}},
            indent=2,
            sort_keys=True,
        )
    )
