#!/usr/bin/env python3
"""Re-run a committed benchmark and check its published claims still hold.

Three benchmarks carry numbers quoted in external write-ups but had no
automated regeneration: their results existed only as committed JSON, which
makes them assertions about a file rather than evidence about the system. This
gates them.

Each benchmark already self-gates — it exits non-zero when its own invariants
break — so simply running it in CI is half the job. The other half is checking
that the freshly produced numbers still match what was published, because a
benchmark can pass its internal gates while quietly reporting something other
than the figure a reader was given.

What is compared, and why the split matters:

``exact``      booleans and integer counts. These are decisions, not estimates:
               a probe either bypassed the gate or it did not. Compared with no
               tolerance, because a single one changing is a real change.
``tol``        float aggregates, compared within a stated band. Floats that
               come out of a long reduction over per-receiver values are not
               bit-stable across hosts or BLAS builds, so demanding equality
               would produce failures that say nothing about the system.
``must_hold``  invariants asserted directly rather than inferred from equality
               above. If the committed and fresh results ever drifted the same
               way, equality alone would stay green while the guarantee was
               broken.
``nonzero``    quantities that must be > 0 so that a vacuous run — one that
               exercised nothing — cannot pass by trivially matching zeros.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SPECS: dict[str, dict[str, Any]] = {
    "projection_capacity_2x2": {
        "exact": [
            "findings.tabular_loss_is_zero_everywhere",
            "findings.no_illegal_action_executed",
            "findings.learning_beats_zero_data_policy_when_optimum_interior",
            "findings.learning_cannot_pay_when_optimum_at_cap",
            "findings.boundary_optimal_hides_the_defect",
            "learning_pays.boundary_optimal_gain",
            "receivers",
            "eirp_cap_dbm",
        ],
        "tol": {
            "learning_pays.interior_optimal_gain_min": 0.002,
            "learning_pays.interior_optimal_gain_max": 0.005,
            "findings.smooth_loss_min_fraction": 0.02,
            "findings.smooth_loss_max_fraction": 0.05,
        },
        "must_hold": [
            # The two capability gates, restated here so they are enforced even
            # if the benchmark's own SystemExit is ever loosened.
            "findings.learning_beats_zero_data_policy_when_optimum_interior",
            "findings.learning_cannot_pay_when_optimum_at_cap",
            "findings.no_illegal_action_executed",
        ],
        "nonzero": ["learning_pays.interior_optimal_gain_min"],
    },
    "integrity_attack_suite": {
        "exact": [
            "total_attacks",
            "total_probes",
            "detected_or_blocked",
            "bypassed",
            "skipped",
            "all_defended",
        ],
        "tol": {},
        "must_hold": ["all_defended"],
        "nonzero": ["total_attacks", "detected_or_blocked"],
    },
    "evasion_suite": {
        # The headline is not a number but a bound that must survive every
        # attack in every propagation regime: the Shield-effective symbol error
        # stays inside the certified 1 dB envelope of the classical baseline.
        # Asserted structurally below rather than by field equality, because
        # the SERs themselves are seed- and host-sensitive.
        "exact": [],
        "tol": {},
        "must_hold": [],
        "nonzero": [],
        "custom": "evasion_shield_never_worse_than_classical",
    },
}

TOL_DEFAULT = 1e-9

# NeuralRxEnvelopeInvariant(tolerance_dB=1.0) — the certified band the Shield
# actually enforces. Kept in one place so prose and gate cannot drift apart.
ENVELOPE_TOLERANCE_DB = 1.0


def _dig(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _evasion_shield_never_worse(fresh: dict[str, Any]) -> list[str]:
    """Shield-effective SER must stay inside the certified envelope, everywhere.

    The guarantee the system actually offers is NOT "never worse than the
    classical receiver". It is ``NeuralRxEnvelopeInvariant`` with a 1 dB
    tolerance: the Shield watches an independently measured windowed SER and
    falls back to the certified classical receiver once the neural one drifts
    more than 1 dB outside its envelope. Inside that band the neural receiver
    is allowed to be slightly worse — that band is the price of not thrashing
    the fallback on measurement noise.

    Writing this check as ``eff <= cls`` would therefore be wrong in two ways
    at once: it would fail honest runs (under fading the effective SER trails
    classical by up to ~0.37 dB, which is inside the envelope and expected),
    and it would quietly encourage the surrounding prose to claim a stronger
    guarantee than the code provides. Check the real bound.
    """
    errors: list[str] = []
    checked = 0
    envelope_linear = 10.0 ** (ENVELOPE_TOLERANCE_DB / 10.0)
    for regime in ("awgn", "fading", "real_raytraced"):
        block = fresh.get(regime)
        if not isinstance(block, dict):
            continue
        attacks = block.get("attacks")
        if not isinstance(attacks, dict):
            continue
        for attack, row in attacks.items():
            if not isinstance(row, dict):
                continue
            eff = row.get("shield_effective_ser")
            cls = row.get("classical_ser")
            if isinstance(eff, dict):
                eff = eff.get("mean")
            if isinstance(cls, dict):
                cls = cls.get("mean")
            if eff is None or cls is None:
                continue
            checked += 1
            if cls <= 0:
                # No classical errors to compare against: the only defensible
                # requirement is that the Shield produced none either.
                if eff > 0:
                    errors.append(
                        f"{regime}/{attack}: classical SER is 0 but "
                        f"shield-effective SER is {eff}"
                    )
                continue
            if eff > cls * envelope_linear:
                errors.append(
                    f"{regime}/{attack}: shield-effective SER {eff:.6g} is "
                    f"{10.0 * math.log10(eff / cls):.3f} dB worse "
                    f"than the classical baseline {cls:.6g}, outside the "
                    f"{ENVELOPE_TOLERANCE_DB} dB certified envelope"
                )
    if checked == 0:
        errors.append(
            "no attack rows carried both shield_effective_ser and classical_ser "
            "— the guarantee was not actually checked"
        )
    return errors


def verify(name: str, committed: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    spec = SPECS[name]
    errors: list[str] = []

    for path in spec["exact"]:
        c, f = _dig(committed, path), _dig(fresh, path)
        if c != f:
            errors.append(f"{path}: committed {c!r} != fresh {f!r}")

    for path, tol in spec["tol"].items():
        c, f = _dig(committed, path), _dig(fresh, path)
        if c is None or f is None:
            errors.append(f"{path}: missing (committed={c!r}, fresh={f!r})")
        elif abs(float(c) - float(f)) > tol:
            errors.append(
                f"{path}: committed {c} vs fresh {f} differs by "
                f"{abs(float(c) - float(f)):.6g} > tolerance {tol}"
            )

    for path in spec["must_hold"]:
        if _dig(fresh, path) is not True:
            errors.append(f"INVARIANT {path} is not true in the fresh run")

    for path in spec["nonzero"]:
        val = _dig(fresh, path)
        if not isinstance(val, (int, float)) or val <= TOL_DEFAULT:
            errors.append(
                f"{path} is {val!r}; a run that exercises nothing must not pass"
            )

    if spec.get("custom") == "evasion_shield_never_worse_than_classical":
        errors.extend(_evasion_shield_never_worse(fresh))

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, choices=sorted(SPECS))
    ap.add_argument("--committed", type=Path)
    ap.add_argument("--fresh", type=Path, required=True)
    args = ap.parse_args()

    committed_path = args.committed or Path(
        f"benchmarks/results/{args.benchmark}.json"
    )
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    fresh = json.loads(args.fresh.read_text(encoding="utf-8"))

    errors = verify(args.benchmark, committed, fresh)
    if errors:
        print(f"{args.benchmark} verification FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"{args.benchmark} verification PASSED against {committed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
