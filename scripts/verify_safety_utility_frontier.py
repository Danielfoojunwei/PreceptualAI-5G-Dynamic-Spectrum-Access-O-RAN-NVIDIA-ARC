#!/usr/bin/env python3
"""Verify a fresh safety-utility frontier run against the committed real-data result.

The frontier loop is deterministic given the same feature bytes, but served
fractions are quantised to 1/4096 and a last-ULP cross-host difference in the
DeepMIMO channel computation can flip a receiver sitting exactly on the SINR
threshold (the same reason the DSA reproducer binds on source_tree_sha256 and
compares floats with tolerance). This verifier therefore checks what is
host-stable and meaningful:

* the fresh run is bound to the canonical, checksum-pinned DeepMIMO build;
* the frontier invariants hold on the freshly rebuilt real data (a looser cap
  never hurts; the constraint genuinely binds at the operational 33 dBm cap;
  tightening the cap loses served fraction somewhere; every executed action is
  legal under its own cap; the free-constraint task costs nothing);
* **every feasible set in the run came from the Shield's own predicate** (P1),
  and every ranked winner — the free case's ``best_subband`` and each cap's
  ``best_feasible_subband`` — lies inside it;
* **the zero-data baseline is reported** (P3): every cap emits
  ``constant_cap_policy_reward`` and ``realised_regret``, and no cap is allowed
  to claim it beat a policy that transmits at the cap with no data at all;
* the host-stable decisions reproduce the committed result within tolerance
  bands (same cap grid, best-feasible served fractions within 0.005, positive
  utility forgone at 33 dBm, zero-cost free case) — NOT byte-compares;
* the trust chain verifies intact, the tampered chain is refused for replay,
  and the tamper index is pinpointed.

**Why no subband index is ever exact-compared.** The argmax subband is not a
reproducible identity: it moves with EIRP (over all six it is 0 at 20 dBm, 1 at
26, 5 at 32, 3 at 33/40/46, 5 at 52) and the whole subband decision is worth
only ~12-15 receivers of 4096, which is inside the last-ULP threshold-flip band
this verifier already tolerates for served fractions. An exact
``best_subband == 2`` assertion would therefore be a latent CI flake that fails
on a re-tie without anything being wrong. What *is* stable and meaningful — and
what this verifier gates on — is the **achieved reward within
``SERVED_FRACTION_TOL``**, plus the structural facts that the winner is
Shield-feasible and that the candidate set is the Shield's.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BAND_LO_HZ = 3.45e9
BAND_HI_HZ = 3.55e9
OPERATIONAL_CAP_DBM = 33.0
SERVED_FRACTION_TOL = 0.005  # ~20 receivers of 4096; generous for threshold flips.


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(*, committed: dict[str, Any], fresh: dict[str, Any], manifest: dict[str, Any]) -> None:
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    # 1. Real-data binding. The fresh run must be bound to its own rebuilt
    #    feature file and derived from the same raw ray-tracing scenario as the
    #    committed result (source_tree_sha256 is the host-independent binding).
    check(
        fresh.get("features_sha256") == manifest.get("features_sha256"),
        "fresh result is not bound to the rebuilt manifest features_sha256",
    )
    check(
        fresh.get("source_tree_sha256") == committed.get("source_tree_sha256"),
        "fresh run derives from a different raw scenario than the committed result",
    )
    check(
        fresh.get("source_tree_sha256") == manifest.get("source_tree_sha256"),
        "fresh result source_tree_sha256 does not match the rebuilt manifest",
    )
    check(fresh.get("receivers") == 4096, "fresh run did not use 4096 receivers")

    frontier = fresh.get("frontier", [])
    check(len(frontier) >= 2, "fresh frontier has fewer than two caps")
    best = [f["best_feasible_reward"] for f in frontier]

    # 2. Frontier invariants on the freshly rebuilt real data.
    check(
        all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:])),
        "best feasible reward is not non-decreasing in the cap (a looser cap hurt)",
    )
    check(
        fresh.get("utility_forgone_at_operational_cap", 0.0) > 0.0,
        "the constraint does not bind at the operational 33 dBm cap",
    )
    check(
        any(
            m["served_fraction_lost_per_db"] > 0.0
            for m in fresh.get("marginal_price_per_db", [])
        ),
        "no adjacent cap pair loses served fraction when tightened",
    )
    for f in frontier:
        cap = f["eirp_cap_dbm"]
        check(
            f["max_executed_eirp_dbm"] <= cap + 1e-9,
            f"an executed action exceeded its own cap ({cap} dBm)",
        )
        check(
            BAND_LO_HZ <= f["executed_frequency_hz"] <= BAND_HI_HZ,
            f"an executed action left the licensed band at cap {cap} dBm",
        )
        check(
            f["realised_mean_reward"] <= f["best_feasible_reward"] + 1e-9,
            f"realised reward exceeded the feasible optimum at cap {cap} dBm",
        )
        check(f["projection_rate"] == 1.0, f"the Shield did not project every step at {cap} dBm")

    free = fresh.get("free_constraint_case", {})
    check(free.get("projection_rate") == 0.0, "free-constraint case was projected")
    check(free.get("corrected") is False, "free-constraint case was corrected")
    check(
        abs(free.get("utility_cost", 1.0)) < 1e-9,
        "free-constraint case has a non-zero utility cost",
    )
    check(
        free.get("max_executed_eirp_dbm", 1e9) <= OPERATIONAL_CAP_DBM + 1e-9,
        "free-constraint case exceeded the operational cap",
    )

    # 2b. P1 — every feasible set is the Shield's, and every winner is in it.
    feasibility = fresh.get("feasibility", {})
    shield_feasible = feasibility.get("shield_feasible_subbands")
    check(
        "Shield.is_feasible" in str(feasibility.get("source", "")),
        "the fresh run did not derive its feasible set from Shield.is_feasible",
    )
    check(
        isinstance(shield_feasible, list) and len(shield_feasible) > 0,
        "the fresh run reported no Shield-feasible subbands",
    )
    if isinstance(shield_feasible, list):
        check(
            free.get("candidate_subbands") == shield_feasible,
            "the free-constraint case ranked a candidate set that is not the Shield's",
        )
        check(
            free.get("best_subband") in shield_feasible,
            "the free-constraint case selected a Shield-infeasible subband",
        )
        for f in frontier:
            check(
                f["best_feasible_subband"] in shield_feasible,
                f"best_feasible_subband at cap {f['eirp_cap_dbm']} dBm is Shield-infeasible",
            )

    # 2c. P3 — the zero-data baseline must be emitted, and must not be beaten.
    for f in frontier:
        cap = f["eirp_cap_dbm"]
        check(
            "constant_cap_policy_reward" in f and "realised_regret" in f,
            f"cap {cap} dBm did not report its zero-data constant-cap baseline",
        )
        check(
            f.get("realised_regret", -1.0) >= -1e-9,
            f"cap {cap} dBm claims to beat a zero-data constant-cap policy",
        )
        check(
            abs(f.get("constant_cap_policy_reward", -1.0) - f["best_feasible_reward"]) < 1e-9,
            f"constant_cap_policy_reward at cap {cap} dBm is not the best feasible reward",
        )
    check(
        "realised_regret" in free and "constant_cap_policy_reward" in free,
        "the free-constraint case did not report its zero-data constant-cap baseline",
    )

    # 3. Host-stable decisions reproduce the committed result (tolerance bands).
    committed_frontier = committed.get("frontier", [])
    check(
        [f["eirp_cap_dbm"] for f in frontier]
        == [f["eirp_cap_dbm"] for f in committed_frontier],
        "fresh run swept a different cap grid than the committed result",
    )
    for cf, ff in zip(committed_frontier, frontier):
        check(
            abs(cf["best_feasible_reward"] - ff["best_feasible_reward"])
            <= SERVED_FRACTION_TOL,
            f"best feasible reward at cap {cf['eirp_cap_dbm']} dBm moved by more than "
            f"{SERVED_FRACTION_TOL} vs the committed result",
        )
        check(
            abs(cf["utility_forgone"] - ff["utility_forgone"]) <= SERVED_FRACTION_TOL,
            f"utility forgone at cap {cf['eirp_cap_dbm']} dBm moved by more than "
            f"{SERVED_FRACTION_TOL} vs the committed result",
        )
    check(
        committed.get("utility_forgone_at_operational_cap", 0.0) > 0.0,
        "committed result does not show a binding constraint at 33 dBm",
    )
    committed_free = committed.get("free_constraint_case", {})
    check(
        committed_free.get("utility_cost") == 0.0,
        "committed free-constraint case is not zero-cost",
    )
    # The free case reproduces on its ACHIEVED REWARD within tolerance, never on
    # its subband index — see the module docstring. The index is power-dependent
    # and the whole decision is worth ~15 receivers of 4096, well inside the
    # threshold-flip band; an exact compare would be a latent flake.
    check(
        abs(committed_free.get("task_optimum_reward", 0.0) - free.get("task_optimum_reward", 1e9))
        <= SERVED_FRACTION_TOL,
        f"free-constraint task optimum moved by more than {SERVED_FRACTION_TOL} vs committed",
    )
    check(
        free.get("candidate_subbands") == committed_free.get("candidate_subbands"),
        "the Shield admitted a different set of subbands than in the committed result",
    )

    # 4. Trust chain.
    tc = fresh.get("trust_chain", {})
    check(tc.get("verify_first_broken_index") == -1, "evidence chain did not verify intact")
    check(tc.get("evidence_chain_length", 0) >= 2, "evidence chain is too short")
    check(
        tc.get("tampered_chain_refused") is True,
        "tampered chain was not refused for replay",
    )
    check(
        tc.get("verify_after_tamper_index") not in (None, -1),
        "tamper was not detected",
    )

    if errors:
        raise AssertionError(
            "safety-utility frontier reproduction failed:\n  - " + "\n  - ".join(errors)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed-result", type=Path, required=True)
    parser.add_argument("--actual-result", type=Path, required=True)
    parser.add_argument("--actual-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(
            committed=_load(args.committed_result),
            fresh=_load(args.actual_result),
            manifest=_load(args.actual_manifest),
        )
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("safety-utility frontier reproduction verified on real DeepMIMO data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
