#!/usr/bin/env python3
"""Verify a fresh exploration-truncation run against the committed real result.

The loop is deterministic given the same feature bytes (fixed seed, no clock
randomness), but its tie-breaking wanders a flat reward plateau by design, and
last-ULP float differences in DeepMIMO's channel computation across hosts can
shift a served-fraction count by a receiver or two. Per-quarter rates are
therefore compared with tolerance bands, never byte-compared. This verifier
checks what is host-stable and meaningful:

* the fresh run is bound to the canonical, checksum-pinned DeepMIMO build;
* the experiment's invariants hold on the freshly rebuilt real data (the
  learner is genuinely pulled toward illegality; correction_aware graduates
  while reward_only does not; the Shield intervenes less for the graduate;
  no executed action is ever illegal);
* the host-stable decisions match the committed result within tolerance
  (graduation ordering, intervention ordering, unobserved-bin count, the
  worst over-cap excess, the feasible optimum);
* the trust chain verifies intact and the tampered chain is refused as
  training data.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RATE_TOL = 0.15  # tolerance band for per-quarter rates (plateau tie-breaks).


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(*, committed: dict[str, Any], fresh: dict[str, Any], manifest: dict[str, Any]) -> None:
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    # 1. Real-data binding. The fresh run must be bound to its own rebuilt
    #    feature file, and derived from the *same raw ray-tracing scenario* as
    #    the committed result. features_sha256 is deliberately NOT compared
    #    cross-host (see the manifest's features_sha256_scope note); the
    #    source tree hash and receiver count are the host-independent binding.
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

    ro = fresh["learners"]["reward_only"]
    ca = fresh["learners"]["correction_aware"]

    # 2. Experiment invariants on the freshly rebuilt real data.
    check(
        ro["illegal_proposal_rate_q1"] > 0.0,
        "reward_only was never pulled toward illegal proposals in Q1",
    )
    check(
        ca["illegal_proposal_rate_q4"] < ca["illegal_proposal_rate_q1"],
        "correction_aware did not graduate (Q4 not below Q1)",
    )
    check(
        ca["illegal_proposal_rate_q4"] < 0.1,
        "correction_aware Q4 illegal rate did not reach ~0",
    )
    check(
        ca["illegal_proposal_rate_q4"] < ro["illegal_proposal_rate_q4"],
        "correction_aware Q4 is not below reward_only Q4",
    )
    check(
        ro["illegal_proposal_rate_q4"] > 0.5,
        "reward_only unexpectedly stopped proposing illegal actions",
    )
    check(
        ca["shield_interventions_total"] < ro["shield_interventions_total"],
        "the Shield did not intervene less for the correction-aware learner",
    )
    for name, learner in fresh["learners"].items():
        check(
            learner["max_executed_eirp_dbm"] <= 33.0 + 1e-9,
            f"{name}: an executed action exceeded the EIRP cap",
        )
        check(
            learner["executed_out_of_band_steps"] == 0,
            f"{name}: an executed action left the band",
        )
        check(
            learner["unobserved_infeasible_bins"]
            == fresh["action_space"]["n_infeasible_arms"],
            f"{name}: some infeasible bin claims a real observation",
        )

    # 3. Host-stable decisions match the committed result (tolerance bands,
    #    not byte-compares of chaotic floats).
    cro = committed["learners"]["reward_only"]
    cca = committed["learners"]["correction_aware"]
    for quarter in ("q1", "q4"):
        key = f"illegal_proposal_rate_{quarter}"
        check(
            abs(ro[key] - cro[key]) <= RATE_TOL,
            f"reward_only {key} drifted beyond tolerance from the committed run",
        )
        check(
            abs(ca[key] - cca[key]) <= RATE_TOL,
            f"correction_aware {key} drifted beyond tolerance from the committed run",
        )
    check(
        ca["graduated"] is cca["graduated"] is True,
        "graduation decision did not reproduce",
    )
    check(
        ro["graduated"] is cro["graduated"] is False,
        "reward_only graduation decision did not reproduce",
    )
    check(
        abs(
            fresh["counterfactual_harm_prevented"]["max_over_cap_excess_db"]
            - committed["counterfactual_harm_prevented"]["max_over_cap_excess_db"]
        )
        <= 1.0,
        "worst over-cap excess did not reproduce",
    )
    check(
        abs(fresh["optimal_feasible_reward"] - committed["optimal_feasible_reward"]) <= 1e-3,
        "feasible optimum drifted beyond float tolerance",
    )

    # 4. Trust chain.
    tc = fresh["trust_chain"]
    check(tc["verify_first_broken_index"] == -1, "evidence chain did not verify intact")
    check(
        tc["tampered_chain_refused"] is True,
        "tampered chain was not refused as training data",
    )
    check(
        tc.get("verify_after_tamper_index") not in (None, -1),
        "tamper was not detected",
    )
    check(
        tc["evidence_chain_length"] == 2 * fresh["training"]["steps_per_learner"],
        "evidence chain does not cover every decision of both learners",
    )

    if errors:
        raise AssertionError(
            "exploration-truncation reproduction failed:\n  - " + "\n  - ".join(errors)
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
    print("exploration-truncation reproduction verified on real DeepMIMO data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
