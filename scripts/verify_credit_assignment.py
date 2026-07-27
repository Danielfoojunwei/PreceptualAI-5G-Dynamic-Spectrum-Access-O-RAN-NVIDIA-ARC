#!/usr/bin/env python3
"""Verify a fresh credit-assignment run against the committed real-data result.

The credit-assignment loop is deterministic given the same feature bytes (fixed
seeds, no wall-clock anywhere), but a cross-host DeepMIMO rebuild can shift a
few receivers' channel gains at the last ULP, which moves served fractions by a
handful of 1/4096 quanta. Floats are therefore compared with tolerance bands,
never byte-compared. What must reproduce exactly is what is host-stable and
scientifically load-bearing:

* the fresh run is bound to the canonical, checksum-pinned DeepMIMO build;
* the Shield genuinely binds (positive safety/utility gap, same sign as
  committed);
* the credit-assignment bias is real and ordered the same way: the naive
  proposed-credit regime's value error over infeasible EIRP bins strictly
  exceeds the projection-aware regime's, and the naive regime alone claims
  values for illegal actions;
* the terminal argmax decisions match (33 dBm for the shielded regimes, the
  grid-top 52 dBm for the oracle) including the naive regime's 20-way tie;
* correction-aware learning graduates (projection rate falls first->last
  quarter) while naive credit does not;
* nothing illegal was ever emitted, the chain verifies intact, the tampered
  chain is refused as training data, and the tamper is pinpointed.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MAE_TOL = 0.02  # served-fraction quantities: generous vs the ~1e-3 rebuild jitter.
RATE_TOL = 0.1  # projection rates over 60-step quarters (seeded rng, coarse grid).
GAP_TOL = 0.05

SHIELDED = ("naive_proposed_credit", "projection_aware", "correction_aware")
ALL_REGIMES = SHIELDED + ("unconstrained_oracle",)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(*, committed: dict[str, Any], fresh: dict[str, Any], manifest: dict[str, Any]) -> None:
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    # 1. Real-data binding. The fresh run must be bound to its own rebuilt
    #    feature file and derive from the same raw ray-tracing scenario as the
    #    committed result. features_sha256 is deliberately NOT compared
    #    cross-host (same last-ULP rationale as the federated-coverage and DSA
    #    verifiers); source_tree_sha256 and the receiver count are the binding.
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

    fr, cr = fresh["regimes"], committed["regimes"]

    # 2. The Shield binds, with the same sign and comparable magnitude.
    check(fresh["safety_utility_gap"] > 0.0, "safety_utility_gap is not positive")
    check(
        abs(fresh["safety_utility_gap"] - committed["safety_utility_gap"]) <= GAP_TOL,
        "safety_utility_gap moved outside the tolerance band",
    )

    # 3. The credit-assignment bias is real and ordered as committed.
    check(
        fr["naive_proposed_credit"]["value_mae_infeasible"]
        > fr["projection_aware"]["value_mae_infeasible"],
        "naive credit is not strictly worse than projection-aware over infeasible bins",
    )
    check(
        fr["naive_proposed_credit"]["value_mae_infeasible"] > 0.1,
        "naive credit's infeasible-bin value error is implausibly small",
    )
    check(
        abs(
            fr["naive_proposed_credit"]["value_mae_infeasible"]
            - cr["naive_proposed_credit"]["value_mae_infeasible"]
        )
        <= MAE_TOL,
        "naive infeasible-bin value MAE moved outside the tolerance band",
    )
    check(
        fr["projection_aware"]["infeasible_bins_claimed"] == 0
        and fr["correction_aware"]["infeasible_bins_claimed"] == 0,
        "projection/correction-aware regimes claimed values for illegal actions",
    )
    check(
        fr["naive_proposed_credit"]["infeasible_bins_claimed"] == 19,
        "naive regime did not claim all 19 infeasible bins",
    )
    check(
        fr["unconstrained_oracle"]["value_mae_infeasible"] <= 1e-9,
        "oracle regime's infeasible-bin values are not exact",
    )
    for name in ALL_REGIMES:
        check(
            fr[name]["value_mae_feasible"] <= MAE_TOL,
            f"{name}: feasible-bin value error is not near zero",
        )

    # 4. Host-stable terminal decisions reproduce the committed result.
    for name in ALL_REGIMES:
        check(
            fr[name]["argmax_proposed_eirp_dbm"] == cr[name]["argmax_proposed_eirp_dbm"],
            f"{name}: terminal argmax EIRP did not reproduce",
        )
        check(
            fr[name]["argmax_tie_count"] == cr[name]["argmax_tie_count"],
            f"{name}: terminal argmax tie structure did not reproduce",
        )
        check(
            abs(fr[name]["realised_mean_reward"] - cr[name]["realised_mean_reward"]) <= MAE_TOL,
            f"{name}: realised mean reward moved outside the tolerance band",
        )
    check(
        fr["naive_proposed_credit"]["argmax_tie_count"] == 20,
        "naive regime's terminal indifference (20-way tie incl. all illegal bins) is gone",
    )

    # 5. Graduation: correction-aware learns feasibility; naive never does.
    ca = fr["correction_aware"]
    check(
        ca["projection_rate_last_quarter"] < ca["projection_rate_first_quarter"],
        "correction-aware learning did not graduate",
    )
    check(
        ca["projection_rate_last_quarter"] <= RATE_TOL,
        "correction-aware last-quarter projection rate is not small",
    )
    check(
        fr["naive_proposed_credit"]["projection_rate_last_quarter"] > 0.5,
        "naive credit unexpectedly stopped leaning on the Shield",
    )

    # 6. Nothing illegal ever reached the RAN, in any regime.
    for name in ALL_REGIMES:
        check(
            fr[name]["max_executed_eirp_dbm"] <= 33.0 + 1e-9,
            f"{name}: an executed action exceeded the EIRP cap",
        )
    check(
        fr["naive_proposed_credit"]["illegal_proposals"] > 0,
        "naive regime proposed nothing illegal (bias would be unmeasurable)",
    )

    # 7. Trust chain.
    tc = fresh["trust_chain"]
    check(tc["verify_first_broken_index"] == -1, "evidence chain did not verify intact")
    check(
        tc["evidence_chain_length"] == tc["replay_transitions"],
        "replay did not recover every chained decision",
    )
    check(tc["replay_matches_online"] is True, "replayed tables diverge from online training")
    check(tc.get("tampered_chain_refused") is True, "tampered chain was not refused as training data")
    check(tc.get("verify_after_tamper_index") not in (None, -1), "tamper was not detected")

    if errors:
        raise AssertionError(
            "credit-assignment reproduction failed:\n  - " + "\n  - ".join(errors)
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
    print("credit-assignment reproduction verified on real DeepMIMO data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
