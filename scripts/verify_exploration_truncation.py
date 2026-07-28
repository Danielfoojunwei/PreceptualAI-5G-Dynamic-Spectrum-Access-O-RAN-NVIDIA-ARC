#!/usr/bin/env python3
"""Verify a fresh exploration-truncation run against the committed real result.

The loop is deterministic given the same feature bytes (fixed seed, no clock
randomness), but its tie-breaking wanders a flat reward plateau by design, and
last-ULP float differences in DeepMIMO's channel computation across hosts can
shift a served-fraction count by a receiver or two. Per-quarter rates are
therefore compared with tolerance bands, never byte-compared. This verifier
checks what is host-stable and meaningful:

* the fresh run is bound to the canonical, checksum-pinned DeepMIMO build;
* **legality is the Shield's own predicate.** This is the gate that would have
  caught the retracted 28-arm bug: if a run scores legality with anything other
  than ``Shield.is_feasible``, then ``illegal_proposal_rate`` and
  ``projection_rate`` stop being the same number and this verifier fails. The
  superseded centre-frequency test produced 0.878125 against a projection rate
  of 0.909375 — a gap this check rejects;
* the experiment's invariants hold on the freshly rebuilt real data (the
  learner is genuinely pulled toward illegality; correction_aware graduates
  while reward_only does not; the Shield intervenes less for the graduate;
  no executed action is ever illegal);
* **the free-lunch control is present and dominates.** ``shield_masked`` — the
  same bandit with the Shield's feasibility mask on its argmax — must reach
  zero illegal proposals with zero interventions, and must not be beaten on
  realised regret by the learner that had to be taught. Without this the
  graduation result reads as a capability rather than as a cost;
* **every learner is scored against the zero-data constant-cap policy.**
  ``realised_regret`` and ``constant_cap_policy_reward`` must be emitted, and
  the regret must be non-negative — i.e. the verifier refuses to let a run hide
  the fact that proposing the cap forever, with no data, is not beaten here;
* **correction_aware reports a shaped score, never a value estimate**, and no
  oracle error is computed against it;
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

    for required in ("reward_only", "correction_aware", "shield_masked"):
        if required not in fresh.get("learners", {}):
            raise AssertionError(
                f"exploration-truncation reproduction failed:\n  - "
                f"learner {required!r} is missing from the fresh result"
            )
    ro = fresh["learners"]["reward_only"]
    ca = fresh["learners"]["correction_aware"]
    sm = fresh["learners"]["shield_masked"]

    # 2a. P1 — legality must be the SHIELD's predicate, not a private
    #     restatement of it. Under the Shield's predicate an action is
    #     infeasible IFF the Shield projects it, so these two rates are the
    #     same number. The retracted centre-frequency test broke exactly this
    #     identity (0.878125 illegal vs 0.909375 projected on reward_only).
    fp = fresh.get("feasibility_predicate")
    check(isinstance(fp, dict), "fresh result does not declare its feasibility predicate")
    if isinstance(fp, dict):
        check(
            "Shield.is_feasible" in str(fp.get("source", "")),
            "feasibility is not derived from Shield.is_feasible",
        )
        check(
            fp.get("illegal_rate_equals_projection_rate") is True,
            "illegal_proposal_rate and projection_rate are not the same quantity",
        )
        check(
            fp.get("n_infeasible_arms_shield") == fresh["action_space"]["n_infeasible_arms"],
            "action_space infeasible count was not derived from the Shield predicate",
        )
    for name, learner in fresh["learners"].items():
        summary = learner["closed_loop_summary"]
        check(
            abs(summary["illegal_proposal_rate"] - summary["projection_rate"]) <= 1e-9,
            f"{name}: illegal_proposal_rate {summary['illegal_proposal_rate']} != "
            f"projection_rate {summary['projection_rate']}; legality is being "
            "scored against a feasible set the Shield does not agree with",
        )
        check(
            learner.get("projection_flag_mismatches") == 0,
            f"{name}: per-step legality flags disagree with the Shield's projections",
        )

    # 2b. P3 — the zero-data baseline must be emitted, and the learners must be
    #     scored against it. A run that omits it, or that claims to beat a
    #     constant-cap policy it never measured, fails here.
    for name, learner in fresh["learners"].items():
        for key in ("realised_regret", "realised_regret_q4", "constant_cap_policy_reward"):
            check(key in learner, f"{name}: {key} is not reported")
        if "constant_cap_policy_reward" in learner:
            check(
                abs(learner["constant_cap_policy_reward"] - fresh["optimal_feasible_reward"])
                <= 1e-3,
                f"{name}: constant_cap_policy_reward is not the feasible optimum",
            )
        if "realised_regret" in learner:
            check(
                learner["realised_regret"] >= -1e-9,
                f"{name}: realised_regret is negative — a shielded learner cannot beat "
                "the feasible optimum; the baseline or the reward is wrong",
            )

    # 2c. P3(d) — correction_aware's number above the cap is a shaped score.
    belief = ca["infeasible_belief"]
    check(
        "shaped_score_at_eirp_50_dbm" in belief,
        "correction_aware does not label its above-cap number as a shaped score",
    )
    check(
        "value_estimate_at_eirp_50_dbm" not in belief,
        "correction_aware still reports a penalised score as a value estimate",
    )
    check(
        "mean_abs_error_vs_oracle_above_cap" not in belief,
        "correction_aware still reports an oracle error against a penalised score "
        "(comparing a penalised score with an unpenalised oracle is a category error)",
    )
    check(
        belief.get("shaped_score_identity_holds") is True,
        "the shaped score is not optimal_feasible_reward - CORRECTION_PENALTY; "
        "the identity that proves it is not a learned quantity no longer holds",
    )

    # 2d. P1 defence in depth — the free-lunch control must be present, must
    #     work, and must not be beaten by the taught learner.
    check(sm.get("action_mask_applied") is True, "shield_masked did not apply the mask")
    check(
        sm["illegal_proposals_blocked"] == 0 and sm["shield_interventions_total"] == 0,
        "the Shield's own feasibility mask did not eliminate illegal proposals",
    )
    check(
        all(sm[f"illegal_proposal_rate_q{q}"] == 0.0 for q in (1, 2, 3, 4)),
        "masked learner proposed an illegal action in some quarter",
    )
    if "realised_regret" in sm and "realised_regret" in ca:
        check(
            sm["realised_regret"] <= ca["realised_regret"] + 1e-9,
            "the masked learner was beaten on realised regret by the taught learner",
        )

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

    # 2e. The above-cap plateau is a property of the Shield, not the reward.
    #     The retracted claim was "zero gradient"; the measured true slope
    #     above the cap is a large fraction of the sub-cap slope. If a run ever
    #     reports a flat true reward again, this fails.
    geom = fresh.get("above_cap_reward_geometry")
    check(isinstance(geom, dict), "the above-cap reward geometry is not reported")
    if isinstance(geom, dict):
        check(
            geom["true_slope_above_cap_per_db"] > 0.0,
            "the true reward above the cap is reported as flat; the plateau the "
            "learner sees comes from the Shield's projection, not from the reward",
        )
        check(
            geom["true_slope_ratio_above_over_below"] > 0.5,
            "above-cap slope collapsed relative to the sub-cap slope; the pull "
            "toward illegality is no longer real on this build",
        )

    # 3. Host-stable decisions match the committed result (tolerance bands,
    #    not byte-compares of chaotic floats).
    cro = committed["learners"]["reward_only"]
    cca = committed["learners"]["correction_aware"]
    # The committed result must itself have been produced under the Shield's
    # predicate. A committed artifact still carrying the retracted 213-arm
    # feasible set fails here, so the fix cannot be half-landed.
    check(
        committed["action_space"]["n_infeasible_arms"]
        == fresh["action_space"]["n_infeasible_arms"],
        "the committed result was measured against a different feasible set than "
        "the fresh run (was the committed artifact regenerated after the "
        "Shield-predicate fix?)",
    )
    check(
        "shield_masked" in committed["learners"],
        "the committed result predates the masked control learner",
    )
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
        tc["evidence_chain_length"]
        == len(fresh["learners"]) * fresh["training"]["steps_per_learner"],
        "evidence chain does not cover every decision of every learner",
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
