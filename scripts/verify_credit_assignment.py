#!/usr/bin/env python3
"""Verify a fresh credit-assignment run against the committed real-data result.

The credit-assignment loop is deterministic given the same feature bytes (fixed
seeds, no wall-clock anywhere), but a cross-host DeepMIMO rebuild can shift a
few receivers' channel gains at the last ULP, which moves served fractions by a
handful of 1/4096 quanta. Floats are therefore compared with tolerance bands,
never byte-compared. What must reproduce exactly is what is host-stable and
scientifically load-bearing:

* the fresh run is bound to the canonical, checksum-pinned DeepMIMO build;
* **the hard safety guarantee**: in every arm and every regime, no executed
  action exceeded the 33 dBm cap and none left the band; the chain verifies
  intact, replays exactly, and the injected tamper is caught at its exact index;
* **P1** — the feasibility mask is the Shield's own predicate (it agrees with
  what the Shield actually projects on every bin), and masking the action space
  collapses the naive regime's 20-way argmax tie to 1, drives every regime's
  illegal proposals to 0, and leaves no regime claiming a value for an
  infeasible bin;
* **P2** — ``CORRECTION_PENALTY`` genuinely changes behaviour in the unmasked
  arm (proposal trajectory AND a realised scalar both move when it is switched
  off), and is provably a no-op once the action space is masked;
* **P3** — accuracy claims are not satisfiable by abstention (a regime that
  claims no infeasible bin reports ``null``, not 0.0, and is flagged as having
  abstained); every regime emits ``realised_regret_*`` and
  ``constant_cap_policy_reward``; the zero-data constant-cap baseline is
  emitted and the reported learning gain over it is not positive;
* the terminal argmax decisions and realised rewards reproduce within tolerance.

What this verifier deliberately no longer checks, because the post-mortem showed
the check was vacuous (see ``deploy/shield-learning/ERRATA.md``):

* ``naive.value_mae_infeasible > projection_aware.value_mae_infeasible``. The
  right-hand side was 0.0 only because ``_mae_over`` scored an EMPTY claimed set
  as perfect. The comparison was satisfiable by abstention and is retracted; it
  is replaced by an explicit "naive claims all 19 bins and is wrong by > 0.1"
  plus "projection-aware abstained, and says so".
* ``naive.argmax_tie_count == 20`` as evidence about the Shield. The tie count
  is identically ``EIRP_MAX_DBM - cap + 1`` — a property of where the grid top
  was placed. It is still compared against the committed run for reproduction,
  but it is no longer asserted to a magic constant.

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
ARMS = ("unmasked", "shield_masked")
EIRP_CAP_DBM = 33.0


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

    for arm in ARMS:
        check(arm in fresh.get("arms", {}), f"fresh result is missing the {arm} arm")
    if errors:
        raise AssertionError(
            "credit-assignment reproduction failed:\n  - " + "\n  - ".join(errors)
        )

    fa, ca = fresh["arms"], committed["arms"]
    fu, cu = fa["unmasked"]["regimes"], ca["unmasked"]["regimes"]
    fm = fa["shield_masked"]["regimes"]

    # 2. The Shield binds, with the same sign and comparable magnitude.
    check(fresh["safety_utility_gap"] > 0.0, "safety_utility_gap is not positive")
    check(
        abs(fresh["safety_utility_gap"] - committed["safety_utility_gap"]) <= GAP_TOL,
        "safety_utility_gap moved outside the tolerance band",
    )

    # 3. THE HARD GUARANTEE — every arm, every regime. This is the claim the
    #    post-mortem could not break, and it is checked first-class.
    for arm in ARMS:
        for name in ALL_REGIMES:
            r = fa[arm]["regimes"][name]
            check(
                r["max_executed_eirp_dbm"] <= EIRP_CAP_DBM + 1e-9,
                f"{arm}/{name}: an executed action exceeded the EIRP cap",
            )
            check(
                r["executed_out_of_band_steps"] == 0,
                f"{arm}/{name}: an executed action left the licensed band",
            )

    # 4. P1 — feasibility comes from the Shield, and masking removes the
    #    pathology without weakening the guarantee.
    am = fresh["action_space_masking"]
    check(
        am.get("mask_agrees_with_projection_on_every_bin") is True,
        "the feasibility mask disagrees with what the Shield actually projects",
    )
    check(
        am.get("mask_matches_analytic_eirp_cap") is True,
        "the Shield-derived mask does not match the analytic EIRP cap on this grid",
    )
    check(
        am.get("feasible_bins") == committed["action_space_masking"]["feasible_bins"],
        "the number of Shield-feasible bins did not reproduce",
    )
    check(
        fu["naive_proposed_credit"]["argmax_tie_count"] > 1,
        "the unmasked naive regime lost its terminal argmax tie (the bias is unmeasurable)",
    )
    for name in ALL_REGIMES:
        check(
            fm[name]["argmax_tie_count"] == 1,
            f"shield_masked/{name}: masking did not collapse the argmax tie",
        )
        check(
            fm[name]["illegal_proposals"] == 0,
            f"shield_masked/{name}: a masked learner still proposed an infeasible bin",
        )
        check(
            fm[name]["infeasible_bins_claimed"] == 0,
            f"shield_masked/{name}: a masked learner claimed a value for an infeasible bin",
        )
    check(
        fu["naive_proposed_credit"]["illegal_proposals"] > 0,
        "unmasked naive regime proposed nothing illegal (bias would be unmeasurable)",
    )
    # The loop must judge legality with the Shield's predicate. At this loop's
    # fixed interior subband the shared substrate's StepResult flag happens to
    # agree; that agreement is verified, not assumed, because the substrate's
    # flag uses a centre-frequency test that is wrong at the band edges.
    for arm in ARMS:
        for name in ALL_REGIMES:
            check(
                fa[arm]["regimes"][name]["illegal_predicate_agrees_with_substrate"] is True,
                f"{arm}/{name}: Shield and substrate legality predicates disagree",
            )

    # 5. P2 — the correction penalty must actually bite, and must be shown to be
    #    a no-op where it is one. A gate keyed on argmax_tie_count would pass
    #    today while proving nothing; this one is keyed on the proposal
    #    trajectory and on a realised scalar.
    probe = fresh["correction_penalty_probe"]
    check(
        probe["unmasked"]["proposal_trajectory_differs"] is True,
        "CORRECTION_PENALTY=0 does not change the unmasked proposal trajectory (penalty is inert)",
    )
    check(
        probe["unmasked"]["realised_scalar_differs"] is True,
        "CORRECTION_PENALTY=0 does not move any realised scalar (penalty is inert)",
    )
    check(probe["unmasked"]["penalty_bites"] is True, "the correction penalty is inert")
    check(
        probe["shield_masked"]["penalty_bites"] is False,
        "the correction penalty is reported as biting under masking, where nothing is projected",
    )
    check(
        probe["unmasked"]["penalty_off"]["projected_steps"]
        > probe["unmasked"]["penalty_on"]["projected_steps"],
        "switching the correction penalty off did not increase the Shield's correction load",
    )

    # 6. P3(a) — accuracy claims are not satisfiable by abstention.
    naive = fu["naive_proposed_credit"]
    check(
        naive["infeasible_bins_claimed"] == fresh["learner"]["infeasible_bins"],
        "unmasked naive regime did not claim every infeasible bin",
    )
    check(
        naive["value_mae_infeasible"] is not None and naive["value_mae_infeasible"] > 0.1,
        "unmasked naive regime's infeasible-bin value error is absent or implausibly small",
    )
    check(
        naive["value_mae_infeasible"] is not None
        and abs(
            naive["value_mae_infeasible"] - cu["naive_proposed_credit"]["value_mae_infeasible"]
        )
        <= MAE_TOL,
        "naive infeasible-bin value MAE moved outside the tolerance band",
    )
    for name in ("projection_aware", "correction_aware"):
        r = fu[name]
        check(
            r["infeasible_bins_claimed"] == 0,
            f"unmasked/{name}: claimed a value for an illegal action",
        )
        check(
            r["value_mae_infeasible"] is None and r["abstained_on_infeasible_bins"] is True,
            f"unmasked/{name}: abstention is being scored as accuracy instead of reported as null",
        )
    # Against the REALISABLE truth R∘Π the naive regime is exactly right and the
    # oracle is the one in error — both columns must be present and ordered.
    check(
        naive["value_mae_infeasible_vs_projected_truth"] is not None
        and naive["value_mae_infeasible_vs_projected_truth"] <= 1e-6,
        "naive regime is not exact against the post-projection truth R∘Pi",
    )
    check(
        fu["unconstrained_oracle"]["value_mae_infeasible_vs_projected_truth"] is not None
        and fu["unconstrained_oracle"]["value_mae_infeasible_vs_projected_truth"] > 0.1,
        "oracle regime is not in error against the post-projection truth R∘Pi",
    )
    check(
        fu["unconstrained_oracle"]["value_mae_infeasible"] is not None
        and fu["unconstrained_oracle"]["value_mae_infeasible"] <= 1e-9,
        "oracle regime's infeasible-bin values are not exact against R",
    )
    # P3(c) — the grid-top sensitivity of the headline MAE must be published.
    sens = fresh["value_mae_infeasible_sensitivity"]["grid_top_dbm"]
    check(
        {"34", "52", "110"} <= set(sens),
        "grid-top sensitivity of value_mae_infeasible is not reported",
    )
    check(
        {"34", "110"} <= set(sens)
        and sens["110"]["value_mae_infeasible_if_naive"]
        > 10 * sens["34"]["value_mae_infeasible_if_naive"],
        "grid-top sensitivity does not reproduce the >10x span (34 dBm vs 110 dBm)",
    )
    published_top = str(
        int(fresh["value_mae_infeasible_sensitivity"]["published_grid_top_dbm"])
    )
    check(
        naive["value_mae_infeasible"] is not None
        and published_top in sens
        and abs(sens[published_top]["value_mae_infeasible_if_naive"] - naive["value_mae_infeasible"])
        <= MAE_TOL,
        "the naive MAE is not reproduced by the pure grid-top formula (the claim is unexplained)",
    )

    # 7. P3(b) — the zero-data baseline is emitted and nothing is overclaimed.
    zdb = fresh["zero_data_baseline"]
    check(
        zdb["equals_optimal_feasible"] is True,
        "the zero-data constant-cap policy no longer equals the feasible optimum "
        "(the baseline must be recomputed, not dropped)",
    )
    check(
        abs(zdb["realised_mean_reward"] - fresh["env"]["optimal_feasible_reward"]) <= MAE_TOL,
        "constant-cap baseline reward drifted from the feasible optimum",
    )
    check(
        zdb["measured_learning_gain_last_quarter"] <= 1e-9,
        "a positive learning gain over the zero-data policy is being claimed; "
        "the committed measurement is exactly 0.000000",
    )
    check(
        zdb["measured_learning_gain_all"] < 0.0,
        "the full-run learning gain over the zero-data policy is not reported as negative",
    )
    for arm in ARMS:
        for name in ALL_REGIMES:
            r = fa[arm]["regimes"][name]
            for field in (
                "constant_cap_policy_reward",
                "realised_regret_last_quarter",
                "realised_regret_all",
                "learning_gain_over_constant_cap_policy_last_quarter",
            ):
                check(r.get(field) is not None, f"{arm}/{name}: {field} is not emitted")

    # 8. Host-stable terminal decisions reproduce the committed result.
    for arm in ARMS:
        for name in ALL_REGIMES:
            fr_, cr_ = fa[arm]["regimes"][name], ca[arm]["regimes"][name]
            check(
                fr_["argmax_proposed_eirp_dbm"] == cr_["argmax_proposed_eirp_dbm"],
                f"{arm}/{name}: terminal argmax EIRP did not reproduce",
            )
            check(
                fr_["argmax_tie_count"] == cr_["argmax_tie_count"],
                f"{arm}/{name}: terminal argmax tie structure did not reproduce",
            )
            check(
                abs(fr_["realised_mean_reward"] - cr_["realised_mean_reward"]) <= MAE_TOL,
                f"{arm}/{name}: realised mean reward moved outside the tolerance band",
            )
            check(
                fr_["value_mae_feasible"] is not None
                and fr_["value_mae_feasible"] <= MAE_TOL,
                f"{arm}/{name}: feasible-bin value error is absent or not near zero",
            )

    # 9. Graduation, stated for what it is. correction_aware's projection rate
    #    falls across the unmasked run; under masking it is 0 from step one
    #    because there is nothing to graduate from.
    cw = fu["correction_aware"]
    check(
        cw["projection_rate_last_quarter"] < cw["projection_rate_first_quarter"],
        "unmasked correction-aware learning did not graduate",
    )
    check(
        cw["projection_rate_last_quarter"] <= RATE_TOL,
        "unmasked correction-aware last-quarter projection rate is not small",
    )
    check(
        fm["correction_aware"]["projection_rate_first_quarter"] == 0.0
        and fm["correction_aware"]["projection_rate_last_quarter"] == 0.0,
        "masked correction-aware still needed the Shield to correct it",
    )
    check(
        fu["naive_proposed_credit"]["projection_rate_last_quarter"] > 0.5,
        "unmasked naive credit unexpectedly stopped leaning on the Shield",
    )

    # 10. Trust chain.
    tc = fresh["trust_chain"]
    check(tc["verify_first_broken_index"] == -1, "evidence chain did not verify intact")
    check(
        tc["evidence_chain_length"]
        == len(ARMS) * len(ALL_REGIMES) * fresh["learner"]["rounds_per_regime"],
        "the chain does not carry every decision of every arm and regime",
    )
    check(
        tc["evidence_chain_length"] == tc["replay_transitions"],
        "replay did not recover every chained decision",
    )
    check(tc["replay_matches_online"] is True, "replayed tables diverge from online training")
    check(
        tc.get("tampered_chain_refused") is True,
        "tampered chain was not refused as training data",
    )
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
