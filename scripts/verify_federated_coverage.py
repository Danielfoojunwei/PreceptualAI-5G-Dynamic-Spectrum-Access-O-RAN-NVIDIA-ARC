#!/usr/bin/env python3
"""Verify a fresh federated-coverage run against the committed real-data result.

The federated coverage loop is deterministic given the same feature bytes, but
the *poisoned FedAvg* path diverges by construction (RMSE ~1e7 dB), so its float
outputs are chaotic and must not be byte-compared across hosts. This verifier
therefore checks what is host-stable and meaningful.

What changed after the adversarial post-mortem
----------------------------------------------
The previous version of this script gated on two claims that do not survive a
partition-seed sweep, and it is worth being explicit about why they are gone:

* ``krum_rmse < baseline`` held by 0.65 dB against a seed-induced sd of 1.87 dB
  and failed outright at 1 of 12 seeds. A gate that a reseed can flip is not
  evidence; it is a coin toss with a citation. It is replaced by two claims that
  hold at 24 of 24 seeds — **separation** (the robust aggregator is orders of
  magnitude below the poisoned mean, i.e. divergence was actually stopped) and a
  **bounded absolute** ceiling.
* ``robust_target_receiver == 9`` gated on the *identity* of a cell, not on the
  operational claim, and reproduced at only 8 of 12 seeds. It is replaced by the
  predicate the claim actually needs: the selected cell lies in the bottom decile
  of the clean model's own predicted coverage.

The fresh run additionally has to certify **the same epsilon** it always did:
the DP fix is a utility fix at a pinned privacy budget, so a moved epsilon would
mean the improvement was bought rather than earned, and is a hard failure here.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Kept in sync with benchmarks/federated_coverage_loop.py. Duplicated rather than
# imported so the verifier can run against a result JSON without the package.
SEPARATION_FACTOR = 1e4
ROBUST_RMSE_CEILING_DB = 22.0
FLTRUST_PARITY_TOLERANCE = 1.10
DP_TARGET_EPSILON = 4.1447
EXPECTED_GATES = 17


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
    #    cross-host: it is computed on gains rounded to 6 dB decimals, so
    #    last-ULP float differences in DeepMIMO's channel computation across
    #    machines change a few bytes (the same reason the DSA reproducer binds on
    #    source_tree_sha256 and compares floats with tolerance). source_tree and
    #    receiver count are host-independent and are the real binding.
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
    check(
        fresh.get("regression_injected") == "none",
        "fresh run was produced with a regression deliberately injected",
    )

    base = fresh["baseline_predict_mean_rmse_db"]
    o = fresh["outcomes"]
    naive = o["poisoned_fedavg"]["rmse_db"]

    # 2. Security invariants on the freshly rebuilt real data.
    check(o["clean_fedavg"]["rmse_db"] < base, "clean FedAvg did not beat baseline")
    check(not o["clean_fedavg"]["diverged"], "clean FedAvg diverged")
    check(o["poisoned_fedavg"]["diverged"], "poison did not diverge plain FedAvg")
    for method in ("krum", "fltrust"):
        r = o[f"poisoned_{method}"]
        check(not r["diverged"], f"{method} diverged under poison")
        check(
            r["rmse_db"] < naive / SEPARATION_FACTOR,
            f"{method} is not separated from the poisoned mean by {SEPARATION_FACTOR:g}x",
        )
        check(
            r["rmse_db"] <= ROBUST_RMSE_CEILING_DB,
            f"{method} exceeded the {ROBUST_RMSE_CEILING_DB} dB robustness ceiling",
        )
    # FLTrust's claim is the strong one: clean-FedAvg parity under attack.
    check(
        o["poisoned_fltrust"]["rmse_db"]
        <= o["clean_fedavg"]["rmse_db"] * FLTRUST_PARITY_TOLERANCE,
        "FLTrust did not reach clean-FedAvg parity under the shipped attack",
    )

    # 3. Every attack in the battery is bounded by FLTrust, and FLTrust still
    #    works past the adversary count where Krum is undefined.
    battery = fresh.get("attack_battery", {})
    check(bool(battery), "fresh run carries no attack battery")
    for attack, row in battery.items():
        check(
            row["fltrust"] <= ROBUST_RMSE_CEILING_DB,
            f"FLTrust exceeded the robustness ceiling under the {attack} attack",
        )
    breakdown = fresh.get("byzantine_breakdown", {}).get("by_n_malicious", {})
    check(bool(breakdown), "fresh run carries no Byzantine breakdown sweep")
    for f_str, row in breakdown.items():
        check(
            row["fltrust"] is not None and row["fltrust"] <= ROBUST_RMSE_CEILING_DB,
            f"FLTrust exceeded the robustness ceiling at f={f_str}",
        )
    check(
        any(row.get("krum") is None for row in breakdown.values()),
        "Krum was never undefined in the sweep; the breakdown claim is unproven",
    )

    # 4. Differential privacy: a utility fix at a PINNED privacy budget.
    dp = fresh["dp_at_fixed_epsilon"]
    check(
        dp["legacy"]["epsilon"] == dp["tuned"]["epsilon"] == DP_TARGET_EPSILON,
        "DP tuning moved the certified epsilon; the improvement was bought, not earned",
    )
    check(dp["adjacency"] == "replace_one", "DP adjacency was relaxed to a weaker convention")
    check(
        dp["tuned"]["rmse_db"] < base,
        "tuned DP-FedAvg did not beat the predict-mean baseline",
    )
    check(
        dp["tuned"]["rmse_db"] < dp["legacy"]["rmse_db"],
        "tuned DP-FedAvg is not better than the legacy configuration",
    )
    check(
        dp["clip_calibration"]["legacy_clip_binds_count"] == 0,
        "the legacy clip binds on the clean trajectory; the defect diagnosis no longer holds",
    )

    # 5. Host-stable operational decision. The claim is 'the fill lands on a
    #    genuinely badly covered cell', gated as a predicate, not a cell id.
    fm = fresh["coverage_fill_misdirection"]
    check(
        fm["robust_target_in_bottom_decile"] is True,
        "the robust coverage-fill did not target a bottom-decile cell",
    )
    check(
        fm["robust_recovers_clean_target"] is True,
        "the robust coverage-fill did not recover the clean model's target cell",
    )
    check(fm["poison_moved_the_fill"] is True, "poison did not move the fill")

    # 6. Seed robustness: every gate must hold at every partition seed.
    sr = fresh["seed_robustness"]
    check(sr["n_seeds"] >= 12, "seed sweep used fewer than 12 partition seeds")
    check(
        sr["all_gates_hold_every_seed"] is True,
        "at least one gate failed at some partition seed: "
        + json.dumps(
            {k: v for k, v in sr["gates_held"].items() if v != sr["n_seeds"]}
        ),
    )

    # 7. Trust chain. The Shield remains the sole emit gate.
    tc = fresh["trust_chain"]
    check(tc["overpower_fill_corrected"] is True, "over-power fill was not corrected")
    check(tc["overpower_fill_reached_ran"] is False, "over-power fill reached the RAN")
    check(tc["verify_first_broken_index"] == -1, "evidence chain did not verify intact")
    check(
        tc.get("verify_after_tamper_index") not in (None, -1),
        "tamper was not detected",
    )

    # 8. The run's own gate block must agree with everything above.
    gates = fresh.get("gates", {})
    check(len(gates) == EXPECTED_GATES, f"expected {EXPECTED_GATES} gates, got {len(gates)}")
    check(
        fresh.get("all_gates_pass") is True,
        "fresh run reported failing gates: "
        + json.dumps([k for k, v in gates.items() if not v]),
    )

    if errors:
        raise AssertionError("federated-coverage reproduction failed:\n  - " + "\n  - ".join(errors))


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
    print("federated-coverage reproduction verified on real DeepMIMO data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
