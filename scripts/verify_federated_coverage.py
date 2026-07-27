#!/usr/bin/env python3
"""Verify a fresh federated-coverage run against the committed real-data result.

The federated coverage loop is deterministic given the same feature bytes, but
the *poisoned FedAvg* path diverges by construction (RMSE ~1e7 dB), so its float
outputs are chaotic and must not be byte-compared across hosts. This verifier
therefore checks what is host-stable and meaningful:

* the fresh run is bound to the canonical, checksum-pinned DeepMIMO build;
* the security invariants hold on the freshly rebuilt real data (clean FL learns,
  the poison diverges plain FedAvg, Krum stays bounded and below baseline);
* the operational decisions from the well-converged models reproduce (the clean
  and robust coverage-fill target cells match the committed result);
* the trust chain verifies intact, refuses the over-power fill, and catches the
  tamper.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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

    base = fresh["baseline_predict_mean_rmse_db"]
    o = fresh["outcomes"]

    # 2. Security invariants on the freshly rebuilt real data.
    check(o["clean_fedavg"]["rmse_db"] < base, "clean FedAvg did not beat baseline")
    check(not o["clean_fedavg"]["diverged"], "clean FedAvg diverged")
    check(o["poisoned_fedavg"]["diverged"], "poison did not diverge plain FedAvg")
    check(
        (not o["poisoned_krum"]["diverged"])
        and o["poisoned_krum"]["rmse_db"] < base
        and o["poisoned_krum"]["rmse_db"] < o["poisoned_fedavg"]["rmse_db"],
        "Krum did not bound the poison below baseline",
    )
    check(
        o["dp_fedavg_clean"]["epsilon"] is not None and o["dp_fedavg_clean"]["epsilon"] > 0.0,
        "DP release has no positive certified epsilon",
    )

    # 3. Host-stable operational decisions reproduce the committed result.
    fm, cm = fresh["coverage_fill_misdirection"], committed["coverage_fill_misdirection"]
    check(
        fm["clean_target_receiver"] == cm["clean_target_receiver"],
        "clean coverage-fill target cell did not reproduce",
    )
    check(
        fm["robust_target_receiver"] == cm["robust_target_receiver"],
        "robust coverage-fill target cell did not reproduce",
    )
    check(fm["poison_moved_the_fill"] is True, "poison did not move the fill")

    # 4. Trust chain.
    tc = fresh["trust_chain"]
    check(tc["overpower_fill_corrected"] is True, "over-power fill was not corrected")
    check(tc["overpower_fill_reached_ran"] is False, "over-power fill reached the RAN")
    check(tc["verify_first_broken_index"] == -1, "evidence chain did not verify intact")
    check(
        tc.get("verify_after_tamper_index") not in (None, -1),
        "tamper was not detected",
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
