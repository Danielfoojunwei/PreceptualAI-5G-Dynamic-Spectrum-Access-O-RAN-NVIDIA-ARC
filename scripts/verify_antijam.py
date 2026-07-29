#!/usr/bin/env python3
"""Verify a fresh anti-jam null-steering run against the committed real result.

The anti-jam loop is deterministic given the same feature bytes, but the
feature bytes themselves are host-dependent at the last rounded decimal (see
``features_sha256_scope`` in the angular manifest), so per-receiver floats must
not be byte-compared across hosts. This verifier therefore checks what is
host-stable and meaningful:

* the fresh run is bound to its own rebuilt angular feature file, and derived
  from the *same raw ray-tracing scenario* as the committed result
  (source_tree_sha256);
* the anti-jam invariants hold on the freshly rebuilt real data (the jammer
  collapses the serving-beam SINR, MVDR restores it by > 5 dB mean, the null
  is > 10 dB deep, more receivers usable after mitigation);
* the host-stable operational decisions reproduce (fixed jammer azimuth and
  JSR, mean SINR/null-depth within tolerance bands, restored-fraction
  ordering) — chaotic last-decimal floats are compared with tolerance, never
  byte-for-byte;
* the trust chain verifies intact, refuses the over-EIRP null-steer, and
  catches the tamper.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SINR_TOL_DB = 1.0  # cross-host tolerance on mean/median SINR aggregates.
NULL_DEPTH_TOL_DB = 3.0  # cross-host tolerance on the mean null depth.
FRACTION_TOL = 0.02  # cross-host tolerance on restored fractions.


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
    #    cross-host: it is computed on angles/gains rounded to 4 decimals, so
    #    last-ULP float differences across machines change a few bytes.
    #    source_tree and receiver count are host-independent and are the real
    #    binding.
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

    # 2. Anti-jam invariants on the freshly rebuilt real data.
    check(fresh["mean_sinr_jammed_db"] < 0.0, "jammer did not collapse the serving-beam SINR")
    check(fresh["mean_sinr_gain_db"] > 5.0, "MVDR did not materially restore SINR (mean gain)")
    check(fresh["mean_null_depth_db"] > 10.0, "jammer was not genuinely nulled (mean depth)")
    check(
        fresh["restored_fraction_antijam"] > fresh["restored_fraction_jammed"],
        "anti-jam did not make more receivers usable",
    )

    # 3. Host-stable operational decisions reproduce the committed result.
    check(
        fresh["jammer"]["azimuth_deg"] == committed["jammer"]["azimuth_deg"],
        "jammer azimuth differs from the committed configuration",
    )
    check(
        fresh["jammer"]["jammer_to_signal_db"] == committed["jammer"]["jammer_to_signal_db"],
        "jammer-to-signal ratio differs from the committed configuration",
    )
    check(
        fresh["jammer"]["note"] == committed["jammer"]["note"],
        "jammer honesty note differs from the committed result",
    )
    for key in (
        "mean_sinr_jammed_db",
        "mean_sinr_antijam_db",
        "mean_sinr_gain_db",
    ):
        check(
            abs(fresh[key] - committed[key]) <= SINR_TOL_DB,
            f"{key} not within {SINR_TOL_DB} dB of the committed result",
        )
    check(
        abs(fresh["mean_null_depth_db"] - committed["mean_null_depth_db"]) <= NULL_DEPTH_TOL_DB,
        f"mean_null_depth_db not within {NULL_DEPTH_TOL_DB} dB of the committed result",
    )
    for key in ("restored_fraction_jammed", "restored_fraction_antijam"):
        check(
            abs(fresh[key] - committed[key]) <= FRACTION_TOL,
            f"{key} not within {FRACTION_TOL} of the committed result",
        )

    # 4. Trust chain.
    tc = fresh["trust_chain"]
    check(tc["overpower_nullsteer_corrected"] is True, "over-EIRP null-steer was not corrected")
    check(tc["overpower_nullsteer_reached_ran"] is False, "over-EIRP null-steer reached the RAN")
    check(tc["verify_first_broken_index"] == -1, "evidence chain did not verify intact")
    check(
        tc.get("verify_after_tamper_index") not in (None, -1),
        "tamper was not detected",
    )

    if errors:
        raise AssertionError("anti-jam reproduction failed:\n  - " + "\n  - ".join(errors))


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
    print("anti-jam null-steering reproduction verified on real DeepMIMO data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
