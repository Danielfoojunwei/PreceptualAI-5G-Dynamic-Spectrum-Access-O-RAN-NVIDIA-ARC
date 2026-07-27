#!/usr/bin/env python3
"""Verify a fresh beam-management run against the committed real-data result.

The beam sweep is deterministic given the same feature bytes, but the angular
feature file itself is host-dependent at the last decimal (angles/gains rounded
to 4 decimals; see ``features_sha256_scope`` in the manifest), so per-receiver
floats must not be byte-compared across hosts. This verifier therefore checks
what is host-stable and meaningful:

* the fresh run is bound to its own rebuilt, checksum-pinned angular build and
  derives from the same raw ray-tracing scenario as the committed result;
* the physical invariants hold on the freshly rebuilt real data (mean array
  gain > 6 dB and under the 10*log10(8) ceiling, steering beats a fixed beam,
  the codebook is genuinely exercised);
* the operational decisions reproduce (modal serving beam, boresight beam,
  beam histogram within a small boundary-flip tolerance, mean gains within a
  tolerance band);
* the trust chain verifies intact, refuses the over-power beam activation, and
  catches the tamper.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

COHERENT_BOUND_DB = 9.0309  # 10*log10(8): the hard 8-element combining ceiling.
MEAN_GAIN_TOL_DB = 0.1  # last-ULP feature differences move the mean far less.
RECOVERY_TOL_DB = 0.5
HIST_TOL = 8  # receivers sitting exactly on a beam boundary may flip.


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(*, committed: dict[str, Any], fresh: dict[str, Any], manifest: dict[str, Any]) -> None:
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    # 1. Real-data binding. The fresh run must be bound to its own rebuilt
    #    angular feature file, and derived from the *same raw ray-tracing
    #    scenario* as the committed result. features_sha256 is deliberately NOT
    #    compared cross-host (rounded floats; see the manifest's
    #    features_sha256_scope). source_tree and receiver count are
    #    host-independent and are the real binding.
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

    # 2. Physical invariants on the freshly rebuilt real data.
    check(
        6.0 < fresh["mean_array_gain_db"] <= COHERENT_BOUND_DB + 0.1,
        "mean array gain is not in the real coherent-combining range (6, 9.03] dB",
    )
    check(
        fresh["max_array_gain_db"] <= COHERENT_BOUND_DB + 0.01,
        "a receiver exceeded the 10*log10(8) combining ceiling",
    )
    check(
        fresh["mean_misalignment_recovery_db"] > 1.0,
        "selected beams did not beat the fixed boresight beam by > 1 dB",
    )
    check(
        fresh["distinct_beams_selected"] >= 4,
        "the real AoD spread did not exercise >= 4 codebook beams",
    )

    # 3. Host-stable operational decisions reproduce the committed result.
    check(
        fresh["modal_beam_index"] == committed["modal_beam_index"],
        "modal serving beam did not reproduce",
    )
    check(
        fresh["codebook"]["boresight_beam_index"]
        == committed["codebook"]["boresight_beam_index"],
        "boresight beam index did not reproduce",
    )
    fh, ch = fresh["beam_selection_histogram"], committed["beam_selection_histogram"]
    check(
        len(fh) == len(ch) == 8
        and all(abs(int(a) - int(b)) <= HIST_TOL for a, b in zip(fh, ch)),
        "beam-selection histogram moved beyond boundary-flip tolerance",
    )
    check(
        abs(fresh["mean_array_gain_db"] - committed["mean_array_gain_db"]) <= MEAN_GAIN_TOL_DB,
        "mean array gain moved beyond tolerance vs the committed result",
    )
    check(
        abs(fresh["mean_misalignment_recovery_db"] - committed["mean_misalignment_recovery_db"])
        <= RECOVERY_TOL_DB,
        "mean misalignment recovery moved beyond tolerance vs the committed result",
    )

    # 4. Trust chain.
    tc = fresh["trust_chain"]
    check(tc["overpower_beam_corrected"] is True, "over-power beam was not corrected")
    check(
        tc["overpower_beam_safe_eirp_dbm"] <= 33.0 + 1e-6,
        "corrected beam EIRP exceeds the legal cap",
    )
    check(tc["overpower_beam_reached_ran"] is False, "over-power beam reached the RAN")
    check(tc["verify_first_broken_index"] == -1, "evidence chain did not verify intact")
    check(
        tc.get("verify_after_tamper_index") not in (None, -1),
        "tamper was not detected",
    )

    if errors:
        raise AssertionError("beam-management reproduction failed:\n  - " + "\n  - ".join(errors))


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
    print("beam-management reproduction verified on real DeepMIMO angular data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
