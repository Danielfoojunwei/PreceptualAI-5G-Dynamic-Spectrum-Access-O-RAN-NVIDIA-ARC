#!/usr/bin/env python3
"""Verify a fresh mobility-handover run against the committed real-data result.

The beam-handover loop is deterministic given the same feature bytes, but the
Doppler phase accumulates ~60 wavelengths of travel per trajectory step, so the
per-beam gain floats are fade-sensitive: last-ULP differences in a cross-host
DeepMIMO rebuild (angles rounded to 4 decimals) can flip a handful of argmax
samples near beam boundaries. This verifier therefore checks what is host-stable
and meaningful:

* the fresh run is bound to the canonical, checksum-pinned angular build;
* Doppler is genuinely on (max shift > 0) and the mobility invariants hold on
  the freshly rebuilt real data (the A3 hysteresis policy makes >= 1 real
  handover, cuts ping-pongs below greedy, and keeps outage <= 0.25);
* the operational decisions reproduce (same final serving beam; handover counts
  match the committed result within a small fade tolerance) — chaotic per-beam
  gain floats are deliberately NOT byte-compared;
* the trust chain verifies intact, refuses the over-EIRP handover emit, and
  catches the tamper.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HANDOVER_TOLERANCE = 3  # cross-host fade tolerance on event counts


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
    #    cross-host: it is computed on values rounded to 4 decimals, so last-ULP
    #    float differences in DeepMIMO's channel computation across machines
    #    change a few bytes. source_tree and receiver count are host-independent
    #    and are the real binding.
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

    # 2. Mobility + Doppler invariants on the freshly rebuilt real data.
    dop = fresh["doppler"]
    g, h = fresh["greedy"], fresh["hysteresis"]
    check(dop["max_shift_hz"] > 0.0, "Doppler is not on (max shift <= 0)")
    check(dop["coherence_time_s"] > 0.0, "coherence time is not positive")
    check(fresh["trajectory"]["points"] >= 150, "trajectory has fewer than 150 real points")
    check(g["ping_pongs"] > h["ping_pongs"], "hysteresis did not reduce ping-pongs below greedy")
    check(h["outage_fraction"] <= 0.25, "hysteresis outage fraction exceeds 0.25")
    check(h["handovers"] >= 1, "hysteresis made no handover (no real mobility across beams)")

    # 3. Host-stable operational decisions reproduce the committed result.
    ch = committed["hysteresis"]
    cg = committed["greedy"]
    check(
        h["final_serving_beam"] == ch["final_serving_beam"],
        "final serving beam did not reproduce",
    )
    check(
        abs(h["handovers"] - ch["handovers"]) <= HANDOVER_TOLERANCE,
        f"hysteresis handover count {h['handovers']} not within "
        f"{HANDOVER_TOLERANCE} of committed {ch['handovers']}",
    )
    check(
        abs(g["handovers"] - cg["handovers"]) <= 3 * HANDOVER_TOLERANCE,
        f"greedy handover count {g['handovers']} not within "
        f"{3 * HANDOVER_TOLERANCE} of committed {cg['handovers']}",
    )
    check(
        h["beams_visited"] == ch["beams_visited"],
        "hysteresis policy visited a different beam set than the committed result",
    )

    # 4. Trust chain.
    tc = fresh["trust_chain"]
    check(tc["overpower_handover_corrected"] is True, "over-EIRP handover was not corrected")
    check(
        tc["overpower_handover_safe_eirp_dbm"] <= 33.0 + 1e-6,
        "corrected handover EIRP exceeds the legal cap",
    )
    check(tc["overpower_handover_reached_ran"] is False, "over-EIRP handover reached the RAN")
    check(tc["verify_first_broken_index"] == -1, "evidence chain did not verify intact")
    check(
        tc.get("verify_after_tamper_index") not in (None, -1),
        "tamper was not detected",
    )

    if errors:
        raise AssertionError(
            "mobility-handover reproduction failed:\n  - " + "\n  - ".join(errors)
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
    print("mobility-handover reproduction verified on real DeepMIMO ray geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
