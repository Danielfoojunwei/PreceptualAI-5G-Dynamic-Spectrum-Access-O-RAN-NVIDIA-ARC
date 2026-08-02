#!/usr/bin/env python3
"""Gate G8 — the E2SM-RC control we encode is one OCUDU would parse.

The claim under gate:

    Horizon's Slice-level PRB quota control carries the RAN Parameter IDs a
    real gNB implementation declares, nested in the order that implementation's
    parser walks, encoded in aligned PER from the vendored O-RAN standard
    ASN.1 — and the identifiers are re-derived from OCUDU's source rather than
    transcribed, so they cannot drift without failing here.

That last clause is what makes this a gate rather than a note. Copying thirteen
parameter IDs into a Python module would be correct today and silently wrong
the next time the submodule moves; the failure would surface as a gNB rejecting
a control message in a lab, weeks later, with no obvious cause.

Seven checks. The load-bearing one is ``catalogue_is_reproducible``: everything
else asserts agreement with the catalogue, and agreement with a stale catalogue
proves nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(REPO / "agentic" / "src"))
sys.path.insert(0, str(HERE))

from extract_catalogue import build as rebuild_catalogue  # noqa: E402
from horizon_ocudu.rc_slice_quota import (  # noqa: E402
    ACTION_ID,
    ACTION_NAME,
    DEDICATED_PRB_POLICY_RATIO,
    MAX_PRB_POLICY_RATIO,
    MIN_PRB_POLICY_RATIO,
    PLMN_IDENTITY,
    RRM_POLICY,
    RRM_POLICY_MEMBER,
    RRM_POLICY_MEMBER_LIST,
    RRM_POLICY_RATIO_GROUP,
    RRM_POLICY_RATIO_LIST,
    S_NSSAI,
    SD,
    SST,
    STYLE_ID,
    SliceQuota,
    SliceQuotaError,
    build_control_message,
    encode_control_message,
    fraction_to_ratio,
    visit_order,
)

from horizon_ric.e2.rc_control import rc_spec  # noqa: E402

CATALOGUE_PATH = HERE / "catalogue" / "ocudu-e2sm-catalogue.json"

PLMN = bytes.fromhex("00f110")

# Every constant the encoder uses, and the name OCUDU gives it. The gate
# asserts this mapping against the freshly extracted catalogue, so a rename or
# renumber upstream fails here.
EXPECTED_NAMES = {
    RRM_POLICY_RATIO_LIST: "RRM Policy Ratio List",
    RRM_POLICY_RATIO_GROUP: "RRM Policy Ratio Group",
    RRM_POLICY: "RRM Policy",
    RRM_POLICY_MEMBER_LIST: "RRM Policy Member List",
    RRM_POLICY_MEMBER: "RRM Policy Member",
    PLMN_IDENTITY: "PLMN Identity",
    S_NSSAI: "S-NSSAI",
    SST: "SST",
    SD: "SD",
    MIN_PRB_POLICY_RATIO: "Min PRB Policy Ratio",
    MAX_PRB_POLICY_RATIO: "Max PRB Policy Ratio",
    DEDICATED_PRB_POLICY_RATIO: "Dedicated PRB Policy Ratio",
}


def _slice_action(catalogue: dict[str, Any]) -> dict[str, Any] | None:
    for action in catalogue["e2sm_rc_control_actions"]:
        if (
            action["node"] == "DU"
            and action["style_id"] == STYLE_ID
            and action["action_id"] == ACTION_ID
        ):
            return action
    return None


def check_catalogue_is_reproducible() -> dict[str, Any]:
    """Re-extract from OCUDU source and compare against the committed file.

    Without this the other checks compare the encoder against a file we wrote,
    which is circular. This is what binds them to OCUDU.
    """
    committed = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    fresh = rebuild_catalogue()
    differences = [
        key
        for key in ("ocudu_commit", "source_digests", "e2sm_rc_control_actions", "e2sm_ccc")
        if committed.get(key) != fresh.get(key)
    ]
    return {
        "id": "catalogue_is_reproducible",
        "passed": not differences,
        "differing_keys": differences,
        "ocudu_commit": fresh.get("ocudu_commit", ""),
        "sources": sorted(fresh.get("source_digests", {})),
    }


def check_action_exists() -> dict[str, Any]:
    catalogue = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    action = _slice_action(catalogue)
    return {
        "id": "slice_quota_action_exists",
        "passed": action is not None and action["action_name"] == ACTION_NAME,
        "action_name": action["action_name"] if action else None,
        "style_id": STYLE_ID,
        "action_id": ACTION_ID,
    }


def check_ids_match_ocudu() -> dict[str, Any]:
    """Every constant the encoder uses is one OCUDU declares, under the same name."""
    catalogue = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    action = _slice_action(catalogue)
    if action is None:
        return {"id": "parameter_ids_match_ocudu", "passed": False, "detail": "no action"}
    declared = {p["id"]: p["name"] for p in action["ran_parameters"]}
    mismatched = {
        pid: {"ours": name, "ocudu": declared.get(pid)}
        for pid, name in EXPECTED_NAMES.items()
        if declared.get(pid) != name
    }
    unused = sorted(set(declared) - set(EXPECTED_NAMES))
    return {
        "id": "parameter_ids_match_ocudu",
        "passed": not mismatched and not unused,
        "mismatched": mismatched,
        "declared_but_unused": unused,
        "count": len(declared),
    }


def check_visit_order() -> dict[str, Any]:
    """The nesting must drive OCUDU's ``.back()`` calls into the right group.

    A tree can decode perfectly and still be wrong here: id 6 appended to a
    group that does not exist yet, or ratios applied to the previous group.
    """
    quota = SliceQuota(plmn=PLMN, sst=1, sd=1, min_ratio=20, max_ratio=60,
                       dedicated_ratio=10)
    order = visit_order(build_control_message([quota]))
    catalogue = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    action = _slice_action(catalogue)
    declared = sorted(p["id"] for p in action["ran_parameters"]) if action else []

    group_at = order.index(RRM_POLICY_RATIO_GROUP)
    member_at = order.index(RRM_POLICY_MEMBER)
    ok = (
        order == declared
        and group_at < member_at
        and member_at < order.index(PLMN_IDENTITY)
        and member_at < order.index(SST)
        and group_at < order.index(MIN_PRB_POLICY_RATIO)
    )
    return {
        "id": "visit_order_matches_parser_contract",
        "passed": bool(ok),
        "order": order,
        "declared": declared,
    }


def check_per_round_trip() -> dict[str, Any]:
    quota = SliceQuota(plmn=PLMN, sst=1, sd=1, min_ratio=20, max_ratio=60,
                       dedicated_ratio=10)
    raw = encode_control_message([quota])
    decoded = rc_spec().decode("E2SM-RC-ControlMessage-Format1", raw)
    again = encode_control_message([quota])
    return {
        "id": "per_round_trip_is_stable",
        "passed": (
            visit_order(decoded) == visit_order(build_control_message([quota]))
            and raw == again
            and len(raw) > 0
        ),
        "octets": len(raw),
        "hex": raw.hex(),
    }


def check_horizon_binding() -> dict[str, Any]:
    """A Horizon PRB fraction becomes the DU's Min PRB Policy Ratio.

    This is the whole point of choosing this action: the floor
    ``ProtectedSliceFloorInvariant`` projects onto and the ratio a DU enforces
    are the same quantity in different units.
    """
    ratio = fraction_to_ratio(0.20)
    quota = SliceQuota(plmn=PLMN, sst=1, min_ratio=ratio)
    decoded = rc_spec().decode(
        "E2SM-RC-ControlMessage-Format1", encode_control_message([quota])
    )

    found: list[int] = []

    def walk(items):
        for item in items:
            arm, payload = item["ranParameter-valueType"]
            if item["ranParameter-ID"] == MIN_PRB_POLICY_RATIO:
                found.append(payload["ranParameter-value"][1])
            if arm == "ranP-Choice-Structure":
                walk(payload["ranParameter-Structure"]["sequence-of-ranParameters"])
            elif arm == "ranP-Choice-List":
                for entry in payload["ranParameter-List"]["list-of-ranParameter"]:
                    walk(entry["sequence-of-ranParameters"])

    walk(decoded["ranP-List"])
    return {
        "id": "horizon_floor_binds_to_min_ratio",
        "passed": found == [20] and ratio == 20,
        "fraction": 0.20,
        "min_prb_policy_ratio_on_the_wire": found,
    }


def check_impossible_quotas_are_refused() -> dict[str, Any]:
    """Nothing that a DU could not honour may reach the encoder."""
    cases: dict[str, bool] = {}

    def refuses(name: str, fn) -> None:
        try:
            fn()
            cases[name] = False
        except SliceQuotaError:
            cases[name] = True

    refuses("min above max", lambda: SliceQuota(PLMN, 1, min_ratio=80, max_ratio=20))
    refuses("dedicated above min", lambda: SliceQuota(PLMN, 1, min_ratio=20,
                                                     dedicated_ratio=50))
    refuses("ratio above 100", lambda: SliceQuota(PLMN, 1, min_ratio=101))
    refuses("short plmn", lambda: SliceQuota(b"\x00\x01", 1, min_ratio=20))
    refuses("empty control", lambda: build_control_message([]))
    refuses(
        "minimums exceed the cell",
        lambda: build_control_message(
            [SliceQuota(PLMN, 1, min_ratio=60), SliceQuota(PLMN, 2, min_ratio=60)]
        ),
    )
    refuses("fraction out of range", lambda: fraction_to_ratio(1.5))
    return {
        "id": "impossible_quotas_refused",
        "passed": all(cases.values()),
        "cases": cases,
    }


def source_digests() -> dict[str, str]:
    paths = [
        HERE / "extract_catalogue.py",
        HERE / "src/horizon_ocudu/rc_slice_quota.py",
        HERE / "src/horizon_ocudu/metrics.py",
        REPO / "src/horizon_ric/e2/rc_control.py",
        REPO / "src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn",
    ]
    return {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.exists()
    }


def run() -> dict[str, Any]:
    checks = [
        check_catalogue_is_reproducible(),
        check_action_exists(),
        check_ids_match_ocudu(),
        check_visit_order(),
        check_per_round_trip(),
        check_horizon_binding(),
        check_impossible_quotas_are_refused(),
    ]
    return {
        "gate": "G8",
        "claim": (
            "the Slice-level PRB quota control Horizon encodes carries the RAN "
            "Parameter IDs OCUDU declares, nested in the order its parser walks, "
            "in aligned PER from the vendored O-RAN E2SM-RC v1.03 ASN.1"
        ),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "source_digests": source_digests(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    for check in result["checks"]:
        print(("  ok    " if check["passed"] else "  FAIL  ") + check["id"],
              file=sys.stderr)
    print("G8 " + ("PASSED" if result["passed"] else "FAILED"), file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
