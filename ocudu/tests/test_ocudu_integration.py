"""OCUDU integration: the control we encode, and the telemetry we consume.

The theme running through this file is that OCUDU is a *real receiving
implementation*, so claims about interoperability can be checked against its
source rather than against a reading of the specification. Every constant here
is asserted against `catalogue/ocudu-e2sm-catalogue.json`, which is extracted
from that source at a pinned commit and re-extracted by gate G8.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from horizon_agentic.telemetry_trust import (
    TelemetryTrustGate,
    TrustPolicy,
    sign_record,
)
from horizon_ocudu.ccc_rrm_policy import (
    DEDICATED_RATIO_ATTR,
    MAX_RATIO_ATTR,
    MEMBER_LIST_ATTR,
    MIN_RATIO_ATTR,
    STRUCTURE_NAME,
    CellIdentity,
    build_rrm_policy_ratio,
    ccc_attributes,
)
from horizon_ocudu.metrics import (
    OCUDU_BOUNDS,
    UE_FIELDS,
    parse_metrics,
    records_from_metrics,
)
from horizon_ocudu.rc_slice_quota import (
    ACTION_ID,
    ACTION_NAME,
    MIN_PRB_POLICY_RATIO,
    STYLE_ID,
    SliceQuota,
    SliceQuotaError,
    build_control_message,
    encode_control_message,
    fraction_to_ratio,
    visit_order,
)

from horizon_ric.e2.rc_control import rc_spec

CATALOGUE = Path(__file__).resolve().parents[1] / "catalogue" / "ocudu-e2sm-catalogue.json"
PLMN = bytes.fromhex("00f110")


def catalogue() -> dict:
    return json.loads(CATALOGUE.read_text(encoding="utf-8"))


def slice_action() -> dict:
    return next(
        a
        for a in catalogue()["e2sm_rc_control_actions"]
        if a["node"] == "DU" and a["action_id"] == ACTION_ID
    )


# ── the catalogue itself ─────────────────────────────────────────────────
def test_the_catalogue_records_which_ocudu_it_came_from() -> None:
    """A catalogue without provenance is a transcription with extra steps."""
    data = catalogue()
    assert len(data["ocudu_commit"]) == 40
    assert data["source_digests"]
    assert all(len(d) == 64 for d in data["source_digests"].values())


def test_the_slice_quota_action_is_style_2_action_6() -> None:
    action = slice_action()
    assert (action["style_id"], action["action_id"]) == (STYLE_ID, ACTION_ID)
    assert action["action_name"] == ACTION_NAME


def test_structural_and_value_parameters_are_distinguished() -> None:
    """Ids 1, 3, 5 and 8 open nesting levels and carry nothing.

    OCUDU spells this as one `if` arm whose body is `// No need to parse`. The
    distinction matters to an encoder: omitting a structural id collapses the
    tree the parser recurses through, while omitting a value id silently drops
    a field.
    """
    action = slice_action()
    assert action["structural_parameter_ids"] == [1, 3, 5, 8]
    assert action["value_parameter_ids"] == [2, 6, 7, 9, 10, 11, 12, 13]


def test_parameter_4_is_absent_as_the_standard_skips_it() -> None:
    action = slice_action()
    assert 4 not in {p["id"] for p in action["ran_parameters"]}


# ── E2SM-RC encoding ─────────────────────────────────────────────────────
def test_every_declared_parameter_is_emitted_in_parser_order() -> None:
    quota = SliceQuota(plmn=PLMN, sst=1, sd=1, min_ratio=20, max_ratio=60,
                       dedicated_ratio=10)
    order = visit_order(build_control_message([quota]))
    assert order == sorted(p["id"] for p in slice_action()["ran_parameters"])


def test_the_group_precedes_its_member_and_its_ratios() -> None:
    """OCUDU mutates the most recently created group and member.

    A tree that decodes perfectly can still append a member to a group that
    does not exist yet, or apply ratios to the previous group.
    """
    quota = SliceQuota(plmn=PLMN, sst=1, min_ratio=20)
    order = visit_order(build_control_message([quota]))
    assert order.index(2) < order.index(6)          # group before member
    assert order.index(6) < order.index(7)          # member before PLMN
    assert order.index(6) < order.index(9)          # member before SST
    assert order.index(2) < order.index(11)         # group before its ratios


def test_per_encoding_round_trips_through_the_vendored_spec() -> None:
    quota = SliceQuota(plmn=PLMN, sst=1, sd=1, min_ratio=20)
    raw = encode_control_message([quota])
    assert raw
    decoded = rc_spec().decode("E2SM-RC-ControlMessage-Format1", raw)
    assert visit_order(decoded) == visit_order(build_control_message([quota]))


def test_encoding_is_deterministic() -> None:
    quota = SliceQuota(plmn=PLMN, sst=1, min_ratio=20)
    assert encode_control_message([quota]) == encode_control_message([quota])


def test_two_slices_produce_two_groups() -> None:
    order = visit_order(
        build_control_message(
            [SliceQuota(PLMN, 1, min_ratio=20), SliceQuota(PLMN, 2, min_ratio=30)]
        )
    )
    assert order.count(2) == 2
    assert order.count(6) == 2


# ── the Horizon binding ──────────────────────────────────────────────────
def test_a_horizon_prb_fraction_becomes_the_min_prb_policy_ratio() -> None:
    """The reason this action was chosen.

    ``ProtectedSliceFloorInvariant`` reserves a fraction of the cell's PRBs;
    ``Min PRB Policy Ratio`` is what a DU enforces. Same quantity, different
    units.
    """
    assert fraction_to_ratio(0.20) == 20
    quota = SliceQuota(PLMN, 1, min_ratio=fraction_to_ratio(0.20))
    decoded = rc_spec().decode(
        "E2SM-RC-ControlMessage-Format1", encode_control_message([quota])
    )
    found = []

    def walk(items):
        for item in items:
            arm, payload = item["ranParameter-valueType"]
            if item["ranParameter-ID"] == MIN_PRB_POLICY_RATIO:
                found.append(payload["ranParameter-value"][1])
            if arm == "ranP-Choice-Structure":
                walk(payload["ranParameter-Structure"]["sequence-of-ranParameters"])
            elif arm == "ranP-Choice-List":
                for e in payload["ranParameter-List"]["list-of-ranParameter"]:
                    walk(e["sequence-of-ranParameters"])

    walk(decoded["ranP-List"])
    assert found == [20]


def test_a_minimum_rounds_down_not_up() -> None:
    """Rounding a floor up reserves PRBs nobody authorised taking."""
    assert fraction_to_ratio(0.205) == 20
    assert fraction_to_ratio(0.999) == 99


@pytest.mark.parametrize(
    "make",
    [
        lambda: SliceQuota(PLMN, 1, min_ratio=80, max_ratio=20),
        lambda: SliceQuota(PLMN, 1, min_ratio=20, dedicated_ratio=50),
        lambda: SliceQuota(PLMN, 1, min_ratio=101),
        lambda: SliceQuota(b"\x00\x01", 1, min_ratio=20),
        lambda: SliceQuota(PLMN, 999, min_ratio=20),
        lambda: build_control_message([]),
        lambda: build_control_message(
            [SliceQuota(PLMN, 1, min_ratio=60), SliceQuota(PLMN, 2, min_ratio=60)]
        ),
        lambda: fraction_to_ratio(1.5),
        lambda: fraction_to_ratio(-0.1),
    ],
)
def test_impossible_quotas_never_reach_the_wire(make) -> None:
    with pytest.raises(SliceQuotaError):
        make()


# ── E2SM-CCC ─────────────────────────────────────────────────────────────
def test_ccc_attribute_names_are_ocudus_spelling() -> None:
    """These are matched as strings; a rename is silently ignored, not rejected."""
    declared = set(ccc_attributes())
    for name in (MEMBER_LIST_ATTR, MIN_RATIO_ATTR, MAX_RATIO_ATTR, DEDICATED_RATIO_ATTR):
        assert name in declared, f"{name} not among OCUDU's writable attributes"
    assert STRUCTURE_NAME in catalogue()["e2sm_ccc"]["ran_configuration_structures"]


def test_ccc_carries_the_same_numbers_as_the_rc_route() -> None:
    """Two encodings of one intent must not disagree."""
    quota = SliceQuota(PLMN, 1, min_ratio=20, max_ratio=60, dedicated_ratio=10)
    ccc = build_rrm_policy_ratio(CellIdentity(PLMN, 0x1234), [quota])
    group = ccc["values"][0]
    assert group[MIN_RATIO_ATTR] == quota.min_ratio == 20
    assert group[MAX_RATIO_ATTR] == quota.max_ratio == 60
    assert group[DEDICATED_RATIO_ATTR] == quota.dedicated_ratio == 10


def test_ccc_shares_the_rc_validation() -> None:
    with pytest.raises(SliceQuotaError):
        build_rrm_policy_ratio(CellIdentity(PLMN, 1), [])
    with pytest.raises(SliceQuotaError):
        build_rrm_policy_ratio(
            CellIdentity(PLMN, 1),
            [SliceQuota(PLMN, 1, min_ratio=60), SliceQuota(PLMN, 2, min_ratio=60)],
        )
    with pytest.raises(SliceQuotaError):
        CellIdentity(b"\x00", 1)


# ── metrics into the trust gate ──────────────────────────────────────────
REPORT = {
    "timestamp": 1000.0,
    "du": {
        "du_high": {
            "mac": {
                "timestamp": 1000.0,
                "cells": [
                    {
                        "cell_metrics": {
                            "pci": 1,
                            "average_latency": 120.0,
                            "max_latency": 900.0,
                            "error_indication_count": 0,
                            "failed_dl_pdcch": 2,
                        },
                        "ue_list": [
                            {"rnti": 17921, "cqi": 12, "dl_mcs": 20, "dl_ri": 2,
                             "dl_brate": 5.0e6, "last_phr": 10},
                            {"rnti": 17922, "cqi": 4, "dl_mcs": 6, "dl_ri": 1,
                             "dl_brate": 1.0e6, "last_phr": -5},
                        ],
                    }
                ],
            }
        }
    },
}


def test_metrics_parse_into_per_cell_records() -> None:
    cells = parse_metrics(REPORT)
    assert len(cells) == 1
    assert cells[0].pci == 1
    assert cells[0].source_id == "ocudu-pci-1"
    assert cells[0].cell["average_latency"] == 120.0


def test_ue_quality_metrics_take_the_worst_not_the_mean() -> None:
    """A comfortable average hides the user at the edge of coverage.

    An enforcement layer reasoning about the mean would approve an action that
    harms exactly the UE it should protect.
    """
    records = records_from_metrics(parse_metrics(REPORT))
    fields = records[0].fields
    assert fields["cqi"] == 4          # the worse of 12 and 4
    assert fields["dl_ri"] == 1
    assert fields["last_phr"] == -5
    assert fields["dl_brate"] == 6.0e6  # rates sum


def test_records_are_unsigned_and_the_gate_refuses_them() -> None:
    """OCUDU emits unauthenticated JSON, and pretending otherwise would be worse.

    Refusal is the correct default for a source nobody has decided to trust.
    """
    policy = TrustPolicy(
        secrets={"ocudu-pci-1": b"k"},
        max_age_s=60.0,
        bounds=dict(OCUDU_BOUNDS),
    )
    gate = TelemetryTrustGate(policy, clock=lambda: 1000.0)
    records = records_from_metrics(parse_metrics(REPORT))
    assert all(r.auth_tag == "" for r in records)
    assert not gate.admit(list(records)).trusted


def test_a_collector_that_signs_them_is_admitted() -> None:
    """The supported deployment: a collector holding a real key."""
    policy = TrustPolicy(
        secrets={"ocudu-pci-1": b"k"},
        max_age_s=60.0,
        bounds=dict(OCUDU_BOUNDS),
        required_fields=frozenset({"cqi"}),
    )
    gate = TelemetryTrustGate(policy, clock=lambda: 1000.0)
    signed = [sign_record(r, b"k") for r in records_from_metrics(parse_metrics(REPORT))]
    verdict = gate.admit(signed)
    assert verdict.trusted, [f"{c.check_id}: {c.detail}" for c in verdict.refusals]
    assert verdict.state["cqi"] == 4


def test_out_of_range_metrics_are_refused_by_the_bounds() -> None:
    """A gNB reporting CQI 99 is broken or lying; either way it is not state."""
    broken = json.loads(json.dumps(REPORT))
    broken["du"]["du_high"]["mac"]["cells"][0]["ue_list"][0]["cqi"] = 99
    broken["du"]["du_high"]["mac"]["cells"][0]["ue_list"][1]["cqi"] = 99
    policy = TrustPolicy(
        secrets={"ocudu-pci-1": b"k"}, max_age_s=60.0, bounds=dict(OCUDU_BOUNDS)
    )
    gate = TelemetryTrustGate(policy, clock=lambda: 1000.0)
    signed = [sign_record(r, b"k") for r in records_from_metrics(parse_metrics(broken))]
    assert not gate.admit(signed).trusted


def test_every_mapped_field_has_a_declared_bound() -> None:
    """An unbounded field is refused by the gate, so mapping one is a dead end."""
    from horizon_ocudu.metrics import CELL_FIELDS

    for name in CELL_FIELDS + UE_FIELDS:
        assert name in OCUDU_BOUNDS, f"{name} is mapped but has no bound"


def test_an_empty_or_foreign_report_yields_nothing() -> None:
    assert parse_metrics({}) == ()
    assert parse_metrics('{"unrelated": 1}') == ()
    assert records_from_metrics(()) == ()


def test_sequence_numbers_advance_per_cell() -> None:
    """Replay defence is per-source; a shared counter would silence a cell."""
    two = json.loads(json.dumps(REPORT))
    cell_b = json.loads(json.dumps(two["du"]["du_high"]["mac"]["cells"][0]))
    cell_b["cell_metrics"]["pci"] = 2
    two["du"]["du_high"]["mac"]["cells"].append(cell_b)
    records = records_from_metrics(parse_metrics(two))
    assert {r.source_id for r in records} == {"ocudu-pci-1", "ocudu-pci-2"}
    assert all(r.sequence == 1 for r in records)
