"""E2SM-RC Style 2 Action 6 — "Slice-level PRB quota", as OCUDU actually parses it.

This is the control action that matters most to Horizon, because it is the same
thing ``ProtectedSliceFloorInvariant`` guards. That invariant reserves a
fraction of a cell's PRBs for a safety-critical slice and projects any planner
proposal back above the floor. E2SM-RC action 2.6 is how a RIC *tells a DU* to
enforce exactly that, and OCUDU implements the receiving end.

Two things had to be true before this could be written honestly, and both now
are.

**The RAN Parameter IDs are no longer placeholders.** Horizon's E2SM-RC
encoder carried local defaults, flagged amber, with the note that E2SM-RC
assigns identifiers per node via ``RANFunctionDefinition-Control-Action-Item``.
The ids used here come from ``ocudu/catalogue/ocudu-e2sm-catalogue.json``,
extracted from OCUDU's own executor source at a pinned commit — so they are the
identifiers one real gNB implementation will parse, and the gate re-extracts and
compares rather than trusting a transcription.

**The nesting is the part that is easy to get wrong.** OCUDU's parser walks the
RAN-parameter tree in document order and mutates the *most recently created*
group and member:

    id 2  →  rrm_policy_ratio_list.emplace_back()                  new group
    id 6  →  .back().policy_members_list.emplace_back()            new member
    id 7/9/10  →  fill .back().policy_members_list.back()          that member
    id 11/12/13  →  set on .back()                                 that group

So a flat list of parameters in the wrong order does not produce a wrong policy
— it produces a policy attached to the wrong group, or crashes on an empty
``back()``. The structure below is built to that contract, and
:func:`visit_order` exposes the sequence a conformance check can assert against.

Ids 1, 3, 5 and 8 carry no value: OCUDU matches them in an arm whose body is
``// No need to parse``. They exist to open a nesting level, and omitting them
would collapse the tree the recursion is walking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from horizon_ric.e2.rc_control import rc_spec

__all__ = [
    "RRM_POLICY_RATIO_LIST",
    "RRM_POLICY_RATIO_GROUP",
    "RRM_POLICY",
    "RRM_POLICY_MEMBER_LIST",
    "RRM_POLICY_MEMBER",
    "PLMN_IDENTITY",
    "S_NSSAI",
    "SST",
    "SD",
    "MIN_PRB_POLICY_RATIO",
    "MAX_PRB_POLICY_RATIO",
    "DEDICATED_PRB_POLICY_RATIO",
    "SliceQuota",
    "SliceQuotaError",
    "catalogue",
    "build_control_message",
    "encode_control_message",
    "visit_order",
    "fraction_to_ratio",
]

CATALOGUE = Path(__file__).resolve().parents[2] / "catalogue" / "ocudu-e2sm-catalogue.json"

STYLE_ID = 2
ACTION_ID = 6
ACTION_NAME = "Slice-level PRB quota"

# Named for readability; every one is asserted against the extracted catalogue
# by `verify_g8_ocudu_conformance.py`, so these constants cannot drift from
# OCUDU without the gate failing.
RRM_POLICY_RATIO_LIST = 1
RRM_POLICY_RATIO_GROUP = 2
RRM_POLICY = 3
RRM_POLICY_MEMBER_LIST = 5
RRM_POLICY_MEMBER = 6
PLMN_IDENTITY = 7
S_NSSAI = 8
SST = 9
SD = 10
MIN_PRB_POLICY_RATIO = 11
MAX_PRB_POLICY_RATIO = 12
DEDICATED_PRB_POLICY_RATIO = 13


class SliceQuotaError(ValueError):
    """Raised for a quota that no DU could honour, before anything is encoded."""


def catalogue() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    return data


def fraction_to_ratio(fraction: float) -> int:
    """Horizon's PRB *fraction* to E2SM-RC's PRB policy *ratio* (percent).

    ``ProtectedSliceFloorInvariant`` and the agentic ``prb_allocation`` both
    express a share as a fraction of the cell; the wire wants an integer
    percentage. Rounding is toward zero for a **minimum**: rounding 0.205 up to
    21% would ask the DU to reserve more than the operator's floor, which is
    not obviously safe — it takes PRBs from other slices that nobody
    authorised.
    """
    if not 0.0 <= fraction <= 1.0:
        raise SliceQuotaError(
            f"PRB fraction {fraction!r} is outside [0, 1]; a ratio cannot be derived"
        )
    return int(fraction * 100.0)


@dataclass(frozen=True)
class SliceQuota:
    """One RRM policy group: a slice, and the PRB ratios reserved for it.

    ``plmn`` is the 3-byte encoded PLMN identity. ``sst`` and ``sd`` identify
    the slice; ``sd`` is optional because a slice may be identified by SST
    alone.
    """

    plmn: bytes
    sst: int
    min_ratio: int
    max_ratio: int = 100
    dedicated_ratio: int = 0
    sd: int | None = None

    def __post_init__(self) -> None:
        if len(self.plmn) != 3:
            # OCUDU refuses a PLMN that is not exactly three octets, logging
            # "encoded not correctly" and returning — silently dropping the
            # member. Failing here is better than producing a message that a
            # DU accepts and half-applies.
            raise SliceQuotaError(
                f"PLMN identity must be exactly 3 bytes, got {len(self.plmn)}"
            )
        if not 0 <= self.sst <= 255:
            raise SliceQuotaError(f"SST {self.sst} outside 0..255")
        for name, value in (
            ("min_ratio", self.min_ratio),
            ("max_ratio", self.max_ratio),
            ("dedicated_ratio", self.dedicated_ratio),
        ):
            if not 0 <= value <= 100:
                raise SliceQuotaError(f"{name}={value} is not a percentage in 0..100")
        if self.min_ratio > self.max_ratio:
            raise SliceQuotaError(
                f"min_ratio {self.min_ratio} exceeds max_ratio {self.max_ratio}; "
                "no allocation satisfies both"
            )
        if self.dedicated_ratio > self.min_ratio:
            # Dedicated capacity is a subset of the guaranteed minimum. A
            # dedicated share above the minimum is a contradiction the DU has
            # no way to resolve.
            raise SliceQuotaError(
                f"dedicated_ratio {self.dedicated_ratio} exceeds min_ratio "
                f"{self.min_ratio}"
            )


def _elem_int(value: int) -> tuple[str, Any]:
    return ("ranP-Choice-ElementFalse", {"ranParameter-value": ("valueInt", value)})


def _elem_octets(value: bytes) -> tuple[str, Any]:
    return ("ranP-Choice-ElementFalse", {"ranParameter-value": ("valueOctS", value)})


def _structure(items: Sequence[dict[str, Any]]) -> tuple[str, Any]:
    return (
        "ranP-Choice-Structure",
        {"ranParameter-Structure": {"sequence-of-ranParameters": list(items)}},
    )


def _list(items: Sequence[dict[str, Any]]) -> tuple[str, Any]:
    return (
        "ranP-Choice-List",
        {"ranParameter-List": {"list-of-ranParameter": [
            {"sequence-of-ranParameters": [item]} for item in items
        ]}},
    )


def _param(param_id: int, value_type: tuple[str, Any]) -> dict[str, Any]:
    return {"ranParameter-ID": param_id, "ranParameter-valueType": value_type}


def _member(quota: SliceQuota) -> dict[str, Any]:
    """One RRM Policy Member: the slice this group's ratios apply to."""
    snssai: list[dict[str, Any]] = [
        _param(SST, _elem_octets(bytes([quota.sst]))),
    ]
    if quota.sd is not None:
        snssai.append(
            _param(SD, _elem_octets(quota.sd.to_bytes(3, "big")))
        )
    return _param(
        RRM_POLICY_MEMBER,
        _structure(
            [
                _param(PLMN_IDENTITY, _elem_octets(quota.plmn)),
                _param(S_NSSAI, _structure(snssai)),
            ]
        ),
    )


def _group(quota: SliceQuota) -> dict[str, Any]:
    """One RRM Policy Ratio Group: its members, then its three ratios.

    Members precede ratios because OCUDU's ``.back()`` handling requires the
    group to exist before a member is appended to it, and the ratios are set on
    the group itself — so any order works for the ratios, but putting them last
    keeps the emitted tree in the same shape as the spec's table.
    """
    return _param(
        RRM_POLICY_RATIO_GROUP,
        _structure(
            [
                _param(
                    RRM_POLICY,
                    _structure([_param(RRM_POLICY_MEMBER_LIST, _list([_member(quota)]))]),
                ),
                _param(MIN_PRB_POLICY_RATIO, _elem_int(quota.min_ratio)),
                _param(MAX_PRB_POLICY_RATIO, _elem_int(quota.max_ratio)),
                _param(DEDICATED_PRB_POLICY_RATIO, _elem_int(quota.dedicated_ratio)),
            ]
        ),
    )


def build_control_message(quotas: Sequence[SliceQuota]) -> dict[str, Any]:
    """The RAN-parameter tree for a Slice-level PRB quota control."""
    if not quotas:
        raise SliceQuotaError(
            "a slice-quota control with no groups would instruct the DU to do "
            "nothing while looking like a policy change"
        )
    total_min = sum(q.min_ratio for q in quotas)
    if total_min > 100:
        # The DU cannot guarantee more than the cell has. This is the same
        # conservation property `PrbConservation` enforces on the bundle side,
        # applied here at the point the intent leaves for the wire.
        raise SliceQuotaError(
            f"minimum ratios sum to {total_min}%, which exceeds the cell's PRBs"
        )
    return {
        "ranP-List": [
            _param(RRM_POLICY_RATIO_LIST, _list([_group(q) for q in quotas]))
        ]
    }


def encode_control_message(quotas: Sequence[SliceQuota]) -> bytes:
    """Aligned-PER bytes, using the vendored O-RAN E2SM-RC v1.03 spec.

    The same ``rc_spec()`` the main package compiles, so these bytes come from
    the same standard ASN.1 that FlexRIC's own asn1c codec was generated from.
    """
    raw: bytes = rc_spec().encode(
        "E2SM-RC-ControlMessage-Format1", build_control_message(quotas)
    )
    return raw


def visit_order(message: dict[str, Any]) -> list[int]:
    """RAN parameter ids in the order OCUDU's recursive parser will see them.

    Exposed so a conformance check can assert the ordering contract directly
    rather than inferring it from a successful decode — a tree can decode
    perfectly and still drive the parser's ``.back()`` calls into the wrong
    group.
    """
    order: list[int] = []

    def walk(items: Sequence[dict[str, Any]]) -> None:
        for item in items:
            order.append(int(item["ranParameter-ID"]))
            arm, payload = item["ranParameter-valueType"]
            if arm == "ranP-Choice-Structure":
                walk(payload["ranParameter-Structure"]["sequence-of-ranParameters"])
            elif arm == "ranP-Choice-List":
                for entry in payload["ranParameter-List"]["list-of-ranParameter"]:
                    walk(entry["sequence-of-ranParameters"])

    walk(message["ranP-List"])
    return order
