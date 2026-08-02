"""The same slice control via E2SM-CCC — O-RRMPolicyRatio, as JSON.

OCUDU implements two ways to set a slice's PRB quota, and it is worth having
both. E2SM-RC action 2.6 is ASN.1 PER over a nested RAN-parameter tree whose
ordering has to satisfy a parser's ``.back()`` calls. E2SM-CCC style 2 carries
the same intent as a named configuration structure with named attributes —
``O-RRMPolicyRatio`` with ``rRMPolicyMinRatio``, ``rRMPolicyMaxRatio`` and
``rRMPolicyDedicatedRatio`` — which is far harder to get subtly wrong.

Both names and both structures come from
``catalogue/ocudu-e2sm-catalogue.json``, extracted from OCUDU's CCC packer, so
the attribute spelling is OCUDU's rather than a guess. That spelling matters:
these are matched as strings on the receiving side, and ``rrmPolicyMinRatio``
would be silently ignored rather than rejected.

This module builds the structure and validates it. It does not encode CCC's
outer ASN.1 wrapper — CCC carries its payload as JSON inside an ASN.1 envelope,
and only the JSON half is produced here. Said plainly rather than implied: this
is a well-formed O-RRMPolicyRatio configuration, not a complete CCC control
message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from horizon_ocudu.rc_slice_quota import SliceQuota, SliceQuotaError

__all__ = [
    "STRUCTURE_NAME",
    "MIN_RATIO_ATTR",
    "MAX_RATIO_ATTR",
    "DEDICATED_RATIO_ATTR",
    "MEMBER_LIST_ATTR",
    "CELL_CONTROL_STYLE",
    "build_rrm_policy_ratio",
    "ccc_attributes",
]

CATALOGUE = Path(__file__).resolve().parents[2] / "catalogue" / "ocudu-e2sm-catalogue.json"

STRUCTURE_NAME = "O-RRMPolicyRatio"
MEMBER_LIST_ATTR = "rRMPolicyMemberList"
MAX_RATIO_ATTR = "rRMPolicyMaxRatio"
MIN_RATIO_ATTR = "rRMPolicyMinRatio"
DEDICATED_RATIO_ATTR = "rRMPolicyDedicatedRatio"

# From the extracted catalogue: {1: node-level, 2: cell-level}. The slice quota
# is cell-level.
CELL_CONTROL_STYLE = 2


def ccc_attributes() -> list[str]:
    """The writable attribute names OCUDU's CCC packer declares."""
    data = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    return list(data["e2sm_ccc"]["writable_attributes"])


@dataclass(frozen=True)
class CellIdentity:
    """The NR cell an O-RRMPolicyRatio applies to."""

    plmn: bytes
    nci: int

    def __post_init__(self) -> None:
        if len(self.plmn) != 3:
            raise SliceQuotaError(
                f"PLMN identity must be exactly 3 bytes, got {len(self.plmn)}"
            )
        if not 0 <= self.nci < (1 << 36):
            raise SliceQuotaError(f"NR Cell Identity {self.nci} outside 36 bits")


def build_rrm_policy_ratio(
    cell: CellIdentity, quotas: Sequence[SliceQuota]
) -> Mapping[str, Any]:
    """An ``O-RRMPolicyRatio`` configuration for one cell.

    Reuses :class:`~horizon_ocudu.rc_slice_quota.SliceQuota`, so the same
    validation applies — ratios in range, dedicated inside minimum, minimums
    summing within the cell. Two encodings of one intent should not disagree
    about what is valid, and sharing the type is the only way to be sure they
    do not.
    """
    if not quotas:
        raise SliceQuotaError(
            "an O-RRMPolicyRatio with no members configures nothing while "
            "appearing to be a policy change"
        )
    total_min = sum(q.min_ratio for q in quotas)
    if total_min > 100:
        raise SliceQuotaError(
            f"minimum ratios sum to {total_min}%, which exceeds the cell's PRBs"
        )

    members = []
    for quota in quotas:
        member: dict[str, Any] = {
            "plmnId": quota.plmn.hex(),
            "snssai": {"sst": quota.sst},
        }
        if quota.sd is not None:
            member["snssai"]["sd"] = quota.sd
        members.append(member)

    # One group per quota, matching the RC encoding's shape: a member list plus
    # the three ratios. Keeping the two routes structurally parallel is what
    # lets a test assert they carry the same numbers.
    return {
        "cellId": {"plmnId": cell.plmn.hex(), "nCI": cell.nci},
        "ranConfigurationStructureName": STRUCTURE_NAME,
        "controlStyle": CELL_CONTROL_STYLE,
        "values": [
            {
                MEMBER_LIST_ATTR: [members[i]],
                MIN_RATIO_ATTR: quota.min_ratio,
                MAX_RATIO_ATTR: quota.max_ratio,
                DEDICATED_RATIO_ATTR: quota.dedicated_ratio,
            }
            for i, quota in enumerate(quotas)
        ],
    }
