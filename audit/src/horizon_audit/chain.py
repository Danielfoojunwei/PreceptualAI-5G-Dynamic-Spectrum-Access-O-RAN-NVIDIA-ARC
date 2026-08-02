"""Offline hash-chain verification for a Horizon evidence export.

Reproduces, from first principles, the tamper-evidence contract of
``horizon_ric.evidence.store.JsonlEvidenceStore`` so a regulator can verify
an evidence export with nothing but the Python standard library.

Format (one JSON object per line in the ``.jsonl`` export)::

    {"hash": "<sha256-hex>", "tenant_id": "<str>", "record": {<record>}}

Chain rule (per line)::

    curr_hash = sha256( bytes.fromhex(prev_hash_hex)
                        + canonical_json(record).encode("utf-8") ).hexdigest()

where ``canonical_json(obj) = json.dumps(obj, sort_keys=True,
separators=(",", ":"))``.

The stored ``record`` object is ALREADY the canonicalised dict (the producer
appends ``json.loads(canonical_json(record))``), so to recompute the hash we
re-canonicalise it with the same ``sort_keys`` / compact-separator rule.

Chains are PER-TENANT: lines are grouped by ``tenant_id`` (defaulting to
``"_unscoped_"``), and each tenant's chain starts from the all-zero genesis
hash independently. ``verify_chain`` reports the first broken *global* line
index — matching ``EvidenceStore.verify()``'s semantics — plus the tenant and
per-tenant index for a human-readable report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Genesis / zero hash that seeds every tenant's chain.
ZERO_HASH_HEX = "0" * 64

# Sentinel tenant used when a record carries no tenant scope. MUST match
# ``horizon_ric.evidence.store._UNSCOPED_TENANT``.
UNSCOPED_TENANT = "_unscoped_"


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace. Matches the producer."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def chain_hash(prev_hex: str, record_obj: Any) -> str:
    """Recompute the chain hash for one record given the previous hash.

    ``sha256( bytes.fromhex(prev_hex) + canonical_json(record).encode() )``.
    """
    h = hashlib.sha256()
    h.update(bytes.fromhex(prev_hex))
    h.update(canonical_json(record_obj).encode("utf-8"))
    return h.hexdigest()


def _line_tenant(obj: dict) -> str:
    """Resolve a line's tenant id the same way the store does on read.

    The top-level ``tenant_id`` wins; otherwise the record's own
    ``tenant_id``; otherwise the unscoped sentinel.
    """
    tid = obj.get("tenant_id")
    if not tid:
        rec = obj.get("record")
        if isinstance(rec, dict):
            tid = rec.get("tenant_id")
    return tid or UNSCOPED_TENANT


@dataclass(frozen=True)
class TenantChainResult:
    """Per-tenant verification outcome."""

    tenant_id: str
    line_count: int
    intact: bool
    # Per-tenant 0-based index of the first broken record, or -1 if intact.
    first_broken_tenant_index: int = -1
    # Global 0-based file line index of the first broken record, or -1.
    first_broken_global_index: int = -1
    detail: str = ""


@dataclass(frozen=True)
class ChainResult:
    """Whole-export verification outcome.

    ``first_broken_index`` is the global 0-based line index of the first
    record whose recomputed hash disagrees with the stored hash — the exact
    value ``horizon_ric.evidence.store.EvidenceStore.verify()`` returns
    (``-1`` when the entire export is intact).
    """

    intact: bool
    line_count: int
    tenant_count: int
    first_broken_index: int = -1
    first_broken_tenant: str | None = None
    tenants: list[TenantChainResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "intact": self.intact,
            "line_count": self.line_count,
            "tenant_count": self.tenant_count,
            "first_broken_index": self.first_broken_index,
            "first_broken_tenant": self.first_broken_tenant,
            "tenants": [
                {
                    "tenant_id": t.tenant_id,
                    "line_count": t.line_count,
                    "intact": t.intact,
                    "first_broken_tenant_index": t.first_broken_tenant_index,
                    "first_broken_global_index": t.first_broken_global_index,
                    "detail": t.detail,
                }
                for t in self.tenants
            ],
            "errors": list(self.errors),
        }


def verify_chain(lines: list[dict]) -> ChainResult:
    """Verify a list of parsed evidence lines (in file/append order).

    Each element must be a dict with ``hash`` and ``record`` keys (and
    optionally ``tenant_id``). Returns a :class:`ChainResult` whose
    ``first_broken_index`` matches ``EvidenceStore.verify()``.
    """
    errors: list[str] = []
    per_tenant_prev: dict[str, str] = {}
    per_tenant_count: dict[str, int] = {}
    per_tenant_result: dict[str, TenantChainResult] = {}
    first_broken_index = -1
    first_broken_tenant: str | None = None

    for i, obj in enumerate(lines):
        if not isinstance(obj, dict):
            errors.append(f"line {i}: not a JSON object")
            if first_broken_index == -1:
                first_broken_index = i
            continue
        if "hash" not in obj or "record" not in obj:
            errors.append(f"line {i}: missing 'hash' or 'record'")
            if first_broken_index == -1:
                first_broken_index = i
            continue

        tid = _line_tenant(obj)
        tenant_idx = per_tenant_count.get(tid, 0)
        prev_hex = per_tenant_prev.get(tid, ZERO_HASH_HEX)
        stored = obj["hash"]

        try:
            expected = chain_hash(prev_hex, obj["record"])
        except (ValueError, TypeError) as exc:
            errors.append(f"line {i}: cannot recompute hash: {exc}")
            expected = None

        if expected != stored:
            # First broken record for this tenant.
            if tid not in per_tenant_result or per_tenant_result[tid].intact:
                per_tenant_result[tid] = TenantChainResult(
                    tenant_id=tid,
                    line_count=per_tenant_count.get(tid, 0) + 1,
                    intact=False,
                    first_broken_tenant_index=tenant_idx,
                    first_broken_global_index=i,
                    detail=(
                        f"recomputed hash {expected} != stored {stored}"
                        if expected is not None
                        else "record could not be canonicalised/hashed"
                    ),
                )
            if first_broken_index == -1:
                first_broken_index = i
                first_broken_tenant = tid
            # Match verify()'s global short-circuit semantics: it returns the
            # first broken GLOBAL index. We keep walking only to complete the
            # per-tenant report, but we do NOT advance this tenant's prev
            # hash past the break (a broken link poisons everything after it).
            per_tenant_count[tid] = tenant_idx + 1
            continue

        # Intact link — advance this tenant's chain.
        per_tenant_prev[tid] = stored
        per_tenant_count[tid] = tenant_idx + 1

    tenants: list[TenantChainResult] = []
    for tid, count in per_tenant_count.items():
        if tid in per_tenant_result:
            tenants.append(per_tenant_result[tid])
        else:
            tenants.append(
                TenantChainResult(
                    tenant_id=tid, line_count=count, intact=True
                )
            )

    return ChainResult(
        intact=(first_broken_index == -1 and not errors),
        line_count=len(lines),
        tenant_count=len(per_tenant_count),
        first_broken_index=first_broken_index,
        first_broken_tenant=first_broken_tenant,
        tenants=sorted(tenants, key=lambda t: t.tenant_id),
        errors=errors,
    )


def parse_jsonl(text: str) -> list[dict]:
    """Parse JSONL text into a list of dicts, skipping blank lines."""
    out: list[dict] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        out.append(json.loads(raw))
    return out


def verify_jsonl_file(path: str | Path) -> ChainResult:
    """Load a ``.jsonl`` evidence export and verify its hash chain(s)."""
    text = Path(path).read_text(encoding="utf-8")
    return verify_chain(parse_jsonl(text))


__all__ = [
    "ZERO_HASH_HEX",
    "UNSCOPED_TENANT",
    "ChainResult",
    "TenantChainResult",
    "canonical_json",
    "chain_hash",
    "parse_jsonl",
    "verify_chain",
    "verify_jsonl_file",
]
