"""Evidence store — tamper-evident persistence for DecisionRecord.

Two backends ship out of the box:

  * `JsonlEvidenceStore`   appends one canonical-JSON record per line to a
                           file, plus a SHA-256 chain so any tamper of an
                           older line breaks the chain at every later line.
  * `SqliteEvidenceStore`  same logic backed by SQLite (sqlalchemy core),
                           suitable for queries and the rApp REST API.

Both implement the same `EvidenceStore` ABC so callers (the A1 adapter,
the rApp REST query handler) don't depend on the backend.

Tamper-evidence:
    sha256_curr = sha256(prev_sha256 || canonical_json(record))
The first record uses the all-zero hash as the prior. `verify()` walks the
chain from the start and returns the first index that fails to validate
(or -1 if the entire chain is intact).
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import (
    Column,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from horizon_ric.evidence.schema import DecisionRecord
from horizon_ric.security.tenant import current_tenant

_ZERO_HASH_HEX = "0" * 64

# Sentinel tenant used when no TenantScope is active. We deliberately do
# NOT silently succeed: the evidence store records compliance-critical
# state, so a missing tenant scope must produce a tenant-tagged record
# under a known label rather than mix into another tenant's chain.
_UNSCOPED_TENANT = "_unscoped_"


def _resolve_tenant() -> str:
    """Return the currently-active tenant, or the unscoped sentinel.

    Callers in fully-tenant-aware code paths (the rApp REST handler, the
    A1 adapter inside a request) must run inside a `TenantScope`. Callers
    that legitimately have no tenant (system-level housekeeping) end up
    in the `_unscoped_` chain which is also tamper-evident."""
    t = current_tenant()
    return t if t else _UNSCOPED_TENANT


def _canonical_json(rec: DecisionRecord) -> str:
    """Serialise to canonical JSON (sorted keys, ISO-8601 timestamps)."""
    # Pydantic's .model_dump_json doesn't sort keys; round-trip through dict.
    obj = json.loads(rec.model_dump_json())
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _chain(prev_hex: str, payload: str) -> str:
    h = hashlib.sha256()
    h.update(bytes.fromhex(prev_hex))
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


class EvidenceStore(ABC):
    """Abstract evidence store with hash-chain tamper-evidence.

    Multi-tenancy: every record is stamped with a `tenant_id` (resolved
    from the active `TenantScope` at append time) and each tenant has
    its own SHA-256 chain. Iterating without a tenant scope yields all
    tenants' records (administrators only); iterating inside a
    `TenantScope` yields only that tenant's chain.
    """

    @abstractmethod
    def append(self, record: DecisionRecord) -> str:
        """Persist a record; return its sha256 chain hash (hex).

        If `record.tenant_id` is None, the value is filled from the
        active `TenantScope`. The chain is computed per-tenant.
        """

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[DecisionRecord, str]]:
        """Yield (record, chain_hash). When called inside a TenantScope
        only that tenant's records are yielded; otherwise all records
        are yielded in append order."""

    def verify(self) -> int:
        """Return the index of the first broken record, or -1 if all intact.

        Verification is per-tenant: each tenant's chain is walked
        independently, since chains are independent. Within the bounds
        of an active `TenantScope`, only that tenant's chain is verified.

        On a non-(-1) result, an X.733 ``processingErrorAlarm`` /
        ``softwareError`` (CRITICAL) is emitted onto the default alarm
        bus so the SMO Fault Management plane sees the tamper detection.
        """
        per_tenant_prev: dict[str, str] = {}
        for i, (rec, stored_hash) in enumerate(self):
            tid = rec.tenant_id or _UNSCOPED_TENANT
            prev_hex = per_tenant_prev.get(tid, _ZERO_HASH_HEX)
            expected = _chain(prev_hex, _canonical_json(rec))
            if expected != stored_hash:
                # X.733 emit (Devil-A Finding #20 / WG10 OAM §6).
                try:
                    from horizon_ric.observability.x733_alarms import default_bus

                    default_bus().emit_event(
                        "horizon.evidence.tamper_detected",
                        additional_information={
                            "first_invalid_index": i,
                            "tenant_id": tid,
                        },
                    )
                except Exception:
                    # Verification must not be blocked by alarm-bus issues.
                    pass
                return i
            per_tenant_prev[tid] = stored_hash
        return -1

    def verify_tenant(self, tenant_id: str) -> int:
        """Walk the chain for exactly one tenant.

        Returns the per-tenant index of the first broken record (counting
        only that tenant's rows), or -1 if intact. A tamper of tenant A
        therefore does NOT break tenant B's verify_tenant() — chains are
        independent.
        """
        prev_hex = _ZERO_HASH_HEX
        idx = 0
        for rec, stored_hash in self:
            tid = rec.tenant_id or _UNSCOPED_TENANT
            if tid != tenant_id:
                continue
            expected = _chain(prev_hex, _canonical_json(rec))
            if expected != stored_hash:
                return idx
            prev_hex = stored_hash
            idx += 1
        return -1


class JsonlEvidenceStore(EvidenceStore):
    """File-backed evidence store. One JSON record + sha256 hash per line.

    Format per line:
        {"hash": "<sha256-hex>", "record": {...DecisionRecord...}}
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        # Cross-thread serialisation for the read-prev-then-append critical
        # section. Without this, two threads can read the same `_last_hash`
        # and both write rows whose `prev` is identical → chain breaks at
        # the second row. (See test_known_bugs::CRIT-01.)
        self._append_lock = threading.Lock()

    def _last_hash_for_tenant(self, tenant_id: str) -> str:
        """Walk the file once and return the latest chain hash for `tenant_id`.

        We deliberately re-scan the file rather than caching: the file is
        the source of truth, and another process (the audit-rotate CLI)
        may have appended in between calls. For high-throughput rApps we
        cache in memory in a follow-up; today this is correct."""
        if self._path.stat().st_size == 0:
            return _ZERO_HASH_HEX
        last_for_tenant = _ZERO_HASH_HEX
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rec_tenant = obj.get("tenant_id") or obj.get("record", {}).get(
                    "tenant_id"
                ) or _UNSCOPED_TENANT
                if rec_tenant == tenant_id:
                    last_for_tenant = obj["hash"]
        return last_for_tenant

    def append(self, record: DecisionRecord) -> str:
        # Stamp the record with the active tenant if it lacks one.
        if record.tenant_id is None:
            record = record.model_copy(update={"tenant_id": _resolve_tenant()})
        tenant_id = record.tenant_id or _UNSCOPED_TENANT
        # The read-prev-then-write critical section MUST be serialised so
        # the chain stays linear. Without this lock concurrent threads
        # can both read the same `_last_hash_for_tenant()` and emit two
        # rows whose `prev` hash is identical → chain breaks at the
        # second row. (Reproducer: tests/test_known_bugs::CRIT-01.)
        with self._append_lock:
            prev = self._last_hash_for_tenant(tenant_id)
            payload = _canonical_json(record)
            chain_hash = _chain(prev, payload)
            line = json.dumps(
                {
                    "hash": chain_hash,
                    "tenant_id": tenant_id,
                    "record": json.loads(payload),
                },
                sort_keys=True,
            )
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        return chain_hash

    def __iter__(self) -> Iterator[tuple[DecisionRecord, str]]:
        active_tenant = current_tenant()
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rec = DecisionRecord.model_validate(obj["record"])
                rec_tenant = obj.get("tenant_id") or rec.tenant_id or _UNSCOPED_TENANT
                # Tenant isolation: when a TenantScope is active, only
                # records belonging to that tenant are visible. This is
                # what blocks an auditor@tenant_A from reading tenant_B.
                if active_tenant is not None and rec_tenant != active_tenant:
                    continue
                yield rec, obj["hash"]

    def __len__(self) -> int:
        return sum(1 for _ in self)


class SqliteEvidenceStore(EvidenceStore):
    """SQLite-backed evidence store with the same hash-chain semantics."""

    def __init__(self, url: str = "sqlite:///horizon_evidence.db"):
        self._engine: Engine = create_engine(url, future=True)
        # Serialise the read-prev-then-insert critical section. SQLite
        # can serialise writes itself but the *prev hash* is computed
        # outside the transaction in this implementation, so we still
        # need an explicit lock for chain correctness under
        # multi-threaded `.append`. (See test_known_bugs::CRIT-02.)
        self._append_lock = threading.Lock()
        self._meta = MetaData()
        self._table = Table(
            "decisions",
            self._meta,
            Column("seq", Integer, primary_key=True, autoincrement=True),
            Column("decision_id", String(64), unique=True, nullable=False),
            Column("tenant_id", String(128), nullable=False, index=True),
            Column("payload_json", Text, nullable=False),
            Column("chain_hash", LargeBinary(32), nullable=False),
        )
        self._meta.create_all(self._engine)

    def _last_hash_for_tenant(self, tenant_id: str) -> str:
        with self._engine.begin() as conn:
            row = conn.execute(
                select(self._table.c.chain_hash)
                .where(self._table.c.tenant_id == tenant_id)
                .order_by(self._table.c.seq.desc())
                .limit(1)
            ).first()
        return row[0].hex() if row else _ZERO_HASH_HEX

    def append(self, record: DecisionRecord) -> str:
        if record.tenant_id is None:
            record = record.model_copy(update={"tenant_id": _resolve_tenant()})
        tenant_id = record.tenant_id or _UNSCOPED_TENANT
        with self._append_lock:
            prev = self._last_hash_for_tenant(tenant_id)
            payload = _canonical_json(record)
            chain_hash_hex = _chain(prev, payload)
            with self._engine.begin() as conn:
                conn.execute(
                    insert(self._table).values(
                        decision_id=record.decision_id,
                        tenant_id=tenant_id,
                        payload_json=payload,
                        chain_hash=bytes.fromhex(chain_hash_hex),
                    )
                )
        return chain_hash_hex

    def __iter__(self) -> Iterator[tuple[DecisionRecord, str]]:
        active_tenant = current_tenant()
        with self._engine.begin() as conn:
            stmt = select(
                self._table.c.payload_json,
                self._table.c.chain_hash,
                self._table.c.tenant_id,
            ).order_by(self._table.c.seq.asc())
            if active_tenant is not None:
                stmt = stmt.where(self._table.c.tenant_id == active_tenant)
            rows = conn.execute(stmt).all()
        for payload_json, chain_hash, _tid in rows:
            rec = DecisionRecord.model_validate_json(payload_json)
            yield rec, chain_hash.hex()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._engine.begin():
            yield


__all__ = [
    "EvidenceStore",
    "JsonlEvidenceStore",
    "SqliteEvidenceStore",
]
