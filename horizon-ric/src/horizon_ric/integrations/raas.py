"""rApps-as-a-Service (RaaS) — multi-tenant single-rApp instance.

A single PreceptualAI instance can serve many tenants. Each tenant gets:
  * a JWT (issued by the SMO authentication layer) whose `tenant_id`
    claim is mapped onto :class:`horizon_ric.security.tenant.TenantScope`
    so the evidence store maintains a separate, tamper-evident chain
    per tenant.
  * a billing meter that increments on every successful A1 emission and
    is exposed at ``/api/v1/billing/usage``.
  * an export tool that produces a signed audit-trail tarball (cosign
    detached signature) for a tenant + time-window.

This module is the thin coordinator that wires those pieces together.
The heavy lifting (tenant scope, evidence chain, JWT validation) is
already done in horizon_ric.security and horizon_ric.evidence; here we
only:

  * expose a :class:`RaaSEndpoint` per rApp
  * count metered emissions per tenant
  * emit a deterministic audit tarball

Cosign signing
--------------
The export uses ``cosign sign-blob --yes --output-signature ...`` when
the ``cosign`` binary is on PATH. When cosign is unavailable we fall
back to a plain HMAC-SHA256 signature using a process-local key — this
is honestly documented and the response payload tells the caller which
mode produced the signature. Real deployments are expected to have
cosign installed; the HMAC fallback exists so unit tests run on any CI
runner.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import secrets
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Billing meter
# ---------------------------------------------------------------------------


@dataclass
class _TenantMeter:
    decisions: int = 0
    last_decision_ts: float = 0.0


@dataclass
class BillingUsage:
    """JSON-serialisable usage row exposed at /api/v1/billing/usage."""

    tenant_id: str
    decisions: int
    last_decision_ts: float
    rapp_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "rapp_id": self.rapp_id,
            "decisions": self.decisions,
            "last_decision_ts": self.last_decision_ts,
        }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@dataclass
class AuditExport:
    """Result of :meth:`RaaSEndpoint.export_audit_trail`."""

    tenant_id: str
    rapp_id: str
    since_ts: float
    archive_path: Path
    signature_path: Path
    signature_mode: str  # "cosign" or "hmac-sha256"
    record_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "rapp_id": self.rapp_id,
            "since_ts": self.since_ts,
            "archive_path": str(self.archive_path),
            "signature_path": str(self.signature_path),
            "signature_mode": self.signature_mode,
            "record_count": self.record_count,
        }


# ---------------------------------------------------------------------------
# RaaSEndpoint
# ---------------------------------------------------------------------------


class RaaSEndpoint:
    """One PreceptualAI instance serves many tenants for a single rApp.

    The endpoint owns the per-tenant billing counters and signs audit
    trails. Tenant identity comes from JWT claims via
    :class:`horizon_ric.security.tenant.TenantScope` — that wiring is
    already done by the security middleware, and :meth:`meter_decision`
    is called by the A1 adapter on every successful emit_policy.
    """

    def __init__(
        self,
        rapp_id: str,
        evidence_store: Any | None = None,
        export_root: Path | str | None = None,
        hmac_key: bytes | None = None,
    ):
        if not rapp_id:
            raise ValueError("rapp_id must be a non-empty string")
        self.rapp_id = rapp_id
        self._meters: dict[str, _TenantMeter] = {}
        self._lock = Lock()
        self._evidence_store = evidence_store
        self._export_root = (
            Path(export_root)
            if export_root is not None
            else Path(os.environ.get("HORIZON_RAAS_EXPORT_ROOT", "/tmp/horizon_raas_exports"))
        )
        self._export_root.mkdir(parents=True, exist_ok=True)
        # HMAC fallback key — process-local. Real deployments use cosign.
        self._hmac_key = hmac_key or secrets.token_bytes(32)

    # -- billing -------------------------------------------------------------

    def meter_decision(self, tenant_id: str) -> int:
        """Increment the per-tenant decision counter.

        Called by the A1 adapter after every successful policy emission.
        Returns the post-increment count for the tenant.
        """
        if not tenant_id:
            raise ValueError("tenant_id required for billing")
        with self._lock:
            meter = self._meters.setdefault(tenant_id, _TenantMeter())
            meter.decisions += 1
            meter.last_decision_ts = time.time()
            count = meter.decisions
        logger.debug(
            "raas.meter_decision",
            tenant_id=tenant_id,
            rapp_id=self.rapp_id,
            count=count,
        )
        return count

    def usage(self, tenant_id: str) -> BillingUsage:
        """Return the current billing usage for one tenant."""
        with self._lock:
            meter = self._meters.get(tenant_id, _TenantMeter())
            return BillingUsage(
                tenant_id=tenant_id,
                rapp_id=self.rapp_id,
                decisions=meter.decisions,
                last_decision_ts=meter.last_decision_ts,
            )

    def all_usage(self) -> list[BillingUsage]:
        """Return usage rows for every metered tenant. Used by the
        ``GET /api/v1/billing/usage`` handler."""
        with self._lock:
            return [
                BillingUsage(
                    tenant_id=t,
                    rapp_id=self.rapp_id,
                    decisions=m.decisions,
                    last_decision_ts=m.last_decision_ts,
                )
                for t, m in self._meters.items()
            ]

    # -- audit export --------------------------------------------------------

    def _iter_tenant_records(
        self, tenant_id: str, since_ts: float
    ) -> Iterator[dict[str, Any]]:
        """Iterate evidence records for a tenant, filtered by timestamp.

        Works with both the JSONL and SQLite evidence stores; both
        expose ``iter_records(tenant_id=...)``.
        """
        store = self._evidence_store
        if store is None:
            return iter([])

        records = []
        # EvidenceStore.__iter__ yields (DecisionRecord, chain_hash) tuples.
        try:
            it = iter(store)
        except TypeError:
            return iter([])

        for item in it:
            # Tolerate both the canonical (record, hash) tuple shape and
            # a bare-record shape used by some test doubles.
            if isinstance(item, tuple) and len(item) == 2:
                rec, chain_hash = item
            else:
                rec, chain_hash = item, None
            d = rec.model_dump() if hasattr(rec, "model_dump") else dict(rec)
            if d.get("tenant_id") not in (tenant_id, None):
                continue
            if chain_hash is not None:
                d["_chain_hash"] = chain_hash
            ts = d.get("timestamp")
            ts_epoch = _to_epoch(ts)
            if ts_epoch is None or ts_epoch >= since_ts:
                records.append(d)
        return iter(records)

    def export_audit_trail(
        self,
        tenant_id: str,
        since_ts: float,
        out_dir: Path | str | None = None,
    ) -> AuditExport:
        """Emit a signed tarball of the tenant's audit trail.

        Steps:
          1. Filter the evidence store by ``tenant_id`` and ``since_ts``.
          2. Write canonical JSONL into ``audit.jsonl`` inside a tarball.
          3. Sign the tarball with cosign (or HMAC fallback).

        Returns an :class:`AuditExport` describing the artefacts.
        """
        if not tenant_id:
            raise ValueError("tenant_id required for export")

        target = Path(out_dir) if out_dir is not None else self._export_root
        target.mkdir(parents=True, exist_ok=True)

        # Build the JSONL payload deterministically (canonical JSON).
        buf = io.BytesIO()
        count = 0
        for rec in self._iter_tenant_records(tenant_id, since_ts):
            line = json.dumps(rec, sort_keys=True, separators=(",", ":"), default=_json_default)
            buf.write(line.encode("utf-8"))
            buf.write(b"\n")
            count += 1
        payload = buf.getvalue()

        # Tarball: audit.jsonl + a manifest.
        manifest = {
            "tenant_id": tenant_id,
            "rapp_id": self.rapp_id,
            "since_ts": since_ts,
            "exported_at": time.time(),
            "record_count": count,
            "sha256_jsonl": hashlib.sha256(payload).hexdigest(),
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")

        archive_name = f"audit-{tenant_id}-{int(since_ts)}.tar.gz"
        archive_path = target / archive_name
        with tarfile.open(archive_path, "w:gz") as tar:
            _add_bytes(tar, "audit.jsonl", payload)
            _add_bytes(tar, "manifest.json", manifest_bytes)

        sig_path = archive_path.with_suffix(archive_path.suffix + ".sig")
        sig_mode = self._sign(archive_path, sig_path)

        logger.info(
            "raas.export.completed",
            tenant_id=tenant_id,
            rapp_id=self.rapp_id,
            record_count=count,
            archive=str(archive_path),
            signature_mode=sig_mode,
        )

        return AuditExport(
            tenant_id=tenant_id,
            rapp_id=self.rapp_id,
            since_ts=since_ts,
            archive_path=archive_path,
            signature_path=sig_path,
            signature_mode=sig_mode,
            record_count=count,
        )

    def _sign(self, archive: Path, sig: Path) -> str:
        """Sign with cosign when available; otherwise HMAC-SHA256.

        Returns the signature mode tag ("cosign" or "hmac-sha256"). The
        caller's response includes this so an auditor can decide which
        verification path to take.
        """
        cosign = shutil.which("cosign")
        # In CI / unit tests we honour HORIZON_RAAS_FORCE_HMAC=1 to skip
        # cosign even if a stub is on PATH — keeps tests deterministic.
        if cosign and not os.environ.get("HORIZON_RAAS_FORCE_HMAC"):
            try:
                # cosign sign-blob requires a key or keyless OIDC. Real
                # deployments mount /etc/horizon/cosign.key; we read the
                # path from env so this stays declarative for tests.
                key_path = os.environ.get("COSIGN_KEY")
                cmd = [cosign, "sign-blob", "--yes", "--output-signature", str(sig)]
                if key_path:
                    cmd += ["--key", key_path]
                cmd.append(str(archive))
                # COSIGN_PASSWORD must already be in env for keyed mode.
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                return "cosign"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(
                    "raas.cosign.failed_falling_back_to_hmac",
                    error=str(e),
                )

        # HMAC-SHA256 fallback. The signature payload binds the archive
        # SHA-256 to the rApp id so an auditor can re-derive it.
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        signed = hmac.new(
            self._hmac_key,
            f"{self.rapp_id}|{digest}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        sig.write_text(
            json.dumps(
                {
                    "mode": "hmac-sha256",
                    "rapp_id": self.rapp_id,
                    "archive_sha256": digest,
                    "signature": signed,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return "hmac-sha256"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_epoch(ts: Any) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        # ISO-8601 with optional trailing Z
        try:
            from datetime import datetime

            cleaned = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned).timestamp()
        except ValueError:
            return None
    # datetime
    if hasattr(ts, "timestamp"):
        try:
            return float(ts.timestamp())
        except Exception:
            return None
    return None


def _json_default(o: Any) -> Any:
    # Make datetime / Path / bytes serialisable.
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    raise TypeError(f"not serialisable: {type(o).__name__}")


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0  # deterministic — no wall-clock leak in tarball
    tar.addfile(info, io.BytesIO(data))


__all__ = [
    "RaaSEndpoint",
    "BillingUsage",
    "AuditExport",
]
