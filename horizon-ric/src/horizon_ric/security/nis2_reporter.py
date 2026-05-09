"""NIS2 24h/72h/1-month incident-reporting clock.

Devil-A Finding #5 / Solver-2 Fix #10 closure.

Directive (EU) 2022/2555 (NIS2) Art. 23(4):

    (a) early warning within 24 hours of becoming aware
    (b) incident notification within 72 hours
    (c) intermediate report on supervisor request
    (d) final report no later than one month after notification under (b)

This module is the rApp's automated trigger surface. It is a webhook
client (``httpx`` based, so unit tests can drive it with
``httpx.MockTransport``) plus a milestone calculator. It records every
trigger into an ``EvidenceStore`` so the audit chain captures the start
time and the three downstream milestones — that is the artefact the
operator's CSIRT presents to the supervisory authority.

The trigger predicates are deliberately simple and operator-tunable:

  1. Audit-chain integrity failure — ``EvidenceStore.verify()`` returns
     a non-(-1) value, meaning a record at position ``k`` has been
     tampered with.
  2. Critical SLA breach — a single SLA breach event with severity
     >= "critical" or breach_probability >= 0.95.
  3. Auth failure rate > 1% over 5 minutes — feed observed counters in
     and the reporter computes the ratio.

The module does **not** decide whether the incident is "significant" in
the Art. 23(3) sense — that is an operator policy call. It does
guarantee a structured event at the moment the rApp reaches the trigger
threshold, so the operator's SOC has a concrete `nis2.clock.started`
timestamp to attach to the supervisory-authority filing.

References
----------
* Directive (EU) 2022/2555 (NIS2) Art. 23 — Reporting obligations.
* ENISA "Reporting Methodology for Significant Incidents" v1.0 (draft).
* 3GPP TS 33.117 §4.2.5 — security audit logs.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

# ─── trigger surface ────────────────────────────────────────────────────


@dataclass(frozen=True)
class NIS2Milestones:
    """The three Art. 23(4) milestones derived from the trigger time.

    Times are UTC; ISO-8601 with explicit ``+00:00`` suffix.
    """

    triggered_at: dt.datetime
    early_warning_due: dt.datetime  # +24h
    incident_notification_due: dt.datetime  # +72h
    final_report_due: dt.datetime  # +30d

    @classmethod
    def from_trigger(cls, triggered_at: dt.datetime) -> "NIS2Milestones":
        if triggered_at.tzinfo is None:
            triggered_at = triggered_at.replace(tzinfo=dt.timezone.utc)
        return cls(
            triggered_at=triggered_at,
            early_warning_due=triggered_at + dt.timedelta(hours=24),
            incident_notification_due=triggered_at + dt.timedelta(hours=72),
            final_report_due=triggered_at + dt.timedelta(days=30),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "triggered_at": self.triggered_at.isoformat(),
            "early_warning_due": self.early_warning_due.isoformat(),
            "incident_notification_due": self.incident_notification_due.isoformat(),
            "final_report_due": self.final_report_due.isoformat(),
        }


@dataclass
class NIS2Incident:
    """Structured representation of an incident notification.

    The fields are the union of the ENISA early-warning template and
    Art. 23(4)(b) notification template — operators can serialise this
    directly into their CSIRT submission.
    """

    incident_id: str
    severity: str  # "critical" | "high" | "medium" | "low"
    scope: str  # textual scope (e.g. "rApp evidence chain", "A1 emit pipeline")
    evidence: dict[str, Any]
    milestones: NIS2Milestones
    trigger_kind: str  # "audit_chain_integrity" / "sla_breach" / "auth_failure_rate"

    def as_payload(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "severity": self.severity,
            "scope": self.scope,
            "evidence": self.evidence,
            "trigger_kind": self.trigger_kind,
            "milestones": self.milestones.as_dict(),
            "directive": "EU 2022/2555 NIS2 Art. 23(4)",
        }


class _AuditChainSource(Protocol):
    def verify(self) -> int: ...


@dataclass
class NIS2IncidentReporter:
    """Webhook client that fires NIS2 24h-clock notifications.

    Args:
        operator_endpoint: HTTPS URL of the operator's NOC/SOC ingestion.
        csirt_endpoint:    HTTPS URL of the national CSIRT submission API.
        client:            optional pre-built ``httpx.Client``; tests pass a
                           client wrapping ``httpx.MockTransport``.
        evidence_sink:     optional callable that accepts a JSON-serialisable
                           dict and persists it to the rApp audit chain. The
                           default is a no-op for unit-test simplicity; in
                           production wire it to ``EvidenceStore.append``.
        clock:             callable returning current UTC ``datetime``. Tests
                           inject a frozen clock.
    """

    operator_endpoint: str
    csirt_endpoint: str
    client: httpx.Client | None = None
    evidence_sink: Any = None  # Callable[[dict], None] | None
    clock: Any = None  # Callable[[], dt.datetime] | None
    auth_failure_window_seconds: int = 300
    auth_failure_threshold_pct: float = 1.0

    _auth_events: deque = field(
        default_factory=lambda: deque(maxlen=10_000), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.Client(timeout=10.0)
        if self.clock is None:
            self.clock = lambda: dt.datetime.now(tz=dt.timezone.utc)

    # ─── public API ──────────────────────────────────────────────────

    def trigger_incident(
        self,
        *,
        severity: str,
        scope: str,
        evidence: dict[str, Any],
        trigger_kind: str = "manual",
        incident_id: str | None = None,
    ) -> NIS2Incident:
        """Fire an NIS2 incident — POST to operator + CSIRT, persist to chain."""
        now = self.clock()
        milestones = NIS2Milestones.from_trigger(now)
        if incident_id is None:
            incident_id = self._derive_incident_id(now, scope)
        incident = NIS2Incident(
            incident_id=incident_id,
            severity=severity,
            scope=scope,
            evidence=evidence,
            milestones=milestones,
            trigger_kind=trigger_kind,
        )
        payload = incident.as_payload()
        self._post(self.operator_endpoint, payload)
        self._post(self.csirt_endpoint, payload)
        self._persist(payload)
        return incident

    # ─── automatic predicates ────────────────────────────────────────

    def auto_check_audit_chain(
        self, store: _AuditChainSource, *, scope: str = "rApp evidence chain"
    ) -> NIS2Incident | None:
        """Auto-fire if ``store.verify()`` returns a non-(-1) value.

        The non-(-1) return means the chain head-to-record-k SHA-256
        validation failed at index k — i.e. tamper-evidence triggered.
        Per NIS2 Art. 23(3), tampering with security-evidence stores is
        a "significant incident" by default.
        """
        result = store.verify()
        if result == -1:
            return None
        return self.trigger_incident(
            severity="critical",
            scope=scope,
            evidence={"first_invalid_index": int(result)},
            trigger_kind="audit_chain_integrity",
        )

    def auto_check_sla_breach(
        self,
        *,
        breach_probability: float,
        slice_id: str,
        threshold: float = 0.95,
    ) -> NIS2Incident | None:
        """Auto-fire if a single SLA breach event crosses the critical threshold."""
        if breach_probability < threshold:
            return None
        return self.trigger_incident(
            severity="critical",
            scope=f"SLA breach on slice {slice_id}",
            evidence={
                "breach_probability": float(breach_probability),
                "threshold": float(threshold),
                "slice_id": slice_id,
            },
            trigger_kind="sla_breach",
        )

    def record_auth_event(self, *, success: bool, when: dt.datetime | None = None) -> None:
        """Record an auth attempt; used by ``auto_check_auth_failure_rate``."""
        when = when or self.clock()
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        self._auth_events.append((when, bool(success)))

    def auto_check_auth_failure_rate(self) -> NIS2Incident | None:
        """Auto-fire if auth-failure rate exceeds threshold over the window."""
        now = self.clock()
        window_start = now - dt.timedelta(seconds=self.auth_failure_window_seconds)
        # Drop events older than the window.
        recent = [e for e in self._auth_events if e[0] >= window_start]
        if len(recent) < 10:
            # Not enough samples — stay below the noise floor.
            return None
        failures = sum(1 for _t, ok in recent if not ok)
        rate_pct = (failures / len(recent)) * 100.0
        if rate_pct <= self.auth_failure_threshold_pct:
            return None
        return self.trigger_incident(
            severity="high",
            scope=f"auth-failure rate {rate_pct:.2f}% over "
            f"{self.auth_failure_window_seconds}s",
            evidence={
                "failure_rate_pct": rate_pct,
                "threshold_pct": self.auth_failure_threshold_pct,
                "window_seconds": self.auth_failure_window_seconds,
                "samples": len(recent),
                "failures": failures,
            },
            trigger_kind="auth_failure_rate",
        )

    # ─── helpers ─────────────────────────────────────────────────────

    def _post(self, url: str, payload: dict[str, Any]) -> None:
        try:
            assert self.client is not None
            r = self.client.post(url, json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            # Persist the delivery failure to the chain so the auditor
            # has evidence the rApp tried; do NOT swallow silently.
            self._persist(
                {
                    "kind": "nis2.delivery_failure",
                    "url": url,
                    "incident_id": payload.get("incident_id"),
                    "error": str(exc),
                }
            )

    def _persist(self, payload: dict[str, Any]) -> None:
        if self.evidence_sink is None:
            return
        try:
            self.evidence_sink(payload)
        except Exception:
            # An evidence-sink failure must not crash the SOC trigger path.
            pass

    @staticmethod
    def _derive_incident_id(when: dt.datetime, scope: str) -> str:
        slug = "".join(ch if ch.isalnum() else "-" for ch in scope).strip("-")
        return f"nis2-{when.strftime('%Y%m%dT%H%M%SZ')}-{slug[:40]}"


__all__ = [
    "NIS2Incident",
    "NIS2IncidentReporter",
    "NIS2Milestones",
]
