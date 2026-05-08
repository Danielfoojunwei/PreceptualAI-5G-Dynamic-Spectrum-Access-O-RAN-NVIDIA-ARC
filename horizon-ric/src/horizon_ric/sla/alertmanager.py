"""Alertmanager v2 webhook client.

Posts breach events to a real Prometheus Alertmanager
(`prom/alertmanager:latest`) using the v2 alerts API:

  POST /api/v2/alerts
  Content-Type: application/json
  Body: [
    {
      "labels":      { "alertname": "...", "severity": "...", ... },
      "annotations": { "summary": "...", "description": "...", ... },
      "startsAt":    "2026-05-06T12:34:56.789Z",
      "endsAt":      "0001-01-01T00:00:00Z",
      "generatorURL": "..."
    }
  ]

Reference: https://github.com/prometheus/alertmanager/blob/main/api/v2/openapi.yaml
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from horizon_ric.sla.policy import SLABreachEvent

logger = structlog.get_logger(__name__)

# Per OpenAPI v2 spec: `endsAt: null` is rejected — the schema requires a
# zero-time RFC-3339 timestamp instead. Alertmanager treats that as "open".
_OPEN_END_TIME = "0001-01-01T00:00:00Z"


def _iso8601(ts: datetime) -> str:
    """RFC-3339 UTC with millisecond precision (Alertmanager-friendly)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def format_breach_for_alertmanager(
    breach: SLABreachEvent,
    *,
    sla_name: str | None = None,
    tenant: str | None = None,
    generator_url: str | None = None,
) -> dict[str, Any]:
    """Build one Alertmanager v2 alert dict from a breach event.

    Schema diffs cleanly against the published OpenAPI v2 `postableAlert`
    object (labels + annotations as `<name,value>` maps, startsAt/endsAt
    as RFC3339 date-times, optional generatorURL).
    """
    labels: dict[str, str] = {
        "alertname": f"SLABreach_{breach.source_metric}",
        "severity": breach.severity,
        "sla_id": breach.sla_id,
        "source_metric": breach.source_metric,
    }
    if sla_name:
        labels["sla_name"] = sla_name
    if tenant:
        labels["tenant"] = tenant

    annotations: dict[str, str] = {
        "summary": (
            f"SLA {breach.sla_id} breached on {breach.source_metric} "
            f"(severity={breach.severity})"
        ),
        "description": (
            f"observed_value={breach.observed_value} "
            f"violates threshold={breach.target_threshold} "
            f"(lasting {breach.lasting_s:.1f}s)"
        ),
        "observed_value": str(breach.observed_value),
        "threshold": str(breach.target_threshold),
        "lasting_s": f"{breach.lasting_s:.3f}",
    }
    if breach.decision_id_at_breach:
        annotations["decision_id"] = breach.decision_id_at_breach

    alert: dict[str, Any] = {
        "labels": labels,
        "annotations": annotations,
        "startsAt": _iso8601(breach.ts_utc),
        "endsAt": _OPEN_END_TIME,
    }
    if generator_url:
        alert["generatorURL"] = generator_url
    return alert


class AlertManagerClient:
    """Real Alertmanager v2 client. POSTs to `{base_url}/api/v2/alerts`."""

    def __init__(
        self,
        base_url: str,
        *,
        auth: tuple[str, str] | None = None,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        tenant: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout
        self.tenant = tenant
        self._client = client  # if injected (e.g. MockTransport in tests)

    def _http(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self.timeout)

    def route_breach_to_alertmanager(
        self,
        breach: SLABreachEvent,
        *,
        sla_name: str | None = None,
        generator_url: str | None = None,
    ) -> httpx.Response:
        """Format `breach` and POST it. Returns the raw Response."""
        alert = format_breach_for_alertmanager(
            breach,
            sla_name=sla_name,
            tenant=self.tenant,
            generator_url=generator_url,
        )
        url = f"{self.base_url}/api/v2/alerts"
        client = self._http()
        try:
            resp = client.post(
                url,
                json=[alert],
                auth=self.auth,
                headers={"Content-Type": "application/json"},
            )
        finally:
            if self._client is None:
                client.close()
        logger.info(
            "horizon.sla.alertmanager.posted",
            breach_id=breach.id,
            status=resp.status_code,
            url=url,
        )
        return resp

    def route_breaches(
        self,
        breaches: list[SLABreachEvent],
        *,
        sla_name_lookup: dict[str, str] | None = None,
        generator_url: str | None = None,
    ) -> httpx.Response | None:
        """Batch-post multiple breaches in a single API call."""
        if not breaches:
            return None
        alerts = [
            format_breach_for_alertmanager(
                b,
                sla_name=(sla_name_lookup or {}).get(b.sla_id),
                tenant=self.tenant,
                generator_url=generator_url,
            )
            for b in breaches
        ]
        url = f"{self.base_url}/api/v2/alerts"
        client = self._http()
        try:
            resp = client.post(
                url,
                json=alerts,
                auth=self.auth,
                headers={"Content-Type": "application/json"},
            )
        finally:
            if self._client is None:
                client.close()
        return resp


__all__ = [
    "AlertManagerClient",
    "format_breach_for_alertmanager",
]
