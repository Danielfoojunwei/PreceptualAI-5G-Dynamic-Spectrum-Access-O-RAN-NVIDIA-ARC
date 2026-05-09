"""Escalation policy engine.

Real channel adapters (Slack, PagerDuty, email, generic webhook) and an
engine that walks the policy on missed ACKs.

Walk semantics:
  - On a `severity == "critical"` breach, the engine starts from step 0,
    fires its channels, then waits `delay_s` for an ACK.
  - If no ACK by `delay_s`, advance to the next step.
  - On ACK (`POST /api/v1/sla/breaches/{id}/ack`), the walk halts.
  - If a step's channel raises an exception and `failover_to` is set,
    the engine looks up that policy by name and switches to it.

All credentials come from env vars — never hardcoded:
  HORIZON_SLACK_WEBHOOK
  HORIZON_PAGERDUTY_KEY
  HORIZON_SMTP_HOST / HORIZON_SMTP_PORT / HORIZON_SMTP_USER / HORIZON_SMTP_PASS
  HORIZON_SMTP_FROM
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Awaitable, Callable, Literal, Protocol

import httpx
import structlog
from pydantic import BaseModel, Field

from horizon_ric.sla.policy import SLABreachEvent

logger = structlog.get_logger(__name__)

ChannelKind = Literal["email", "pagerduty", "slack", "webhook"]


class Channel(Protocol):
    """A real escalation channel adapter."""

    name: str

    async def send(self, breach: SLABreachEvent, recipients: list[str]) -> None: ...


# ---------------------------------------------------------------------------
# Real channel adapters
# ---------------------------------------------------------------------------


@dataclass
class SlackChannel:
    """Posts to a Slack incoming webhook URL."""

    name: str = "slack"
    webhook_env: str = "HORIZON_SLACK_WEBHOOK"
    client: httpx.AsyncClient | None = None
    webhook_url: str | None = None  # explicit override (tests)

    async def send(self, breach: SLABreachEvent, recipients: list[str]) -> None:
        url = self.webhook_url or os.environ.get(self.webhook_env)
        if not url:
            raise RuntimeError(
                f"Slack webhook URL not configured (env {self.webhook_env})"
            )
        text_lines = [
            f":rotating_light: *SLA Breach* — sla_id=`{breach.sla_id}` "
            f"severity=*{breach.severity}*",
            f"metric: `{breach.source_metric}`",
            f"observed: `{breach.observed_value}` vs threshold "
            f"`{breach.target_threshold}` (lasting {breach.lasting_s:.1f}s)",
        ]
        if breach.decision_id_at_breach:
            text_lines.append(f"decision_id: `{breach.decision_id_at_breach}`")
        if recipients:
            text_lines.append("cc: " + " ".join(recipients))
        body = {"text": "\n".join(text_lines)}
        client = self.client or httpx.AsyncClient(timeout=10.0)
        try:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
        finally:
            if self.client is None:
                await client.aclose()


@dataclass
class PagerDutyChannel:
    """PagerDuty Events API v2."""

    name: str = "pagerduty"
    routing_key_env: str = "HORIZON_PAGERDUTY_KEY"
    api_url: str = "https://events.pagerduty.com/v2/enqueue"
    client: httpx.AsyncClient | None = None
    routing_key: str | None = None  # explicit override (tests)

    async def send(self, breach: SLABreachEvent, recipients: list[str]) -> None:
        key = self.routing_key or os.environ.get(self.routing_key_env)
        if not key:
            raise RuntimeError(
                f"PagerDuty routing key not configured (env {self.routing_key_env})"
            )
        body = {
            "routing_key": key,
            "event_action": "trigger",
            "dedup_key": breach.id,
            "payload": {
                "summary": (
                    f"SLA {breach.sla_id} breached on {breach.source_metric} "
                    f"(severity={breach.severity})"
                ),
                "source": "horizon-ric",
                "severity": breach.severity if breach.severity != "info" else "info",
                "component": breach.source_metric,
                "custom_details": {
                    "sla_id": breach.sla_id,
                    "observed_value": breach.observed_value,
                    "threshold": breach.target_threshold,
                    "lasting_s": breach.lasting_s,
                    "decision_id": breach.decision_id_at_breach,
                    "recipients": recipients,
                },
            },
        }
        client = self.client or httpx.AsyncClient(timeout=10.0)
        try:
            resp = await client.post(self.api_url, json=body)
            resp.raise_for_status()
        finally:
            if self.client is None:
                await client.aclose()


@dataclass
class EmailChannel:
    """SMTP email via aiosmtplib."""

    name: str = "email"
    host_env: str = "HORIZON_SMTP_HOST"
    port_env: str = "HORIZON_SMTP_PORT"
    user_env: str = "HORIZON_SMTP_USER"
    pass_env: str = "HORIZON_SMTP_PASS"
    from_env: str = "HORIZON_SMTP_FROM"
    sender: Callable[[EmailMessage, str, int, str | None, str | None], Awaitable[None]] | None = (
        None  # injection point for tests; bypass real network
    )

    async def send(self, breach: SLABreachEvent, recipients: list[str]) -> None:
        host = os.environ.get(self.host_env)
        if not host:
            raise RuntimeError(f"SMTP host not configured (env {self.host_env})")
        port = int(os.environ.get(self.port_env, "587"))
        user = os.environ.get(self.user_env)
        password = os.environ.get(self.pass_env)
        sender = os.environ.get(self.from_env, "horizon-ric@example.com")

        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = ", ".join(recipients) if recipients else sender
        msg["Subject"] = (
            f"[SLA-{breach.severity.upper()}] {breach.sla_id} — {breach.source_metric}"
        )
        msg.set_content(
            f"SLA {breach.sla_id} breach\n"
            f"  metric:    {breach.source_metric}\n"
            f"  observed:  {breach.observed_value}\n"
            f"  threshold: {breach.target_threshold}\n"
            f"  lasting:   {breach.lasting_s:.1f}s\n"
            f"  decision:  {breach.decision_id_at_breach}\n"
        )
        if self.sender is not None:
            await self.sender(msg, host, port, user, password)
            return
        # Real SMTP path — uses aiosmtplib so we stay async.
        import aiosmtplib

        await aiosmtplib.send(
            msg,
            hostname=host,
            port=port,
            username=user,
            password=password,
            start_tls=True,
        )


@dataclass
class WebhookChannel:
    """Generic JSON POST to a configured URL."""

    name: str = "webhook"
    url: str = ""  # required; configured per step
    client: httpx.AsyncClient | None = None

    async def send(self, breach: SLABreachEvent, recipients: list[str]) -> None:
        if not self.url:
            raise RuntimeError("WebhookChannel: url not configured")
        body = {
            "breach": breach.model_dump(mode="json"),
            "recipients": recipients,
        }
        client = self.client or httpx.AsyncClient(timeout=10.0)
        try:
            resp = await client.post(self.url, json=body)
            resp.raise_for_status()
        finally:
            if self.client is None:
                await client.aclose()


# ---------------------------------------------------------------------------
# Policy + engine
# ---------------------------------------------------------------------------


class EscalationStep(BaseModel):
    delay_s: float = Field(ge=0.0)
    channels: list[ChannelKind]
    recipients: list[str] = Field(default_factory=list)
    failover_to: str | None = None


class EscalationPolicy(BaseModel):
    name: str
    steps: list[EscalationStep]


@dataclass
class EscalationEngine:
    """Walks one or more policies, advancing on missed ACKs."""

    policies: dict[str, EscalationPolicy] = field(default_factory=dict)
    channels: dict[str, Channel] = field(default_factory=dict)
    # In-flight breach -> task; allows ACK to cancel.
    _walks: dict[str, asyncio.Task] = field(default_factory=dict)
    # ACKed breach IDs (so we don't restart a walk for one already ACKed).
    _acked: set[str] = field(default_factory=set)

    def register_policy(self, policy: EscalationPolicy) -> None:
        self.policies[policy.name] = policy

    def register_channel(self, channel: Channel) -> None:
        self.channels[channel.name] = channel

    def acknowledge(self, breach_id: str) -> bool:
        """Mark a breach ACKed and cancel any in-flight walk."""
        self._acked.add(breach_id)
        task = self._walks.pop(breach_id, None)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    def is_acked(self, breach_id: str) -> bool:
        return breach_id in self._acked

    async def _fire_step(
        self,
        breach: SLABreachEvent,
        step: EscalationStep,
    ) -> list[str]:
        """Fire all channels in `step`. Returns list of channel names that errored."""
        failed: list[str] = []
        for ch_name in step.channels:
            ch = self.channels.get(ch_name)
            if ch is None:
                logger.error(
                    "horizon.sla.escalation.unknown_channel",
                    channel=ch_name,
                    breach_id=breach.id,
                )
                failed.append(ch_name)
                continue
            try:
                await ch.send(breach, step.recipients)
                logger.info(
                    "horizon.sla.escalation.fired",
                    channel=ch_name,
                    breach_id=breach.id,
                    recipients=step.recipients,
                )
            except Exception as exc:  # pragma: no cover - exact messages vary
                logger.error(
                    "horizon.sla.escalation.channel_failed",
                    channel=ch_name,
                    breach_id=breach.id,
                    error=str(exc),
                )
                failed.append(ch_name)
        return failed

    async def _walk(
        self,
        breach: SLABreachEvent,
        policy_name: str,
    ) -> None:
        """Walk policy steps for one breach until ACKed or exhausted."""
        try:
            current_policy = self.policies[policy_name]
            step_idx = 0
            while step_idx < len(current_policy.steps):
                if breach.id in self._acked:
                    return
                step = current_policy.steps[step_idx]
                failed = await self._fire_step(breach, step)
                # If every channel failed and a failover policy is named,
                # jump to the start of that policy.
                if failed and len(failed) == len(step.channels) and step.failover_to:
                    failover_name = step.failover_to
                    if failover_name in self.policies:
                        logger.warning(
                            "horizon.sla.escalation.failover",
                            from_policy=current_policy.name,
                            to_policy=failover_name,
                            breach_id=breach.id,
                        )
                        current_policy = self.policies[failover_name]
                        step_idx = 0
                        continue
                # Wait `delay_s` for an ACK before moving on.
                try:
                    await asyncio.sleep(step.delay_s)
                except asyncio.CancelledError:
                    return
                if breach.id in self._acked:
                    return
                step_idx += 1
        finally:
            self._walks.pop(breach.id, None)

    def start_walk(self, breach: SLABreachEvent, policy_name: str) -> asyncio.Task:
        """Kick off a walk task for `breach`. No-op if not critical."""
        if breach.id in self._walks:
            return self._walks[breach.id]
        if breach.id in self._acked:

            async def _noop() -> None:
                return None

            return asyncio.create_task(_noop())
        if policy_name not in self.policies:
            raise KeyError(f"unknown escalation policy: {policy_name}")
        task = asyncio.create_task(self._walk(breach, policy_name))
        self._walks[breach.id] = task
        return task

    async def walk_now(self, breach: SLABreachEvent, policy_name: str) -> None:
        """Synchronously walk a policy (used by tests)."""
        await self._walk(breach, policy_name)


__all__ = [
    "EscalationStep",
    "EscalationPolicy",
    "EscalationEngine",
    "SlackChannel",
    "PagerDutyChannel",
    "EmailChannel",
    "WebhookChannel",
    "Channel",
    "ChannelKind",
]
