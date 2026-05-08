"""Tests for the escalation engine + channel adapters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from horizon_ric.sla.escalation import (
    EmailChannel,
    EscalationEngine,
    EscalationPolicy,
    EscalationStep,
    PagerDutyChannel,
    SlackChannel,
    WebhookChannel,
)
from horizon_ric.sla.policy import SLABreachEvent


def _breach() -> SLABreachEvent:
    return SLABreachEvent(
        id="breach-esc-1",
        sla_id="sla-test",
        ts_utc=datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc),
        severity="critical",
        observed_value=25.0,
        target_threshold=10.0,
        source_metric="latency_p99_ms",
        lasting_s=5.0,
    )


@dataclass
class _RecordingChannel:
    """Test double that just records every send call."""

    name: str = "rec"
    fail: bool = False
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    async def send(self, breach: SLABreachEvent, recipients: list[str]) -> None:
        self.calls.append((breach.id, list(recipients)))
        if self.fail:
            raise RuntimeError("simulated channel failure")


@pytest.mark.asyncio
async def test_step_advances_on_missed_ack():
    """No ACK by `delay_s` ⇒ engine advances to the next step."""
    ch = _RecordingChannel(name="slack")
    eng = EscalationEngine()
    eng.register_channel(ch)
    eng.register_policy(
        EscalationPolicy(
            name="tier1",
            steps=[
                EscalationStep(delay_s=0.05, channels=["slack"], recipients=["#noc"]),
                EscalationStep(delay_s=0.05, channels=["slack"], recipients=["#sre"]),
                EscalationStep(delay_s=0.0, channels=["slack"], recipients=["#cto"]),
            ],
        )
    )
    await eng.walk_now(_breach(), "tier1")
    # All three steps should have fired.
    assert len(ch.calls) == 3
    assert ch.calls[0][1] == ["#noc"]
    assert ch.calls[1][1] == ["#sre"]
    assert ch.calls[2][1] == ["#cto"]


@pytest.mark.asyncio
async def test_ack_halts_escalation():
    """ACK during step 0 wait ⇒ later steps never fire."""
    ch = _RecordingChannel(name="slack")
    eng = EscalationEngine()
    eng.register_channel(ch)
    eng.register_policy(
        EscalationPolicy(
            name="tier1",
            steps=[
                EscalationStep(delay_s=0.5, channels=["slack"], recipients=["#noc"]),
                EscalationStep(delay_s=0.0, channels=["slack"], recipients=["#sre"]),
            ],
        )
    )
    breach = _breach()
    task = eng.start_walk(breach, "tier1")
    # Give step 0 a moment to fire, then ACK.
    await asyncio.sleep(0.05)
    eng.acknowledge(breach.id)
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert len(ch.calls) == 1  # only step 0
    assert ch.calls[0][1] == ["#noc"]


@pytest.mark.asyncio
async def test_failover_on_channel_failure():
    """Every channel in a step fails ⇒ engine jumps to `failover_to` policy."""
    failing = _RecordingChannel(name="slack", fail=True)
    backup = _RecordingChannel(name="webhook")
    eng = EscalationEngine()
    eng.register_channel(failing)
    eng.register_channel(backup)
    eng.register_policy(
        EscalationPolicy(
            name="primary",
            steps=[
                EscalationStep(
                    delay_s=0.0,
                    channels=["slack"],
                    recipients=["#noc"],
                    failover_to="backup",
                ),
            ],
        )
    )
    eng.register_policy(
        EscalationPolicy(
            name="backup",
            steps=[
                EscalationStep(delay_s=0.0, channels=["webhook"], recipients=["sre"]),
            ],
        )
    )
    await eng.walk_now(_breach(), "primary")
    assert len(failing.calls) == 1  # primary tried once
    assert len(backup.calls) == 1   # then failed over


@pytest.mark.asyncio
async def test_slack_channel_posts_real_webhook():
    """SlackChannel POSTs to the webhook URL with `text` body."""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        captured["body"] = req.content
        return httpx.Response(200, text="ok")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ch = SlackChannel(
        webhook_url="https://hooks.slack.com/services/T/B/X",
        client=client,
    )
    await ch.send(_breach(), ["#noc"])
    assert captured["url"] == "https://hooks.slack.com/services/T/B/X"
    assert b'"text"' in captured["body"]
    assert b"SLA Breach" in captured["body"]
    await client.aclose()


@pytest.mark.asyncio
async def test_pagerduty_channel_v2_payload():
    """PagerDutyChannel posts the Events API v2 envelope."""
    import json

    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        captured["body"] = json.loads(req.content)
        return httpx.Response(202, json={"status": "success"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ch = PagerDutyChannel(routing_key="TEST_KEY_123", client=client)
    await ch.send(_breach(), ["sre@example.com"])
    body = captured["body"]
    assert body["routing_key"] == "TEST_KEY_123"
    assert body["event_action"] == "trigger"
    assert body["dedup_key"] == "breach-esc-1"
    assert body["payload"]["severity"] == "critical"
    assert body["payload"]["custom_details"]["sla_id"] == "sla-test"
    await client.aclose()


@pytest.mark.asyncio
async def test_email_channel_uses_injected_sender(monkeypatch):
    """EmailChannel composes a real EmailMessage and dispatches."""
    sent: dict = {}

    async def fake_send(msg, host, port, user, password):
        sent["host"] = host
        sent["port"] = port
        sent["subject"] = msg["Subject"]
        sent["body"] = msg.get_content()

    monkeypatch.setenv("HORIZON_SMTP_HOST", "smtp.test")
    monkeypatch.setenv("HORIZON_SMTP_PORT", "2525")
    monkeypatch.setenv("HORIZON_SMTP_FROM", "alerts@horizon")
    ch = EmailChannel(sender=fake_send)
    await ch.send(_breach(), ["cto@example.com"])
    assert sent["host"] == "smtp.test"
    assert sent["port"] == 2525
    assert "[SLA-CRITICAL]" in sent["subject"]
    assert "latency_p99_ms" in sent["subject"]
    assert "observed:  25.0" in sent["body"]


@pytest.mark.asyncio
async def test_already_acked_breach_skips_walk():
    """If a breach is ACKed BEFORE the walk starts, walk is a no-op."""
    ch = _RecordingChannel(name="slack")
    eng = EscalationEngine()
    eng.register_channel(ch)
    eng.register_policy(
        EscalationPolicy(
            name="tier1",
            steps=[EscalationStep(delay_s=0.0, channels=["slack"])],
        )
    )
    breach = _breach()
    eng.acknowledge(breach.id)  # pre-ACK
    task = eng.start_walk(breach, "tier1")
    await task
    assert ch.calls == []
