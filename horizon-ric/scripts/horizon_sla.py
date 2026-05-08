"""horizon-sla — SLA admin CLI.

Real `typer` CLI backed by the SQLite store at `data/horizon_sla.db`.

Examples
--------
horizon-sla create --file my-sla.yaml
horizon-sla list
horizon-sla breaches --since 24h
horizon-sla ack <breach_id>
horizon-sla policy create-escalation \\
    --name on-call-tier-1 \\
    --steps slack@5min,pagerduty@15min,email-cto@30min
horizon-sla seed-defaults
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
import yaml

from horizon_ric.sla.escalation import EscalationPolicy, EscalationStep
from horizon_ric.sla.policy import (
    SLA,
    SLOTarget,
    ack_breach,
    default_engine,
    list_breaches,
    list_slas,
    persist_sla,
)

app = typer.Typer(help="PreceptualAI SLA administration.")
policy_app = typer.Typer(help="Manage escalation policies.")
app.add_typer(policy_app, name="policy")


_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|sec|m|min|h|hr|d|day|days)$", re.IGNORECASE)


def _parse_duration(s: str) -> timedelta:
    """Parse '5m', '24h', '300s' → timedelta. Raises ValueError on junk."""
    m = _DURATION_RE.match(s.strip())
    if not m:
        raise ValueError(f"unparseable duration: {s!r}")
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("s", "sec"):
        return timedelta(seconds=value)
    if unit in ("m", "min"):
        return timedelta(minutes=value)
    if unit in ("h", "hr"):
        return timedelta(hours=value)
    if unit in ("d", "day", "days"):
        return timedelta(days=value)
    raise ValueError(f"unknown unit in duration: {s!r}")


# ---------------------------------------------------------------------------
# create / list / breaches / ack
# ---------------------------------------------------------------------------


@app.command("create")
def create(
    file: Path = typer.Option(..., "--file", "-f", exists=True, readable=True),
    db_url: Optional[str] = typer.Option(None, "--db-url"),
) -> None:
    """Create one or many SLAs from a YAML file."""
    raw = yaml.safe_load(file.read_text())
    engine = default_engine(db_url)
    items = raw.get("slas", [raw]) if isinstance(raw, dict) else raw
    for item in items:
        sla = SLA(
            id=item.get("id"),
            name=item["name"],
            description=item.get("description", ""),
            scope=item.get("scope", {}),
            targets=[SLOTarget(**t) for t in item["targets"]],
            severity_levels=item.get("severity_levels", {}),
        ) if item.get("id") else SLA(
            name=item["name"],
            description=item.get("description", ""),
            scope=item.get("scope", {}),
            targets=[SLOTarget(**t) for t in item["targets"]],
            severity_levels=item.get("severity_levels", {}),
        )
        persist_sla(engine, sla)
        typer.echo(f"created sla id={sla.id} name={sla.name!r}")


@app.command("list")
def list_cmd(
    db_url: Optional[str] = typer.Option(None, "--db-url"),
) -> None:
    """List active SLAs."""
    engine = default_engine(db_url)
    slas = list_slas(engine)
    if not slas:
        typer.echo("(no SLAs configured)")
        return
    for s in slas:
        typer.echo(
            f"{s.id}\t{s.name}\ttargets={len(s.targets)}\tscope={json.dumps(s.scope, sort_keys=True)}"
        )


@app.command("breaches")
def breaches(
    since: str = typer.Option("24h", "--since", help="e.g. 30m / 24h / 7d"),
    sla_id: Optional[str] = typer.Option(None, "--sla-id"),
    db_url: Optional[str] = typer.Option(None, "--db-url"),
) -> None:
    """List breaches newer than `since`."""
    engine = default_engine(db_url)
    cutoff = datetime.now(timezone.utc) - _parse_duration(since)
    rows = list_breaches(engine, sla_id=sla_id, since=cutoff)
    if not rows:
        typer.echo("(no breaches)")
        return
    for b in rows:
        ack = "ACK" if b.acknowledged else "open"
        typer.echo(
            f"{b.id}\t{b.ts_utc.isoformat()}\t{b.sla_id}\t{b.severity}\t"
            f"{b.source_metric}={b.observed_value} (thr={b.target_threshold}) [{ack}]"
        )


@app.command("ack")
def ack(
    breach_id: str = typer.Argument(...),
    by: str = typer.Option("cli", "--by"),
    db_url: Optional[str] = typer.Option(None, "--db-url"),
) -> None:
    """Acknowledge a breach by id."""
    engine = default_engine(db_url)
    ok = ack_breach(engine, breach_id, by=by)
    typer.echo(f"ack {breach_id}: {'ok' if ok else 'not found'}")
    if not ok:
        raise typer.Exit(code=1)


@app.command("seed-defaults")
def seed_defaults(
    file: Path = typer.Option(
        Path("data/sla_defaults.yaml"), "--file", "-f", exists=True
    ),
    db_url: Optional[str] = typer.Option(None, "--db-url"),
) -> None:
    """Load `data/sla_defaults.yaml` into the SLA DB."""
    create(file=file, db_url=db_url)


# ---------------------------------------------------------------------------
# escalation policies
# ---------------------------------------------------------------------------

# Escalation policies are stored as JSON in a sibling table reusing the
# slas DB. Simpler than adding a second migration; the structure is
# {"name": str, "steps": [...]}, persisted under id "policy:<name>".

_STEP_RE = re.compile(r"^([a-z0-9_-]+)@(\d+(?:\.\d+)?(?:s|m|h|d))$", re.IGNORECASE)


def _parse_step(step_str: str) -> EscalationStep:
    """Parse 'slack@5min' or 'pagerduty@15m' → EscalationStep."""
    m = _STEP_RE.match(step_str.strip())
    if not m:
        raise ValueError(f"unparseable step: {step_str!r} (expected channel@delay)")
    channel = m.group(1).lower()
    delay = _parse_duration(m.group(2)).total_seconds()
    # We accept "email-cto" etc. as a recipient hint.
    if "-" in channel:
        ch_kind, recipient = channel.split("-", 1)
        recipients = [recipient]
    else:
        ch_kind = channel
        recipients = []
    if ch_kind not in ("slack", "pagerduty", "email", "webhook"):
        raise ValueError(f"unknown channel kind: {ch_kind}")
    return EscalationStep(
        delay_s=delay,
        channels=[ch_kind],  # type: ignore[list-item]
        recipients=recipients,
    )


@policy_app.command("create-escalation")
def create_escalation(
    name: str = typer.Option(..., "--name"),
    steps: str = typer.Option(..., "--steps", help="comma-list, e.g. slack@5min,pagerduty@15min"),
    db_url: Optional[str] = typer.Option(None, "--db-url"),
) -> None:
    """Create or update an escalation policy."""
    parsed = [_parse_step(s) for s in steps.split(",")]
    policy = EscalationPolicy(name=name, steps=parsed)
    # Reuse the SLA DB; store as a single-target SLA-shaped row under id `policy:<name>`.
    # This keeps migrations simple while remaining queryable.
    from horizon_ric.sla.policy import (
        create_schema,
        slas_table,
        load_slas as _load,  # noqa: F401
    )
    from sqlalchemy import insert, delete

    engine = default_engine(db_url)
    create_schema(engine)
    pid = f"policy:{policy.name}"
    payload = json.dumps(policy.model_dump(), sort_keys=True)
    with engine.begin() as conn:
        conn.execute(delete(slas_table).where(slas_table.c.id == pid))
        conn.execute(
            insert(slas_table).values(
                id=pid,
                name=f"escalation::{policy.name}",
                description=payload,
                scope_json="{}",
                targets_json="[]",
                severity_levels_json="{}",
                created_at=datetime.now(timezone.utc),
            )
        )
    typer.echo(f"created escalation policy {policy.name} with {len(parsed)} steps")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
