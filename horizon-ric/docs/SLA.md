# SLA Management Surface

PreceptualAI ships a full SLA management surface for operators: define
targets, persist them, evaluate observations every tick, alert on
breaches, escalate on missed ACKs, and annotate the audit chain with
breach context so a regulator can prove which decision was in flight
during a breach.

All metric names are 3GPP TS 28.554 §6 KPIs or rApp-internal indicators.

## Default SLA Templates (data/sla_defaults.yaml)

Five production-ready SLAs ship out of the box. Seed them with:

    horizon-sla seed-defaults

| ID                              | Target                                | Window |
| ------------------------------- | ------------------------------------- | ------ |
| sla-latency-p99                 | latency_p99_ms ≤ 10                   | 5 min  |
| sla-a1-emit-success             | availability_pct ≥ 99.99              | 24 h   |
| sla-audit-chain-integrity       | availability_pct ≥ 100 (verify == -1) | inst.  |
| sla-epfd-margin                 | epfd_margin_dB ≥ 5                    | inst.  |
| sla-decision-latency-p99        | latency_p99_ms ≤ 100                  | 5 min  |

## Custom SLAs (YAML)

```yaml
slas:
  - name: "NTN feeder availability"
    description: "Maritime feeder link must hit 4-9s availability"
    scope:
      tenant: maritime
      gateway: G3
    targets:
      - metric: availability_pct
        comparison: ">="
        threshold: 99.99
        window_s: 3600
    severity_levels:
      critical:
        metrics: [availability_pct]
```

Apply:

    horizon-sla create --file my-sla.yaml

`metric` must be one of:
`latency_p99_ms`, `throughput_mbps`, `ber`, `sla_risk_30s`,
`epfd_margin_dB`, `availability_pct`.

## Engine Behaviour

`SLAEvaluator.evaluate(observation, decision_record)` runs in the
rApp main loop. For each (SLA, target):

1. Compare `observation[metric]` against `threshold`.
2. On first failed comparison, start a per-target timer.
3. After `window_s` of sustained failure, emit ONE `SLABreachEvent` and
   continue updating its `lasting_s` field on subsequent ticks (no flood).
4. On recovery, clear the timer; the next breach is a NEW event.

## Alerting

`AlertManagerClient(base_url).route_breach_to_alertmanager(breach)`
posts a v2-spec alert to `{base_url}/api/v2/alerts`. Payload labels
include `alertname`, `severity`, `sla_id`, `tenant`, `source_metric`;
annotations carry `summary`, `description`, `decision_id`,
`observed_value`, `threshold`, `lasting_s`.

A Prometheus-native fallback ships at
`deploy/prometheus/sla_rules.yml` for the cases where the rApp itself
is degraded.

## Escalation

Critical breaches walk an `EscalationPolicy`:

```python
policy = EscalationPolicy(
    name="on-call-tier-1",
    steps=[
        EscalationStep(delay_s=300, channels=["slack"], recipients=["#noc"]),
        EscalationStep(delay_s=900, channels=["pagerduty"], recipients=["sre"]),
        EscalationStep(delay_s=1800, channels=["email"], recipients=["cto@example.com"]),
    ],
)
```

Each step fires its channels, then waits `delay_s` for an ACK at
`POST /api/v1/sla/breaches/{id}/ack`. No ACK ⇒ advance.

### Channel setup (env vars only — never hardcoded)

| Channel    | Required env vars                                                   |
| ---------- | ------------------------------------------------------------------- |
| Slack      | `HORIZON_SLACK_WEBHOOK`                                             |
| PagerDuty  | `HORIZON_PAGERDUTY_KEY`                                             |
| Email      | `HORIZON_SMTP_HOST`, `HORIZON_SMTP_PORT`, `HORIZON_SMTP_USER`, `HORIZON_SMTP_PASS`, `HORIZON_SMTP_FROM` |
| Webhook    | (URL passed per-channel)                                            |

## Audit-log annotation

Right before `EvidenceStore.append(record)`:

```python
breaches = evaluator.evaluate(observation, record)
annotate_decision_record(record, breaches)
store.append(record)
```

`record.sla_breach_context` is now a list of breach dicts.  Because
the JSON canonicalisation hashes the entire record, the breach context
is part of the chain and any tamper is detected by `store.verify()`.
