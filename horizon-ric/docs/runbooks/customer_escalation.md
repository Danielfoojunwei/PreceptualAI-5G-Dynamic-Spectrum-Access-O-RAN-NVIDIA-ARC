# Runbook — Customer Escalation

For the on-call when a tenant raises a Sev1/2 ticket, an SLA breach alert
fires, or a customer NOC pages directly. Pairs with `oncall.md` for
rotation, `security_incident.md` for security-flavoured escalations.

```sh
NS=${HORIZON_NS:-horizon}
TENANT=${HORIZON_TENANT:-acme}
SDK_URL=${HORIZON_RIC_URL:-https://horizon.${TENANT}}
```

## Tier definitions

| Tier | Owner                                | Pager target              | SLA target |
| ---- | ------------------------------------ | ------------------------- | ---------- |
| T1   | Operator NOC (customer-side)         | their own NOC tooling     | 5 min ack  |
| T2   | PreceptualAI engineer on-call          | PagerDuty `horizon-ric`   | 15 min ack |
| T3   | Engineering lead + CTO               | PagerDuty `horizon-exec`  | 1 hour ack |

## Auto-escalation matrix

| Source alert (Prometheus / runbook)             | Auto-page tier | Notes                                           |
| ----------------------------------------------- | -------------- | ----------------------------------------------- |
| `HorizonRAppDown` / `HorizonRAppDegraded`       | T2             | See `deploy/RUNBOOK.md` rapp-down / rapp-degraded |
| `/healthz` 503 sustained > 60 s                 | T2             | See `deploy/RUNBOOK.md` healthz-503             |
| `HorizonA1EmitFailureRate > 1% for 5 min`       | T2             | See `deploy/RUNBOOK.md` a1-emit-failure         |
| Audit-chain verify fail (`EvidenceStore.verify`)| T3 + Sev1 sec  | Pivot to `security_incident.md` immediately     |
| Tenant SLA breach: latency p99 > target × 1.5  | T2             | See `src/horizon_ric/sla/engine.py`             |
| Tenant SLA breach > 30 min unmitigated          | T3             | Auto-bumps; CTO joins bridge                    |
| `HorizonQueueDepth` > 1k for 5 min              | T2             | See `deploy/RUNBOOK.md` queue-depth             |

Manual triage: any non-alert customer ticket. T1 customer triages first;
they escalate to T2 by paging or by replying with `@horizon-oncall` in the
shared Slack channel.

## Severity / impact

| Sev | Trigger                                                                          | Impact                                          |
| --- | -------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1   | Contractual SLA breach: `deploy/SLO.md` row 1 (availability < 99.999% / `/healthz` 5xx) sustained > 60 s | tenant page, regulator clock starts            |
| 2   | 3+ failed A1 emits visible to operator within 5 min, OR `deploy/SLO.md` row 4 / row 4a (constrained-Orin) burn | partial degradation, contractual A1 success bar |
| 3   | Latency p99 > target × 1.5 (rows 2/2a/2b/2c/2d), single tenant                   | warning, ticket                                 |

## Detection

Prometheus alerts (from `deploy/prometheus/rules.yml`, mapped to SLO rows):

- `HorizonRAppDown` — SLO row 1 (`/healthz` 200 < 99.999%)
- `HorizonA1EmitFailureRate` — SLO row 4 (≥ 99.9% post-retry)
- `HorizonA1EmitFailureRateOrin` — SLO row 4a (constrained-Orin envelope, ≥ 99.5%)
- `HorizonDecisionLatencyP99High` — SLO row 2 (cluster path > 200 ms)

Operator-visible log lines (structlog JSON, on `kubectl logs deployment/horizon-rapp`):

```
{"event":"a1.emit","status":"failed","attempt":3,...}        # 3 strikes -> escalate
{"event":"rapp.state","from":"running","to":"degraded",...}  # safe-state entry
{"event":"healthz","ok":false,"checks":["watchdog_stale"]}   # /healthz 503 source
```

## Triage flow (T2 receives a page) — 5-minute clock

Single most critical 4-line block, paste **before** anything else:

```sh
curl -sS -o /dev/null -w "healthz=%{http_code}\n" http://horizon-rapp.${NS}:8081/healthz
curl -sS -o /dev/null -w "readyz=%{http_code}\n"  http://horizon-rapp.${NS}:8081/readyz
curl -sS http://horizon-rapp.${NS}:8081/metrics | grep -E '^horizon_a1_policies(_emitted|_rolled_back)_total'
horizon-ric-sdk state && horizon-ric-sdk audit verify
```

Endpoints are wired in `src/horizon_ric/rapp/health.py:122` (`/healthz`),
`:135` (`/readyz`), `:172` (`/metrics`); the canonical Prometheus metric
names (`horizon_a1_policies_emitted_total`,
`horizon_a1_policies_rolled_back_total`) are declared at
`src/horizon_ric/rapp/health.py:55-64`. Lifecycle state machine is
`src/horizon_ric/rapp/lifecycle.py:42-48` (`RAppState`); a
`/readyz` 503 with `rapp_state="degraded"` is the safe-state, NOT an
intrusion.

```sh
# 1. Confirm scope
kubectl -n $NS get pods,svc,hpa,pdb -l app.kubernetes.io/name=horizon-ric
horizon-ric-sdk sla timeline --minutes 60          # last-hour breach map

# 2. Classify
#    - Single tenant, single policy -> T2 mitigation, 1h SLA
#    - Multi-tenant or chain break  -> T3, 4h SLA, security_incident.md
#    - Platform-wide control plane  -> T3, declare incident
```

Single most critical command — the one operators paste before anything else:

```sh
horizon-ric-sdk state && horizon-ric-sdk audit verify
```

If `audit verify` returns `verified: false`, stop. Pivot to
`security_incident.md` Sev1 step 4.

## Customer-facing communication templates

All templates use plain text — Markdown headers strip in some ticketing
tools and we want zero ambiguity.

### T+15 min — initial response (open ticket / page acknowledgement)

```
Subject: PreceptualAI investigating SLA breach — Ticket <TICKET_ID>

Hi <NOC contact>,

We have your page. PreceptualAI engineer <NAME> is on the bridge.

Symptom we see on our side: <one line — match what they reported>.
Mitigation in flight: <one line — e.g. "rolled emit dry-run, restart in progress">.
Next update: in 45 minutes, on this thread.

If urgent escalation is needed, reply with "ESCALATE" and we will page T3.
```

### T+1 hour — status update

```
Subject: PreceptualAI update — Ticket <TICKET_ID>

What changed in the last hour:
  * <bullet — what we tried>
  * <bullet — what we observed>
  * <bullet — current state>

Service status: <RESTORED | DEGRADED | DOWN>
SLA target this hour: <met | breached, breach <duration>>

Next update: <UTC timestamp + 1h>. We will not let this thread go quiet.
```

### T+4 hours — resolution + post-mortem ETA

```
Subject: PreceptualAI resolved — Ticket <TICKET_ID>

Service restored at <UTC timestamp>. Total user-visible breach: <duration>.

Root cause (preliminary): <one paragraph>.
Customer action required: <none | rotate <X> | restart <Y>>.

Post-mortem document will be shared by <UTC timestamp + 5 business days>.
SLA credit calculation will follow the post-mortem.
```

If the breach exceeded contractual SLA, a credit calculation is mandatory
(see `docs/SLA.md`) and is attached to the post-mortem.

## Customer-side runbook excerpts

When the customer asks "what do we do on our side", paste the relevant block:

### "Our rApp suddenly went DEGRADED"

```
1. Confirm SMO connectivity from your side (Non-RT RIC -> SMO health endpoint).
2. If your SMO is healthy, no action required — DEGRADED is a deliberate
   safe-state. We will restore as soon as we identify the upstream issue.
3. If your SMO is also unhealthy, please page your SMO vendor in parallel.
```

### "We see /healthz 503"

```
1. /healthz returns 503 when the rApp ASGI server is up but lifecycle has
   not reached RUNNING. This is by design — see horizon_ric/rapp/lifecycle.py.
2. Do not auto-restart on 503 alone. Wait for /readyz to flip back, OR
   contact us if the state is stuck > 5 minutes.
```

### "Our policies stopped emitting"

```
1. Confirm A1 dry-run is OFF on your side: env var HORIZON_A1_DRY_RUN must
   not be "true" in your override config.
2. If your dashboards show our state as RUNNING but no policy emit traffic,
   page T2.
```

## Per-tier SLA targets (contractual)

| Tier | Ack    | First mitigation update | Resolution target | Post-mortem |
| ---- | ------ | ----------------------- | ----------------- | ----------- |
| T1   | 5 min  | 15 min                  | 1 h               | n/a         |
| T2   | 15 min | 30 min                  | 4 h               | 5 biz days  |
| T3   | 1 h    | 1 h                     | 8 h               | 5 biz days  |

Miss two consecutive ack targets in the same shift -> handover early to the
next region per `oncall.md`.

## Mitigation (15 min) — step-by-step

1. **If `/healthz` 503 sustained** (`src/horizon_ric/rapp/health.py:122`):
   the liveness registry has a failed signal. Pull the structured body —
   it names the failed check (`watchdog_stale`, `loop_lag`, `planner_idle`)
   per `src/horizon_ric/rapp/health.py:130-133`. Roll the pod **only** if
   `planner_idle` (the planner-task heartbeat) — the other two recover
   when the asyncio loop unwedges.
2. **If A1 emit failure burst** — the breaker is OPEN. Confirm with
   `curl /metrics | grep horizon_a1_policies_rolled_back_total` and
   compare to emitted total. Engage A1 dry-run if the upstream Near-RT
   RIC is the problem (see `deploy/RUNBOOK.md` a1-emit-failure).
3. **If `RAppState=DEGRADED`** (`lifecycle.py:46`): the
   `DegradationController` flipped to safe-state. **Do not auto-restart**
   — it un-degrades on its own once the upstream recovers
   (`src/horizon_ric/runtime/graceful_degradation.py`).
4. **If row 4a (constrained-Orin) burning**: the 99.5% bar leaves
   ≤ 3.6 h / month budget. Three consecutive failed emits visible to
   the operator means ~10 % of the monthly budget burned in one minute
   — page T3.

## Recovery (60 min) — to RUNNING

```sh
# 1. Confirm root cause has cleared (upstream SMO reachable, breaker closing).
kubectl -n $NS logs deploy/$APP --tail=200 | jq 'select(.event=="a1.emit" and .status=="ok")' | tail -3
# 2. /readyz returns 200 + rapp_state=running.
curl -sS http://horizon-rapp.${NS}:8081/readyz | jq '{status, rapp_state, degraded_state}'
# 3. SLA engine reports breach closed.
horizon-ric-sdk sla timeline --minutes 5
# 4. Send T+1h or T+4h customer template.
```

`RUNNING` is the only state that satisfies row 1; `lifecycle.py:44`
sets the value, `health.py:162` veto-checks it on every `/readyz`.

## Postmortem prompts

1. Did detection (Prometheus alert) lead the customer page, or did the
   customer page first? If customer-first, what gap did the Prometheus
   rule miss?
2. Was the `/healthz` 503 source visible in the alert payload, or did
   the operator have to shell into the pod to find it?
3. Which `deploy/SLO.md` row burned, and by how much of the monthly
   budget?
4. Did the breaker / retry layer absorb what it was supposed to, or did
   the `99.9% post-retry` bar (row 4) hide the real upstream behaviour?
5. Was the customer-facing template sent on time (T+15, T+1h, T+4h)? If
   not, what blocked it?

## References

- `RELIABILITY.md` — SLO bar derivation + breaker config
- `deploy/SLO.md` — rows 1, 2, 2a-2d, 4, 4a
- `deploy/RUNBOOK.md` — operator runbook (rapp-down, a1-emit-failure)
- `src/horizon_ric/rapp/health.py:55-95` — Prometheus metric definitions
- `src/horizon_ric/rapp/health.py:122-182` — `/healthz`, `/readyz`, `/metrics`
- `src/horizon_ric/rapp/lifecycle.py:42-48` — `RAppState` enum
- `src/horizon_ric/sla/engine.py` — SLA breach timeline source
- `docs/runbooks/oncall.md`, `docs/runbooks/security_incident.md`,
  `docs/runbooks/cert_rotation.md`

## Honest gaps

- `horizon-ric-sdk sla timeline` (`sdk/python/horizon_ric_sdk/cli.py:131`)
  exists but currently returns last-15-min by default; the `--minutes 60`
  flag works against the API but the Grafana panel pinned in the runbook
  also needs updating.
- The "auto-bump after 30 min unmitigated" rule lives in PagerDuty config
  today, not in the repo. There is no in-tree alertmanager rule that
  enforces it; it relies on the on-call honouring the runbook.
