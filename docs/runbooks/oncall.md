# Runbook — On-call Rotation (First 5 Minutes of a Page)

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


The rules of the chair. Read this on Day 1. Re-read at handover.

## Severity / impact

| Sev | Acks                            | Wakes | Examples                                                    |
| --- | ------------------------------- | ----- | ----------------------------------------------------------- |
| 1   | primary on-call within 5 min    | yes   | `HorizonRAppDown`, audit-chain break, `/healthz` 5xx > 60 s |
| 2   | primary on-call within 15 min   | yes   | `HorizonA1EmitFailureRate` > 5% / 5 min, FL non-converging  |
| 3   | next business hour              | no    | `HorizonRAppDegraded` < 30 min, queue depth 100–1000        |

Severity classification is mechanical: the alert rule sets the page
priority; the on-call does not re-classify on receipt unless they
discover wider impact. A page that escalates to multi-tenant or evidence
chain break **always** promotes to Sev1 and pivots to
`security_incident.md`.

## Detection — where the page comes from

| Source                                                    | Where to look                                                   |
| --------------------------------------------------------- | --------------------------------------------------------------- |
| Prometheus alert (`HorizonRAppDown` / `…Degraded` / etc.) | Grafana `deploy/grafana/dashboards/horizon-counterfactual.json` |
| `/healthz` 503 watcher                                    | `src/horizon_ric/rapp/health.py:122` — body names failed check  |
| Customer page                                             | `customer_escalation.md` triage flow                            |
| Audit-chain break                                         | `EvidenceStore.verify()` returns non-`-1` — Sev1 immediate      |
| FL non-convergence / drift                                | `horizon_drift_fired_total` > 0 — escalate per `customer_escalation.md` |

```sh
NS=${HORIZON_NS:-horizon}
REL=${HORIZON_RELEASE:-horizon-ric}
APP=${REL}-horizon-ric
TENANT_VALUES=deploy/helm/horizon-ric/values.yaml
```

## Triage (5 min) — the moment the pager fires

```sh
# 1. Acknowledge in PagerDuty (do this BEFORE typing anything else).
# 2. Pod liveness, single command — copy/paste verbatim:
kubectl get pods -n horizon
kubectl logs -f deployment/horizon-rapp --tail=200      # Ctrl-C after ~10 s
# 3. Probe the rApp directly:
curl -sS http://horizon-rapp.horizon:8081/healthz | jq .
curl -sS http://horizon-rapp.horizon:8081/readyz  | jq .
# 4. Pull lifecycle state + audit-chain status from the v1 API (dashboard, 8083).
#    (There is no packaged horizon-ric-sdk CLI; call the HTTP endpoints directly.)
curl -sS http://horizon-rapp.horizon:8083/v1/state | jq '.state'
curl -sS http://horizon-rapp.horizon:8083/v1/audit/verify | jq   # {ok:true, first_bad_index:-1} = intact
```

Open the Grafana board:
`deploy/grafana/dashboards/horizon-counterfactual.json` — the panels in
order are (a) `horizon_rapp_state` per state value, (b)
`horizon_a1_policies_emitted_total` rate, (c) audit verify latency
histogram. Source-of-truth metric definitions:
`src/horizon_ric/rapp/health.py:45-95`.

## Page-source → runbook decision tree

```
Page received
│
├─ Source = HorizonRAppDown / /healthz 503 sustained > 60 s
│   └─ deploy/RUNBOOK.md §"rapp-down" + this runbook §"Mitigation"
│
├─ Source = HorizonRAppDegraded
│   └─ ack only if > 30 min; deploy/RUNBOOK.md §"rapp-degraded"
│
├─ Source = HorizonA1EmitFailureRate (cluster) or …Orin (constrained)
│   └─ customer_escalation.md  (SLA breach is contractual)
│
├─ Source = HorizonAuditChainBroken / EvidenceStore.verify() != -1
│   └─ security_incident.md Sev1 step 4 (do NOT investigate; isolate first)
│
├─ Source = horizon_cert_expiry_days < 7
│   └─ cert_rotation.md (the cert that fired is named in the alert label)
│
└─ Source = unrecognised
    └─ default to customer_escalation.md T2; promote on impact
```

## Mitigation (15 min) — first stabilising action

The single goal of the 15-minute window is to put the rApp into a
**known** state — RUNNING, DEGRADED (safe), or quarantined for
forensics. Pick **one** of:

1. **Roll the pod** if `/healthz` reports `watchdog_stale` (the
   loop/watchdog heartbeat is stale), per
   `src/horizon_ric/rapp/health.py:130-133`:

   ```sh
   kubectl -n horizon rollout restart deployment/horizon-rapp
   kubectl -n horizon rollout status  deployment/horizon-rapp --timeout=2m
   ```
2. **Engage A1 dry-run** if upstream Near-RT RIC is the failure source —
   `deploy/RUNBOOK.md` §"a1-emit-failure".
3. **Quarantine** if any audit-chain check returned False —
   `security_incident.md` Sev1 steps 2–3.

Do NOT roll on `loop_lag_high` alone — transient loop lag normally
clears within `DEFAULT_WATCHDOG_INTERVAL_S`
(`src/horizon_ric/runtime/watchdog.py`).

## Recovery (60 min) — back to RUNNING

```sh
kubectl get pods -n horizon                        # all replicas Running 1/1
curl -sS http://horizon-rapp.horizon:8081/readyz   # status=ok rapp_state=running
curl -sS http://horizon-rapp.horizon:8083/v1/audit/verify | jq   # {ok:true, first_bad_index:-1}
curl -sS "http://horizon-rapp.horizon:8083/v1/sla/timeline?from=$(date -u -d '30 min ago' +%Y-%m-%dT%H:%M:%SZ)&to=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | jq   # no open breach rows
```

If `rapp_state=running` but `/readyz` is still 503, one of (a) connector
RUNNING, (b) evidence store writable, (c) R1 registered — see
`src/horizon_ric/rapp/health.py:135-170`.

## Communication template (status page / Slack #incidents)

```
[<UTC>] Horizon-RIC <SEV> — <ALERT_NAME>
Owner:        <on-call name>
Started:      <UTC>
Customer-visible: <YES | NO>
Symptom:      <one line — what users / dashboards see>
Status:       <triage | mitigation | recovery | resolved>
Next update:  <UTC + 30 min>
Page source:  <HorizonRAppDown | … | customer ticket>
Runbook:      docs/runbooks/<name>.md
```

Post once per status transition, plus every 30 min while open.

## Postmortem prompts

1. Was the page actionable on its own (alert label + dashboard link), or
   did the on-call have to dig into JSON logs to know what to do?
2. Did the page source map to exactly one runbook in the decision tree?
   If multiple matched, what was missing?
3. Was the 5-minute triage block enough to classify, or was a wider net
   required?
4. Did `horizon_rapp_state` flip in the order the lifecycle expects
   (`src/horizon_ric/rapp/lifecycle.py:42-48`), or did we observe an
   illegal transition?
5. Did anyone get woken who shouldn't have? (Sev3 noise is the silent
   tax — count it.)

## References

- `deploy/SLO.md` — fault-tree, SLO derivation; every alert maps back to one row
- `deploy/RUNBOOK.md` — change-management, rapp-down / rapp-degraded /
  a1-emit-failure / queue-depth / healthz-503
- `deploy/grafana/dashboards/horizon-counterfactual.json` — primary
  on-call dashboard
- `src/horizon_ric/rapp/health.py:122` — `/healthz`
- `src/horizon_ric/rapp/health.py:135` — `/readyz`
- `src/horizon_ric/rapp/health.py:172` — `/metrics`
- `src/horizon_ric/rapp/lifecycle.py:42` — `RAppState`
- Sister runbooks: `customer_escalation.md`, `security_incident.md`,
  `cert_rotation.md`

## Schedule

24 / 7 follow-the-sun, 3 regions:

| Region | Hours (UTC)   | Local handover (typical)         |
| ------ | ------------- | -------------------------------- |
| APAC   | 22:00 – 06:00 | 06:00 UTC -> EU on-call          |
| EU     | 06:00 – 14:00 | 14:00 UTC -> US on-call          |
| US     | 14:00 – 22:00 | 22:00 UTC -> APAC on-call        |

Primary + secondary per region. Secondary covers if primary acks fail twice
in a row (PagerDuty escalation policy `horizon-ric` step 2).

## Handover ritual (15 min, every region boundary)

1. Outgoing engineer dumps to the shared bridge:

   ```sh
   curl -sS http://horizon-rapp.horizon:8083/v1/state | jq '.state'
   curl -sS http://horizon-rapp.horizon:8083/v1/audit/verify | jq
   curl -sS "http://horizon-rapp.horizon:8083/v1/sla/timeline?from=$(date -u -d '480 min ago' +%Y-%m-%dT%H:%M:%SZ)&to=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | jq   # last shift
   kubectl -n $NS get events --sort-by='.lastTimestamp' --field-selector type!=Normal | tail -20
   ```

2. Walk through any open incidents, paged tickets, deferred tickets.
3. Hand the pager — confirm in the bridge: "I have the pager,
   <DATETIME UTC>." PagerDuty schedule reflects this automatically;
   verbal confirmation is for the bridge log.
4. If anything was deferred from the previous shift, the incoming on-call
   acknowledges it and owns it.

## The four daily checks

Run at the top of shift (and at 4-hour mark on a long shift). Paste output
into the on-call diary for the day.

### 1. All pods running

```sh
kubectl -n $NS get pods -l app.kubernetes.io/name=horizon-ric
# expected (one row per replica):
# NAME                            READY   STATUS    RESTARTS   AGE
# horizon-ric-horizon-ric-<hash>  1/1     Running   0          7d
```

Anything but `Running 1/1` means triage per `deploy/RUNBOOK.md` rapp-down.

### 2. Public readiness

```sh
for T in $(yq '.tenants[].id' $TENANT_VALUES); do
  curl -sS -o /dev/null -w "$T %{http_code}\n" https://horizon.${T}/readyz
done
# expected: every line ends in " 200"
```

A 503 -> `deploy/RUNBOOK.md` healthz-503. A connection refusal ->
ingress / DNS triage, not a Horizon-RIC issue.

### 3. API round-trip

There is no packaged `horizon-ric-sdk` CLI in the repo today (see "Honest
gaps" below). Exercise the same intent against the `/v1` HTTP API
(`src/horizon_ric/rapp/api_v1.py`, served on the dashboard port 8083):

```sh
curl -fsS http://horizon-rapp.horizon:8083/v1/state > /dev/null \
  && curl -fsS "http://horizon-rapp.horizon:8083/v1/policies?limit=1" > /dev/null \
  && curl -fsS http://horizon-rapp.horizon:8083/v1/audit/verify > /dev/null
echo "exit=$?"
# expected: exit=0
```

A non-zero exit means either the API is unhealthy (T2) or the operator
JWT is expired (rotate per `cert_rotation.md` §3).

### 4. Audit chain across all tenants

```sh
for T in $(yq '.tenants[].id' $TENANT_VALUES); do
  kubectl -n $NS exec deploy/$APP -- python -c "
from horizon_ric.evidence.store import SqliteEvidenceStore
from horizon_ric.security.tenant import TenantScope
import os
with TenantScope('$T'):
    s = SqliteEvidenceStore('sqlite:///'+os.environ.get('HORIZON_EVIDENCE_DB','/var/lib/horizon/evidence.db'))
    print('$T verify=', s.verify())   # -1 == chain intact
"
done
# expected: every line "<tenant> verify= -1"
```

Single most critical command of the shift — `EvidenceStore.verify()`
returns `-1` when the chain is intact and the index of the first broken
record otherwise. If any line raises an exception or prints a non-`-1`
value, declare Sev1 and pivot to `security_incident.md` step 4.

## Pager rules

Auto-page (P1, wakes you):

- `HorizonRAppDown`
- `/healthz` 503 sustained > 60 s
- `EvidenceStore.verify()` failure
- Any tenant chain break detected by the daily check
- `HorizonA1EmitFailureRate > 5%` for 5 min

Queue (P2, business-hours triage):

- `HorizonRAppDegraded` (DEGRADED is the safe-state, do not wake on it
  unless > 30 min)
- `HorizonA1EmitFailureRate` between 1% and 5%
- `HorizonQueueDepth` between 100 and 1000
- Any cert-expiry warning > 7 days out (per `cert_rotation.md` triggers)

Anything outside this list -> ticket, not page.

## Post-incident: post-mortem template

Blameless. 5-whys structure. Use the standard action-item table format so
action items get tracked alongside roadmap gaps:

```
# Post-mortem — <INCIDENT_ID> — <ONE-LINE SYMPTOM>

*Date: <UTC>. Authors: <on-call> + <reviewer>. Severity: <Sev1|Sev2|Sev3>.*

## Timeline (UTC)
- HH:MM   Detection: <how>
- HH:MM   Page acked
- HH:MM   First mitigation
- HH:MM   Service restored
- HH:MM   All-clear

## What happened (3-5 sentences)

## Why — five whys
1. ...
2. ...
3. ...
4. ...
5. ...

## What worked
- ...

## What didn't
- ...

## Action items (table)

| # | Action | Owner | Due | Status |
|---|--------|-------|-----|--------|
| 1 | <action>           | <owner>      | <date>  | OPEN |
| 2 | <action>           | <owner>      | <date>  | OPEN |
```

File the post-mortem in `docs/post_mortems/<INCIDENT_ID>.md` and link it
from the roadmap-gap tracker if any action item closes a roadmap gap.

## Burnout rule

No engineer holds the primary pager for more than 14 consecutive days.
Hard limit. If the schedule shows any individual exceeding 14 days
(holiday cover, sickness backfill, etc), the engineering lead must
swap them out before day 15. PagerDuty schedule export is checked
weekly by the lead; there is no in-repo enforcement.

## Honest gaps

- There is no packaged `horizon-ric-sdk` CLI and no SDK health-wrapper
  script in the repo. Daily check #3 calls the `/v1` HTTP API
  (`src/horizon_ric/rapp/api_v1.py`) directly via `curl`. A thin CLI/health
  wrapper over `/v1/state`, `/v1/policies`, and `/v1/audit/verify` that
  exits non-zero on any failure is a small follow-up.
- The 4-tenant loop in checks #2 and #4 reads tenants from
  `deploy/helm/horizon-ric/values.yaml`; a registry-backed list would be
  more correct but the values file is the source of truth today.
- Burnout rule has no tooling enforcement. Lead checks PagerDuty
  manually.
