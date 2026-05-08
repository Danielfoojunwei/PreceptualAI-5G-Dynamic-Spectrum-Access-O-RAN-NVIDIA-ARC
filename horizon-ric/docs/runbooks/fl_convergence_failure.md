# Runbook — Federated-Learning Convergence Failure

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


For the on-call when a federated round's loss is not decreasing across the
client cohort, or a single client's update has anomalous magnitude. Pairs
with `customer_escalation.md` for tenant communication.

```sh
NS=${HORIZON_NS:-horizon}
REL=${HORIZON_RELEASE:-horizon-ric}
APP=${REL}-horizon-ric
ROUND_LOG=/var/lib/horizon/fl/round.log
GLOBAL_DIR=/var/lib/horizon/fl/global
SNAP_DIR=/var/lib/horizon/fl/snapshots
```

## Severity / impact

| Sev | Trigger                                                                                            | Impact                                            |
| --- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 2   | `horizon_drift_fired_total` > 0 for ≥ 3 consecutive rounds (`src/horizon_ric/continual/drift_detector.py:16,238-289`) | global model regresses; tenants stay on local fine-tunes |
| 2   | FedProx aggregated loss diverges across the cohort (`aggregate_fedprox` in `src/horizon_ric/federated/aggregator.py:110-121`) | next round will compound — pause FL              |
| 3   | Single divergent client > 5× cohort `grad_norm` median for 2 rounds                                | one tenant; allowlist mitigation                  |

## Detection

Prometheus alerts (registered via the private collector in
`src/horizon_ric/continual/drift_detector.py:238-289`):

- `HorizonFLNonConverging` — 5 consecutive rounds with non-decreasing loss
- `HorizonDriftFired` — `horizon_drift_fired_total` increasing
  (label `detector` = `ks` or `page_hinkley`)
- `horizon_drift_pvalue` low watermark < 1e-3 sustained

Log lines to look for:

```
{"event":"fl.round","status":"diverged","round":N,"loss":<rising>,...}
{"event":"drift.fired","detector":"ks","pvalue":1.2e-04,...}
{"event":"fl.client.train","client_id":"<X>","grad_norm":<>5*median,...}
```

## Trigger

Any of:

- **`horizon_drift_fired_total > 0` for ≥ 3 consecutive rounds** (drift
  detectors fire on telemetry distribution shift; see
  `src/horizon_ric/continual/drift_detector.py:1-23`).
- **FedProx loss diverges**: aggregated loss has not decreased for **5 or
  more consecutive rounds** (Prometheus alert `HorizonFLNonConverging`).
- Operator inspection of `$ROUND_LOG`: monotonically rising or plateaued
  loss series.

Implementation reference: `src/horizon_ric/federated/aggregator.py`
(`aggregate_fedavg:82`, `aggregate_fedprox:110`, `FedProx:145`,
`default_aggregator:176`, `DEFAULT_FEDPROX_MU:171`),
`src/horizon_ric/federated/secure_aggregation.py` (Shamir share layer
over FedAvg, `ShamirSecretSharing:84`, threshold reconstruction
`reconstruct:132`), and `src/horizon_ric/federated/sparsifier.py`
(`top_k_sparsify`, `sign_sgd_compress`, `quantize_int8`).

## Diagnostic flow

### 1. Inspect per-client loss

```sh
kubectl -n $NS exec deploy/$APP -- tail -n 200 $ROUND_LOG | jq \
  'select(.event=="fl.round") | {round, client_id, n, loss, grad_norm}'
```

Expected one record per client per round. Look for:

- A single client with `grad_norm` an order of magnitude above the cohort
  median -> divergent client.
- All clients moving in the same wrong direction -> learning-rate or
  global-optimizer issue, **not** a per-client problem.
- One client dominating sample weighting (`n >> sum(others)/k`) -> FedAvg
  is just averaging that client's overfit.

### 2. Identify the divergent client

```sh
kubectl -n $NS exec deploy/$APP -- python -c "
import json, sys
from collections import defaultdict
import statistics
rounds = defaultdict(list)
for line in open('$ROUND_LOG'):
    r = json.loads(line)
    if r.get('event') == 'fl.round':
        rounds[r['round']].append(r)
last = max(rounds)
gn = [(c['client_id'], c['grad_norm']) for c in rounds[last]]
median = statistics.median(g for _, g in gn)
for cid, g in sorted(gn, key=lambda x: -x[1]):
    print(f'{cid}\\t{g:.3f}\\t{(g/median):.2f}x median')
"
```

A client at >5x the cohort median in two consecutive rounds is the suspect.

### 3. Robust-aggregation override

`federated/aggregator.py` exports `FedAvg` and `FedProx` only — there is no
Krum / median-of-means / trimmed-mean implementation today (see "Honest
gaps"). Mitigations available **right now**:

- Drop the divergent client from the next round (server-side allowlist):

  ```sh
  kubectl -n $NS set env deploy/$APP \
    HORIZON_FL_DENY_CLIENTS=<divergent-client-id>
  kubectl -n $NS rollout status deploy/$APP --timeout=60s
  ```

- Switch from FedAvg to FedProx with `μ=0.01` to bound per-round drift:

  ```sh
  kubectl -n $NS set env deploy/$APP \
    HORIZON_FL_AGGREGATOR=fedprox HORIZON_FL_FEDPROX_MU=0.01
  ```

  `aggregate_fedprox` in `src/horizon_ric/federated/aggregator.py` is
  already wired; the proximal term lives client-side, and the server-side
  weighted mean is identical to FedAvg.

### 4. Check for label flip / data poisoning

Top-k sparsification (`top_k_sparsify` in
`src/horizon_ric/federated/sparsifier.py`) keeps only the largest |Δw|
coordinates (default 1%). It **masks but does not detect** a label flip —
a poisoner can keep their flipped gradient inside the top-k mass exactly
because the magnitude is large.

Manual checks (no automated detector ships today):

```sh
# (a) Compare the divergent client's per-class loss to the cohort.
kubectl -n $NS exec deploy/$APP -- jq \
  'select(.event=="fl.client.train") | {round, client_id, per_class_loss}' \
  $ROUND_LOG | head -50

# (b) Side-channel: ask the tenant to confirm their training data manifest
#     hash matches what they sent in their last attestation. The manifest
#     hash is recorded in the DecisionRecord audit chain by the rApp.
horizon-ric-sdk audit verify
```

If the per-class loss profile is wildly different from the cohort and the
manifest hash has changed without notice, treat as Sev2 per
`security_incident.md`.

### 5. Roll back to last good global model

```sh
LAST_GOOD=$(ls -1t $SNAP_DIR/global-*.pt | head -1)
kubectl -n $NS exec deploy/$APP -- cp $LAST_GOOD $GLOBAL_DIR/global.pt
kubectl -n $NS rollout restart deploy/$APP
```

Per-tenant local snapshots are preserved on the client devices; only the
**global** state regresses. Tenants do not lose their personalised
fine-tunes.

## Triage (5 min)

```sh
# 1. Confirm the trigger fired:
curl -sS http://horizon-rapp.${NS}:8081/metrics | \
  grep -E '^horizon_drift_(fired_total|pvalue)'
# 2. Pull the last 3 rounds of FL events:
kubectl -n $NS logs deploy/$APP --tail=2000 | \
  jq 'select(.event=="fl.round" or .event=="drift.fired")' | tail -50
# 3. Lifecycle state — refusing to RUNNING means FL is already paused:
horizon-ric-sdk state
```

## Mitigation (15 min) — pause FL + fall back

Goal: prevent the next round from compounding the divergence; keep
tenant traffic on the **last good global model** while you investigate.

```sh
# 1. Pause the FL round scheduler (do not kill mid-round — let in-flight
#    client uploads complete; the aggregator just won't run).
kubectl -n $NS set env deploy/$APP HORIZON_FL_PAUSE=true
# 2. Fall back to the last good global model:
LAST_GOOD=$(ls -1t $SNAP_DIR/global-*.pt | head -1)
kubectl -n $NS exec deploy/$APP -- cp $LAST_GOOD $GLOBAL_DIR/global.pt
# 3. Bump the aggregator's μ if you elect to resume cautiously:
kubectl -n $NS set env deploy/$APP \
  HORIZON_FL_AGGREGATOR=fedprox HORIZON_FL_FEDPROX_MU=0.01
```

Reference points: pause flag is read by the FL coordinator that drives
`default_aggregator()` at `src/horizon_ric/federated/aggregator.py:176`;
`DEFAULT_FEDPROX_MU=0.01` is the system default
(`src/horizon_ric/federated/aggregator.py:171`). The Shamir-shared
secure-aggregation path
(`src/horizon_ric/federated/secure_aggregation.py:84,132`) is **not**
bypassed by this mitigation — clients still upload shares; only the
reconstruction step is gated on the pause flag.

## Mitigation summary (single most critical command)

The fastest knob — bound divergence without dropping any client:

```sh
kubectl -n $NS set env deploy/$APP \
  HORIZON_FL_AGGREGATOR=fedprox HORIZON_FL_FEDPROX_MU=0.01
```

Watch the next two rounds in `$ROUND_LOG`. If aggregated loss starts
decreasing again, leave FedProx engaged for the rest of the day and file
a ticket to investigate the divergent client offline. If loss still flat,
escalate per "Diagnostic flow" §4.

## Recovery (60 min) — resume FL with bumped μ + smaller cohort

```sh
# 1. Bump μ to 0.05 (5× the default). Tighter proximal term, smaller
#    per-round step, slower convergence but bounded drift on heterogeneous
#    clients (Li et al., MLSys 2020).
kubectl -n $NS set env deploy/$APP HORIZON_FL_FEDPROX_MU=0.05
# 2. Halve the participation rate so any single divergent client
#    contributes less per round.
kubectl -n $NS set env deploy/$APP HORIZON_FL_PARTICIPATION=0.5
# 3. Un-pause:
kubectl -n $NS set env deploy/$APP HORIZON_FL_PAUSE=false
# 4. Watch 3 rounds. If horizon_drift_fired_total stays flat AND loss
#    decreases monotonically, restore the defaults at end of shift:
#       HORIZON_FL_FEDPROX_MU unset (-> 0.01), HORIZON_FL_PARTICIPATION=1.0
kubectl -n $NS logs deploy/$APP --tail=400 -f | \
  jq 'select(.event=="fl.round")|{round, loss, n_clients}'
```

Recovery exit gate: `horizon_drift_fired_total` does **not** advance for
3 consecutive rounds AND `/readyz` reports `rapp_state=running`. The
`FedProx(mu=0.05)` configuration is honoured by
`src/horizon_ric/federated/aggregator.py:145-156`; the validation
`_validate_updates` (`aggregator.py:65-80`) gates malformed shares
before they reach the weighted mean.

## Communication template

```
[<UTC>] PreceptualAI FL pause — round <N>
Tenants:        <list>
Trigger:        horizon_drift_fired_total advanced 3 consecutive rounds
                 OR FedProx loss diverged (round <N-2..N>)
Action:         FL paused; global model rolled to <snapshot path>
Tenant impact:  none — local fine-tunes intact, training data on-device
Resume plan:    μ=0.05, participation=0.5, watch 3 rounds
Next update:    <UTC + 1h>
Runbook:        docs/runbooks/fl_convergence_failure.md
```

## Postmortem prompts

1. Which detector fired first — K-S or Page-Hinkley
   (`drift_detector.py:56,238`)? Did the slower detector add value, or
   was it noise we could prune?
2. Was the divergent client actually divergent, or was the global
   learning rate the real culprit?
3. Did the Shamir secure-aggregation path
   (`secure_aggregation.py:84-132`) mask any signal we'd have caught
   pre-quantisation?
4. Was bumping μ to 0.05 enough, or did we need to drop participation
   too? Could we have tightened μ earlier?
5. Should `HorizonFLNonConverging` and `HorizonDriftFired` page the
   on-call (Sev2) or queue (Sev3)? What does the on-call cost vs. the
   compounding-divergence cost?

## References

- `RELIABILITY.md` — federated-learning posture
- `deploy/SLO.md` — does NOT cover model-quality SLO (excluded)
- `src/horizon_ric/federated/aggregator.py:65-184` — FedAvg, FedProx,
  defaults
- `src/horizon_ric/federated/secure_aggregation.py:84-316` — Shamir
  share + reconstruct
- `src/horizon_ric/continual/drift_detector.py:1-23` (metrics names),
  `:56-235` (detectors), `:238-289` (registry)
- `docs/runbooks/customer_escalation.md` — tenant comms tempo
- `docs/runbooks/oncall.md` — on-call rotation

## Communication to the affected tenant

Use the T2 template in `customer_escalation.md`. Specific language for
the FL case:

```
Your tenant data has not been compromised — federated learning kept your
training data on-device throughout. The global model is temporarily
behind your local model because we rolled the global state back to the
last verified-converging snapshot. Your local fine-tunes are intact and
continue to serve traffic. We will resume global aggregation as soon as
the divergent client is identified and addressed; we will share the
post-mortem within 5 business days.
```

This is true today: the wire format is weights only (see the
`federated/__init__.py` docstring and `aggregator.py` module docstring),
not telemetry.

## Honest gaps

- **Krum / median-of-means / trimmed-mean are not implemented.** The
  runbook recommends FedProx + client allowlist as the available
  mitigation. `GAPS_TO_PILOT.md` row 15 already tracks this as
  SCAFFOLD-level FL maturity. Action: add `aggregate_krum`,
  `aggregate_median`, `aggregate_trimmed_mean` to
  `src/horizon_ric/federated/aggregator.py`.
- No DP accountant ships. If a regulator asks about ε-bound after a
  round, the answer today is "not bounded by the platform" — that is
  also tracked in row 15 of `GAPS_TO_PILOT.md`.
- `HORIZON_FL_DENY_CLIENTS` and `HORIZON_FL_AGGREGATOR` env-var hooks
  are referenced as the operator switches; the wiring exists in the
  rApp config layer but the names are operational conventions, not
  defined as constants in the source — if you change them, also update
  this runbook and `deploy/helm/horizon-ric/values.yaml` env block.
- The Prometheus alert `HorizonFLNonConverging` is referenced in
  `deploy/prometheus/` rules but is not yet emitted by the rApp's
  observability layer end-to-end. Manual monitoring of `$ROUND_LOG` is
  the fallback today.
