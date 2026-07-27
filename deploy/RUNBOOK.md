# Horizon-RIC On-Call Runbook

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`../README.md`](../README.md) for the 49-section deep dive.*


For operators paged on a Horizon-RIC rApp alert. Each scenario below has:

1. **What you'll see** — the symptom in Grafana / kubectl / journalctl.
2. **Diagnose** — exact commands, in order.
3. **Mitigate** — fastest known restore-to-service path.
4. **Root-cause** — where to dig once service is restored.

Conventions: `$NS` is the rApp namespace (default `horizon`); `$REL` is the
Helm release name (default `horizon-ric`).

```sh
NS=horizon
REL=horizon-ric
APP=$REL-horizon-ric
```

---

## rapp-down — `HorizonRAppDown`

**You'll see:** Grafana → "rApp State (RUNNING)" panel red. Prometheus → no
samples for `horizon_rapp_state` for ≥60 s. The rApp `/healthz` is unreachable.

### Diagnose

```sh
# Pod state — restarts, age, image.
kubectl -n $NS get pods -l app.kubernetes.io/name=horizon-ric -o wide
kubectl -n $NS describe pod -l app.kubernetes.io/name=horizon-ric | head -120

# Recent logs (last 200 lines, JSON via structlog).
kubectl -n $NS logs -l app.kubernetes.io/name=horizon-ric --tail=200 | jq .

# Probe the health server directly. Port story: in pipeline mode (the chart
# default, pipeline.enabled=true) /healthz, /readyz and /metrics are ALL on
# the metrics port 8082; in bare-lifecycle mode they are all on 8081.
kubectl -n $NS port-forward svc/$APP 8082:8082 &
curl -fsS http://localhost:8082/healthz
curl -fsS http://localhost:8082/readyz
# Bare-lifecycle mode (pipeline.enabled=false) instead:
#   kubectl -n $NS port-forward svc/$APP 8081:8081 &
#   curl -fsS http://localhost:8081/healthz
```

On bare-metal:

```sh
sudo systemctl status horizon-rapp
sudo journalctl -u horizon-rapp -n 200 --no-pager
sudo journalctl -u horizon-rapp --since "5 min ago" | grep -iE 'error|fatal|degraded'
```

### Mitigate

```sh
# K8s — force a rollout (PDB allows minAvailable=1, so this is safe).
kubectl -n $NS rollout restart deploy/$APP
kubectl -n $NS rollout status deploy/$APP --timeout=2m

# Bare metal — restart the unit.
sudo systemctl restart horizon-rapp
```

### Root-cause

* OOMKill: `kubectl describe` shows `Last State: Terminated, Reason: OOMKilled`. → Bump `resources.limits.memory` in values.yaml.
* CrashLoopBackOff with `R1AdapterError`: check `HORIZON_SMO_URL` reachable from the pod, NetworkPolicy permits it.
* Image pull error: confirm registry creds in `image.pullSecrets`.

---

## rapp-degraded — `HorizonRAppDegraded`

**You'll see:** Grafana → state gauge shows DEGRADED for >5 min. The pod is
healthy but A1 / R1 calls are failing.

### Diagnose

```sh
kubectl -n $NS logs -l app.kubernetes.io/name=horizon-ric --tail=500 \
  | jq 'select(.event | test("r1|a1|degraded"; "i"))'

# Check egress to SMO.
kubectl -n $NS exec deploy/$APP -- \
  python -c "import httpx,os; print(httpx.get(os.environ['HORIZON_SMO_URL']+'/health',timeout=5).status_code)"

# Check A1.
kubectl -n $NS exec deploy/$APP -- \
  python -c "import httpx,os; print(httpx.get(os.environ['HORIZON_NEAR_RT_RIC_URL']+'/health',timeout=5).status_code)"
```

### Mitigate

* If SMO is the issue, do **not** restart the rApp — DEGRADED is the correct posture; wait for SMO to recover. Confirm the Non-RT RIC team sees the same.
* If only A1 emit is broken, flip the dry-run kill switch:

  ```sh
  kubectl -n $NS set env deploy/$APP HORIZON_A1_DRY_RUN=true
  ```

  With `HORIZON_A1_DRY_RUN` truthy the pipeline still runs the planner,
  Shield and guard chain per event, but **skips the A1 PUT**: each
  suppressed decision is appended to the audit chain with
  `chosen_action.dry_run: true` and counted in the
  `horizon_dry_run_decisions_total` metric (the daemon report counts it
  under `dry_run`, not `accepted`). Unset the variable (and restart) to
  resume live emits.

### Root-cause

* TLS handshake failures → certs rotated by IDP; check `secret/$APP-credentials`.
* Sustained 5xx from Near-RT RIC → see *a1-emit-failure* below.

---

## a1-emit-failure — `HorizonA1EmitFailureRate`

**You'll see:** rolled-back / emitted ratio > 1% over 5 min.

### Diagnose

```sh
# Per-policy-type rollback rate.
curl -s http://localhost:8082/metrics \
  | grep -E 'horizon_a1_policies_(emitted|rolled_back)_total'

# Recent A1 emit logs with status codes.
kubectl -n $NS logs deploy/$APP --tail=500 | jq 'select(.event=="a1.emit")'
```

### Mitigate

```sh
# Pause emits, keep observing. Decisions still run the Shield + guards and
# land in the audit chain flagged `dry_run: true`; the A1 PUT is skipped and
# horizon_dry_run_decisions_total counts each suppressed emit.
kubectl -n $NS set env deploy/$APP HORIZON_A1_DRY_RUN=true
```

### Root-cause

* 4xx from Near-RT RIC → policy schema drift; verify `policy_types` in the A1 adapter match the RIC's registered types.
* 5xx → Near-RT RIC overloaded; coordinate with the RIC team.

---

## healthz-503 — `/healthz returns 503`

`/healthz` returns 200 only while the process is alive. A 503 indicates the
ASGI server is up but the lifecycle never reached RUNNING (it's checking
`/readyz` semantics).

### Diagnose

```sh
kubectl -n $NS exec deploy/$APP -- \
  python -c "from horizon_ric.rapp.lifecycle import RAppState; print('rApp lifecycle states:', [s.value for s in RAppState])"

# Probe both:
curl -i http://localhost:8081/healthz
curl -i http://localhost:8081/readyz
```

### Mitigate

* Same as *rapp-down* — restart and re-trigger boot.

---

## audit-chain-broken — `HorizonAuditChainBroken`

**Critical, regulatory-impact.** The on-disk audit chain has a hash mismatch.

### Diagnose

```sh
# Snapshot the audit file before doing anything.
kubectl -n $NS cp $POD:/var/lib/horizon/audit.jsonl /tmp/audit-$(date +%s).jsonl
sha256sum /tmp/audit-*.jsonl

# Run verify offline.
python - <<'PY'
from horizon_ric.evidence.store import JsonlEvidenceStore
s = JsonlEvidenceStore("/tmp/audit-XXXXX.jsonl")
print("first broken index:", s.verify())
PY
```

### Mitigate

1. **Do not** delete the audit file. It is evidence.
2. Quarantine the pod:

   ```sh
   kubectl -n $NS label pod $POD horizon.preceptual.ai/quarantine=true
   kubectl -n $NS scale deploy/$APP --replicas=0
   ```

3. Open an incident in the compliance tracker. The compliance lead must sign off before resuming traffic.

### Root-cause

* Disk corruption (`dmesg | grep -i error`).
* Concurrent writers (multiple replicas writing to the same PVC). The chart
  enforces `ReadWriteOnce` so this should be impossible — investigate
  storage class.
* Tamper. Treat as a security incident.

---

## queue-backpressure — `HorizonTelemetryQueueDepthGrowing`

Queue depth >1000 sustained for 5 min: planner is not draining.

### Diagnose

```sh
# Latency may also be elevated.
curl -s http://localhost:8082/metrics | grep horizon_decision_latency_seconds

# Encoder / planner CPU saturation?
kubectl -n $NS top pod -l app.kubernetes.io/name=horizon-ric
```

### Mitigate

```sh
# Scale up — HPA target is 80% CPU; bumping minReplicas is faster.
kubectl -n $NS scale deploy/$APP --replicas=3
```

### Root-cause

* Hot-path regression in the planner; bisect against the last green deploy.
* Upstream burst (operator override flood, weather event); confirm via the
  source connector's metrics.

---

## Full-pipeline mode and A1 dialect/auth configuration

The chart runs the container in full-pipeline mode by default
(`pipeline.enabled=true`): the pod gets
`args: ["--source-config", "/etc/horizon/source.yaml"]`, where `source.yaml`
is the `<fullname>-source` ConfigMap rendered verbatim from
`pipeline.sourceConfig` (telemetry source → planner → Shield → A1 emit).
Set `pipeline.enabled=false` to fall back to the bare lifecycle daemon
(register with the SMO and idle) — the pre-wave behavior.

### Environment contract → Helm values

All knobs are env vars read by the daemon at boot (`_config_from_env()` in
`horizon_ric.rapp.lifecycle`). Non-secret vars land in the `-config`
ConfigMap; secrets in the `-credentials` Secret. Empty values are *omitted*
from the ConfigMap so the adapter's defaults apply.

| Env var | Helm value | Default / notes |
| --- | --- | --- |
| `HORIZON_A1_DIALECT` | `config.a1Dialect` | `osc` — OSC NONRTRIC PMS, the reference SMO path. Also: `legacy`, `osc_a1`, `eiap`, `mantaray`. |
| `HORIZON_A1_RIC_ID` | `config.a1RicId` | empty → per-dialect default. Per-dialect overrides `HORIZON_A1_OSC_RIC_ID` / `HORIZON_A1_EIAP_RIC_ID` / `HORIZON_A1_MANTARAY_RIC_ID` can be injected via `extraVolumes`-style env if you run mixed SMOs. |
| `HORIZON_A1_SERVICE_ID` | `config.a1ServiceId` | empty → adapter default. |
| `HORIZON_A1_TIMEOUT_S` | `config.a1TimeoutS` | `10` |
| `A1_CLIENT_TOKEN` | `secret.data.a1ClientToken` | static bearer; see flow note below. |
| `HORIZON_A1_OAUTH_TOKEN_URL` | `config.auth.oauthTokenUrl` | OAuth2 client-credentials token endpoint. |
| `HORIZON_A1_OAUTH_CLIENT_ID` | `config.auth.oauthClientId` | |
| `HORIZON_A1_OAUTH_CLIENT_SECRET` | `secret.data.a1OauthClientSecret` | secret — never in the ConfigMap. |
| `HORIZON_A1_OAUTH_SCOPE` | `config.auth.oauthScope` | |
| `HORIZON_A1_CLIENT_CERT_PATH` | `config.auth.clientCertPath` | mTLS client cert; mount the material via `extraVolumes`/`extraVolumeMounts`. |
| `HORIZON_A1_CLIENT_KEY_PATH` | `config.auth.clientKeyPath` | |
| `HORIZON_A1_CA_BUNDLE_PATH` | `config.auth.caBundlePath` | |
| `HORIZON_A1_VERIFY_TLS` | `config.auth.verifyTls` | `"true"`/`"false"`; empty → adapter default. |
| `HORIZON_PRODUCTION_MODE` | `config.auth.productionMode` | `"true"` hard-fails insecure A1 config. |
| `HORIZON_SHIELD_BAND_LO_HZ` | `pipeline.shield.bandLoHz` | `3.40e9` |
| `HORIZON_SHIELD_BAND_HI_HZ` | `pipeline.shield.bandHiHz` | `3.50e9` |
| `HORIZON_SHIELD_MAX_EIRP_DBM` | `pipeline.shield.maxEirpDbm` | `33.0` |
| `HORIZON_DECISION_BUDGET_MS` | `pipeline.decisionBudgetMs` | `1000` |
| `HORIZON_A1_STATUS_POLL_ATTEMPTS` | `pipeline.statusPoll.attempts` | `3` |
| `HORIZON_A1_STATUS_POLL_INTERVAL_S` | `pipeline.statusPoll.intervalS` | `1.0` |
| `HORIZON_ONCE_REQUIRE_ACCEPTED` | — (set via `kubectl set env` / CI) | `--once` smoke gates on ACCEPTED policy status when `1`. |

On bare metal the same contract applies — see the commented `Environment=`
block and the `--source-config` ExecStart example in
`deploy/systemd/horizon-rapp.service`; secrets go in
`/etc/horizon/horizon.env` (EnvironmentFile), never in the unit file.

### Token flow (static bearer)

`A1_CLIENT_TOKEN` now flows end-to-end: chart Secret (`-credentials`) →
pod env via `envFrom.secretRef` → `AuthConfig` in the daemon →
`Authorization: Bearer <token>` header on every A1 call. Rotating the
Secret and restarting the deployment rotates the header.

Precedence (matches `horizon_ric.rapp.auth.build_secure_async_client`):
a configured **static bearer token beats OAuth2** — when both
`A1_CLIENT_TOKEN` and the OAuth2 triple are set, the static token is
sent and the OAuth2 client-credentials flow is not used. mTLS is
orthogonal and **composes with either** bearer mechanism (the client
cert/key ride on the TLS layer regardless of which Authorization header
is chosen).

### Vendor SMO dialects

* `eiap` (Ericsson Intelligent Automation Platform) and `mantaray`
  (Nokia MantaRay SMO) reuse the same env contract with their own URL and
  payload shapes — see `deploy/onboarding/` for the vendor onboarding
  packages and `docs/SMO_INTEGRATION.md` for the dialect matrix.
* Wire behavior per dialect is covered by `tests/test_a1_eiap_dialect.py`
  and `tests/test_a1_mantaray_dialect.py`.

---

## change-management

All production-affecting changes (HPA bounds, NetworkPolicy, image bump) must:

1. Go through `helm template` + `helm diff`.
2. Be staged in a non-prod release first.
3. Carry a rollback plan in the PR description.
4. Trigger a fresh audit-chain `verify()` post-deploy.

---

## Useful one-liners

```sh
# Top-of-page status — paste in incident channel.
kubectl -n $NS get pods,svc,hpa,pdb -l app.kubernetes.io/name=horizon-ric

# Latest decision record (last line in the audit log).
kubectl -n $NS exec deploy/$APP -- tail -n1 /var/lib/horizon/audit.jsonl | jq

# Audit chain length (gauge).
curl -s http://localhost:8082/metrics | grep '^horizon_audit_chain_length'

# State checkpoint (for crash recovery debug).
kubectl -n $NS exec deploy/$APP -- cat /var/lib/horizon/state.json
```
