# PreceptualAI On-Call Runbook

For operators paged on a PreceptualAI rApp alert. Each scenario below has:

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

# Probe the health server directly.
kubectl -n $NS port-forward svc/$APP 8081:8081 &
curl -fsS http://localhost:8081/healthz
curl -fsS http://localhost:8081/readyz
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
* If only A1 emit is broken, throttle the rApp to read-only:

  ```sh
  kubectl -n $NS set env deploy/$APP HORIZON_A1_DRY_RUN=true
  ```

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
# Pause emits, keep observing.
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
