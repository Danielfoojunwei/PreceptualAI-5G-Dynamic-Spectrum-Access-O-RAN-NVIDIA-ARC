# Horizon-RIC Disaster-Recovery Plan

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`../README.md`](../README.md) for the 49-section deep dive.*


| Field            | Value |
|------------------|-------|
| **Owner**        | Horizon-RIC SRE on-call |
| **Cadence**      | Hourly snapshots, monthly full drill |
| **RPO target**   | **≤ 1 hour** (worst case = one hourly snapshot lost) |
| **RTO target**   | **≤ 4 hours** (restore + chain verify + handover) |
| **Tooling**      | `deploy/helm/horizon-ric/` + `scripts/backup_and_restore.sh` |
| **Backing services** | TimescaleDB StatefulSet, MinIO StatefulSet, optional cosign for blob signatures |

This plan closes Row 38 of the gap matrix. Every step references real
artifacts checked into the repository — no manual edits, no out-of-band
runbooks. The CI gates `tests/test_state_recovery.py` and
`tests/test_loop_state.py` cover the state-persistence and crash-recovery
logic this plan relies on: `test_state_recovery.py` exercises atomic
save/load round-trips, corrupt/partial-write recovery, and periodic
checkpointing, while `test_loop_state.py` locks the lifecycle state machine
and its append-only, replayable history. The `scripts/backup_and_restore.sh`
driver (signature + `pg_restore` + chain re-verify) is the operational
backup tool referenced throughout.

---

## 1. Architecture

```
   ┌─────────────────────────────────────────────────────────────┐
   │                     Horizon-RIC rApp                        │
   │                                                             │
   │   evidence.JsonlEvidenceStore  ──►  TimescaleDB (StatefulSet)
   │      (in-pod tamper-evident         hypertable: `decisions`
   │       chain, per tenant)            chunk_time_interval = 1d
   │                                     retention = 7 days
   └─────────────────────────────────────────────────────────────┘
                                │
              hourly snapshot   │   pg_dump --format=custom
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │   MinIO (StatefulSet)                                       │
   │   Bucket: `horizon-evidence-archive`                        │
   │   Versioned, GOVERNANCE retention: 30 days                  │
   │   Each object: `decisions-<ISO8601>.dump` + `.dump.sig`     │
   └─────────────────────────────────────────────────────────────┘
                                │
                                │   weekly cross-region mirror
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │   Off-site S3 mirror (DR cluster)                           │
   └─────────────────────────────────────────────────────────────┘
```

The chart deploys both StatefulSets via:

* `deploy/helm/horizon-ric/templates/statefulset-timescaledb.yaml`
* `deploy/helm/horizon-ric/templates/statefulset-minio.yaml`

with values in `deploy/helm/horizon-ric/values.yaml` under the `timescaledb:`
and `minio:` blocks.

## 2. Backup retention

| Tier    | Count | Implementation |
|---------|-------|----------------|
| Hourly  | 24    | All hourly snapshots within the day, pruned by `enforce_retention()`. |
| Daily   | 30    | First snapshot of each day kept; pruned by `HORIZON_DR_RETENTION_DAILY` (default 30). |
| Monthly | 12    | First snapshot of each month, lifecycle-tier transitioned to MinIO `STANDARD-IA`. |
| Yearly  | 5     | First snapshot of each year, transitioned to off-site S3 mirror. |

Retention is enforced both client-side (`scripts/backup_and_restore.sh
enforce_retention`) and server-side (MinIO bucket lifecycle rules set up
during the `helm install` post-install Job).

## 3. Monthly DR drill checklist

Run on the first Tuesday of each month:

1. [ ] Confirm last 24 hourly snapshots are present:
   ```
   kubectl exec -it sts/horizon-ric-minio -- mc ls local/horizon-evidence-archive/decisions/
   ```
2. [ ] Run a full backup-and-restore on the staging cluster:
   ```
   kubectl exec deploy/horizon-ric -- /opt/horizon/scripts/backup_and_restore.sh backup
   kubectl exec deploy/horizon-ric -- /opt/horizon/scripts/backup_and_restore.sh restore decisions-<TS>.dump
   ```
3. [ ] Verify the restored hash chain:
   ```
   kubectl exec deploy/horizon-ric -- python -c "
   from horizon_ric.evidence.store import JsonlEvidenceStore
   s = JsonlEvidenceStore('/var/lib/horizon/audit.jsonl')
   rc = s.verify()
   print('verify =', rc); raise SystemExit(0 if rc == -1 else 1)
   "
   ```
   `EvidenceStore.verify()` returns `-1` iff the chain is intact (any other
   value is the index of the first broken record); the snippet exits 0 only
   when every record validates.
4. [ ] Time the round-trip; record in the SRE log. RTO target is ≤ 4 hours.
5. [ ] Page the on-call (`/dr-drill-fire`) and have them execute the failover
   playbook below from cold context. Stop-watch ≤ 4 hours.
6. [ ] Rotate the cosign signing key
   (`cosign generate-key-pair --kms ...`) and update the
   `horizon-ric-cosign` Secret. Re-sign the most recent snapshot.

## 4. Failover playbook (primary cluster gone)

When the primary cluster is unreachable, the on-call follows these steps in
order. Total time budget = 4 h.

### 4.1 Spin up the DR cluster

```
helm upgrade --install horizon-ric \
  oci://ghcr.io/preceptualai/horizon-ric \
  -f deploy/helm/horizon-ric/values-dr.yaml \
  --namespace horizon \
  --create-namespace
```

`values-dr.yaml` differs from prod by pointing `minio.endpoint` at the
off-site mirror's S3 endpoint.

### 4.2 Wait for backing services to be ready

```
kubectl -n horizon wait --for=condition=ready pod \
  -l app.kubernetes.io/component=timescaledb --timeout=10m
kubectl -n horizon wait --for=condition=ready pod \
  -l app.kubernetes.io/component=minio --timeout=10m
```

### 4.3 Identify the most recent verified snapshot

```
kubectl -n horizon exec -it sts/horizon-ric-minio -- \
  mc ls local/horizon-evidence-archive/decisions/ | tail -n 5
# Pick the newest decisions-<ISO>.dump that has a matching .sig.
```

### 4.4 Restore

```
KEY=decisions-20260506T180000Z.dump   # the chosen snapshot
kubectl -n horizon exec deploy/horizon-ric -- \
  /opt/horizon/scripts/backup_and_restore.sh restore "$KEY"
```

This step both:

1. Verifies the cosign signature on `$KEY.sig` (or fails closed when
   `HORIZON_DR_REQUIRE_COSIGN=1`).
2. Runs `pg_restore --clean --if-exists` against the freshly-provisioned
   TimescaleDB pod.

### 4.5 Verify the hash chain

```
kubectl -n horizon exec deploy/horizon-ric -- python -c "
from horizon_ric.evidence.store import JsonlEvidenceStore
s = JsonlEvidenceStore('/var/lib/horizon/audit.jsonl')
rc = s.verify()
print('verify =', rc); raise SystemExit(0 if rc == -1 else 1)
"
# exit 0 = chain intact (EvidenceStore.verify() == -1)
# exit 1 = chain broken; STOP — do not promote the DR cluster.
```

### 4.6 Promote DNS / RIC handover

```
# Switch the SMO non-RT-RIC R1 endpoint over to the DR cluster's gateway.
kubectl -n horizon patch service horizon-ric \
  --type=json -p='[{"op":"replace","path":"/spec/type","value":"LoadBalancer"}]'
# Update the global SMO ingress weight (this is operator-specific; example
# below is for the AWS Load-Balancer-Controller weighted target group).
aws elbv2 modify-target-group-attributes \
  --target-group-arn $TG_ARN \
  --attributes Key=stickiness.enabled,Value=true \
               Key=stickiness.lb_cookie.duration_seconds,Value=86400
```

### 4.7 Smoke-test

```
curl -fsS https://horizon-dr.smo.example.com/healthz
curl -fsS https://horizon-dr.smo.example.com/readyz
curl -fsS https://horizon-dr.smo.example.com/api/v1/evidence | jq '.count'
```

`evidence.count` must be within 1 hour of the pre-disaster value.

### 4.8 Re-arm hourly backups on the DR cluster

The CronJob is created automatically via `backup.enabled=true` in
`values-dr.yaml`. Verify:

```
kubectl -n horizon get cronjob horizon-ric-backup
kubectl -n horizon logs -l job-name=horizon-ric-backup-<TS>
```

## 5. Failure modes the plan covers

| Scenario | Tier  | Recovery action |
|----------|-------|-----------------|
| Single TimescaleDB pod crash | RPO=0 | StatefulSet recreates the pod with the same PVC. |
| TimescaleDB PVC corruption   | RPO≤1h | Restore from the most recent hourly MinIO snapshot. |
| Whole namespace lost         | RPO≤1h | `helm install` + restore. |
| Primary cluster lost         | RPO≤24h (cross-region mirror lag) | Failover playbook §4. |
| Region-wide outage           | RPO≤24h | Failover playbook §4 against the off-site mirror. |
| Hash-chain mismatch on restore | aborts restore | `verify_blob()` in the script returns non-zero, restore is rolled back. |
| Cosign key compromise | escalation | Rotate during the next monthly drill (see §3 step 6). |

## 6. References

* Bash driver: [`scripts/backup_and_restore.sh`](../scripts/backup_and_restore.sh)
* Helm template (TimescaleDB): [`templates/statefulset-timescaledb.yaml`](helm/horizon-ric/templates/statefulset-timescaledb.yaml)
* Helm template (MinIO): [`templates/statefulset-minio.yaml`](helm/horizon-ric/templates/statefulset-minio.yaml)
* State-recovery CI tests: [`tests/test_state_recovery.py`](../tests/test_state_recovery.py), [`tests/test_loop_state.py`](../tests/test_loop_state.py)
* RPO/RTO governance: [`deploy/SLO.md`](SLO.md).
