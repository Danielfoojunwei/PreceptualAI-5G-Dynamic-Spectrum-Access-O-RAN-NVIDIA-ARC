# Runbook — Security Incident Response

Activates when a PreceptualAI tenant chain breaks, an IDS hit lands on the rApp
pod, or a vendor / regulator notification arrives. Pair this runbook with
`cert_rotation.md` (credential rotation step) and `customer_escalation.md`
(communication tempo).

```sh
NS=${HORIZON_NS:-horizon}
REL=${HORIZON_RELEASE:-horizon-ric}
APP=${REL}-horizon-ric
TENANT=${HORIZON_TENANT:-acme}
EVIDENCE_DB=${HORIZON_EVIDENCE_DB:-/var/lib/horizon/evidence.db}
SNAP_DIR=/var/horizon/incident-$(date +%Y%m%dT%H%M%SZ)
```

## Severity classification

| Sev | Trigger                                                                          | Page              |
| --- | -------------------------------------------------------------------------------- | ----------------- |
| 1   | Active intrusion (RCE, lateral movement, evidence chain break, exfil)            | on-call + DPO + CISO immediately |
| 2   | Suspected intrusion (anomalous auth, unexplained restart, unrecognised binary)   | on-call within 15 min |
| 3   | Suspicious activity (failed-login burst, unusual A1 churn, JWT verify spikes)    | queue, triage in business hours |

## Sev1 — active intrusion playbook

Time budget: contain within 30 min; preserve evidence within 60 min;
NIS2 24-hour clock starts at the moment of detection.

### Step 1 — page

`PagerDuty` service `horizon-ric-sec`. Manual escalations: tenant DPO,
internal CISO. Open incident bridge.

### Step 2 — isolate the rApp pod

```sh
POD=$(kubectl -n $NS get pod -l app.kubernetes.io/name=horizon-ric \
       -o jsonpath='{.items[0].metadata.name}')
NODE=$(kubectl -n $NS get pod $POD -o jsonpath='{.spec.nodeName}')

# Quarantine label freezes the pod for forensics. NetworkPolicy in
# deploy/helm/horizon-ric/templates/networkpolicy.yaml denies all egress
# for pods carrying horizon.preceptual.ai/quarantine=true.
kubectl -n $NS label pod $POD horizon.preceptual.ai/quarantine=true --overwrite
kubectl -n $NS scale deploy/$APP --replicas=0
kubectl drain $NODE --ignore-daemonsets --delete-emptydir-data --force
```

### Step 3 — snapshot before touching anything else

```sh
mkdir -p $SNAP_DIR
kubectl -n $NS cp $POD:/var/lib/horizon/evidence.db $SNAP_DIR/evidence.db
kubectl -n $NS cp $POD:/var/lib/horizon/audit.jsonl $SNAP_DIR/audit.jsonl
kubectl -n $NS cp $POD:/var/lib/horizon/state.json  $SNAP_DIR/state.json
kubectl -n $NS logs $POD --all-containers --previous=true \
  > $SNAP_DIR/logs-previous.log 2>&1 || true
kubectl -n $NS logs $POD --all-containers > $SNAP_DIR/logs-current.log
sha256sum $SNAP_DIR/* > $SNAP_DIR/MANIFEST.sha256
```

### Step 4 — verify every tenant chain

`EvidenceStore.verify()` is at `src/horizon_ric/evidence/store.py:106`. ANY
chain break = compromise.

```sh
for T in $(yq '.tenants[].id' deploy/helm/horizon-ric/values.yaml); do
  kubectl -n $NS exec deploy/$APP -- python -c "
from horizon_ric.evidence.store import open_default_store
from horizon_ric.security.tenant import TenantScope
with TenantScope('$T'):
    s = open_default_store()
    print('$T', s.verify())
" || echo "$T BROKEN"
done | tee $SNAP_DIR/chain-verify.txt
```

Expected good output: `acme 1742` (records verified, no exception). A
`BROKEN` line, an exception, or a record count regressing vs the last known
state in `state.json` are all sev1 confirmations.

### Step 5 — rotate ALL credentials

Run every rotation in `cert_rotation.md` §1–6, in order. Land each rotation
as a `DecisionRecord` with `operator_override=True` and the incident ID in
`operator_override_reason`.

### Step 6 — start the NIS2 24-hour clock

Note the detection timestamp in the bridge. The 24-hour initial notification
template is in §"Reporting templates" below.

### Step 7 — forensic image

```sh
# Use kubectl debug + crictl on the (drained) node to extract the running
# container filesystem before the kubelet GC reclaims it.
kubectl debug node/$NODE -it --image=alpine:3.20 -- chroot /host bash
# inside the debug shell:
crictl ps -a | grep horizon-ric
crictl inspect <CONTAINER_ID> > /tmp/inspect.json
crictl exec <CONTAINER_ID> tar cf - /var/lib/horizon /etc/horizon \
  | gzip > /tmp/forensic-$(date +%s).tar.gz
```

Copy `/tmp/forensic-*.tar.gz` off the node to evidence storage.

### Step 8 — restore from last good chain

```sh
LAST_GOOD=$(ls /backups/horizon-evidence/*.db | tail -1)
kubectl -n $NS create configmap horizon-evidence-restore \
  --from-file=evidence.db=$LAST_GOOD
# Update Helm values to mount the restore configmap, redeploy on a clean node.
helm upgrade $REL deploy/helm/horizon-ric -n $NS --reuse-values \
  --set persistence.restoreFrom=horizon-evidence-restore
```

Run `EvidenceStore.verify()` again before scaling replicas back up.

## Sev2 — suspected intrusion

Skip drain + forensic image. Do steps 3 (snapshot), 4 (verify chains),
5 (rotate JWT signing key only), and queue full Sev1 if any chain breaks
or a second indicator lands within 4 hours.

## Sev3 — suspicious activity

Triage in business hours. Required actions:

1. `kubectl -n $NS logs deploy/$APP --tail=2000 | jq 'select(.event | test("auth|jwt|rbac"))'`
2. Open a ticket; attach the log slice.
3. If the burst > 5 events/min for > 10 min, promote to Sev2.

## Reporting templates

### NIS2 — 24-hour initial (Art. 23(4)(a))

```
Subject: Initial incident notification — PreceptualAI tenant <TENANT> — <INCIDENT_ID>
Detected: <UTC timestamp>
Service affected: PreceptualAI rApp / Near-RT RIC policy emit
Suspected cause: <one line>
Cross-border impact: <yes/no, list MS>
Mitigation in place: pod quarantined, credentials rotated, evidence preserved
Next update: <UTC timestamp + 48h>
```

### NIS2 — 72-hour full (Art. 23(4)(b))

Add: timeline, IOCs (hashes/IPs/CVEs), affected tenants, customer comms log,
preliminary attribution. Reference `$SNAP_DIR/MANIFEST.sha256`.

### NIS2 — 1-month detailed (Art. 23(4)(c))

Add: full root-cause, all DecisionRecords minted between detection and
restoration (`horizon-ric-sdk audit verify --since <ts>`), corrective and
preventive actions, post-mortem link.

### EU AI Act Art. 73 — 15-day serious incident report

Required for any incident affecting `policy.constraints` violations or human-
oversight bypass. Template lives in `docs/compliance/ai_act_art73.md`
(referenced from `GAPS_TO_PILOT.md` row 36 — currently MISSING; placeholder
text in this runbook acts as fallback until shipped).

## Communication playbook

| Audience              | Who tells them          | When                 | Channel                  |
| --------------------- | ----------------------- | -------------------- | ------------------------ |
| Internal CISO         | On-call                 | T+0                  | Pager + bridge           |
| Tenant DPO            | On-call                 | T+15 min             | Encrypted email + phone  |
| Affected customer NOC | T2 engineer             | T+30 min             | Per `customer_escalation.md` 15-min template |
| Regulator (NIS2 CSIRT)| CISO                    | T+24h / 72h / 30d    | National CSIRT portal    |
| Regulator (AI Act)    | CISO + Legal            | T+15 days            | EU AI Office portal      |
| Public statement      | CEO + Comms             | After regulator      | Press release + status page |

## Honest gaps

- `EvidenceStore.verify()` runs per-tenant via `TenantScope`; the runbook
  shells one invocation per tenant. There is no `verify_all_tenants()` helper
  in `evidence/store.py` today — file as a follow-up.
- The `kubectl debug node/...` flow assumes a permissive PSP/PSA; on
  cluster profiles that block privileged debug pods, fall back to
  `crictl` direct on a node SSH session.
- `docs/compliance/ai_act_art73.md` is referenced but not present in repo
  (see `GAPS_TO_PILOT.md` row 36).
