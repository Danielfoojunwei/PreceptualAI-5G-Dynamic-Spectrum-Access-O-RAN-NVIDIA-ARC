# Runbook — Certificate / Key Rotation

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


For the on-call when a cert-expiry alert fires, a CA-compromise notice arrives,
or the quarterly rotation cron rings.

Conventions (export once at top of shift):

```sh
NS=${HORIZON_NS:-horizon}                           # k8s namespace
REL=${HORIZON_RELEASE:-horizon-ric}                 # Helm release name
APP=${REL}-horizon-ric                              # k8s workload name
TENANT=${HORIZON_TENANT:-acme}                      # Casbin domain
EVIDENCE_DB=${HORIZON_EVIDENCE_DB:-/var/lib/horizon/evidence.db}
```

## Triggers

| Trigger                                       | Severity | Clock        |
| --------------------------------------------- | -------- | ------------ |
| Prometheus `horizon_cert_expiry_days < 30`    | warning  | 30 days      |
| Prometheus `horizon_cert_expiry_days < 7`     | critical | 24 hours     |
| Internal CA / vendor disclosure of compromise | sev1     | rotate now   |
| Scheduled quarterly rotation                  | planned  | within 7 d   |

Every rotation **MUST** land a `DecisionRecord` in the evidence chain with
`operator_override=True` and `operator_override_reason="<rotation kind>: <ticket>"`.
That record is the legal trail under `evidence/store.py` and AI Act Art. 12.

---

## 1. mTLS client cert (R1Adapter outbound to SMO)

Used by `src/horizon_ric/integrations/raas.py` and the R1 register call.
Cert lives in `Secret/${APP}-credentials` keys `r1.client.crt`, `r1.client.key`.

```sh
# 1. Generate new keypair (P-256, 90-day lifetime)
openssl ecparam -name prime256v1 -genkey -noout -out /tmp/r1.client.key
openssl req -new -key /tmp/r1.client.key \
  -subj "/CN=${REL}.${NS}/O=PreceptualAI/OU=R1Adapter" \
  -out /tmp/r1.client.csr
# 2. Submit /tmp/r1.client.csr to the operator's CA. Save signed cert as /tmp/r1.client.crt.
# 3. Patch the Secret in-place (Helm-managed; --dry-run first to confirm diff).
kubectl -n $NS create secret generic ${APP}-credentials \
  --from-file=r1.client.crt=/tmp/r1.client.crt \
  --from-file=r1.client.key=/tmp/r1.client.key \
  --dry-run=client -o yaml | kubectl -n $NS apply -f -
# 4. Rolling restart picks up the new mount.
kubectl -n $NS rollout restart deploy/$APP
kubectl -n $NS rollout status deploy/$APP --timeout=2m
```

Validation:

```sh
kubectl -n $NS logs deploy/$APP --tail=50 | jq 'select(.event=="r1.register")'
# expected: {"event":"r1.register","status":"REGISTERED",...}
```

Rollback: `kubectl -n $NS rollout undo deploy/$APP` reverts to the previous
secret revision and pod template.

## 2. mTLS server cert (health/api endpoints behind TLS terminator)

Termination is at the cluster Ingress / sidecar; PreceptualAI itself listens on
plaintext 8081 inside the pod. Rotate the Ingress secret:

```sh
kubectl -n $NS create secret tls ${APP}-server-tls \
  --cert=/tmp/server.crt --key=/tmp/server.key \
  --dry-run=client -o yaml | kubectl -n $NS apply -f -
kubectl -n $NS rollout restart deploy/ingress-nginx-controller
```

Validation: `curl --cacert /tmp/ca.crt https://horizon.${TENANT}/healthz` returns 200.

## 3. JWT signing key (RS256)

Implementation: `src/horizon_ric/security/jwt.py::JWTManager.rotate_signing_key`.
The 1-hour overlap window is `_OVERLAP_WINDOW_DEFAULT_SEC = 3600`. Tokens
issued by the old key still verify until either their `exp` or the overlap
deadline elapses, whichever is first.

```sh
# 1. Generate the new RSA-2048 key.
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 \
  -out /tmp/jwt.signing.new.pem
# 2. Drop into the running pod and exercise the rotation path so you see
#    the new public-key chars + retired-key count in JSON.
kubectl -n $NS cp /tmp/jwt.signing.new.pem $APP-0:/etc/horizon/jwt.signing.new.pem
kubectl -n $NS exec deploy/$APP -- \
  python scripts/horizon_security.py audit-rotate \
    --new-key /etc/horizon/jwt.signing.new.pem \
    --overlap-seconds 86400          # 24-hour overlap for cross-tenant tokens
# 3. Persist by replacing the Secret + restarting (uses the new key on boot).
kubectl -n $NS create secret generic ${APP}-credentials \
  --from-file=jwt.signing.pem=/tmp/jwt.signing.new.pem \
  --dry-run=client -o yaml | kubectl -n $NS apply -f -
kubectl -n $NS rollout restart deploy/$APP
```

Expected `audit-rotate` output:

```json
{ "ok": true, "active_public_key_chars": 451, "retired_keys": 1, "overlap_seconds": 86400 }
```

Rollback: keep the old `.pem` for 24 h. If new tokens fail to verify, restore
the old Secret and `kubectl rollout undo`. The retired-key list is held in
memory only — a restart drops it, so do not restart twice within the overlap.

## 4. cosign image signing key

Per `deploy/cosign/README.md`. Quarterly or on suspected exposure.

```sh
cd deploy/cosign
COSIGN_PASSWORD=$NEW_PASSWORD cosign generate-key-pair
git add cosign.pub && git commit -m "rotate cosign key $(date -I)"
# Update the COSIGN_KEY + COSIGN_PASSWORD GitHub Actions secrets, then re-sign:
COSIGN_PASSWORD=$NEW_PASSWORD cosign sign --key cosign.key \
  --yes horizonric/rapp:$(git describe --tags --abbrev=0)
cosign verify --key cosign.pub horizonric/rapp:$(git describe --tags --abbrev=0)
```

Validation: `cosign verify` returns `Verification for ... -- The cosign claims were validated`.

## 5. Kubernetes Secret (Helm chart)

Template: `deploy/helm/horizon-ric/templates/secret.yaml`. Three keys today —
`SPACE_TRACK_IDENTITY`, `SPACE_TRACK_PASSWORD`, `A1_CLIENT_TOKEN`. Rotation:

```sh
helm upgrade $REL deploy/helm/horizon-ric -n $NS \
  --reuse-values \
  --set secret.data.a1ClientToken=$(openssl rand -hex 32)
kubectl -n $NS rollout status deploy/$APP --timeout=2m
```

## 6. SSH host key for NETCONF (E2/O1 to E2 nodes)

User-space netconfd lives at `/tmp/nc-server` per `deploy/start_netconf_server.sh`.
On each E2 node host:

```sh
pkill -f 'sshd.*8830' || true
ssh-keygen -t ed25519 -N '' -f /tmp/nc-server/etc/ssh/ssh_host_ed25519_key
bash deploy/start_netconf_server.sh -d
ssh-keyscan -p 8830 -t ed25519 127.0.0.1   # capture new fingerprint
```

Distribute the new fingerprint to every rApp's `known_hosts` and bounce the O1
adapter pods (`kubectl -n $NS rollout restart deploy/$APP`).

---

## Audit trail (mandatory for every rotation)

```sh
kubectl -n $NS exec deploy/$APP -- python -c "
from horizon_ric.evidence.store import open_default_store
from horizon_ric.evidence.schema import DecisionRecord
from horizon_ric.security.tenant import TenantScope
import os
with TenantScope(os.environ['HORIZON_TENANT']):
    s = open_default_store()
    rec = DecisionRecord.new(
        decision_id='rotate-'+os.urandom(4).hex(),
        operator_override=True,
        operator_override_reason='cert_rotation: JIRA SEC-1234 jwt key rotation',
    )
    print(s.append(rec))   # prints chain hash
"
```

Then verify the chain still links:

```sh
horizon-ric-sdk audit verify
# expected: {"verified": true, "records": <N+1>, ...}
```

If `verified: false` after a rotation, treat as **Sev1** and pivot to
`security_incident.md`.

## Honest gaps

- The Helm `secret.yaml` template does **not** carry the JWT signing key
  today — it has to be applied as a side-channel `kubectl create secret`.
  Folding it into `values.yaml` is open work.
- `horizon_cert_expiry_days` Prometheus metric is referenced above but is
  not currently exported by `observability/`. The 30/7 day alerts fire on
  external cert-manager metrics, not from horizon-ric itself.
