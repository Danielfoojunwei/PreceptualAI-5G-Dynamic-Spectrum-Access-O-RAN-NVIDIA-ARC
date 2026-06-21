# Horizon-RIC RBAC + Multi-Tenancy

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


The Horizon-RIC rApp ships with a production-grade RBAC + JWT-bearer
auth layer built on **Casbin** (Apache 2.0) and **python-jose** (Apache
2.0). This document is the operator's reference: roles, custom-role
recipes, CI bot tokens, key rotation, and the audit log.

## 1. Default roles

| Role        | Resources                                  | Actions                          |
|-------------|--------------------------------------------|----------------------------------|
| `admin`     | `*` (all)                                  | `*` (all)                        |
| `operator`  | `policies`, `policies/*`                   | `read`, `emit`, `rollback`       |
| `operator`  | `lifecycle`                                | `read`, `degrade`                |
| `auditor`   | `audit`, `audit/*`                         | `read`, `verify`                 |
| `auditor`   | `policies`, `policies/*`                   | `read`                           |
| `regulator` | `audit`, `audit/*`                         | `read`, `verify`                 |
| `regulator` | `policies`, `policies/*`                   | `read`, `explain`                |
| `api-user`  | `state`, `policies`                        | `read`                           |

The seed lives in `src/horizon_ric/security/rbac_policy.csv` and uses
the canonical Casbin **RBAC-with-domains** model. The "domain" is the
Horizon-RIC **tenant**.

## 2. Adding a custom role

Edit `src/horizon_ric/security/rbac_policy.csv`:

```
p, my_custom_role, default, my_resource/*, read
p, my_custom_role, default, my_resource/*, emit
```

Then either restart the rApp or call `Casbin.reload()` from a Python
shell. Existing role-to-user bindings (`g, ...` lines) survive the
edit untouched.

To bind a user:

```sh
.venv/bin/python scripts/horizon_security.py user add \
    --tenant default --user alice --role my_custom_role
```

## 3. Mint a token for a CI bot

```sh
.venv/bin/python scripts/horizon_security.py user add \
    --tenant default --user ci-bot --role api-user

.venv/bin/python scripts/horizon_security.py token mint \
    --user ci-bot --tenant default --ttl 86400
```

Store the resulting JWT as a CI secret (e.g. GitHub Actions
`CI_BOT_JWT`). Tokens are RS256-signed; the verifier only needs the
public key. Bots should rotate their tokens when the signing key
rotates (see §4).

## 4. Zero-downtime signing key rotation

`JWTManager` keeps a single **active** signing key (used for new mints)
and zero or more **retired** keys (verification only). Rotation:

```sh
openssl genrsa -out keys/jwt_signing_2026Q2.pem 2048

.venv/bin/python scripts/horizon_security.py audit-rotate \
    --new-key keys/jwt_signing_2026Q2.pem \
    --overlap-seconds 3600
```

Behaviour:

* **0–3600s after rotation**: new tokens signed by the new key. Old
  tokens still verify under the retired key.
* **After 3600s**: any token signed by the retired key is rejected.

In production, rotate when ≥ 90 % of issued tokens have an `exp`
within the overlap window. The rApp emits `auth.token_invalid` on a
verification miss so SOC tooling can confirm the cutover.

## 5. Audit log

Every authentication decision lands in the structured logger
`horizon_ric.security` with one of these stable event names:

| Event                | Emitted when                                         |
|----------------------|------------------------------------------------------|
| `auth.granted`       | JWT verified + Casbin allowed → request proceeds     |
| `auth.denied`        | Casbin denied (role, tenant, or resource mismatch)   |
| `auth.token_invalid` | JWT missing/malformed/bad-signature/expired/wrong-aud|

Fields always include `subject`, `tenant`, `path`, plus event-specific
fields (`reason`, `roles`, `obj`, `act`).

Default landing zones:

* **stdout / stderr** — the structlog default. Fluent Bit / Vector
  pick this up in the standard rApp deployment.
* **Loki / Elasticsearch** — via the SMO log pipeline; query
  `{job="horizon_ric.security"}` and filter on `event=auth.denied`.

## 6. Tenant isolation guarantees

* The Casbin matcher binds every check to a specific tenant
  (`r.dom == p.dom`); a user in `tenant_A` cannot enforce against
  `tenant_B` even if they share a name.
* The evidence store maintains **one SHA-256 hash chain per tenant**.
  An auditor scoped to `tenant_A` (`with TenantScope("tenant_A")`)
  cannot read or verify `tenant_B`'s records — `__iter__` filters by
  the active tenant.
* The middleware enforces the `X-Tenant` header equals the JWT's
  `tenant` claim; cross-tenant resource access is a 403 (`auth.denied`,
  `reason="tenant_mismatch"`).

## 7. CLI reference

```
horizon-security user add       --user X --role Y --tenant Z
horizon-security user remove    --user X --role Y --tenant Z
horizon-security user list      [--role Y] [--tenant Z]
horizon-security token mint     --user X --tenant Z [--ttl N]
horizon-security token verify   <jwt>
horizon-security policy show
horizon-security audit-rotate   --new-key path/to/new.pem
                                [--old-key path] [--overlap-seconds N]
```
