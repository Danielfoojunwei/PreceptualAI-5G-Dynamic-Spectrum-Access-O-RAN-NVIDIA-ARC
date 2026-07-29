# Horizon-RIC API surfaces

The repo contains **three** parallel REST API modules under
`src/horizon_ric/rapp/`. Only one of them is served by the running rApp.
This page is the map; the per-module `STATUS` docstrings point back here.

| Module | Served in production? | Auth model | Purpose |
|---|---|---|---|
| `dashboard_api.py` | **Yes** — mounted by `HorizonRAppLifecycle._serve_api()` on port 8083 | HS256 JWT (`HORIZON_API_JWT_SECRET`) + per-endpoint role enforcement; PBKDF2-hashed credentials in `HORIZON_API_USERS` | Operator dashboard surface consumed by the Next.js frontend; reads the real evidence chain |
| `api.py` | No — security scaffold | RS256 (`security.jwt.JWTManager`) + Casbin RBAC (`security.rbac`) middleware | Hosts `JWTAuthMiddleware`/`attach_security` for the security tests; its handlers are mostly empty demos. Also re-exports `build_api` from `api_v1` |
| `api_v1.py` | No — SDK quickstart/demo | HS256 JWT (`HORIZON_RIC_JWT_SECRET`, per-process random fallback in dev) with `scopes` claims | Seeded with **fabricated** DecisionRecords and synthetic SLA timelines for SDK quickstarts and demo tests. Its output is not live data |

## Which one runs

`HorizonRAppLifecycle.run_forever()` starts `_serve_api()`
(`src/horizon_ric/rapp/lifecycle.py`), which:

* refuses to start unless `HORIZON_API_JWT_SECRET` is set (and honours
  `HORIZON_API_DISABLE`);
* builds **`dashboard_api.build_dashboard_api(self)`** and serves it with
  uvicorn on `HORIZON_API_HOST`/`HORIZON_API_PORT` (default `0.0.0.0:8083`).

Neither `api.py` nor `api_v1.py` is mounted anywhere in the lifecycle.

## OpenAPI artifact provenance

`docs/openapi/horizon-ric-rapp.yaml` **was** generated from the seeded demo
module `api_v1.py`, which was misleading: it documented `/v1/*` routes and
scope-based auth that the deployed rApp never served. It is now generated
from `dashboard_api.build_dashboard_api()` — the surface that is actually
served — and carries a provenance comment header with the regeneration
one-liner:

```bash
python - <<'EOF'
import os, yaml
os.environ.setdefault("HORIZON_API_JWT_SECRET", "spec-generation-only")
os.environ.setdefault("HORIZON_EVIDENCE_PATH", "/tmp/spec-audit.jsonl")
from horizon_ric.rapp.dashboard_api import build_dashboard_api
print(yaml.safe_dump(build_dashboard_api(lifecycle=None).openapi(), sort_keys=False))
EOF
```

## Dashboard API hardening (2026-07 audit response)

External-audit findings against `dashboard_api.py` and their fixes:

1. **Plaintext credentials compared with `==`** →
   `HORIZON_API_USERS` entries now support
   `username:pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>:role`
   (stdlib `hashlib.pbkdf2_hmac`, constant-time `hmac.compare_digest`,
   600 000 iterations by default). Generate entries with:

   ```bash
   # bare hash (prompts for the password — keeps it out of shell history):
   python -m horizon_ric.rapp.dashboard_api --hash-password
   # full user:hash:role entry:
   python -m horizon_ric.rapp.dashboard_api --hash-password 's3cret' --user carol --role admin
   ```

   Plaintext entries still work **only** when `HORIZON_PRODUCTION_MODE`
   is explicitly falsy (`0`/`false`/`no`); they log a loud
   `plaintext credentials — dev only` warning on every use. With
   production mode truthy — **the default when unset** — plaintext
   entries are refused at login (401, with a structured error log
   explaining why).

2. **Role claim minted but never enforced** → per-endpoint role
   enforcement via the `require_role(...)` FastAPI dependency (403 on
   mismatch). The JWT shape (`iss`, `aud`, `sub`, `role`, `iat`, `exp`)
   is unchanged, so previously issued tokens keep working.

   | Endpoint | Method | Required role |
   |---|---|---|
   | `/api/v1/auth/token` | POST | public (credential exchange) |
   | `/api/v1/state` | GET | any authenticated role |
   | `/api/v1/policies` | GET | any authenticated role |
   | `/api/v1/policies/{id}` | GET | any authenticated role |
   | `/api/v1/sla/timeline` | GET | any authenticated role |
   | `/api/v1/circuit-breakers` | GET | any authenticated role |
   | `/api/v1/connectors` | GET | any authenticated role |
   | `/api/v1/audit/verify` | POST | `admin` **or** `operator` (viewer → 403) |

   `operator` is deliberately kept on the privileged list for backward
   compatibility with tokens issued before the hardening; the endpoint
   is a full-chain verify walk (privileged/expensive) rather than a
   mutation.

3. **Evidence path drift** → the default evidence path is now
   `/var/lib/horizon/audit.jsonl`, matching the audit chain written by
   `scripts/run_horizon_rapp.py` (and the Helm/systemd mounts).
   `HORIZON_EVIDENCE_PATH` still overrides it; dev checkouts should set
   it to a writable location.

4. **HS256 vs the RS256/Casbin layer** → *not changed in this pass.*
   The dashboard keeps its HS256 mini-auth (now with hashed credentials
   and role enforcement) because the RS256/Casbin scaffold in `api.py`
   has no real handlers yet. Migrating the dashboard onto
   `attach_security()` (RS256 + Casbin, multi-tenant) is the intended
   end state; until then this page and the module docstrings make the
   split explicit.

Security regression tests: `tests/test_dashboard_api_security.py`
(hashed login, plaintext refusal in production, role matrix, JWT
tampering, CLI helper round-trip, evidence-path alignment) alongside the
original `tests/test_api.py` endpoint suite.
