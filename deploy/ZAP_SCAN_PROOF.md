# OWASP ZAP Baseline Scan Proof (Row 37)

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`../README.md`](../README.md) for the 49-section deep dive.*


_Scan recorded_: 2026-05-06T22:43:23.669624+00:00
_Target_: http://127.0.0.1:8083
_Tool_: ZAP 2.16.0 JAR (cross-platform), JDK 21 aarch64
_Scan duration_: 21s (0.3 min)

> **Provenance note.** This is a recorded scan result from 2026-05-06. The
> alert counts and durations below are a historical capture; re-run ZAP
> against the dashboard API to reproduce.

## Headline

**ZAP PASS**: 3 alerts total — High=0 Critical=0 Medium=0 Low=2 Informational=1, scan duration 21s.

## Alerts by severity

| Severity | Count |
| --- | ---: |
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 2 |
| Informational | 1 |

## Alerts (first 50)

| # | Severity | Name |
| --: | --- | --- |
| 1 | Low | Timestamp Disclosure - Unix |
| 2 | Low | X-Content-Type-Options Header Missing |
| 3 | Informational | Authentication Request Identified |

## Reports

* JSON: [`deploy/zap_report.json`](./zap_report.json) (committed in this repo)
* HTML/XML: produced by ZAP under `/tmp/zap_work/` at scan time

## How this was generated

The scan boots `horizon_ric.rapp.dashboard_api:create_app`
(`src/horizon_ric/rapp/dashboard_api.py`) with `HORIZON_API_JWT_SECRET` set,
mints a real JWT via `POST /api/v1/auth/token`, and runs ZAP 2.16 in
`-autorun` mode with the OpenAPI + reports add-ons. ZAP imports the live
`/openapi.json`, spiders the dashboard, runs passive analysis, and exports
the traditional-json report (committed as `deploy/zap_report.json`). No
mocks: a real Java ZAP daemon scans a real FastAPI HTTP endpoint over a real
loopback socket.

> The `scripts/run_zap_scan.sh` wrapper that orchestrated this run is not
> currently checked into `scripts/`. The dashboard app it scans
> (`create_app`) and its JWT auth are exercised in CI by
> `tests/test_jwt.py`, `tests/test_security_middleware.py`, and
> `tests/test_api.py`.
