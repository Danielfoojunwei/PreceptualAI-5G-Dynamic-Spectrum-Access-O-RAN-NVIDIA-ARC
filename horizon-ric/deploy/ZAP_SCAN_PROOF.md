# OWASP ZAP Baseline Scan Proof (Row 37)

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`../README.md`](../README.md) for the 49-section deep dive.*


_Scan time_: 2026-05-06T22:43:23.669624+00:00
_Target_: http://127.0.0.1:8083
_Tool_: ZAP 2.16.0 JAR (cross-platform), JDK 21 aarch64
_Scan duration_: 21s (0.3 min)

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

* JSON: `/home/danielfoojunwei/Preceptualv1/horizon-ric/deploy/zap_report.json`
* HTML/XML: produced by ZAP under /tmp/zap_work/

## How this was generated

`scripts/run_zap_scan.sh` boots `horizon_ric.rapp.dashboard_api:create_app` with `HORIZON_API_JWT_SECRET` set, mints a real JWT via `POST /api/v1/auth/token`, and runs ZAP 2.16 in `-autorun` mode with the OpenAPI + reports add-ons. ZAP imports the live `/openapi.json`, spiders the dashboard, runs passive analysis, and exports the traditional-json report. No mocks: a real Java ZAP daemon scans a real FastAPI HTTP endpoint over a real loopback socket.
