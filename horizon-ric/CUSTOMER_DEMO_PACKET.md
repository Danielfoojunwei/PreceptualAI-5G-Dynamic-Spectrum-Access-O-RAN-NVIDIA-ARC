# PreceptualAI — Customer Demo Packet

**Version**: 0.2.0   **Date**: 2026-05-07   **Repo SHA**: `d23f1215156d00b7f16ab0c6934b3342f6f25900`

Pointers: [`FINAL_PROGRESS.md`](FINAL_PROGRESS.md) · [`GAPS_TO_PILOT.md`](GAPS_TO_PILOT.md) · [`RELIABILITY.md`](RELIABILITY.md) · [`deploy/SLO.md`](deploy/SLO.md)

---

## Page 1 — The 60-second pitch

**PreceptualAI = the regulator-defensible audit-rApp that runs alongside Ericsson EIAP / Nokia MantaRay.**

### Three differentiators

| # | What it is | Where it lives |
|---|---|---|
| 1 | **Counterfactual envelope** — every A1 policy carries chosen + rejected actions with predicted SLA risk at 30 s / 1 min / 5 min, signed and stored. | `src/horizon_ric/evidence/schema.py` |
| 2 | **Hash-chained audit** — SHA-256 chain plus RFC 3161 anchor, 7-year retention, **18 µs verify per record**. | `src/horizon_ric/evidence/{store,rfc3161}.py` |
| 3 | **ITU-R S.1503 EPFD in-loop** — every NTN-touching decision precomputes downlink EPFD against the GSO arc and refuses to emit if the mask is breached. | `src/horizon_ric/data/itu_r.py`, `benchmarks/epfd_10k.json` |

### What we are NOT

- **No production cells.** No tier-1 reference deployment; we augment a prime SMO, we do not replace one (`DEVIL_D_RFP.md:5`).
- **No FIPS 140-3 module today.** Crypto primitives are correct; the validated module is on the roadmap (`DEVIL_D_RFP.md:83`).
- **No 24×7 NOC.** No contractual Sev-1 MTTR. Pilot operations only (`DEVIL_D_RFP.md:69`).

---

## Page 2 — The 7 numbers a hostile architect will demand

| # | Metric | Value | Source |
|---|---|---:|---|
| 1 | Edge decision p99 latency, constrained 2-core Orin-Nano envelope, 10 K steps | **86.5 ms** | `deploy/ORIN_CONSTRAINED_SOAK_PROOF.md:90` |
| 2 | A1 emit success rate, 24-h shadow soak, 17 248 emits | **99.93 %** | `deploy/SOAK_24H_PROOF.md:9` |
| 3 | Audit chain integrity across both 24-h soaks | **2 880 / 2 880** verifies intact (1 440 unconstrained + 1 440 constrained) | `deploy/SOAK_24H_PROOF.md:18` + `deploy/ORIN_CONSTRAINED_SOAK_PROOF.md:46` |
| 4 | Audit chain verify cost per record | **18 µs** | `benchmarks/bench_audit_verify.py`, `BENCHMARK_HEAD_TO_HEAD.md:42` |
| 5 | EPFD violation rate, 10 000 real-orbit Starlink scenarios × 200 sats | **0.19 %** (19 / 10 000) | `benchmarks/epfd_10k.json:21` |
| 6 | OWASP ZAP 2.16 baseline scan against live `/api/v1` + dashboard | **0 Critical / 0 High / 0 Medium** (2 Low, 1 Info) | `deploy/ZAP_SCAN_PROOF.md:11` |
| 7 | Subsystem health smoke test | **57 / 57 green in 3.05 s** | `scripts/lightup_all_subsystems.py`, `DEMO_LIGHTUP.md:5` |

**Honest disclosure on #1**: 86.5 ms was measured on a GB10 aarch64 host with `taskset -c 0-1` (2 cores pinned) — a stricter compute budget than a real Orin Nano. Physical Jetson Orin Nano hardware soak is procurement-blocked, not software-blocked (`GAPS_TO_PILOT.md:230`). Constrained soak A1 success was 99.60 % (Orin bar 99.5 %); unconstrained 24-h soak was 99.93 %.

---

## Page 3 — The 5-minute walkthrough script

Copy-paste, top to bottom. No mocks, no fakes — every step exercises real code paths.

```bash
git clone <repo-url> && cd horizon-ric
.venv/bin/python -m horizon_ric.cli demo --quick
```
**Architect should see**: a single A1 policy emitted with a counterfactual envelope JSON (chosen action + 3 rejected alternatives, each with `sla_risk_30s/1min/5min`).
**Point out**: the rejected alternatives — this is what every other vendor's rApp throws away.

```bash
.venv/bin/python scripts/lightup_all_subsystems.py
```
**Architect should see**: `lightup_all: 57/57 subsystems green, total wall-clock ~3.05 s`.
**Point out**: every public subsystem in `PRD.md` runs a real op (no stubs). This is the canonical smoke test.

```bash
.venv/bin/python scripts/edge_benchmark_arm64.py --steps 1000
```
**Architect should see**: p50 / p95 / p99 latency printed; full 4-layer MLP head + audit-append per decision.
**Point out**: this is the same script that produced the 86.5 ms constrained-Orin number — reproducible on their hardware in seconds.

```bash
.venv/bin/python scripts/run_horizon_rapp.py --once
```
**Architect should see**: one full rApp tick — telemetry ingest → risk head → policy → A1 emit → evidence append → chain verify.
**Point out**: the audit chain's `verify=-1` return value (the index of the first broken link; -1 means intact).

```bash
curl http://localhost:8083/healthz
curl http://localhost:8083/metrics | head -30
```
**Architect should see**: `{"status":"ok"}`, then Prometheus counters for A1 emits, breaker state, audit verifies, EPFD checks.
**Point out**: the Grafana panel at `deploy/grafana/` plots these directly; the SLO bars in `deploy/SLO.md` are wired to these counters.

---

## Page 4 — RFP scorecard summary (link-out)

Bottom line from [`DEVIL_D_RFP.md`](DEVIL_D_RFP.md):

- **38 / 100** on a stock tier-1 RFP scorecard against Ericsson EIAP / Nokia MantaRay (`DEVIL_D_RFP.md:259`).
- **4 / 4 on the differentiator axis** — counterfactual envelope, hash-chained audit, ITU-R S.1503 EPFD in-loop, TS 28.105 model card. Ericsson and Nokia have no public response on any of the four (`DEVIL_D_RFP.md:271`).
- **Posture**: "augment Ericsson, do not replace Ericsson."

Full 15-question breakdown, gap-to-parity costs, and the 6 we cannot fix on a 12-month budget: see `DEVIL_D_RFP.md`.

---

*This packet is hand-out grade. Numbers cite `file:line` against repo SHA `d23f1215156d00b7f16ab0c6934b3342f6f25900`. If a number changes, this document is stale — regenerate from the proof MDs in `deploy/` and `benchmarks/`.*

---

## Page 5 — Demo 4: CFO ROI exhibit (5-minute CFO conversation)

*Added 2026-05-08. Source: `~/.claude/plans/AUDIT_ROI_GAP.md` §4–§5. This is the financial-buyer demo. Use it whenever the architect demos (Demos 1–3 above) are not sufficient — i.e. whenever the spend authorization sits with the CFO, not the CTO.*

### Why this demo exists

Demos 1–3 are architect-facing. They prove the system works. They do **not** translate the wedge into the financial language a Tier-1 CFO uses to approve the line item. Demo 4 takes the audit-chain output from any of Demos 1–3 and projects it onto a Tier-1 P&L view. **Inputs:** operator turnover, claimed AI-RAN annual value, regulatory jurisdiction. **Outputs:** (a) % of claimed value defensible without our chain, (b) % defensible with it, (c) estimated annual fine exposure with vs without the chain, (d) audit headcount delta. Every number sources to `AUDIT_ROI_GAP.md` or a regulator-published fine tier.

### The value-capture table

For a Tier-1 deploying AI-RAN at scale (≥30k macro sites, $0.7–2.4B 5-yr capex), the conservative annual recovery:

| # | Leak | Annual exposure (Tier-1) | PreceptualAI feature that plugs it | $ recovered (Tier-1/yr) |
|--:|---|---:|---|---:|
| L1 | Attribution failure (30–50% of claimed gain undefendable to auditor) | **$300–500M** | Per-decision counterfactual envelope + pinned RNG seed (M1, `policy/counterfactual.py`, `evidence/explanation.py`) | **$150–300M** |
| L2 | Drift erosion (10–25% of original gain disappears within 12 mo) | **$100–250M** | KS + Page-Hinkley drift detectors on input distribution (`continual/drift_detector.py`) | **$60–150M** |
| L3 | Catastrophic regression on bad model promotion (2–4 incidents/yr × $5–20M) | **$10–80M** | SHA-256 chain + RFC 3161 anchor + bit-identical artefact replay (existing + M8) | **$8–60M** |
| L4 | Regulatory clawback — EU AI Act high-risk fine up to 3% global turnover | **$0–1,200M** episodic | TS 28.105 §7.4 model card chain consumable by regulator (M5) | **$50–500M** amortized |
| L5 | Audit headcount — 10–30 internal FTE + Big-4 external | **$5–19M** recurring | Automated evidence pipeline; auditor consumes the chain directly (M5) | **$4–18M** |
| | **Total annual exposure** | **$415M – $2.05B** | | **$272M – $1.03B/yr recovered** |

**License-model implication.** Per-network subscription target **$5–25M/yr** is **0.5–9%** of recovered value — a **10:1 to 200:1 value-to-price ratio**. This is the CFO no-brainer threshold.

### The 5-minute CFO conversation script

*Drop-in voiceover; no engineering jargon; CFO language only — capex preservation, opex reduction, regulatory liability avoidance, board defensibility. Full source: `AUDIT_ROI_GAP.md` §5.*

**[0:00 — Opening, 20 s]**
> "You've signed off on roughly **$1.5 billion of AI-RAN capex over the next 5 years**. Your operator's pitch deck says that buys you a $5–10 billion AI-inference revenue opportunity plus throughput, energy, and footprint gains. I'm not here to argue with the upside. I'm here to talk about the **part of that decision your auditor and your regulator are going to test**, and what it costs you if you can't pass the test."

**[0:20 — The proof gap, 40 s]**
> "When your CTO tells the board 'AI-RAN delivered a 15% throughput lift this quarter,' three questions come back: *Was it actually the AI, or was it the new spectrum band? Is the gain still there, or has it silently degraded? Can you prove to a regulator exactly which decisions the AI made and why?* Today, the answer to all three is **'we have the dashboards.'** Dashboards are not evidence. The AI-RAN Alliance work items, NVIDIA Aerial, Nokia MantaRay, Ericsson IAP — none of them ship the proof apparatus. They ship the AI. The proof apparatus is what we ship."

**[1:00 — The five leaks, 60 s]**
> "Let me put numbers on it.
> **One.** Without counterfactuals, **30 to 50% of your claimed gain can't be defended** when an auditor asks. On a $1B/yr value claim, that's $300–500M of vapor.
> **Two.** Production AI silently drifts. Conservative literature says **10 to 25% of the gain disappears within 12 months** — and your dashboards stay green while it happens. That's another $100–250M.
> **Three.** When a model promotion goes bad — and it will, two to four times a year — without replay, your post-mortem costs three times more, takes three times longer, and doesn't satisfy the SLA-credit dispute from your B2B customer.
> **Four.** EU AI Act, NIS2, parallel state regimes. **One high-risk non-compliance finding is up to 3% of global turnover.** For you, that's about €1.2 billion. **One fine wipes out a full year of AI-RAN ROI.**
> **Five.** Just keeping the lights on for AI governance — internal headcount plus Big-4 audit — runs $5 to $19 million per year **before any fine**.
> Add it up: **$400M to $2B per year of exposure your CFO peers are not pricing in**."

**[2:00 — The fix, 45 s]**
> "We are the proof layer. We sit on top of whatever you've already bought — Aerial, MantaRay, IAP, VIAVI, doesn't matter. For every AI decision your network makes, we emit a tamper-evident, regulator-replayable record: what was decided, what was rejected, why, and the random seed needed to reproduce it bit-for-bit. We detect drift in days. We make a model rollback a one-day forensic exercise instead of a three-week one. We give your regulator a signed chain they can verify themselves. Conservatively, we recover **$270 million to $1 billion per year** of the value your AI-RAN spend would otherwise leak. We charge you about 1% of that. **The math is not subtle.**"

**[2:45 — Close, 15 s]**
> "If your AI-RAN spend is going to clear board scrutiny in the next quarterly review, you need this layer in place **before**, not after, the first regulator inquiry. We can have a pilot running on your existing stack in **8 weeks**. What's the right next conversation to set up?"

### How to run Demo 4 in a meeting

1. **Pre-meeting (10 min).** Customize the value-capture table: replace "$1B claimed value" with the operator's actual published AI-RAN revenue/savings target; replace "€40B turnover" with the operator's actual global revenue. Recompute the L4 row using the operator's jurisdiction (EU AI Act 3%, US state regimes, etc.).
2. **Live (5 min).** Walk the table top-to-bottom; speak the script verbatim. Hand the printed one-pager across the table at the [2:45] close.
3. **Post-meeting (15 min).** Auto-generate a board-ready one-pager from the operator's own audit-chain output (real numbers, not the synthetic $1B example) using `scripts/cfo_one_pager.py` (planned M9 deliverable; until then, manually populate the template in `docs/CFO_ONE_PAGER_TEMPLATE.md`).

### Acceptance criterion

A sales engineer can run Demo 4 with a finance buyer (not architect) in <5 minutes and produce a board-ready one-pager at the end. If the meeting goes longer than 5 minutes on the script alone, return to Demo 1 (counterfactual envelope) — that is the single most CFO-legible architect demo and earns the next meeting.

---
