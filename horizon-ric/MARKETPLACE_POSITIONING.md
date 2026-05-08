# MARKETPLACE_POSITIONING — How PreceptualAI sells against Ericsson EIAP / Nokia MantaRay / VIAVI

**Version:** 0.2.0
**Date:** 2026-05-06
**Audience:** sales engineering, marketing, partner success, customer-facing engineering.
**Companion docs:** `DEVIL_D_RFP.md` (the hostile RFP scorecard this document responds to), `MARKETPLACE_GAPS.md` (the four RFP-medium gaps and the cost to close them), `REFERENCE_CASE_STUDIES.md` (pre-pilot reference scenarios), `OPERATOR_DEPLOYER_DUTIES.md` (EU AI Act deployer-duty inheritance), `PILOT.md` (90-day pilot pack).

This document replaces the implicit "next-generation full-stack RIC" framing with the explicit **augment-don't-replace** framing the procurement team can defend to the CTO. It is the file every sales-engineering presentation, partner pitch, and marketplace listing should align to. It is one page on purpose.

---

## 1. The "we win, they win, we lose" frame

Three columns. Read top to bottom — every line is a rubric item we can be scored on.

| WHERE WE WIN (today)                                                                 | WHERE THEY WIN (incumbents)                                                          | WHERE NEITHER OF US YET WIN (greenfield)                                              |
|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| **Per-policy counterfactual envelope** (paradigm H1) — `evidence/schema.py:67-74` `RejectedAlternative`, 9-cause taxonomy in `evidence/explanation.py:16-56`, tested by `tests/test_emit_guards_counterfactual.py`. No incumbent rApp ships this. | **≥3 live tier-1 deployments ≥12 mo** — Ericsson MasOrange + AT&T, Nokia DOCOMO + du. We have **zero** (`COMPETITIVE_LANDSCAPE.md §4.1`). | **RFC 3161-anchored audit timestamps** — neither PreceptualAI nor any incumbent rApp anchors the SHA-256 chain to an external Roughtime/RFC 3161 TSA. Differentiator on the table. |
| **SHA-256 hash-chain tamper-evident audit** (paradigm H1+H2) — `evidence/store.py:75-79, 106-142`, validated by `tests/test_evidence_store.py`. | **24×7 NOC, MTTR ≤30 min Sev-1** — Ericsson 6-NOC, Nokia Networks Care 24/7. We are best-effort follow-the-sun (`docs/runbooks/oncall.md`). | **Formal mathematical proof of world-model correctness** — neither vendor publishes a bisimulation or contraction-mapping theorem on its compositional WM. Solver 1 is closing this gap (see `THEOREMS.md`). |
| **In-loop ITU-R S.1503 EPFD enforcement** — `planner/physics/epfd.py` Annex 1 Eq. (1), `tests/test_epfd_time_cdf.py` benched at 0.19 s for a 500-sat Walker constellation. No SMO vendor publishes this as an in-loop rApp surface. | **NESAS / TM Forum 3rd-party certifications** — Ericsson FS.16 + 200-SKU FS.13 reports. We have repo-level SBOM + cosign + nist_csf, **self-attested**. | **EU AI Act voluntary high-risk controls** — neither incumbent advertises Articles 9-15 voluntary uptake. We do (`docs/compliance/eu_ai_act.md` §4). |
| **Open weights + TS 28.105 model cards** — `checkpoints/*.md` (sha256+size), `tests/test_ts28105_model_card_emit.py`, SBOM (CycloneDX 1.5) at `deploy/sbom/`. Incumbents ship proprietary binaries an auditor cannot meaningfully inspect. | **BSS/OSS integrations** — Amdocs, BSCS, CBIS, Netcracker connectors live with both Ericsson and Nokia. We have no BSS/OSS bridge today. | **Cross-RAT audit interoperability** — no vendor today exposes a standard external audit-API for hash-chained AI decisions. We can lead the standard. |
| **NTN-native physics** — SGP4 Walker-Δ, S.1428 antenna, P.838 rain, TR 38.811 channel-state, NTN K-offset/RACH timing all in `planner/physics/`. Incumbents treat NTN as a roadmap line item. | **Marketplace listings** — Ericsson rApp Directory + Nokia MantaRay marketplace each have 12+ third-party listings live. Our `marketplace/{oran-sc,ericsson,nokia}/` is **submission stubs**. | **Reproducible-counterfactual standard** — TS 28.105 has no normative section yet on counterfactual envelopes. We are early enough to influence it. |

---

## 2. The pitch sentence

> **PreceptualAI is the audit-first, NTN-native rApp that runs on top of your existing Ericsson EIAP, Nokia MantaRay, or open-source SMO — adding the per-policy counterfactual envelope, SHA-256-chained tamper-evident decision log, and in-loop ITU-R S.1503 EPFD compliance your current rApp catalogue does not ship — without you having to rip and replace the SMO you already trust.**

Three things this sentence does:

1. **Concedes** the SMO-prime competition we cannot win in this RFP cycle. We are not asking the customer to displace Ericsson or Nokia.
2. **Names the three differentiators** D1/D2/D3 from the Devil-D scorecard so the buyer knows what they are paying for.
3. **Reframes the buy** from vendor-replacement to low-risk augment, which is the only commercial path the procurement team can defend to the CTO.

This sentence appears verbatim in `README.md` (tagline area), `PILOT.md` (elevator pitch), every marketplace listing under `marketplace/*/`, and the front of every customer deck.

---

## 3. The 5 hostile RFP questions and our honest answers

These are the five RFP questions that close every tier-1 procurement. We answer them honestly because pretending otherwise gets us caught at exactly the moment that costs the most.

| RFP question (procurement phrasing)                              | Our honest answer                                                                                                                                                                                                                  |
|------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "≥3 live tier-1 deployments?"                                    | **No. We are pre-pilot.** What we offer instead: published reference architecture (`ARCHITECTURE.md`), signed conformance test reports (`docs/conformance/CONFORMANCE.md`), and a **30-day no-cost shadow pilot** in your lab to derisk this. Production references roadmap is 12-18 months. |
| "24×7 NOC, MTTR ≤30 min Sev-1?"                                  | **No. We are best-effort follow-the-sun engineering** (`docs/runbooks/oncall.md`). For pilot Tier-1 we offer **named on-call rotation** (PagerDuty `horizon-ric`, 15 min ack T2). Tier-2 contractual NOC is the v1.0 roadmap; today we recommend integrating through Capgemini RATIO / Tech Mahindra as the prime. |
| "FIPS 140-3 module?"                                             | **No.** We use OpenSSL system-default crypto via the Python `cryptography` library for SHA-256 chaining, mTLS, and Paillier (`evidence/store.py`, `rapp/auth.py`, `trading/auction.py`). FIPS-validated module is on the v1.0 roadmap; on RHEL 9 with FIPS-validated OpenSSL we get partial credit at zero cost. |
| "TM Forum ODA third-party certification?"                        | **No, self-attested via `docs/oda/component.yaml`.** Third-party audit through TM Forum's certification programme is **fundable as part of the pilot** (~$30-80k, 3-4 mo). It is the cheapest cert relative to its scoring weight; we will recommend doing it before the second pilot. |
| "BSS/OSS integration (Amdocs / Ericsson BSCS / Nokia CBIS)?"     | **Out of scope today.** Our integration surface is **A1/R1/O1 with the SMO**; BSS/OSS integration happens at the SMO layer, not the rApp layer. If a trouble-ticket bridge is required by RFP, we will ship a ServiceNow / Amdocs CES webhook for SLA-risk events (~4 wk effort, see `MARKETPLACE_GAPS.md` §BSS/OSS). |

The honesty here is strategic, not defensive. Procurement teams have run into vendors who said "yes" to every question and got caught at PoC. Ours is the only RFP response in their pile that names the gaps and tells them the cost-to-close.

---

## 4. The compensating differentiators

For each "we lose" question above, we lead with a thing the incumbents do not ship. These compensating differentiators are not nice-to-haves; they are the reason a regulator-aware CTO buys an augment-rApp on top of an EIAP they already paid for.

| RFP loss                              | Compensating differentiator we lead with                                                                                                                                                                                                                                                                                                            |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Q1 — no live tier-1 deployments        | **Counterfactual envelope** (D1). Every emitted A1 policy carries the top-K rejected alternatives with a 9-cause structured rejection reason an auditor or regulator can read without an ML degree (`evidence/schema.py:67-74`, `evidence/explanation.py:16-56`, see `docs/COUNTERFACTUAL_USER_GUIDE.md` §4). Ericsson, Nokia, NVIDIA do not ship this. |
| Q3 — no 24×7 contractual NOC          | **Hash-chained decision log** (D2). The audit chain is unforgeable byte-for-byte; an operator who reads the chain post-incident gets a deterministic record nobody — including PreceptualAI engineers — can rewrite. This is a zero-trust audit primitive incumbents do not ship (`evidence/store.py:75-79, 106-142`).                                  |
| Q4 — no FIPS module                   | **In-loop EPFD compliance** (D3). PreceptualAI enforces ITU-R S.1503 EPFD time-CDF inside the planner before A1 emit, saving the operator the cost of a separate spectrum-licensing-compliance tool and turning S.1503 from a post-hoc filing into a pre-emit gate (`planner/physics/epfd.py`, `tests/test_epfd_time_cdf.py`).                          |
| Q9 — no TM Forum ODA 3rd-party audit  | **Open weights + TS 28.105 model cards + SBOM** (D4). The operator can independently reproduce ANY decision from `(state_hash, model_versions, checkpoint sha256)` triple. An incumbent's proprietary binary cannot be reproduced this way; ours can (`checkpoints/*.md`, `tests/test_ts28105_model_card_emit.py`, `deploy/sbom/`).                  |
| Q12 — no BSS/OSS integration          | **NTN-native + EU AI Act voluntary controls.** Our R1/A1/O1 surface is the standard SMO surface; BSS/OSS integration belongs at that layer. We instead invest the equivalent effort in NTN physics and AI Act Art 9-15 controls (`docs/compliance/eu_ai_act.md` §4), which are differentiators incumbents have not yet adopted.                       |

---

## 5. How to use this document

- **Sales engineering:** lead every customer call with the §2 sentence. Walk the §1 table left-to-right. When the customer asks a §3 question, answer with the §3 row + the §4 compensating differentiator.
- **Marketing / web:** the §2 sentence is the homepage hero. The §1 three-column table is the marketplace-listing layout. The §4 differentiators are the four feature blocks below the fold.
- **Partner / channel:** prime integrators (Capgemini RATIO, Tech Mahindra, Wipro, HCL, Tata Consultancy) get §3 to reuse in their bids. They cover Q1 (their references), Q3 (their NOC), Q5 (their security audit credentials); we cover §4 (the differentiators).
- **Customer-facing engineering:** every concession in §3 maps to a `GAPS_TO_PILOT.md` row. Do not invent new concessions in customer meetings; cite this document and link the gap-tracker row.

---

## 6. FIPS readiness

**FIPS readiness:** deployable on FIPS-mode RHEL with a documented crypto-primitive inventory; module-level FIPS 140-3 validation is on the Phase-2 roadmap. See `docs/compliance/fips_readiness.md`.

The short version for a procurement conversation:

- PreceptualAI has **no module-level CMVP certificate today** (we will not claim one we do not have).
- Every cryptographic primitive PreceptualAI uses (TLS 1.3 cipher suites, RS256 JWT signing, SHA-256 audit hash chain, RFC 3161 timestamp verification) is sourced from CPython `ssl` and `cryptography` and **inherits the validation of the host OpenSSL provider** when deployed on FIPS-mode RHEL 9 / Ubuntu 22.04 LTS.
- The full inventory with `file:line` citations and a 6-week path to a defensible "FIPS-Inside" claim (vendoring `pyjwt` over `cryptography`, FIPS-mode CI lane, external crypto review) is in `docs/compliance/fips_readiness.md`.

---

**End of positioning.** Sources: `DEVIL_D_RFP.md` (Devil-D hostile RFP scorecard, 38/100), `COMPETITIVE_LANDSCAPE.md`, `PARADIGMS.md` (paradigm H1 / H2), `evidence/schema.py`, `evidence/explanation.py`, `evidence/store.py`, `planner/physics/epfd.py`, `checkpoints/*.md`, `docs/compliance/eu_ai_act.md`, `docs/runbooks/oncall.md`, `docs/runbooks/customer_escalation.md`, `MARKETPLACE_GAPS.md`, `OPERATOR_DEPLOYER_DUTIES.md`, `REFERENCE_CASE_STUDIES.md`.
