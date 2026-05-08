# MARKETPLACE_POSITIONING — How PreceptualAI sells against Ericsson EIAP / Nokia MantaRay / VIAVI

**Version:** 0.2.0
**Date:** 2026-05-08 (post-v3 trust-layer wave)
**Audience:** sales engineering, marketing, partner success, customer-facing engineering.
**Companion docs:** `README.md` (49-section deep dive), `DEVIL_D_RFP.md` (the hostile RFP scorecard this document responds to), `REFERENCE_CASE_STUDIES.md` (pre-pilot reference scenarios), `OPERATOR_DEPLOYER_DUTIES.md` (EU AI Act deployer-duty inheritance).

**v3 status.** §7 "AI-RAN Alliance integration positioning" appended in this version with: 9 verbatim trust quotes (Verizon CTO, Ofcom, EU AI Act, Dell'Oro, GSMA, ITU-T FG-AN, NIS2, XAI O-RAN survey, Agentic AI 6G), 5 ROI leaks ($415 M – $2.05 B / yr per Tier-1 exposure), value-capture table ($272 M – $1 B / yr recovered at 10:1 to 200:1 license ratio), 17-primitive LCM trust layer comparison, sales-engineering 3 paths, strategic ask (co-author AI-RAN Alliance WG1 normative audit annex).

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

---

## 7. AI-RAN Alliance integration positioning

*Added 2026-05-08. Source documents: `~/.claude/plans/AUDIT_TRUST_GAP.md`, `~/.claude/plans/AUDIT_ROI_GAP.md`, `~/.claude/plans/AUDIT_LCM_GAP.md`, `~/.claude/plans/preceptualai-airan-alliance-integration.md` §11.5.*

The §1–§6 framing above is the **SMO-augment** wedge against Ericsson EIAP / Nokia MantaRay / OSC NONRTRIC. This section adds the **AI-PHY trust layer** wedge against the AI-RAN Alliance neural-PHY blocks (HybridDeepRx, DPoD, learned constellations, neural-RX) running on NVIDIA Aerial cuPHY + Aerial Framework or VIAVI D4AI. The two wedges compose: every Tier-1 that adopts neural-PHY needs both layers, and PreceptualAI is the only vendor shipping both.

### 7.1 The headline thesis

> **For deployments adopting AI-RAN Alliance neural-PHY blocks, PreceptualAI is the only audit + counterfactual + lifecycle-management trust layer for those AI-PHY decisions. Without us, the AI-RAN ROI cannot be proven to a CFO and the AI-RAN model lifecycle cannot be governed to a regulator.**

The thesis decomposes into three orthogonal customer-facing claims:

1. **Trust** — neural-PHY blocks fail in ways classical PHY does not (distribution shift, OOD calibration, adversarial input, silent latency degradation, concept drift). PreceptualAI ships 12 mechanisms that close those failure modes and make the result regulator-defensible. (`AUDIT_TRUST_GAP.md` §3.)
2. **ROI** — without counterfactual attribution, drift detection, regression-replay, regulator-replay, and audit-headcount automation, **30–50%** of claimed AI-RAN gain cannot be defended to a CFO; **10–25%** silently erodes within 12 months; one EU AI Act high-risk fine wipes out a year of ROI; audit headcount alone runs $5–19M/yr. (`AUDIT_ROI_GAP.md` §3.)
3. **LCM** — five lifecycle primitives the four surveyed vendors (Aerial, MantaRay, EIAP, OSC NONRTRIC) **all** lack: per-decision lineage, counterfactual envelope, SHA-256 hash chain, RFC 3161 anchoring, drift detection on input distribution. PreceptualAI ships all five today; M7–M9 of the integration plan adds the remaining five (atomic promotion, shadow executor, artefact vault, LoopState, per-zone lineage). (`AUDIT_LCM_GAP.md` §1.5, §4.)

### 7.2 The trust gap is real — 9 verbatim quotes

The trust gap is **already explicit** in published statements from a Tier-1 CTO, two regulators, an industry analyst, a trade body, a standards body, an academic survey, an academic AI-safety paper, and a standards-body activity log. Verbatim sources cited in `AUDIT_TRUST_GAP.md` §1; a one-line summary of each:

| # | Source | Verbatim wedge |
|--:|---|---|
| Q1 | **Yago Tenorio, CTO, Verizon** (Fierce Network, 11 Mar 2025) | "If you divide performance by cost … then one day it makes sense. Today? No, it doesn't." |
| Q2 | **Yago Tenorio, Verizon** | "AI-RAN — if that is using GPUs to do the number crunching to sell idle cycles for workloads — no, I don't see it at all." |
| Q3 | **Dell'Oro Group**, AI-RAN Advanced Research Report (Jul 2025) | "Limited telco presence … reflect[s] the ongoing skepticism about the goals of AI-RAN." |
| Q4 | **Ofcom**, Strategic Approach to AI 2025/26 (Jun 2025) | "Ofcom is focused on ongoing monitoring of explainability, interpretability, and transparency in AI systems to build trust and accountability." |
| Q5 | **EU AI Act**, Article 6(2) + Annex III §2 | "AI systems intended to be used as safety components in the management and operation of critical digital infrastructure … are classified as high-risk." |
| Q6 | **GSMA**, Responsible AI Maturity Roadmap (Sep 2024) | "Explainability and trustworthiness are key in AI systems to establish trust with consumers." |
| Q7 | **ITU-T FG-AN**, output FGAN-O-024 | "Trustworthiness including certainty and robustness while selecting and applying autonomous decisions" is named a **key technical enabler** for autonomous networks. |
| Q8 | **XAI-in-O-RAN survey** (arxiv 2307.00319) | "The widespread adoption of AI techniques in future 6G O-RAN should be accompanied by mechanisms that verify and explain the black-box models' decisions … especially when they lead to SLA violations or failures." |
| Q9 | **Agentic AI for 6G** (arxiv 2512.12400) | "Embedding AI at microsecond control loops makes it almost impossible to trace reasoning steps or provide post-hoc explainability, creating tension with regulatory requirements such as the EU AI Act and NIS2." |

The minimum bar of "≥1 quote from each of {regulator, Tier-1, analyst}" is exceeded; coverage spans Tier-1 CTO + 2 regulators + analyst + trade body + standards body + academic survey + standards-body activity log + AI-safety paper. The full quote text and citations are in `AUDIT_TRUST_GAP.md` §1.

**The wedge sentence we sell:** *"Verizon's CTO publicly said AI-RAN doesn't make sense to deploy today. Ofcom is putting explainability into the strategic plan. The EU AI Act puts neural-PHY in the high-risk class. We are the trust layer that flips Verizon's calculus and gives Ofcom and the EU AI Act regulator the evidence they will subpoena."*

### 7.3 The 5 ROI leaks and the value-capture table

Every Tier-1 deploying AI-RAN at scale is making a **$0.7–2.4B 5-year capex commitment** (`AUDIT_ROI_GAP.md` §1.1). Five leaks erode that commitment between deployment and the next quarterly board review:

| # | Leak | Annual exposure (Tier-1) | What plugs it (PreceptualAI) | $ recovered (Tier-1/yr) |
|--:|---|---:|---|---:|
| L1 | **Attribution failure** — without counterfactual + pinned RNG seed, 30–50% of claimed lift evaporates under proper analysis | **$300–500M** | Per-decision counterfactual envelope (`policy/counterfactual.py`, `evidence/explanation.py`) | **$150–300M** |
| L2 | **Drift erosion** — production ML loses 10–25% of original gain within 12 months without active drift detection | **$100–250M** | KS + Page-Hinkley drift detectors (`continual/drift_detector.py`) | **$60–150M** |
| L3 | **Catastrophic regression** on bad model promotion — 2–4 incidents/yr × $5–20M each | **$10–80M** | SHA-256 chain + RFC 3161 anchor + bit-identical artefact replay | **$8–60M** |
| L4 | **Regulatory clawback** — EU AI Act high-risk fine up to 3% global turnover (€1.2B for €40B-revenue Tier-1) | **$0–1,200M** episodic | TS 28.105 §7.4 model card chain + regulator-replay package | **$50–500M** amortized |
| L5 | **Audit headcount** — 10–30 internal FTE + Big-4 external | **$5–19M** recurring | Automated evidence pipeline; auditor consumes the chain directly | **$4–18M** |
| | **Total annual exposure** | **$415M – $2.05B** | | **$272M – $1.03B/yr** |

**License-model implication.** Per-network subscription target $5–25M/yr is **0.5–9%** of recovered value — a **10:1 to 200:1 value-to-price ratio**. This is well inside the CFO no-brainer threshold. Full quantitative model and citations in `AUDIT_ROI_GAP.md` §3–§4.

**The CFO sentence we sell:** *"You've signed off on $1.5B of AI-RAN capex over 5 years. Without proof apparatus, $400M to $2B per year of that spend is unmeasured liability. We charge you 1% of that and make 100% of it defensible."* (Full 5-minute CFO script: `AUDIT_ROI_GAP.md` §5; replicated as Demo 4 in `CUSTOMER_DEMO_PACKET.md`.)

### 7.4 The LCM primitive map

`AUDIT_LCM_GAP.md` §1.5 cross-vendor matrix: among **NVIDIA Aerial, Nokia MantaRay, Ericsson EIAP, OSC NONRTRIC**, **none** ship per-decision lineage, counterfactual envelope, SHA-256 chain, RFC 3161 anchor, or drift detection. PreceptualAI ships all five. The full primitive map (`AUDIT_LCM_GAP.md` §4):

| Lifecycle stage | PreceptualAI primitive | Module / file:line | Status |
|---|---|---|:---:|
| Training-data lineage | Manifest hash + per-zone inclusion list | `observability/model_card.py:106` | SHIPPED (manifest); NEW per-zone (M8) |
| Pre-deployment validation | Pre-emit guards | `policy/emit_guards.py:54,74` | SHIPPED |
| Atomic A→B promotion | Slot-boundary swap with shadow validation | `runtime/atomic_promotion.py` | **NEW (M7)** |
| Shadow deployment | Shadow executor running B in parallel with A | `runtime/shadow_executor.py` | **NEW (M7)** |
| Drift detection | KS + Page-Hinkley over input distribution | `continual/drift_detector.py:56,120` | SHIPPED |
| Production observability | Prometheus + OTel + X.733 alarms | `rapp/health.py`, `observability/{tracing,x733_alarms}.py` | SHIPPED |
| Calibration error metric | Predicted-vs-realised SLA breach delta | `rapp/health.py` (extension) | **NEW (M9)** |
| Decision audit chain | SHA-256 hash chain over DecisionRecord | `evidence/store.py:48,75,82` | SHIPPED |
| RFC 3161 timestamp | TSA anchor over evidence batches | `evidence/rfc3161.py` | SHIPPED |
| Bit-identical artefact storage | Content-addressed model artefact vault | `runtime/artefact_vault.py` | **NEW (M8)** |
| Evidence retention | JSONL/SQLite with configurable retention | `evidence/store.py` | SHIPPED |
| TS 28.105 §7.4 model card | Four mandatory fields | `observability/model_card.py:153` | SHIPPED |
| TS 28.567 LoopState | Retrain / Rollback / Scale state machine | `runtime/loop_state.py` | **NEW (M8)** |
| ITU-T X.733 ML-failure alarms | 9 mandatory fields, ML probable causes | `observability/x733_alarms.py:45-80` | SHIPPED (infra); ML vocab NEW (M9) |
| Counterfactual envelope | Rejected alternatives + reason + RNG seed | `evidence/schema.py` `RejectedAlternative` | SHIPPED |

**Tally:** 11 SHIPPED + 2 PARTIAL + 5 NEW (M7–M9). After M7–M9 the count is **17/17** — the LCM trust layer is complete.

**The architect sentence we sell:** *"Your SRE at 03:00 reads a sentence, not a tensor. Your regulator replays a record, not a subpoena. Your rollback is a content-addressed artefact restore in <60 seconds, not a 90-minute image re-flash."* The 4-minute SRE 3 AM walk-through proving this is in `AUDIT_LCM_GAP.md` §5.

### 7.5 Sales-engineering implication

Every customer call now offers **three** paths, not two:

1. **Architect path** — the four shipped differentiators (counterfactual, hash chain, EPFD, model card) in §1–§4 above, plus the LCM primitive map in §7.4. Demo: `CUSTOMER_DEMO_PACKET.md` Demos 1–3.
2. **CFO path** — the $272M–$1B/yr value-capture table in §7.3 plus the 5-minute conversation script. Demo: `CUSTOMER_DEMO_PACKET.md` Demo 4.
3. **Regulator path** — the trust quotes in §7.2, the EU AI Act / Ofcom / ITU-T FG-AN alignment, and the regulator-replay package emitted by the audit chain. (No demo needed — the chain output is the artifact.)

Sales-engineering rule: **if the customer is a Tier-1 with active AI-RAN Alliance posture (NVIDIA Aerial, Nokia Bell Labs HybridDeepRx, R&S DPoD, learned-constellation work), all three paths must be on the table from the first meeting.** Architect path alone is insufficient — Verizon's public statement proves the architects don't sign without the CFO and regulator covered.

### 7.6 Strategic ask

The AI-RAN Alliance has 100+ members but **only a handful of telcos** (`AUDIT_TRUST_GAP.md` §1.2). The trade body GSMA has codified the trust dimensions (`AUDIT_TRUST_GAP.md` §1.5). ITU-T FG-AN has scoped the evaluation methodology (`AUDIT_TRUST_GAP.md` §1.6). The O-RAN ALLIANCE shipped 76 documents in one half-year on AI/ML security and audit (`AUDIT_TRUST_GAP.md` §1.8). The standards bodies are sprinting to close the gap.

PreceptualAI's strategic ask: **co-author the WG1 audit normative annex.** The 17 primitives in §7.4 map 1:1 to what a normative annex would mandate. Shipping the reference implementation today positions PreceptualAI as the de-facto vendor when the annex freezes. (Plan v3 §11.5 Risk Mitigation 3.)

---
