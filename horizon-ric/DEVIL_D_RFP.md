# DEVIL_D_RFP — Hostile RFP Evaluation: PreceptualAI vs Ericsson EIAP / Nokia MantaRay / VIAVI

> **Reviewer profile.** 25 years buyer-side at Vodafone, Telefonica, Verizon. I have PreceptualAI's marketplace pitch, Ericsson EIAP's pitch, Nokia MantaRay's pitch, and VIAVI's pitch on my desk. My job is to find every reason to recommend the incumbent and minimise procurement risk to my CTO.
>
> **Verdict up top.** PreceptualAI is a credible **R&D / regulatory-augment rApp**, not a credible **prime SMO**. On a standard 100-point tier-1 telco RFP scorecard, PreceptualAI loses ~62 of 100 points to Ericsson EIAP. It can win three categories outright (auditability, NTN physics, EPFD compliance) but cannot prime a tier-1 procurement. The right purchase posture is "augment Ericsson, do not replace Ericsson."
>
> Generated 2026-05-06; refreshed 2026-05-08 with v3 trust-layer wave evidence.

> **v3 refresh (2026-05-08).** The verdict above stands — PreceptualAI is still an audit-augment rApp, not a prime SMO. What changed: the trust-layer differentiator is now backed by **17/17 LCM primitives shipped with 189/189 fast-pack tests green**. The four winnable categories (auditability, NTN physics, EPFD compliance, AI-PHY model card lineage) now score 4/4 against every incumbent — see [`README.md`](README.md) §7 (the four shipped differentiators) and §8 (the 17-primitive LCM trust layer with vendor-by-vendor matrix). The ROI leak quantification ($415 M – $2.05 B / yr per Tier-1 exposure; $272 M – $1 B / yr recovered at 10:1 to 200:1 license ratio) gives the augment posture a defensible CFO conversation that the original 100-point scorecard does not capture.

---

## Scorecard summary

| RFP | Question (procurement phrasing) | Max pts | PreceptualAI | Lost |
|-----|----------------------------------|---------|------------|------|
| Q1  | Live tier-1 deployments ≥3 ×12mo | 10      | 0          | 10   |
| Q2  | A1+R1+O1+E2 OSC J-release        | 8       | 5          | 3    |
| Q3  | 24×7 NOC, MTTR ≤30 min Sev-1     | 8       | 0          | 8    |
| Q4  | FIPS 140-3 validated module      | 5       | 0          | 5    |
| Q5  | GSMA NESAS certification         | 5       | 1          | 4    |
| Q6  | Multi-vendor xApp interop report | 6       | 1          | 5    |
| Q7  | 5-yr price model (1k → 10k cells)| 5       | 0          | 5    |
| Q8  | 3+ named, callable references    | 6       | 0          | 6    |
| Q9  | TM Forum ODA third-party audit   | 5       | 1          | 4    |
| Q10 | 18-mo contracted roadmap         | 4       | 1          | 3    |
| Q11 | Localisation (non-English UI)    | 3       | 0          | 3    |
| Q12 | BSS/OSS integration (Amdocs etc.)| 5       | 0          | 5    |
| Q13 | Vendor-neutral + proprietary SMO | 5       | 2          | 3    |
| Q14 | Largest production deployment    | 5       | 0          | 5    |
| Q15 | Largest training corpus (hours)  | 5       | 1          | 4    |
| **Differentiators we score on:** | | | | |
| D1  | Per-policy counterfactual envelope | 4    | 4          | 0    |
| D2  | Tamper-evident SHA-256 audit chain | 4    | 4          | 0    |
| D3  | ITU-R S.1503 EPFD in-loop          | 4    | 4          | 0    |
| D4  | Open weights + TS 28.105 model card| 3    | 3          | 0    |
| **Total** | | **100** | **38** | **62** |

**PreceptualAI scores 38 / 100.** Ericsson EIAP would be expected to score 78–84 / 100 on the same rubric. Nokia MantaRay 75–80. VIAVI does not bid as a prime; it bids as a test instrument.

---

## Q1 — "Vendor must demonstrate live deployment in ≥3 production tier-1 networks for ≥12 months."

1. **RFP question.** *Provide evidence of at least three (3) production deployments in tier-1 mobile network operators (≥10 M subscribers each) that have been operating for a minimum of twelve (12) consecutive months as of RFP issue date. Each reference must include operator name, deployment scope, in-service date, and current status.*

2. **PreceptualAI's response (best case).** Zero production deployments. We are pre-pilot. `COMPETITIVE_LANDSCAPE.md §4.1` states "we have **zero** production cells." `PILOT.md` offers a 30/90/180-day Tier 1/2/3 pilot path, not production references. `GAPS_TO_PILOT.md` is the gap tracker.

3. **Why it loses to incumbent.** Ericsson cites MasOrange (2025 press release) and AT&T (2025 case study) — both live, both >12 months when the RFP closes. Nokia cites NTT DOCOMO MantaRay SON (Nov 2025) and du autonomous slicing (Dec 2025). Ericsson sales engineer's slide reads: *"50+ tier-1 deployments globally; PreceptualAI: 0."* That's a hard zero on a binary requirement.

4. **Score impact.** **−10 / 10.** This is a knockout question on most tier-1 RFPs. Procurement will mark PreceptualAI non-responsive on this category.

5. **Parity fix.** Land three lighthouse pilots, contract an integrator (Capgemini RATIO, Tech Mahindra, Wipro) to act as prime so they can list the operator references, and accept that 12 months of clock cannot be compressed. Realistic horizon: 18–24 months from today.

---

## Q2 — "Vendor must ship A1 + R1 + O1 + E2 conformance against the OSC J-release reference."

1. **RFP question.** *List interface conformance against the O-RAN ALLIANCE J-release specifications and the OSC reference implementation. Required: A1AP-v05.00, R1AP-v06.00, O1 (TS 28.541 NRM, NETCONF), and E2AP. Provide test reports and CI gates.*

2. **PreceptualAI's response (best case).** A1 + R1 + O1 are wired and tested: `src/horizon_ric/rapp/a1_adapter.py:190` (PUT/GET/DELETE + EI jobs), `r1_adapter.py:99` (register/deregister), `o1_adapter.py:103/145/165` (NETCONF connect/get/edit-config). CI gates in `tests/test_a1_osc_dialect.py`, `tests/test_o1_live.py`, `docs/conformance/CONFORMANCE.md`. **E2 is not in scope** — PreceptualAI is a Non-RT rApp; E2 is the Near-RT RIC's xApp interface and lives in the sister Preceptual UHCI product (per the closing note in `README.md`).

3. **Why it loses to incumbent.** Ericsson EIAP markets the full A1+R1+O1+E2 stack across EIAP+EIC. The buyer's rubric does not care that E2 is architecturally a different RIC layer — the question reads "must ship", and EIAP does. Ericsson's slide reads: *"One vendor, full O-RAN stack. PreceptualAI ships 3 of 4."* Architecturally we are right; on the rubric we still lose 3 points.

4. **Score impact.** **−3 / 8.** Partial credit for what we do ship; lost on E2.

5. **Parity fix.** Either (a) bundle the Preceptual UHCI Near-RT xApp into the same procurement line so the joint bid covers E2, or (b) explicitly call out that the RFP question is mis-phrased (Non-RT vs Near-RT are distinct RIC tiers per WG2) and request a scoring amendment. Option (a) is the only one procurement actually accepts.

---

## Q3 — "24×7 NOC coverage with contractual MTTR ≤ 30 minutes for Sev-1."

1. **RFP question.** *Vendor shall provide 24×7×365 NOC coverage with named on-shift engineers in at least three time zones, contractual Mean Time To Restore of ≤30 minutes for Sev-1 incidents, and quarterly availability reporting against a 99.99 % SLA.*

2. **PreceptualAI's response (best case).** Best-effort follow-the-sun engineering team. `RELIABILITY.md` documents internal MTTR target of **5 minutes for the rApp itself** and 99.999 % availability of the daemon — but that is component reliability, not a contracted human-NOC SLA. There is no named NOC, no Sev-1 escalation runbook signed by a director, no quarterly SLA report template. `deploy/RUNBOOK.md` is an engineer's operational handbook, not a customer-facing SLA contract.

3. **Why it loses to incumbent.** Ericsson runs named NOCs in 6 time zones and underwrites the SLA against penalties. Nokia has Networks Care 24/7 with contractual credits. Ericsson sales engineer's slide reads: *"Six named NOCs, contractual SLA credits, 25 years of carrier-grade support. PreceptualAI: best effort by an engineering team."*

4. **Score impact.** **−8 / 8.** This is a binary "yes/no" that procurement scores 0 if not contractually backed.

5. **Parity fix.** Contract 24×7 NOC services through Capgemini, Tech Mahindra, or HCL. Stamp a real SLA with credit schedules. Cost: ~$1.5–3M/yr operating expense. Without this, no tier-1 buys direct from us.

---

## Q4 — "FIPS 140-3 validated cryptographic module."

1. **RFP question.** *All cryptographic operations (hashing, signing, key exchange, AES) shall be performed via a FIPS 140-3 validated cryptographic module. Provide CMVP certificate number and module boundary documentation.*

2. **PreceptualAI's response (best case).** We use system OpenSSL plus `cryptography` (Python) for SHA-256 chaining in `evidence/store.py`, mTLS in `rapp/auth.py`, and Paillier in `trading/auction.py`. We are **not** module-level FIPS 140-3 validated. `PILOT.md §Regulatory officer` already concedes: *"DON'T volunteer: that we have no FIPS-validated crypto module today."* We can run on a FIPS-validated OpenSSL build (RHEL 9) but we have no CMVP certificate of our own.

3. **Why it loses to incumbent.** Ericsson ships their cryptographic libraries with active CMVP certificates and module-boundary documentation. Government and defence-adjacent buyers (Verizon Public Sector, BT Defence, Telefonica gov contracts) treat this as a hard gate. Ericsson slide: *"FIPS 140-3 Level 1 module #4xxx, valid through 2028."*

4. **Score impact.** **−5 / 5.** Hard binary. Federal-adjacent procurements drop us at this question.

5. **Parity fix.** Either (a) re-platform onto a CMVP-validated module (BoringSSL FIPS, OpenSSL 3.x FIPS provider) and self-attest module boundary, or (b) acquire a CMVP certificate ($150–400k, ~12 months). Option (a) gets us partial credit at zero cost; option (b) gets us full credit.

---

## Q5 — "GSMA NESAS certification."

1. **RFP question.** *Vendor shall provide evidence of GSMA Network Equipment Security Assurance Scheme (NESAS) certification for all in-scope products, with development-process audit (FS.16) and product-evaluation report (FS.13/14) issued by a GSMA-accredited test laboratory within the last 24 months.*

2. **PreceptualAI's response (best case).** We have **repo-level conformance**: SBOM (CycloneDX 1.5) at `deploy/sbom/`, cosign signatures at `deploy/cosign/`, OWASP ZAP scan in `deploy/zap_report.json`, `docs/compliance/nist_csf.md`, signed images in CI. We have **no third-party NESAS audit**. `COMPETITIVE_LANDSCAPE.md §4.4` admits: *"GSMA-NESAS, TM Forum Open API certifications, customer references, completed plugfests on operator labs — all built over years. We have repository-level conformance claims; we have not yet been independently certified."*

3. **Why it loses to incumbent.** Ericsson and Nokia have NESAS-audited dev processes covering hundreds of products and refresh annually. Ericsson slide: *"NESAS FS.16 audit refreshed 2025; FS.13 product reports for 200+ SKUs. PreceptualAI: self-attested only."*

4. **Score impact.** **−4 / 5.** One point credit for documented internal security hygiene.

5. **Parity fix.** Engage an accredited NESAS test lab (Atsec, Riscure, NCC Group). Cost ~$200–500k. Timeline ~6–9 months. Recommend doing this in parallel with first pilot so the audit closes before procurement on pilot #2.

---

## Q6 — "Vendor must provide a multi-vendor xApp interoperability test report."

1. **RFP question.** *Provide a third-party multi-vendor interoperability test report demonstrating successful deployment of at least three (3) third-party xApps from at least two (2) other vendors on your platform, including E2 service models KPM v3, RC, and CCC.*

2. **PreceptualAI's response (best case).** We have A1 multi-vendor *dialect* tests (`osc`, `eiap`, `mantaray`, `legacy` per `docs/SMO_INTEGRATION.md`) which prove our rApp PUTs policies into multiple vendor SMOs. We do **not** have an xApp catalogue, an xApp at all (we are Non-RT rApp), or a third-party xApp interop report. The marketplace listings under `marketplace/{oran-sc,ericsson,nokia}/` are **submission stubs**, not accepted listings — `marketplace/README.md` admits the partner-portal fields are unfilled.

3. **Why it loses to incumbent.** Ericsson's rApp Directory has multiple third-party rApps live (Capgemini, AirHop, NetCracker). Nokia's MantaRay rApp marketplace has 12+ partners. Ericsson slide: *"15 third-party rApps in production interop, joint VIAVI TeraVM RIC Test reports on file. PreceptualAI: zero third-party apps tested."*

4. **Score impact.** **−5 / 6.** One point for our A1 dialect breadth.

5. **Parity fix.** Get PreceptualAI accepted into the O-RAN-SC rApp catalogue, the Ericsson rApp Directory, and the TIP Exchange. Engage VIAVI TeraVM RIC Test for a joint report (they sell exactly this service). Cost: ~$80–150k for the test campaign.

---

## Q7 — "5-year price/cost model for 1000 cell sites with growth to 10,000."

1. **RFP question.** *Provide a 5-year fully-loaded TCO model for an initial deployment of 1,000 cell sites, with linear scale to 10,000 cell sites by year 5. Include licence, maintenance, support, professional services, hardware, and uplift assumptions. Volume-discount tiers required.*

2. **PreceptualAI's response (best case).** No published pricing model. `PILOT.md` mentions "fixed-fee Tier 1 and Tier 2; success-fee component on Tier 3" but no $ figures, no per-cell unit price, no volume tiers, no 5-year curve. We cannot answer this question without inventing numbers, and inventing them is exactly what procurement will catch us on.

3. **Why it loses to incumbent.** Ericsson and Nokia have published reference price books, volume-discount waterfalls, and 5-year TCO templates that BAU procurement teams have benchmarked across deals. Ericsson slide: *"Reference price $X per cell-site-year, 12 % volume discount @ 5k cells, capex/opex split documented. PreceptualAI: TBD."*

4. **Score impact.** **−5 / 5.** Procurement cannot evaluate "TBD" against a number.

5. **Parity fix.** Build a defensible per-cell-per-year list price with a documented basis of estimate, three discount tiers (1k / 5k / 10k), a 12-month support uplift, and a professional-services day-rate. Even with very wide bands this is better than no answer. 2 weeks of CFO+sales work.

---

## Q8 — "Reference customer call list (3+ named customers willing to be referenced)."

1. **RFP question.** *Provide a list of three (3) reference customers, with named contact (title VP-Network or above), their email/phone, and written consent to be contacted during the bid evaluation window. References must be currently in production, not pilot.*

2. **PreceptualAI's response (best case).** **Zero references.** No production customers exist (see Q1). Pilot tier referenceability is not contracted in `PILOT.md` and the IP/data clauses there are deliberately customer-friendly, not vendor-friendly for marketing.

3. **Why it loses to incumbent.** Ericsson has a directory of >100 reference VPs across global operators. Nokia has 50+. Ericsson slide: *"References available in your region: <CTO of operator A>, <VP Network of operator B>, <Director RAN of operator C>. PreceptualAI: no references."*

4. **Score impact.** **−6 / 6.** Hard binary.

5. **Parity fix.** Same path as Q1. Land pilots, negotiate referenceability into Tier 2/3 commercial terms, build the reference list one logo at a time. Realistic horizon: 12–18 months for first 3 referenceable logos.

---

## Q9 — "TM Forum ODA conformance certification (third-party audited, not self-attested)."

1. **RFP question.** *Provide TM Forum Open Digital Architecture (ODA) Component Conformance Certification, third-party audited (not self-attested), with the ODA Component Specification version, certificate number, and audit date.*

2. **PreceptualAI's response (best case).** We have a **self-attested ODA component manifest** at `docs/oda/component.yaml`. There is no third-party TM Forum certification on file. `COMPETITIVE_LANDSCAPE.md §4.4` is honest about this gap.

3. **Why it loses to incumbent.** Ericsson and Nokia have TM Forum audited certifications across dozens of components, refreshed annually. Ericsson slide: *"15 TM Forum-certified ODA components, audited by <named auditor>. PreceptualAI: self-attested manifest."*

4. **Score impact.** **−4 / 5.** One point for having authored the manifest.

5. **Parity fix.** Engage TM Forum's certification programme. Cost ~$30–80k per component. Timeline ~3–4 months. This one is genuinely cheap relative to its scoring weight; do it before the next major bid.

---

## Q10 — "Roadmap for next 18 months with named milestones and contractual delivery."

1. **RFP question.** *Provide an 18-month product roadmap with named milestones, target dates, contractual delivery commitments, and SLA-credit terms for each missed milestone.*

2. **PreceptualAI's response (best case).** `PILOT.md` and `PILOT_NEXT_WEEK.md` contain a roadmap-style narrative (Tier 1/2/3 deliverables, demo run-of-show, credibility-closer list in `REVIEW.md`). It is **not contracted**. There are no SLA-credit terms tied to roadmap slips. `README.md` "What's NOT in main yet" is honest gap-tracking, not a delivery contract.

3. **Why it loses to incumbent.** Ericsson and Nokia issue contractually-binding roadmap commitments backed by credits. Ericsson slide: *"Roadmap-on-letterhead with 12 named features, dates, and 5 % credits per slipped quarter. PreceptualAI: README aspirations."*

4. **Score impact.** **−3 / 4.** One point for clarity of internal planning artifacts; lost on contractuality.

5. **Parity fix.** Pick 6 high-confidence milestones from `PILOT.md` and the `REVIEW.md` credibility-closers, attach dates we can defend, attach modest credits (1–3 %), put it on letterhead. Customers do not need 12 contracted milestones; 6 well-chosen ones are enough to score well.

---

## Q11 — "Localization support for non-English languages (operator dashboards)."

1. **RFP question.** *Operator dashboards must support at minimum five (5) of: Spanish, Portuguese, French, German, Italian, Mandarin, Japanese, Arabic, Hindi, Russian. Translation must be professionally reviewed; pseudo-translation or machine-only is not acceptable.*

2. **PreceptualAI's response (best case).** `frontend/README.md` describes a Next.js 14 + TypeScript + shadcn/ui dashboard. It is **English-only**. There is no `next-intl`, no translation files, no locale selector, no translation memory, no review process. This is a small product issue but a binary RFP line item.

3. **Why it loses to incumbent.** Ericsson and Nokia ship 10+ locales reviewed by in-country teams. Ericsson slide: *"Dashboards in 14 languages, professionally reviewed annually. PreceptualAI: English."*

4. **Score impact.** **−3 / 3.** Binary; small but no partial credit.

5. **Parity fix.** Add `next-intl`, externalise strings, ship 5 locales. ~3 weeks for a developer + ~$5–15k professional translation. Cheap relative to score recovered.

---

## Q12 — "Demonstrated integration with the operator's BSS/OSS (Amdocs / Ericsson BSCS / Nokia OSS)."

1. **RFP question.** *Demonstrate integration with the operator's chosen BSS/OSS — Amdocs CES, Ericsson BSCS iX, Nokia CBIS, Netcracker, or equivalent. Include billing-event flow, customer-360 enrichment, and trouble-ticket lifecycle.*

2. **PreceptualAI's response (best case).** **No BSS/OSS integration.** PreceptualAI integrates with the SMO (A1/O1/R1), with NVIDIA Aerial telemetry, with Space-Track and CelesTrak, with ITU-R P-series — but not with Amdocs, BSCS, CBIS, or Netcracker. The dashboard is rApp-internal; there is no billing or trouble-ticket bridge.

3. **Why it loses to incumbent.** Ericsson natively integrates with BSCS (in-house product) and partners with Amdocs and Netcracker. Nokia same with CBIS. Ericsson slide: *"Pre-built BSS/OSS connectors for Amdocs, Netcracker, and BSCS, certified by both vendors. PreceptualAI: not integrated."*

4. **Score impact.** **−5 / 5.** Binary.

5. **Parity fix.** Strictly speaking, Non-RT RIC rApps do not normally cross into BSS/OSS — but the rubric counts it. Build a webhook adapter for ServiceNow / Amdocs Customer Experience Suite trouble-ticket creation when our SLA-risk head fires. ~4 weeks. Document the boundary clearly so we are not over-promising customer-360.

---

## Q13 — "Vendor must support both vendor-neutral SMO (OSC) and proprietary SMO (Ericsson EIC, Nokia MantaRay)."

1. **RFP question.** *Vendor must demonstrate live operation against (a) the open-source OSC NONRTRIC reference SMO, AND (b) at least one proprietary SMO (Ericsson EIC, Nokia MantaRay). Provide test logs from each.*

2. **PreceptualAI's response (best case).** We have **A1 dialects** for `osc`, `eiap`, `mantaray`, and `legacy` (`docs/SMO_INTEGRATION.md`). We have unit tests on each dialect's URL/body shape (`tests/test_a1_osc_dialect.py`). We have **live integration only with OSC NONRTRIC** via `deploy/docker-compose.osc-nonrtric.yml` and `deploy/OSC_NONRTRIC_PROOF.md`. We do **not** have live integration logs against an actual Ericsson EIAP or Nokia MantaRay sandbox — the dialect tests run against mocks of the documented URL shapes.

3. **Why it loses to incumbent.** Ericsson EIAP self-tests against EIC trivially (same vendor); Nokia MantaRay self-tests against itself. They each get full credit on the proprietary leg. Ericsson slide: *"Native EIAP↔EIC, live OSC NONRTRIC integration in 2024 plugfest. PreceptualAI: dialect adapters tested against mocks."*

4. **Score impact.** **−3 / 5.** Two points credit for the OSC live integration.

5. **Parity fix.** Apply for an Ericsson EIAP partner account (`marketplace/ericsson/listing.yaml` already drafted), get sandbox access, run a live emit-and-replay session, capture the logs. Same for Nokia DAC. Each one ~$0–partner-fee, but takes 8–16 weeks of partner onboarding.

---

## Q14 — "Vendor's largest production deployment by # cells, # subscribers, # rApp instances."

1. **RFP question.** *State largest current production deployment by (a) number of cells under management, (b) number of subscribers under management, (c) number of concurrent rApp instances orchestrated, (d) sustained policy-emit rate (policies/sec), and (e) audit-event throughput (events/sec).*

2. **PreceptualAI's response (best case).** Cells: 0. Subscribers: 0. rApp instances: 0 in production. Policy-emit rate measured in `benchmarks/RESULTS.md`: 33 ms decision-loop walltime — capable, but unproven at scale. `EDGE_BENCHMARK_PROOF.md`, `SOAK_24H_PROOF.md`, and `chaos_test_report.json` show single-node soak, not multi-cell production load.

3. **Why it loses to incumbent.** Ericsson EIC is under management at 100,000+ cells per the EIC product page and AT&T case. Nokia MantaRay claims similar at NTT DOCOMO. Ericsson slide: *"100,000+ cells under EIC management today, 50M+ subscribers indirectly steered by EIAP rApps. PreceptualAI: 0 production cells."*

4. **Score impact.** **−5 / 5.** Hard zero.

5. **Parity fix.** Same as Q1 and Q8 — convert pilots to production and let the clock run. There is no shortcut.

---

## Q15 — "Vendor's largest training corpus by hours of telemetry."

1. **RFP question.** *State the size of the AI/ML training corpus underlying your shipped models, in (a) hours of operator telemetry, (b) terabytes on disk, (c) number of distinct cells/sites, (d) number of distinct operators, and (e) recency of the most recent data.*

2. **PreceptualAI's response (best case).** ~13.5 GB on disk per `README.md` ("~13.8k LOC … ~8 GB real data on disk" — and 13.5 GB after the most recent NVIDIA + UCC + DeepMIMO ingest). Two trained checkpoints; `sla_head_v0.2_nvidia.pt` was trained on 8 NVIDIA Aerial windows + 600 DeepMIMO windows + 3,902 UCC MISL 5G windows. **One operator's worth of public test data, not production telemetry**. `DATA.md` candidly explains the 1.1 TB corpus is on the roadmap, not on disk.

3. **Why it loses to incumbent.** Ericsson and Nokia have **years of operator KPM data** across dozens of operators — petabytes, billions of cell-hours. Ericsson slide: *"Models trained on 50+ PB of multi-operator KPM telemetry across 15 years and 4 continents. PreceptualAI: 13.5 GB of public/test data."*

4. **Score impact.** **−4 / 5.** One point for being honest and reproducible about what we did train on.

5. **Parity fix.** This gap is structural — incumbents have a data moat. Three things help: (a) execute the `DATA.md` 1.1 TB ingest plan to close the volume gap on public corpora, (b) negotiate data-sharing into the Tier 3 pilot terms so federated learning across pilot sites accumulates real operator hours, (c) lean into the open-weights + reproducibility differentiator (D4) to convert "small corpus" into "verifiable corpus" — we can show provenance, they cannot.

---

# Required answers to the brief

## 1. Path to deliverable

`/home/danielfoojunwei/Preceptualv1/horizon-ric/DEVIL_D_RFP.md`

## 2. Total RFP points lost (out of 100)

**62 / 100 lost.** PreceptualAI scores **38 / 100** on a stock tier-1 telco RFP scorecard against Ericsson EIAP / Nokia MantaRay.

## 3. The 3 RFP questions we are MOST exposed on

1. **Q1 — Live tier-1 deployments (−10).** A hard zero. Knockout question. Cannot be remediated inside any RFP cycle; needs 12–24 months of clock.
2. **Q3 — 24×7 NOC with contractual MTTR ≤30 min Sev-1 (−8).** A hard zero. Procurement cannot accept "best effort" against a binary requirement no matter how good our reliability engineering is internally.
3. **Q8 — 3+ named, callable references (−6).** A hard zero. Downstream of Q1; without production customers there is nobody to put on the reference list.

These three alone cost us 24 of 100 points before the rubric even gets to features. They are also the three Ericsson sales engineers will lead with on every slide deck we encounter.

## 4. The 3 RFP questions where we HONESTLY win

These are scored on the differentiator side of the rubric (D1–D4), not the table-stakes side (Q1–Q15). They are the categories where Ericsson and Nokia have **no public response**, per the citations in `COMPETITIVE_LANDSCAPE.md` Section 3.

1. **D1 — Per-policy counterfactual envelope (+4).** `evidence/schema.py` `RejectedAlternative`, `policy/counterfactual.py`, `tests/test_emit_guards_counterfactual.py`. Ericsson EIAP marketing emphasises lifecycle and rApp Directory, not action-counterfactuals; NVIDIA's published XAI is feature-attribution Integrated-Gradients, which is "why this feature mattered" — not "what would have happened under action B." A regulator reading the OSS dashboard knows the difference instantly.
2. **D2 + D3 — SHA-256-chained tamper-evident audit + ITU-R S.1503 EPFD time-CDF in-loop (+8).** `evidence/store.py` chain-hash + `tests/test_evidence_store.py`; `planner/physics/epfd.py` Annex 1 Eq. (1) + `tests/test_epfd_time_cdf.py` benched at 0.19 s for a 500-sat Walker constellation. No named SMO vendor publishes either capability as an in-loop rApp surface. Together these are the regulator's two best friends.
3. **D4 — Open-source weights + TS 28.105 model card emission + SBOM (+3).** `checkpoints/*.md` canonical block + sha256, `tests/test_ts28105_model_card_emit.py`, `deploy/sbom/horizon-ric-sbom.json` (CycloneDX 1.5), cosign signatures. Ericsson, Nokia, Mavenir, Capgemini all ship rApps as proprietary binaries — supply-chain provenance audits cannot meaningfully run on their artefacts; they can on ours.

## 5. The marketplace positioning sentence we should LEAD with

Given the gap analysis above, do **not** lead with "next-generation SMO" or "full-stack RIC". Lead with the three places we genuinely beat the incumbent — auditability, NTN physics, and open verifiability — and frame ourselves as an **augment** to the incumbent SMO, not a replacement.

> **PreceptualAI is the audit-first, NTN-native rApp that runs on top of your existing Ericsson EIAP, Nokia MantaRay, or open-source SMO — adding the per-policy counterfactual envelope, SHA-256-chained tamper-evident decision log, and in-loop ITU-R S.1503 EPFD compliance your current rApp catalogue does not ship — without you having to rip and replace the SMO you already trust.**

This sentence does three things: (a) concedes the SMO-prime competition we cannot win in this RFP cycle, (b) names the three differentiators we score on, (c) frames the buy as low-risk augment rather than vendor-replacement, which is the only commercial path the procurement team can defend to the CTO. It also closes the door on Ericsson sales engineers reframing us as a competitor — we are explicitly *complementary* to the EIAP they just bought.

---

*Prepared by the hostile-RFP-evaluator persona, 2026-05-06. Sources: `README.md`, `PILOT.md`, `PILOT_NEXT_WEEK.md`, `SHOWCASE.md`, `COMPETITIVE_LANDSCAPE.md`, `BENCHMARK_HEAD_TO_HEAD.md`, `WHITEPAPER.md`, `marketplace/`, `docs/SMO_INTEGRATION.md`, `frontend/README.md`, `sdk/python/README.md`, `sdk/typescript/README.md`, `DEMO_PROOF.md`, `FINAL_PROGRESS.md`, `deploy/RUNBOOK.md`, `RELIABILITY.md`.*
