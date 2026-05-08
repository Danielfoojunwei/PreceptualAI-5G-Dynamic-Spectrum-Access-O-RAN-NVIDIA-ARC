# EU AI Act — Classification and Voluntary Controls Memo

**Version:** 0.2.0
**Date:** 2026-05-06
**Audience:** Operator compliance officer, EU operator's AI Act register
**Component:** PreceptualAI rApp (`src/horizon_ric/`)
**Regulation cited:** Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence ("AI Act"), in force as of 2 August 2026.

---

## 1. Purpose

This memorandum classifies PreceptualAI under the EU AI Act (Regulation (EU) 2024/1689) and enumerates the high-risk-grade controls PreceptualAI voluntarily adopts so an EU operator's compliance officer can append a self-attestation to their internal AI Act register. The document is auditor-facing: every claim cites either a file:line in this repository or a numbered article of the regulation.

## 2. Classification

**PreceptualAI is NOT a high-risk AI system under Annex III of the AI Act.** It is consequently not subject to the conformity assessment, CE marking, registration, or mandatory post-market monitoring obligations of Articles 16-49.

**PreceptualAI voluntarily adopts the substantive controls of Articles 9 (risk management), 10 (data governance), 11 (technical documentation), 12 (record-keeping), 13 (transparency), 14 (human oversight), and 15 (accuracy / robustness / cybersecurity)** because the largest EU operators (Tier-1) treat AI Act-grade controls as a procurement floor regardless of the regulation's own scope language.

This is a deliberate stance, not a regulatory requirement. It is documented here so an operator's compliance officer can rely on it.

## 3. Why PreceptualAI is not high-risk under Annex III

Annex III lists eight categories of high-risk AI systems. PreceptualAI falls outside all of them.

### 3.1 Categories that are clearly inapplicable

| Annex III §  | Category | PreceptualAI position |
|---|---|---|
| §1 | Biometric identification / categorisation | N/A — no biometric pipeline. The state encoder consumes 3GPP TS 28.552 KPIs, not biometric inputs. (`src/horizon_ric/encoder/`) |
| §3 | Education and vocational training | N/A — no education use case |
| §4 | Employment, workers management, access to self-employment | N/A — no employment use case |
| §5(a) | Public benefits eligibility evaluation | N/A — not a benefits system |
| §5(b) | Creditworthiness | N/A |
| §5(c) | Risk assessment in life and health insurance | N/A |
| §5(d) | Emergency call dispatch and triage | N/A — PreceptualAI is RAN scheduling, not emergency dispatch |
| §6 | Law enforcement | N/A — see also `docs/compliance/li_applicability.md`: PreceptualAI is not an LI function |
| §7 | Migration, asylum and border control management | N/A |
| §8 | Administration of justice and democratic processes | N/A |

### 3.2 Annex III §2 — Critical infrastructure (the only category requiring detailed analysis)

Annex III §2 covers AI systems "intended to be used as safety components in the management and operation of critical digital infrastructure, road traffic and the supply of water, gas, heating and electricity."

**Construction:** the qualifier "safety components" is load-bearing. The Annex III §2 entry is restricted to *safety-of-persons* infrastructure management. Recital 55 of the AI Act clarifies: "It is appropriate to classify as high-risk the AI systems used as safety components of the management and operation of […] critical digital infrastructure as listed in point (8) of the Annex to Directive (EU) 2022/2557, where their failure or malfunctioning may put at risk the life and health of persons at large scale". RAN scheduling and policy optimisation does not place safety-of-life at risk on failure: degraded scheduling lowers throughput, increases SLA breach probability, and may cause handover oscillation, but does not endanger lives.

**Concrete operator-side controls already separate PreceptualAI from safety-of-life telecoms paths:**

- PreceptualAI emits A1 *intents* (`src/horizon_ric/rapp/a1_adapter.py:190`, `emit_policy`). It does not directly drive the gNB or UPF.
- The Near-RT RIC and gNB enforce their own safety bounds on top of any A1 intent.
- Emergency call dispatch (E.164 / E.112) is handled by the operator's IMS / E-CSCF, not by PreceptualAI.
- Hard regulatory constraints (LI, EPFD, ITU-R spectrum masks) are enforced **before** A1 emit by the constraint layer (`src/horizon_ric/policy/li_constraint.py`, `src/horizon_ric/planner/physics/epfd.py`, `src/horizon_ric/planner/physics/s1428.py`). These are hard rejections, not soft penalties (`policy/li_constraint.py:30-34, 100`).

**Conclusion:** PreceptualAI operates on the **performance** side of telecoms management (RAN policy optimisation), not the **safety** side. It is therefore outside Annex III §2.

### 3.3 Operator review is the ultimate control

Article 14 of the AI Act ("Human oversight") is satisfied even were PreceptualAI reclassified as high-risk: every A1 emit can be vetoed by the SOC operator, and every override is recorded in the audit chain (`src/horizon_ric/evidence/schema.py:128-129`, `operator_override` and `operator_override_reason`).

## 4. The nine voluntary high-risk-grade controls PreceptualAI adopts

For each Article 8-15 obligation, PreceptualAI has implemented or documented a control. References below are to file paths in this repository.

### 4.1 Risk management system (Art. 9)

- `REVIEW.md` — pre-pilot independent review with a numbered findings list
- `AUDIT_NO_FAKES.md` — assertion that no mocks/fakes/hardcoded values masquerade as real implementations; reviewed pre-PR
- `STANDARDS.md` — 22-row gap analysis of every standards/regulation row
- `GAPS_TO_PILOT.md` — open risks tracked to closure

These four documents constitute the iterative risk management process required by Art. 9(2)-(5).

### 4.2 Data governance (Art. 10)

- Per-tenant evidence chain with explicit canonicalisation and tamper-evidence: `src/horizon_ric/evidence/store.py:106-142` (`verify`, `verify_tenant`)
- TS 28.105 model cards with sha256 + size verification on every shipped checkpoint: `checkpoints/*.md`, validated by `tests/test_ts28105_model_card_emit.py`
- Decision-time provenance (`ModelVersions`): `src/horizon_ric/evidence/schema.py:18-30` records encoder / risk_heads / dyna / policy / constraint_layer / rapp version tags on every decision

Training data provenance is captured in each model card (e.g. `checkpoints/jepa_encoder_v0.1.md`).

### 4.3 Technical documentation (Art. 11, Annex IV)

The repository's MD documentation set provides the substantive content of an Annex IV technical file:

- `ARCHITECTURE.md` — system architecture and intended purpose
- `WHITEPAPER.md` — design rationale
- `PARADIGMS.md` — design patterns (paradigm H1 = counterfactual envelope)
- `STANDARDS.md` — standards conformance map
- `RELIABILITY.md` — reliability characteristics
- `RBAC.md`, `docs/RBAC.md` — security model
- `docs/conformance/CONFORMANCE.md` — conformance report

Honest gap, see §5.2 below: the content is present but not yet tabulated to the exact Annex IV table-of-contents structure.

### 4.4 Record-keeping (Art. 12)

- Append-only hash-chained DecisionRecord store: `src/horizon_ric/evidence/store.py:82-310` (both `JsonlEvidenceStore` and `SqliteEvidenceStore`)
- Auditor-facing chain verification endpoint: `src/horizon_ric/rapp/api_v1.py:392-398` (`GET /api/v1/audit/verify`)
- Structured event names with stable schema for SOC ingestion: `src/horizon_ric/runtime/state_recovery.py:24-31` (named events: `horizon.state.saved`, `horizon.state.corrupt`, etc.) and `src/horizon_ric/security/middleware.py:38` (event names `auth.token_invalid`, `auth.denied`, `auth.granted`).

### 4.5 Transparency to deployers (Art. 13)

- `PILOT.md` — concession map listing every honest concession or limitation operators must understand before deployment
- `docs/compliance/li_applicability.md` — explicit residual-risk disclosure (LI section §6)
- This document — explicit AI Act position
- Deployer-facing user guide for the counterfactual envelope: `docs/COUNTERFACTUAL_USER_GUIDE.md`

### 4.6 Human oversight (Art. 14)

- Operator override field on every decision: `src/horizon_ric/evidence/schema.py:128-129`
- rApp lifecycle accepts manual stop signals (SIGTERM/SIGINT): `src/horizon_ric/rapp/lifecycle.py:253-272` — operator can halt the daemon at any time without data loss (atomic checkpointer at `src/horizon_ric/runtime/state_recovery.py:50-60`)
- Dashboard API exposes the policy emit / rollback path explicitly: `src/horizon_ric/rapp/api_v1.py:349, 410`

### 4.7 Accuracy, robustness and cybersecurity (Art. 15)

- Accuracy claims documented and CI-tested: `RELIABILITY.md`, plus model card metrics (e.g. `checkpoints/sla_head_v0.4_jepa.md`)
- Cybersecurity controls aligned to O-RAN.WG11 §6 cipher allow-list: `src/horizon_ric/rapp/auth.py:34-38`, applied at lines 155-200
- TLS-1.3 best-effort honest-failure logging: `auth.py:178-200` (we honestly log the OpenSSL TLS-1.3 limitation rather than silently degrade)
- RBAC via Casbin: `src/horizon_ric/security/rbac.py`, `rbac_model.conf`, `rbac_policy.csv`
- mTLS with hard-banned production cert-skipping: `auth.py:111-117`
- JWT RS256 with zero-downtime key rotation: `src/horizon_ric/security/jwt.py:180-198`

### 4.8 Conformity assessment (analogue of Art. 43)

We do NOT undergo a notified-body conformity assessment because we are not classified as high-risk. We do publish a self-conformity report:

- `docs/conformance/CONFORMANCE.md` — row-by-row conformance to O-RAN WG2/WG11, 3GPP TS 28.105/28.541/28.552, ITU-R, and TM Forum ODA, with ✅/⚠️/🟡/❌ status glyphs and file:line references

The report is gated by CI (`docs/conformance/CONFORMANCE.md` §"Continuous validation").

### 4.9 Post-market monitoring (analogue of Art. 72)

- Prometheus metrics at `/metrics` per `src/horizon_ric/rapp/health.py`
- Alertmanager rules: `deploy/prometheus/sla_rules.yml`, `deploy/prometheus/rules.yml`
- SLA breach engine: `src/horizon_ric/sla/engine.py:78-187` — emits structured `horizon.sla.breach` events on sustained breach
- Audit log persistence: `src/horizon_ric/evidence/store.py` — every decision recorded for retroactive analysis with actual_outcome_30s/1min/5min fields filled in retroactively (`evidence/schema.py:122-125`)

## 5. Three controls PreceptualAI explicitly does NOT yet have

Honest enumeration of gaps that an operator's compliance officer should know about.

### 5.1 GPAI provider duties (Art. 53-55) — N/A by design

PreceptualAI ships task-specific models (`checkpoints/*.pt`): a JEPA encoder, an SLA risk head, a CfC cell, a TD-MPC value/policy. None of these are general-purpose AI models within the meaning of Art. 3(63) of the regulation. The thresholds in Art. 51 (10^25 FLOPs training compute) are several orders of magnitude above the training budget for these specialised heads.

**Disposition:** Not applicable. We are not a GPAI provider.

### 5.2 Annex IV technical file — content present, structure pending

The substantive content for an Annex IV file (intended purpose, system architecture, training data, performance metrics, risk management, post-market monitoring plan) is present across the repository's MD set (see §4.3 above). It has NOT yet been tabulated to the exact §1-§9 structure of Annex IV.

**Disposition: NOT YET — planned in Phase 3** (Q4 2026). The closure path is a single document `docs/compliance/annex_iv.md` that copies / cross-references existing content into the Annex IV layout. Estimated effort: 3 person-days. We do not consider this a substantive gap; an auditor can today reconstruct an Annex IV file from the existing documentation.

### 5.3 CE mark — not applicable

CE marking under Art. 16(g) is required only of high-risk AI systems. PreceptualAI is not high-risk (see §3 above). If a future Annex III amendment brings RAN policy optimisation into scope, a CE mark with notified-body involvement would be required; today it is not applicable.

**Disposition:** Not applicable while §3 classification holds.

## 6. Conclusion — operator self-attestation template

An EU operator's compliance officer may append the following statement to their internal AI Act register:

> *We have evaluated PreceptualAI v0.2.0 (third-party rApp from [vendor]) for inclusion in our AI Act register. We classify PreceptualAI as **outside** Annex III on the basis of: (a) it is not a safety component of safety-of-life critical infrastructure; (b) all outputs are operator-reviewable through the SOC override path documented in `src/horizon_ric/evidence/schema.py:128-129`; (c) it does not process biometric, employment, education, law-enforcement, migration, justice, or benefits data. We have nevertheless verified that PreceptualAI implements the substantive controls of Articles 9-15 as documented in `docs/compliance/eu_ai_act.md` §4. Two known gaps (Annex IV tabulation, live LI rule-list refresh) are tracked in `GAPS_TO_PILOT.md` with closure dates. We rely on PreceptualAI's per-tenant audit chain (`src/horizon_ric/evidence/store.py`) to satisfy our own Art. 12 record-keeping obligation.*

---

**End of memo.** Spec citations are to Regulation (EU) 2024/1689 as adopted on 13 June 2024 (OJ L 2024/1689 of 12.7.2024).
