# OPERATOR_DEPLOYER_DUTIES — EU AI Act Art. 26 inheritance for PreceptualAI deployers

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


**Version:** 0.2.0
**Date:** 2026-05-06
**Audience:** EU operator's compliance officer, DPO, AI Act register owner.
**Companion docs:** `docs/compliance/eu_ai_act.md` (provider-side voluntary controls), `docs/compliance/gdpr_dpia.md` (DPIA template), `PILOT.md` (pilot SLA template), `MARKETPLACE_POSITIONING.md` (commercial framing).
**Closes Devils-Advocate finding:** #21 "EU AI Act Art 26 deployer duty inheritance — operator unaware".

This document tells you, the operator, exactly which obligations the AI Act puts on **you** (the deployer) versus on **us** (PreceptualAI, the provider) when you deploy PreceptualAI into production. It exists because Devil-A flagged that the operator's compliance team may not realise that buying an audit-grade rApp does NOT discharge their Art. 26 duties; it shifts them.

---

## 1. Definition of "deployer" under the AI Act

Under Regulation (EU) 2024/1689 ("AI Act"), Art. 3(4):

> "**deployer** means a natural or legal person, public authority, agency or other body using an AI system under its authority **except** where the AI system is used in the course of a personal non-professional activity"

You — the operator — are the **deployer** when PreceptualAI runs in your network under your operational authority, processing your subscribers' management-plane KPIs and emitting policies into your Near-RT RIC. We — PreceptualAI the rApp — are the **provider** under Art. 3(3): we develop the system and place it on the market under our name.

The split matters because Art. 16 obligations bind the **provider** (us) and Art. 26 obligations bind the **deployer** (you). Buying a compliant rApp does not make you compliant; it gives you the substrate on which you discharge your own obligations.

Note: Regardless of whether PreceptualAI is classified as high-risk under Annex III (we argue it is not — see `docs/compliance/eu_ai_act.md` §3), Art. 26 deployer duties are most expansive when the system **is** high-risk. The seven obligations enumerated below are the high-risk-grade duties; we voluntarily structure our offering so you can treat PreceptualAI as if it were high-risk for procurement-floor purposes (consistent with the largest tier-1 EU operators' policy of treating Annex III as the procurement floor).

---

## 2. The 7 deployer obligations under Art. 26

These derive from Art. 26(1)–(11) of the AI Act for a deployer of a high-risk AI system. Some are absolute (you must do them); some require us as provider to give you the inputs to do them; some require joint agreement.

| #   | Obligation                                                                                              | Source                                  |
|-----|----------------------------------------------------------------------------------------------------------|------------------------------------------|
| D1  | Use the AI system **in accordance with the instructions for use** accompanying it.                       | Art. 26(1)                              |
| D2  | Assign **human oversight** to natural persons with the necessary competence, training, and authority.    | Art. 26(2), referring to Art. 14         |
| D3  | Ensure that **input data** is relevant and sufficiently representative in view of the intended purpose.  | Art. 26(4)                              |
| D4  | **Monitor the operation** of the high-risk AI system and inform the provider/distributor of risks and serious incidents.   | Art. 26(5)                              |
| D5  | **Keep automatically generated logs** for at least 6 months unless other Union/national law specifies otherwise. | Art. 26(6)                              |
| D6  | Inform **workers' representatives and affected workers** before deploying or using a high-risk AI system in the workplace. | Art. 26(7)                              |
| D7  | Carry out a **fundamental rights impact assessment (FRIA)** before deployment, where required by Art. 27. | Art. 26(9), referring to Art. 27         |

Two more deployer-side duties cut across the seven:

- **Cooperation duty** (Art. 26(11)) — cooperate with competent authorities on any action concerning the high-risk system.
- **Annex III update duty** — if the deployer makes a substantial modification, the deployer becomes a provider under Art. 25(1)(c) and inherits Art. 16 obligations. Do not modify the rApp's WM, planner, or evidence chain in production; configure only.

---

## 3. Who handles which duty (PreceptualAI vs operator vs joint)

Three columns. Read top-to-bottom; each row is a duty above.

| Duty | PreceptualAI handles                                                                                                                                                                   | Operator handles                                                                                                                                                                                          | Joint / agreement-required                                                                                                                                                                                                |
|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| D1   | We ship `PILOT.md`, `docs/runbooks/*.md`, `docs/COUNTERFACTUAL_USER_GUIDE.md`, `docs/compliance/eu_ai_act.md` as the instructions-for-use bundle.                                       | You read them and configure the rApp accordingly. You distribute them internally to your SOC and SRE teams.                                                                                              | Treatment of any operator-specific extension (e.g. custom A1 policy types) requires a written supplement to the IFU. Template in `PILOT.md` Annex.                                                                       |
| D2   | We ship `RBAC.md`, the operator-override field on every decision (`evidence/schema.py:128-129`), and the explainability surface (`docs/COUNTERFACTUAL_USER_GUIDE.md`) so a non-ML operator can read every decision. | You assign named SOC engineers with authority to override. You train them. You document the training.                                                                                                  | We jointly design the override threshold (when does an alert escalate from auto-policy to human-in-the-loop?). Default thresholds in `deploy/prometheus/sla_rules.yml` are starting points only.                       |
| D3   | We document the training-data lineage in `checkpoints/*.md` (TS 28.105 model cards) and the input schema in `src/horizon_ric/io/schemas.py`. We expose data-drift metrics at `/metrics`.   | You decide what KPIs to feed in (O1 NETCONF subscription set), confirm representativeness of your network vs the training corpus, and re-train via federated learning if drift is material.                | We jointly agree what counts as drift severe enough to trigger model rollback or re-training. Template SLA language in §4 below.                                                                                       |
| D4   | We emit structured monitoring events (`security/middleware.py:38`, `runtime/state_recovery.py:24-31`, `sla/engine.py:78-187`) to `/metrics` and to the audit chain. We provide `docs/runbooks/customer_escalation.md` and `docs/runbooks/fl_convergence_failure.md`. | You run Prometheus + Alertmanager, you triage, you escalate, you decide what is a "serious incident" under Art. 73, and you file the report to the relevant market-surveillance authority within 15 days. | We must be informed of any serious incident the operator detects that traces to the rApp's behaviour, so we can fulfil our Art. 20 corrective-action duty as provider. Template incident-report flow in §4 below.       |
| D5   | We persist DecisionRecord audit chain via `evidence/store.py` (Jsonl + SQLite, hash-chained, per-tenant). The chain integrity is verifiable via `GET /api/v1/audit/verify`.            | You configure retention (we recommend ≥6 months in `docs/compliance/gdpr_dpia.md` §1.6 to align with both Art. 26(6) and your own regulatory log retention). You operate WORM/archive policy.            | We jointly agree the rotation/archive cadence and the chain-checkpoint signing convention so deletions are detectable and lawful (see `docs/compliance/gdpr_dpia.md` Risk-2 mitigation).                                |
| D6   | N/A — we do not employ your workforce.                                                                                                                                                | You inform your workers' representatives (works council, union) before deployment, including: what the rApp does, the override surface, the explainability tools you give your SOC engineers.            | We supply factual material for the consultation: paradigm explainer (`PARADIGMS.md`), human-oversight surface (`docs/COUNTERFACTUAL_USER_GUIDE.md`), audit chain explainer (`docs/compliance/gdpr_dpia.md` §4.2).        |
| D7   | We provide the substantive material an FRIA needs: the affected-rights analysis (no biometric, no employment, no benefits — see `docs/compliance/eu_ai_act.md` §3.1), the LI applicability memo (`docs/compliance/li_applicability.md`).        | You conduct the FRIA, where required by Art. 27 (public-sector deployers, banking, etc.). You sign off and lodge it with the supervisory authority where required.                                       | We jointly review the FRIA's risk register so any vendor-side gap (e.g. drift, model card lineage) is mirrored on both sides. FRIA template language in §4.                                                            |

The joint-agreement column is the source of most procurement disputes. The pilot SLA template (§4) names each joint item explicitly and assigns a named owner on both sides.

---

## 4. The compliance contract / pilot SLA template language

The text below is drop-in language for the Schedule that accompanies your pilot agreement. It is consistent with `PILOT.md`'s 90-day pilot pack and with Art. 26 of the AI Act.

```
SCHEDULE C — AI ACT ART. 26 DEPLOYER OBLIGATIONS

C.1 Roles
C.1.1 The Operator is the deployer under AI Act Art. 3(4). PreceptualAI is the provider under
      Art. 3(3). Neither party shall represent the other as bearing the other's obligations.

C.2 Instructions for use (Art. 26(1))
C.2.1 PreceptualAI shall maintain in the repository:
        - PILOT.md (operational use)
        - docs/COUNTERFACTUAL_USER_GUIDE.md (explainability use)
        - docs/runbooks/oncall.md, customer_escalation.md, fl_convergence_failure.md,
          security_incident.md, cert_rotation.md (incident & lifecycle use)
        - docs/compliance/eu_ai_act.md, gdpr_dpia.md, nist_csf.md, li_applicability.md
        - this OPERATOR_DEPLOYER_DUTIES.md
      and shall keep them current within 14 days of any material rApp change.
C.2.2 The Operator confirms it has read these documents and shall distribute them to its
      SOC, SRE, DPO, and compliance teams before go-live.

C.3 Human oversight (Art. 26(2))
C.3.1 The Operator shall assign at least two (2) named SOC engineers with authority to
      override any A1 policy emit through the operator_override field
      (`src/horizon_ric/evidence/schema.py:128-129`).
C.3.2 The Operator shall maintain training records for those engineers and shall make them
      available on request.
C.3.3 The default rApp action is no-op on internal error (fail-closed); the operator
      override is for the case where the rApp's auto-policy is wrong, not for the case
      where the rApp itself fails.

C.4 Input data representativeness (Art. 26(4))
C.4.1 The Operator confirms that the O1 NETCONF subscription set fed to PreceptualAI
      represents the operational scope the rApp will manage. Material drift triggers a
      mutual notice within 5 business days.
C.4.2 PreceptualAI shall expose drift telemetry at /metrics and document the threshold above
      which retraining is recommended in the model card.

C.5 Monitoring & serious-incident reporting (Art. 26(5), Art. 73)
C.5.1 The Operator shall monitor /metrics, the audit chain, and the structured events
      enumerated in `src/horizon_ric/security/middleware.py:38` and
      `src/horizon_ric/sla/engine.py:78-187`.
C.5.2 If the Operator detects a serious incident as defined in Art. 3(49), the Operator
      shall (a) notify PreceptualAI's incident contact within 24 hours and (b) file the
      Art. 73 report to the relevant market-surveillance authority within 15 days. The
      Operator owns the regulator-facing report; PreceptualAI supplies factual material on
      request.

C.6 Log retention (Art. 26(6))
C.6.1 The Operator shall retain DecisionRecord audit-chain records for at least six (6)
      months and shall verify chain integrity at least monthly via
      `GET /api/v1/audit/verify`.
C.6.2 The Operator shall configure retention/archive in the operator-side WORM store as
      described in `docs/compliance/gdpr_dpia.md` Mitigation-Risk-2.

C.7 Workforce information (Art. 26(7))
C.7.1 The Operator confirms it has informed its workers' representatives of the rApp
      deployment in accordance with applicable national co-determination law (e.g. Germany
      BetrVG, France Code du travail) before go-live.

C.8 Fundamental Rights Impact Assessment (Art. 26(9), Art. 27)
C.8.1 Where Art. 27 applies (public-sector deployers; banking; insurance), the Operator
      shall complete a FRIA before go-live and lodge it where required.
C.8.2 PreceptualAI shall provide on request: training-data lineage (model cards), the LI-
      applicability memo, and the AI Act classification memo (`docs/compliance/eu_ai_act.md`).

C.9 Cooperation with authorities (Art. 26(11))
C.9.1 Both parties shall cooperate with the relevant market-surveillance authority. Each
      party shall designate a named contact in Annex 1 to this Schedule.

C.10 Termination of this Schedule
C.10.1 This Schedule survives termination of the master agreement for the duration of the
       Operator's regulatory log-retention obligation, but in no event less than six (6)
       months.
```

The template is designed to be initialled at pilot kick-off and incorporated into the master subscription agreement at conversion.

---

## 5. What we explicitly do NOT take on

To avoid mis-selling, we explicitly disclaim five things at the deployer-duty layer:

1. **The Art. 73 serious-incident filing.** That is yours; we supply facts on request.
2. **The FRIA itself.** We supply substantive material; the operator signs and lodges.
3. **Any worker-consultation discharge.** That is governed by national labour law and is the operator's exclusive duty.
4. **Any classification of the AI system as your deployer's risk-management tool.** We classify the AI system from our side (`docs/compliance/eu_ai_act.md`); you classify from yours.
5. **Any liability for the operator's input-data choice.** If the operator subscribes O1 to a non-representative cell set, drift is a deployer-side issue.

---

**End of memo.** Spec citations are to Regulation (EU) 2024/1689 (AI Act), in force from 2 August 2026. Cross-references are to file paths in this repository.
