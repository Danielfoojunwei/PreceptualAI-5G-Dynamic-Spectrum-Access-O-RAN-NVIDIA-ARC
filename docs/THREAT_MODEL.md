# Horizon-RIC — AI/ML Security Threat Model

**Component:** Horizon-RIC — an O-RAN Non-RT-RIC **rApp** that is a *security & trust layer*
for AI-RAN neural-PHY and RIC decisions.
**Document type:** AI/ML threat model & control mapping.
**Status:** Living document. Each control below is explicitly tagged **[SHIPPED]** or **[ROADMAP]**.
**Last updated:** 2026-06-21.

> Honesty note: this document maps a defensive rApp against the O-RAN WG11 / OWASP-ML /
> ENISA AI/ML threat catalogue. Where Horizon-RIC has no control, the entry says **GAP / roadmap**
> rather than claiming coverage. Where a referenced spec clause could not be byte-verified via
> public search, it is flagged inline.

---

## 1. Scope & system context

Horizon-RIC sits in the O-RAN **Non-RT-RIC** (with sampled hooks into Near-RT and dApp loops) and
acts as a **shield and auditor** over AI-RAN decisions — it does **not** train or run the neural-PHY
model. It ingests RAN telemetry (NVIDIA Aerial cuBB inner-receiver / CSI metrics and O-RAN E2 KPM
KPIs), observes the action proposed by an AI-PHY / RIC inference engine, **bounds that action against
physics and regulatory invariants**, and only then permits an A1 policy to be emitted — recording a
tamper-evident evidence chain for every decision. The model artifact and its weights are produced and
served by a *separate* AI-RAN owner; Horizon-RIC treats the model as **untrusted** by design and never
needs to trust the model's internals to keep the network safe and legal.

**Trust boundary:** Horizon-RIC does **NOT** train, fine-tune, or execute the neural-PHY model.
It SHIELDS the model's proposed actions (project-to-safe / fall back to a certified classical baseline)
and AUDITS the decisions it produces (signed evidence). A poisoned, backdoored, or adversarially
driven model is therefore *contained* at the action boundary, not trusted at the weight level.

```
                                  TRUST BOUNDARY (Horizon-RIC owns everything right of "::")
                                  |
 telemetry                       |       neural-PHY / RIC                Horizon-RIC SHIELD
 (untrusted source)             |       (UNTRUSTED model)               (this rApp)
 ┌───────────────────────┐     |   ┌───────────────────────┐    ::   ┌──────────────────────────┐
 │ NVIDIA Aerial cuBB    │     |   │ neural-RX / scheduler │    ::   │ Decision Safety Shield   │
 │ (inner-RX, CSI, TBLER)│ ──► |   │ inference → proposes  │ ──►::──►│  • physics invariants    │
 │ O-RAN E2 KPM KPIs     │     |   │   candidate action    │    ::   │  • regulatory invariants │
 └───────────────────────┘     |   └───────────────────────┘    ::   │  • AI-PHY invariants     │
                                  |                                ::   │  • LI fail-closed        │
                                  |                                ::   └───────────┬──────────────┘
                                  |                                ::               │ project-to-safe
                                  |                                ::               │  OR classical fallback
                                  |                                ::               ▼
                                  |                                ::   ┌──────────────────────────┐
                                  |                                ::   │ A1 policy emit (guarded) │
                                  |                                ::   └───────────┬──────────────┘
                                  |                                ::               ▼
                                  |                                ::   ┌──────────────────────────┐
                                  |                                ::   │ Evidence chain           │
                                  |                                ::   │  SHA-256 hash-chain +    │
                                  |                                ::   │  RFC-3161 + SafetyCert + │
                                  |                                ::   │  counterfactual          │
                                  |                                ::   └──────────────────────────┘
```

The architecture context follows the O-RAN ALLIANCE AI/ML-in-RIC model: AI/ML running as xApps/rApps
in the Near-/Non-RT RIC is a core O-RAN element, and "without protections on the AI and ML, the
xApps and rApps can be compromised, leading to data leakage and degraded, or even denied, service"
([O-RAN ALLIANCE Security Update 2025](https://www.o-ran.org/blog/o-ran-alliance-security-update-2025)).

---

## 2. Assets

| # | Asset | Why it matters | Owner |
|---|-------|----------------|-------|
| A1 | **The AI/ML decision** (proposed neural-PHY/RIC action) | A wrong/illegal action degrades or denies RAN service, or breaches spectrum/LI law | AI-RAN model owner (proposes); Horizon-RIC (gates) |
| A2 | **Model artifact / weights** | A poisoned or backdoored artifact silently changes behaviour | AI-RAN model owner (Horizon-RIC verifies on promotion) |
| A3 | **Training-data lineage / manifest** | Poisoned or skewed training data is the root cause of most AI/ML threats | AI-RAN model owner (Horizon-RIC pins the manifest hash) |
| A4 | **Evidence chain** (DecisionRecords) | The audit/regulatory record; if forgeable, accountability collapses | Horizon-RIC |
| A5 | **A1 policy** | The actuated output to the RAN; integrity here = service integrity | Horizon-RIC emits |
| A6 | **Tenant data / KPIs / CSI** | Confidentiality + cross-tenant isolation (model inversion / membership inference target) | Tenant; Horizon-RIC isolates |

---

## 3. Threat catalogue

Threats follow the **OWASP Machine Learning Security Top 10 (2023)** items
([OWASP ML Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)), which O-RAN
WG11 adopted — together with the **ENISA "Securing Machine Learning Algorithms" (2021)** report — to
add ~39 AI/ML-specific threats to the **WG11 Threat Model and Risk Analysis** in 2024
([O-RAN Security Update 2025](https://www.o-ran.org/blog/o-ran-alliance-security-update-2025)).
STRIDE categories: **S**poofing, **T**ampering, **R**epudiation, **I**nformation disclosure,
**D**enial of service, **E**levation of privilege.

| OWASP-ML / Threat | STRIDE | Attack surface in an AI-RAN rApp | Horizon-RIC control |
|-------------------|--------|----------------------------------|---------------------|
| **ML02 Data Poisoning** | T | Training set / federated client updates feeding the neural-PHY model | **Robust federated aggregation** (Krum / coord-median / trimmed-mean) bounds Byzantine clients **[SHIPPED]**; **Safety Shield** contains poisoned-model output at action time **[SHIPPED]**; **training-manifest hash pinning** for lineage **[SHIPPED]** |
| **ML10 Model Poisoning / backdoor** | T, E | Substituted/backdoored weights served to inference | **Model provenance & integrity**: signature over `(weights ‖ training-manifest)` verified on promotion **[SHIPPED, see §4.3]**; **Safety Shield** makes a triggered backdoor unable to emit an unsafe/illegal action **[SHIPPED]** |
| **ML01 Input Manipulation / adversarial-evasion** | T, D | Crafted IQ / CSI / KPM telemetry that pushes neural-RX into a bad demap/scheduling decision | **Safety Shield** AI-PHY invariants (neural-RX TBLER/demap-confidence envelope vs LMMSE baseline, learned-constellation legality, PAPR ceiling) + project-to-safe / classical fallback **[SHIPPED design; envelope thresholds calibrated per-deployment]** |
| **ML03 Model Inversion** | I | Querying / observing the model to reconstruct tenant CSI/training inputs | Tenant isolation (Casbin RBAC + domains) **[SHIPPED]**; no DP guarantee — **GAP / roadmap** (no DP accountant) |
| **ML04 Membership Inference** | I | Determining whether a tenant's data was in the training set | Tenant isolation **[SHIPPED]**; differential-privacy accounting — **GAP / roadmap** |
| **ML05 Model Theft / extraction** | I, E | Repeated inference queries to clone the model | mTLS + JWT + RBAC restrict who can query **[SHIPPED]**; per-tenant inference-API extraction-rate limiting — **GAP / roadmap** |
| **ML06 AI Supply Chain / MLOps compromise** | T, E | Compromised artifact registry, pipeline, or third-party dependency | **Content-addressed artefact vault (SHA-256)** + signature-on-promotion **[SHIPPED]**; HSM-held signing keys **[SHIPPED interface; production custody documented not bundled]** |
| **ML09 Output Integrity** | T, R | Altering the model's output between inference and actuation (A1) | **Pre-emit guards** + **Safety Shield** re-validate every action before A1 emit **[SHIPPED]**; **hash-chained + RFC-3161-anchored** evidence makes output tampering detectable & non-repudiable **[SHIPPED]** |
| **ML07 Transfer Learning Attack** | T | Malicious pretrained base model fine-tuned into the deployed model | **Provenance** chains base→fine-tuned via the pinned training-manifest **[PARTIAL — manifest captures lineage; base-model attestation is ROADMAP]**; **Safety Shield** contains malicious behaviour at action time **[SHIPPED]** |
| **ML08 Model Skewing** | T, D | Feedback-loop / drift manipulation that gradually biases the model | **Evidence chain** + counterfactual records expose drift in rejected-vs-chosen actions over time **[SHIPPED data; automated skew alerting is ROADMAP]**; **Safety Shield** invariants are static and physics-derived, so they do not skew **[SHIPPED]** |

**Cross-cutting:** Spoofed telemetry (S) — Horizon-RIC's source authentication of Aerial/E2 feeds
depends on the vendor transport; see §5. Repudiation (R) of any decision is mitigated by the
RFC-3161-anchored hash chain (§4.2).

---

## 4. Horizon-RIC controls

### 4.1 Decision Safety Shield — *the AI-RAN-native control* **[SHIPPED core; thresholds calibrated per-deployment]**

Bounds **every** neural-PHY/RIC action against three invariant classes plus a fail-closed legal gate.
It is implemented in `src/horizon_ric/shield/` as `Shield.dispose(action) -> (safe_action,
SafetyCertificate)`: a chain of non-trainable invariants that **projects an infeasible action onto the
nearest safe action**, or falls back to a certified classical baseline, before any A1 emission.
Because the gate is mathematical and independent of the model, **a poisoned/backdoored/adversarially-
driven model still cannot emit an unsafe or illegal policy**, and every disposition yields a signed
`SafetyCertificate` recording the invariants checked, the margin to each bound, and any fallback.

- **Terrestrial RAN physics + spectrum-regulatory invariants** — spectral emission mask, max EIRP /
  Tx-power. These map to **3GPP TS 38.104** (NR Base Station radio transmission & reception), which
  defines BS rated/declared EIRP and out-of-band emission limited by the spectrum emission mask and
  ACLR ([TS 38.104 overview](https://www.tech-invite.com/3m38/tinv-3gpp-38-104.html)). *Note: the
  exact numeric mask/EIRP limits live in the normative annexes of TS 38.104 and were not byte-verified
  here; deployments must pin the band-specific values from the controlling release.*
- **AI-PHY invariants** — neural-RX predicted-TBLER / demap-confidence envelope checked against a
  classical **LMMSE** baseline; **learned-constellation legality**; **PAPR ceiling**. If the neural-RX
  confidence or predicted TBLER falls outside the envelope, the Shield prefers the certified classical
  receiver/baseline.
- **LI fail-closed** — lawful-interception capability must remain intact; the Shield refuses to emit
  any action that would disable or evade LI, per the 3GPP LI architecture **TS 33.127**
  ([TS 33.127 — LI architecture & functions](https://itecspec.com/archive/3gpp-specification-ts-33-127/))
  (requirements in TS 33.126, stage-3 in TS 33.128).

On violation the Shield projects to the nearest safe action (or classical fallback) **and emits a
signed `SafetyCertificate`** recording the violation, the projection, and the resulting action.
*Mitigates:* ML01, ML09, and contains ML02/ML07/ML08/ML10 at the action boundary.

### 4.2 At-decision-time evidence chain **[SHIPPED]**

Every decision produces a **`DecisionRecord`** that is **SHA-256 hash-chained** and
**RFC-3161-anchored** (trusted timestamp), and includes a **counterfactual**: the rejected
alternatives, the reason each was rejected, and a **pinned RNG seed** for reproducibility.
Recording is **loop-tiered** to fit each control loop's latency budget:

- **dApp (sub-ms):** sampled records.
- **Near-RT:** per-decision record.
- **Non-RT:** full counterfactual record.

This gives non-repudiation (defeats **R**) and makes output tampering (**ML09**) and gradual skew
(**ML08**) detectable after the fact. Implementation: `evidence/store.py`, `evidence/rfc3161.py`,
`policy/counterfactual.py`.

### 4.3 Model provenance & integrity **[SHIPPED]**

Two layers, because byte-identity alone is insufficient:

1. **Content-addressed artefact vault (SHA-256)** — guarantees the served bytes are exactly the
   promoted bytes (`runtime/artefact_vault.py`).
2. **Cryptographic signature over `(weights ‖ training-manifest)`**, verified **on promotion** —
   closes the *"byte-consistent but untrusted/backdoored model"* gap: a model can be perfectly
   content-addressed yet still poisoned. Binding the signature to the training-manifest also pins
   **training-data lineage** (asset A3).

*Mitigates:* ML06, ML10, and the lineage half of ML02/ML07.

### 4.4 Robust federated aggregation **[SHIPPED]**

When model updates are aggregated across clients, Horizon-RIC supports **Krum**, **coordinate-wise
median**, and **trimmed-mean** to bound the influence of Byzantine (poisoning) clients
(`federated/robust.py`), plus Shamir secure aggregation for update privacy (`federated/secure.py`).
This is the ENISA-recommended class
of robustness control applied across the ML lifecycle
([ENISA — Securing ML Algorithms](https://www.enisa.europa.eu/sites/default/files/publications/ENISA%20Report%20-%20Securing%20Machine%20Learning%20Algorithms.pdf)).
*Mitigates:* ML02 (federated data/model poisoning).

### 4.5 Zero-Trust platform controls (supporting / table-stakes) **[SHIPPED]**

Per **NIST SP 800-207**, these are necessary-but-not-AI-specific supports (see §6):

- **mTLS** on all interfaces; **JWT** authn (`security/jwt.py`).
- **Casbin RBAC with tenant domains** for least-privilege, tenant-isolated authorization
  (`security/rbac.py`, `security/tenant.py`).
- **HSM-held keys** for signing/timestamping (`security/hsm.py`) — *interface shipped; production
  custody documented, not bundled, see §5*.
- **NIS2 24-hour incident reporter** (`security/nis2_reporter.py`), aligned to the NIS2 "early
  warning within 24 hours" obligation.

---

## 5. Residual risks / honest GAP register

| Gap | Threat exposed | Status |
|-----|----------------|--------|
| **No differential-privacy accountant** | ML03 model inversion, ML04 membership inference | **ADDRESSED** (§9) — DP-FedAvg + a real Rényi-DP accountant now *bound* leakage; a committed membership-inference benchmark shows AUC 0.97→~0.5 as ε falls |
| **No inference-API extraction rate-limiting** | ML05 model theft / extraction | **ROADMAP** — authn restricts *who* queries, not *how much* |
| **Telemetry source authentication depends on the vendor** | Spoofed Aerial cuBB / E2 KPM feeds (S) | **VENDOR-DEPENDENT** — Horizon-RIC trusts the Aerial/E2 transport's authentication; it does not independently attest the sensor |
| **HSM production custody not bundled** | Key compromise of signing/timestamp keys | **DOCUMENTED, NOT BUNDLED** — production deployments must wire CloudHSM / Thales Luna; the code ships an HSM *interface* and software fallback for dev |
| **Safety-Shield envelope thresholds are deployment-calibrated** | A mis-calibrated TBLER/confidence envelope could be too loose | **OPERATIONAL** — invariants are physics/regulatory-hard; the AI-PHY *envelope* must be tuned and re-validated per band/deployment |
| **Base-model attestation for transfer-learning** | ML07 transfer-learning attack via a malicious pretrained base | **PARTIAL** — manifest captures lineage; cryptographic base-model attestation is roadmap |
| **Automated model-skew alerting** | ML08 model skewing | **PARTIAL** — evidence chain records the data; automated drift/skew alerting is roadmap |

Regulatory context (informational, not a Horizon-RIC claim of certification): an AI system gating
RAN spectrum behaviour may fall under **EU AI Act** high-risk obligations (risk management, data
governance, technical documentation, record-keeping, human oversight, post-market monitoring), and
operators may carry **NIS2** "significant incident" reporting duties (24h early warning → 72h update
→ 1-month final report). The evidence chain (§4.2) and NIS2 reporter (§4.5) are designed to *support*
these duties, not to discharge them on the operator's behalf.

---

## 6. Zero-Trust posture (NIST SP 800-207 mapping)

Horizon-RIC adopts "never trust, always verify": no network location is inherently trusted, access is
per-request, and trust is continuously re-evaluated from telemetry
([NIST SP 800-207 guidance](https://www.paloaltonetworks.com/cyberpedia/what-is-nist-sp-800-207)).
The notable AI-RAN extension is that Horizon-RIC applies zero-trust **to the model itself** — the
neural-PHY model is a non-trusted subject whose every output is verified at the action boundary.

| NIST SP 800-207 tenet (paraphrased) | Horizon-RIC realization |
|-------------------------------------|--------------------------|
| All data sources & computing services are resources | Model, telemetry feed, A1 policy, evidence store all treated as protected resources |
| All communication secured regardless of network location | mTLS on all interfaces |
| Access granted per-session, per-request | JWT + Casbin per-request authorization |
| Access determined by dynamic policy (identity, posture, context) | RBAC with tenant domains + the Safety Shield as the *action*-level policy decision point |
| Integrity & security posture of all assets monitored | Content-addressed vault + signature verification on promotion; hash-chained evidence |
| Authentication & authorization strictly enforced before access | mTLS + JWT + RBAC before any inference/emit |
| Collect telemetry to continuously improve security posture | RFC-3161-anchored DecisionRecords + counterfactuals feed continuous verification & post-incident review |

The "verify the model's *output*, continuously, at every decision" stance is what turns generic
zero-trust into an AI-RAN-specific defense: even a fully compromised model is a *contained* subject,
not a trusted one.

---

## 7. Empirical adversarial evaluation — what holds, what does not

We ran a real attack campaign against our own system (numpy, no mocks) across five
families. Every attack is implemented, run, and committed with results under
`benchmarks/results/`; **75 adversarial tests** gate them in CI. The honest summary:
the deterministic perimeter (integrity, audit, output-shielding) holds completely;
the ML layer is beatable by adaptive adversaries — as it is across the field — and
we report exactly where.

| Family | Results file(s) | What HOLDS | Where the defense FAILS (reported, not hidden) |
|---|---|---|---|
| Integrity / audit / supply-chain | `integrity_attack_suite.json` | **8/8** probes detected/blocked: model-swap, weight/manifest tamper, evidence tamper/reorder/replay, self-report spoof, illegal-emit, LI fail-closed | none of the 8 bypassed |
| Adversarial evasion (neural-RX) | `evasion_suite.json`, `neural_rx_pgd.json`, `phy_fading.json` | Shield falls back via *independent* CRC measurement (~94% of windows in AWGN), restoring the link; guarantee `effective ≤ classical + tol` | white-box FGSM/BIM/MIM/**transfer** crush the neural-RX **~28×** in AWGN; gap **collapses to ~1.1×** under realistic fading (both receivers vulnerable) |
| Physical-layer jamming | `jamming_suite.json` | helps under imperfect-CSI asymmetry (routes to the better receiver) | **barrage jamming degrades BOTH receivers** — the Shield is a router, not a denoiser; it cannot restore a noisier channel |
| FL model-poisoning | `fl_poisoning_suite.json` | naive Gaussian Byzantine fully bounded by median / trimmed-mean / Krum | **Min-Max/Min-Sum (Shejwalkar NDSS-21) and Fang (USENIX-20) DEFEAT Krum/median/trimmed-mean** near breakdown; FedAvg unbounded |
| FL data-poisoning / backdoor (DSA) | `dsa_poison_suite.json`, `secure_dsa.json` | Shield+audit keep every emitted (channel, power) **LEGAL** and the chain intact even from a fully backdoored policy (**0 illegal emits**); **certified unlearning removes the backdoor post-hoc** (§8) | a single-row **backdoor SURVIVES coordinate-median at its 50% breakdown point** (success 1.0); robust agg only *bounds* reward poisoning |

**The thesis this evidence supports.** Robust aggregation is a *bound*, not a cure —
adaptive poisoning beats it, exactly as the literature predicts (Baruch et al.
NeurIPS-19; Fang et al. USENIX-Sec-20; Shejwalkar & Houmansadr NDSS-21). The two
guarantees that DO hold are **deterministic and model-independent**: (a) the
tamper-evident audit perimeter, and (b) the Decision Safety Shield bounding the
*emitted action* to legal spectrum even when the model is fully compromised. That is
the point of a defense-in-depth design — the AI layer can be defeated; the safety +
audit layer is what holds the line, and that is the AI-RAN-native contribution. We
publish the failures rather than hide them, which is the correct posture for a
security submission.

---

## 8. Closing the backdoor FAIL — certified federated unlearning

Robust aggregation only *bounds* a poisoner; it never *removes* the influence that
already leaked into the global policy (the §7 backdoor row). The repair mechanism
is **machine unlearning**, and it is exactly the active research line of the same
NTU/DTC group this project builds on — Prof Kwok-Yan Lam's federated-unlearning
work (see `docs/RESEARCH_ALIGNMENT.md`): *Privacy-Preserving Federated Unlearning
with Certified Client Removal* (Liu, Ye, Jiang, Shen, Guo, Tjuawinata & Lam,
arXiv:2404.09724) and the *Survey on Federated Unlearning* (Liu, Jiang, Shen, Peng,
Lam, Yuan & Liu, ACM Comput. Surv. 2024); plus the backdoor-as-verification idea of
Han et al. (arXiv:2412.11476).

We implement it torch-free on the federated DSA policy
(`src/horizon_ric/federated/unlearning.py`; `benchmarks/results/federated_unlearning.json`;
7 tests in `tests/test_federated_unlearning.py`). Honest results, multi-seed:

| | Backdoor success (trigger) | Certified L2 distance to retrain | Signed certificate |
|---|---|---|---|
| Poisoned (undefended FedAvg) | **1.00** | — | — |
| **Retrain-from-scratch unlearning** | **0.04** (clean floor) | 0.0 (it *is* the gold standard) | verifies; rejects the poisoned model & manifest/weight tamper |
| Efficient replay unlearning | 1.00 (**insufficient**) | **~14 (large)** | verifies, but the large bound flags it as untrustworthy |

So **certified retrain-from-scratch unlearning removes the backdoor an undefended
aggregator let through (1.00 → 0.04)** and binds the removal to an RSA-PSS-signed
`UnlearningCertificate` (the Starfish-style bound = distance to the gold standard;
backdoor-probe success before/after; audit-chainable). Three honest caveats: (a)
the *cheap* replay unlearner is **insufficient** for a backdoor that propagated
through warm-starting — and its large certified distance-to-retrain says so rather
than hiding it; (b) unlearning needs **attribution** — whole-vector detection flags
large-norm poisoning (crafted Q-table, reward poisoning, and the boost-50 backdoor
here), but a *norm-matched* stealthy backdoor would evade it; (c) at that breakdown
point the deterministic **Decision Safety Shield remains the backstop**. Unlearning
is a new layer that *repairs* an attributed compromise; it does not replace the
output-shield that holds when attribution fails.

---

## 9. Privacy, verifiable aggregation, and the right to be forgotten

Three more trust mechanisms close the remaining privacy gaps in §5, each torch-free,
each grounded in the NTU/DTC research line (`docs/RESEARCH_ALIGNMENT.md`) and each
reporting honestly where it costs something. Results are committed under
`benchmarks/results/`; **24** tests gate them.

| Mechanism | Derives from | What it does | Honest cost / caveat |
|---|---|---|---|
| **DP-FedAvg + Rényi-DP accountant** (`federated/dp.py`, `dp_privacy.json`) | Liu, Jiang, Lam et al., *Efficient FU with Adaptive DP*, IEEE BigData 2024; accounting per Mironov CSF-17 / Abadi CCS-16 | Per-client L2 clipping + Gaussian noise *bound* privacy leakage with a real (ε, δ) accountant. A committed membership-inference benchmark shows the leak close as ε falls: **MIA AUC 0.97 (no DP) → 0.69 (ε≈5.3) → ~0.47 (ε≈0.62)** | Privacy costs **utility**: throughput 1.09 → 0.05 over the same range. We report the full ε/utility/leakage curve, not a single flattering point; the clip bound is data-dependent (a mild leak we disclose) |
| **Verifiable two-server secure aggregation** (`federated/verifiable_secagg.py`, `verifiable_secagg.json`) | Liu, Ye, Jiang, Shen, Guo, Tjuawinata & Lam, Starfish, arXiv:2404.09724 | 2-of-2 additive secret sharing (a malicious or curious **server learns nothing**) + Feldman commitments over a 2048-bit safe prime so a server that **tampers or drops** a contribution is **detected** — tamper & drop detection **1.0** (exact, by discrete-log binding); reconstruction matches plaintext to ~3e-5 | Privacy holds **only under non-collusion** of the two servers; Feldman commitments are *binding, not hiding*; public verifiability costs one 2048-bit modexp **per coordinate** (dim 128 ≈ 9 s in pure Python) |
| **Subject-level certified erasure** (`federated/erasure.py`, `subject_erasure.json`) | Lam et al., *Certifying the Right to be Forgotten: Primal-Dual … in Vertical FL*, IEEE TIFS | GDPR Art. 17 erasure of **one data subject** (not a whole client): exact recompute + re-aggregate, **verified against an independent from-scratch retrain (certified distance 0.0)** and bound to a signed `ErasureCertificate` | Exactness is for the single-round transition model; a *cheap* linear shortcut is exact for FedAvg but leaves a measured **~0.57** residual under non-linear median (reported), so we recompute |

These do not change the thesis: the deterministic perimeter is the guarantee. They
close the *privacy* face of the threat model — bounding leakage (DP), removing trust
in the server (verifiable agg), and honouring erasure (Art. 17) — with the same
"publish the cost" honesty as the attack campaign.

---

## Sources

- [O-RAN ALLIANCE Security Update 2025 — WG11 Threat Model & Risk Analysis, ~39 AI/ML threats added in 2024](https://www.o-ran.org/blog/o-ran-alliance-security-update-2025)
- [OWASP Machine Learning Security Top 10 (2023)](https://owasp.org/www-project-machine-learning-security-top-10/)
- [OWASP ML02 — Data Poisoning Attack](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML02_2023-Data_Poisoning_Attack)
- [OWASP ML03 — Model Inversion Attack](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML03_2023-Model_Inversion_Attack)
- [OWASP ML05 — Model Theft](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML05_2023-Model_Theft)
- [OWASP ML08 — Model Skewing](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML08_2023-Model_Skewing)
- [OWASP ML10 — Model Poisoning](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML10_2023-Model_Poisoning)
- [ENISA — Securing Machine Learning Algorithms (Dec 2021)](https://www.enisa.europa.eu/sites/default/files/publications/ENISA%20Report%20-%20Securing%20Machine%20Learning%20Algorithms.pdf)
- [3GPP TR 33.898 — Study on security and privacy of AI/ML-based services and applications in 5G](https://www.3gpp.org/ftp/Specs/archive/33_series/33.898/)
- [NIST SP 800-207 — Zero Trust Architecture (final)](https://csrc.nist.gov/pubs/sp/800/207/final)
- [NIST SP 800-207 — Zero Trust overview (Palo Alto Networks)](https://www.paloaltonetworks.com/cyberpedia/what-is-nist-sp-800-207)
- [3GPP TS 38.104 — NR Base Station radio transmission & reception (overview)](https://www.tech-invite.com/3m38/tinv-3gpp-38-104.html)
- [3GPP TS 33.127 — Lawful Interception architecture and functions](https://itecspec.com/archive/3gpp-specification-ts-33-127/)
- [EU AI Act vs NIS2 — overlapping cyber/AI compliance (context)](https://www.isms.online/nis-2/vs/eu-ai-act/)
- [EU AI Act — serious incident reporting deep dive (context)](https://www.taylorwessing.com/en/insights-and-events/insights/2025/10/eu-ai-act-deep-dive)
