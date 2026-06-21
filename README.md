# Horizon-RIC — an AI-for-RAN trust & audit enabler

> O-RAN Non-RT-RIC **rApp** that makes a federated AI-RAN spectrum agent's
> decisions **auditable by construction**: every decision a poisoned, drifted,
> or adversarial agent emits is projected onto enumerated, independently-measured
> radio invariants and bound to a per-decision, tamper-evident, replayable
> evidence record a regulator can verify.

**Status:** torch-free, installs on stock Python 3.10+ with no accelerator ·
the full suite (490+ tests) is green in CI (ruff + mypy + pytest + docker +
yang-strict) · benchmark + threat model committed · Apache-2.0.

> Naming: the project is **Horizon-RIC** everywhere; the GitHub repository is
> being renamed to **horizon-ric**.

This README is also the AI-RAN Alliance *Call for Innovation* proposal
(deadline 31 July 2026; no membership required). See the companion documents:
- [`docs/EVALUATION_CRITERIA.md`](docs/EVALUATION_CRITERIA.md) — our anticipated,
  weighted evaluation rubric with honest self-scores.
- [`docs/RESEARCH_ALIGNMENT.md`](docs/RESEARCH_ALIGNMENT.md) — the NTU / SCRIPTS /
  DTC research lineage, the team, and the FCP funding context.

---

## Executive summary

The AI-RAN field is shipping learned control onto live spectrum: neural
physical-layer blocks and, central to this proposal, **federated deep-RL
dynamic-spectrum-access (DSA) agents** that decide which channel and which power
a cell or satellite link should use. The open problem is **not** the
decision-making capability — that is being solved by groups like the NTU/SCRIPTS
team this project builds on (see [`docs/RESEARCH_ALIGNMENT.md`](docs/RESEARCH_ALIGNMENT.md)).
The open problem is **trust**: when a federated agent is poisoned, drifted, or
adversarially driven, what bounds it from steering a cell out of band, over its
power licence, or — for a satellite link — over a power-flux-density ceiling, and
how does an operator *prove* to a regulator what the agent actually did?

**Horizon-RIC is an AI-for-RAN trust & audit layer.** It does not train or run
the agent; it sits beside the SMO / Non-RT RIC and wraps every decision the agent
emits in eight mechanisms:

1. **Decision Safety Shield** — projects each proposed action onto enumerated,
   independently-measured radio invariants (spectral mask, EIRP, NTN/LEO PFD
   ceiling, AI-PHY envelope, lawful-intercept), emitting a signed
   `SafetyCertificate`.
2. **At-decision-time evidence** — a SHA-256 hash-chained, RFC-3161-anchored
   `DecisionRecord` with a replayable counterfactual.
3. **Model provenance** — a signature over `(weights ‖ training-manifest)`,
   verified on promotion.
4. **Federated aggregation** — robust aggregation (Krum / median / trimmed-mean)
   bounding a poisoning client's pull on the shared DSA policy, plus Shamir
   secure aggregation for update privacy.
5. **Certified federated unlearning** — once a poisoning client is attributed,
   *removes* its contribution from the shared DSA policy and binds the removal to
   a signed `UnlearningCertificate` (distance-to-retrain bound + backdoor-probe).
   The **repair** layer for what robust aggregation only bounds — bridging the
   NTU/DTC federated-unlearning research (Lam et al., arXiv:2404.09724).
6. **DP-FedAvg + Rényi-DP accountant** — clipping + Gaussian noise with a real
   (ε, δ) budget that *bounds* membership/inversion leakage (MIA AUC 0.97→~0.5 as
   ε falls); reports the honest privacy/utility curve.
7. **Verifiable two-server secure aggregation** — 2-of-2 additive sharing (the
   server learns nothing) + Feldman commitments, so a malicious server that tampers
   or drops a contribution is *detected* (Starfish model, arXiv:2404.09724).
8. **Subject-level certified erasure (GDPR Art. 17)** — provable, audit-bound
   erasure of one *data subject*, verified against an independent retrain.

Mechanisms 4–8 are the **federated trust stack**; the privacy/erasure additions
(6–8) close the privacy face of `docs/THREAT_MODEL.md` (§9). All eight are
torch-free and grounded in the NTU/DTC/Lam research lineage
(`docs/RESEARCH_ALIGNMENT.md`).

**Measured result (`benchmarks/results/poisoning_shield.json`, deterministic —
reproduce with `python benchmarks/poisoning_shield_benchmark.py --decisions 10000
--poison-rate 0.3`):** on a 10,000-decision stream that is 30% poisoned, an
unguarded emit path puts **1,983** out-of-spec policies on the air interface plus
**489** in-spec-but-harmful ones (graded by an *independent* emission-mask / ACLR
oracle, not the Shield's own constants); with the Shield, **0** of each, and every
decision carries a certificate. The federated half (16 honest / 4 Byzantine, dim
200) is reported honestly: robust aggregation is a *partial bound, not a cure* —
under adaptive Fang-median the FedAvg distance from the honest mean is 8.23 while
median is 4.96 and trimmed-mean 5.81, but **Krum is worse at 12.2**, and under
small-step ALIE the robust aggregators are *further* from the honest mean than
FedAvg. The load-bearing guarantee is the deterministic Shield (0 illegal emits),
not the aggregators — see `docs/THREAT_MODEL.md` §7 and "Honest status and gaps".

**Adversarial-robustness result (`benchmarks/results/neural_rx_pgd.json`):** the
Shield faces an attack it was *not* hand-coded against — a real white-box **PGD**
perturbation of the received signal, crafted on a real numpy neural receiver's
input gradient (gradient verified against finite differences in
`tests/test_neural_rx_pgd.py`). At 16-QAM / 22 dB / ε=0.12, both receivers are
error-free on clean input; under the attack the neural receiver's symbol-error
rate is **~22×** the certified classical demapper's. The Shield watches the
*independently measured* (CRC/HARQ) block-error rate, detects the neural receiver
leaving its envelope, and falls back to the classical demapper — cutting the
attack's block-error impact **~12×**. (Honest scope: no-worse-than-the-certified-
baseline under an unseen attack, not "immune".)

---

## What is actually new — prior art & honest novelty scoping

We make **exactly one** novelty claim, and we state the prior art for everything
else plainly.

**The genuine contribution — and the only place we claim state-of-the-art — is
the decision-level evidence / audit binding:** a per-decision `SafetyCertificate`
bound to a SHA-256 hash chain, an RFC-3161 timestamp anchor, a replayable
counterfactual, and model-provenance threading, composed around a federated
AI-RAN agent's decisions so a regulator can replay and verify them.

What is **not** novel here, with prior art:

- **The safety shield itself is prior art.** Two-stage O-RAN action shielding is
  done by DeRAN (arXiv:2605.10648); shielded RL was introduced by Alshiekh et al.
  (AAAI-18, *"Safe Reinforcement Learning via Shielding"*); provably safe RL via
  reachable-set propagation is Kochdumper et al. (arXiv:2210.10691). Our Shield
  is an engineering composition of this established idea, specialised to RAN
  spectrum invariants — not a new safety method.
- **The aggregators are not novel and are known-defeated.** Krum, coordinate-wise
  median, and trimmed-mean are all pre-2019. They are explicitly broken by
  Baruch et al. (NeurIPS-19, *"A Little Is Enough"*) and Fang et al.
  (USENIX Security-20). We use them as a baseline, not a frontier, and we do not
  claim robustness against adaptive attackers.
- **The crypto is standard.** SHA-256, RSA-PSS, Shamir secret sharing, and
  RFC-3161 are off-the-shelf. We compose them; we do not invent them.

So: **no "SOTA" claim anywhere except, narrowly, the audit / certificate
binding.** The reframing is deliberate — Horizon-RIC is an AI-for-RAN *enabler
and benchmark* for the trust gap, building on the NTU/SCRIPTS federated-DSA line,
not a new safety or aggregation algorithm. The criterion-by-criterion honest
self-assessment is in [`docs/EVALUATION_CRITERIA.md`](docs/EVALUATION_CRITERIA.md).

---

## Problem statement and regulatory relevance

**The trust gap is the binding constraint on deploying federated AI-RAN agents on
licensed spectrum, not the decision-making gap.**

- **Singapore / IMDA (primary).** Singapore's Infocomm Media Development
  Authority (IMDA) licenses spectrum and sets emission / EIRP conditions. A
  federated DSA agent operating on licensed bands must demonstrably stay within
  those licence conditions — and, for the satellite case, within a
  power-flux-density (PFD) ceiling protecting co-channel terrestrial services.
  The Shield's per-decision projection plus a replayable evidence record is the
  artefact such a regulator would require. This framing matches the funder: the
  research lineage (see [`docs/RESEARCH_ALIGNMENT.md`](docs/RESEARCH_ALIGNMENT.md))
  is funded under Singapore's **FCP** (Future Communications R&D Programme,
  IMDA + NRF), whose thrusts include NTN, network orchestration, MEC, and
  security.
- **NTN / LEO (first-class scenario).** The satellite power-control line this
  project builds on drives a **PFD-ceiling invariant** in the Shield. Non-
  terrestrial links are treated as a first-class scenario, not an afterthought.
- **EU (secondary, illustrative analogue).** The same trust gap surfaces in the
  EU: the EU AI Act classifies AI in critical infrastructure as high-risk
  (Art. 6 + Annex III → Art. 9–15 controls); NIS2 Art. 23 mandates 24-hour
  incident notification; Ofcom's 2025/26 AI approach asks for explainability
  "sufficient to support post-incident regulatory review." We retain this as a
  *secondary* demonstration that the gap is multi-jurisdictional; details in
  `docs/compliance/eu_ai_act.md`.
- **Security.** O-RAN WG11 added AI/ML-specific threats (data / model poisoning,
  adversarial input, model inversion / extraction, supply-chain) to its Threat
  Model; 3GPP TR 33.898 studies AI/ML security for the RAN. These threats
  currently have no shipped rApp-level control.

> **Illustrative market note (not a sourced figure).** Industry commentary
> suggests Tier-1 operators are likely to under-deploy AI-RAN in the near term
> because capex follows the *trust* curve, not the technology curve. We previously
> cited a specific "Dell'Oro 30–40%" figure; we cannot link an exact public
> source for that number, so we **label it illustrative** and do not use it as
> evidence. The qualitative point — that the decision-level evidence layer is the
> missing piece in today's SMOs — is what we stand behind.

---

## Innovative solution and technical approach

The core idea is **AI-RAN-native runtime assurance**: bind a federated agent's
*output* to the RAN's own physical and regulatory invariants, at the RAN's own
timescales, and to a replayable evidence record.

**The Shield** (`src/horizon_ric/shield/`) — `Shield.dispose(action) →
(safe_action, SafetyCertificate)` runs an ordered, model-independent invariant
chain:

| Invariant | Basis | Action on violation |
|---|---|---|
| Lawful intercept | 3GPP TS 33.127, **fail-closed** | refuse emit if LI scope unknown |
| Spectral mask | 3GPP TS 38.104 (band-specific numbers calibrated per-deployment in M1–2) | clip carrier inside the licensed channel |
| Max EIRP / Tx power | block-edge / licence limit (calibrated per-deployment in M1–2) | reduce power to the ceiling |
| NTN / LEO PFD ceiling | power-flux-density limit at the surface | reduce satellite-link power below the PFD bound |
| Neural-RX envelope | TBLER vs classical LMMSE baseline + demap confidence | **fall back to the certified classical receiver** |
| Constellation legality + PAPR | legal M-QAM orders, PAPR ceiling | snap to nearest legal order / classical QAM |

Because the gate is mathematical and **independent of the agent**, a
poisoned / backdoored / adversarial agent **cannot violate the enumerated,
independently-measured invariants** — every disposition yields a certificate
recording the invariants checked, the margin to each bound, and any fallback.
*Coverage caveat:* this guarantees only the **enumerated** invariants; an unsafe
behaviour that is not expressible as one of the listed invariants is outside the
guarantee. The invariant set is a living register, not a completeness claim.

**The trust chain runs agent → decision → evidence end-to-end:** `provenance/`
signs and verifies the model artefact (RSA-PSS over weights ‖ manifest, HSM-held
key); `federated/robust.py` bounds a poisoning client's influence (Krum / median
/ trimmed-mean — *baseline only, see novelty scoping*); `federated/secure.py`
hides individual updates (Shamir `(t,n)` over GF(2¹²⁷−1)); `evidence/`
hash-chains every `DecisionRecord` and anchors the chain head to **real public
RFC-3161 TSAs** (freetsa, DigiCert) with nonce + imprint verification.

> **Secure-aggregation caveat.** The Shamir secure-aggregation scheme is
> **honest-but-curious only** (cf. Bonawitz et al., CCS-17): it hides individual
> updates from a curious-but-non-deviating server, not from a malicious one. It
> also sits in tension with robustness — **secure aggregation hides exactly the
> per-client information a robust aggregator needs to detect a poisoning client**
> (the secure-agg ⊥ robustness tension). Reconciling the two (e.g. via verifiable
> secret sharing) is a roadmap item (M3–4), not a solved problem here.

What's novel: not the individual mechanisms, but **composing a model-independent
invariant projection + at-decision-time, replayable evidence around a federated
AI-RAN spectrum agent** — and binding each decision to a regulator-verifiable
record. The demonstrator that exercises this end-to-end is the federated-DSA
bridge below.

### Demonstrator — federated DSA on licensed / NTN spectrum

To make the AI-for-RAN story concrete, a parallel work-stream is building a
demonstrator that runs a federated DSA agent (in the spirit of the NTU/SCRIPTS
publications) through the Horizon-RIC trust layer:

- Federated DSA agent — `src/horizon_ric/spectrum/`.
- Benchmark — `benchmarks/secure_dsa_benchmark.py`, results
  `benchmarks/results/secure_dsa.json`.
- Committed dataset — `datasets/spectrum_dsa/` with `DATASHEET.md`.
- An **LEO PFD invariant** in the Shield.

These artefacts are **in build**; the underlying trust mechanisms they depend on
(Shield, robust / secure aggregation, evidence store) are already working code
with tests. See [`docs/RESEARCH_ALIGNMENT.md`](docs/RESEARCH_ALIGNMENT.md) for the
built-vs-roadmap breakdown.

---

## Deployment feasibility

- **Drops in beside any O-RAN SMO.** Real R1 (registration), A1 (policy emit,
  OSC / EIAP / MantaRay dialects), and O1 (NETCONF / YANG via `ncclient`)
  adapters, behind circuit breakers, mTLS, OAuth2, and Casbin RBAC with tenant
  domains.
- **No accelerator, no lock-in.** Torch-free (numpy + pydantic); the Shield
  validates the *data* an agent emits (in production, from O-RAN E2 KPM / vendor
  telemetry), so it never needs to run the model. `pip install -e .` completes in
  seconds on a stock host.
- **Runs where AI-RAN runs.** Three shapes: Helm-on-Kubernetes, systemd
  edge-envelope (aarch64), or bare-metal Docker. Prometheus `/metrics`, Grafana
  dashboards, RFC-3161-anchored JSONL / SQLite evidence store.
- **Operationally safe.** Hash-chained audit, atomic A→B model promotion with
  bit-identical rollback, graceful degradation, systemd watchdog.

```bash
pip install -e ".[dev]"
pytest tests/ -m "not integration and not slow" -q          # full suite (490+ tests) green
python benchmarks/poisoning_shield_benchmark.py --decisions 10000 --poison-rate 0.30
horizon-rapp --once                                          # boot + readiness smoke
```

---

## 12-month timeline with milestones

| Months | Milestone | Output |
|---|---|---|
| **M1–2** | Per-band invariant calibration — calibrate band-specific TS 38.104 emission masks and EIRP limits for FR1/FR3 target bands per deployment; land the federated-DSA demonstrator (`src/horizon_ric/spectrum/`), the committed `datasets/spectrum_dsa/` + `DATASHEET.md`, and the LEO PFD invariant; live R1/A1 against OSC NONRTRIC. | Calibrated Shield configs + DSA demonstrator + dataset |
| **M3–4** | Strengthen FL security & benchmark — verifiable secret sharing (Feldman / Pedersen) toward malicious-server resistance; differential-privacy accountant for membership-inference / inversion; **add adaptive ALIE (Baruch) and Fang attackers to the benchmark**. | Upgraded `federated/`; adaptive-attack benchmark |
| **M5–6** | Live AI-PHY integration — run a real neural receiver / learned constellation through the Shield on real I/Q telemetry. | End-to-end demo against real I/Q |
| **M7–8** | Loop-tiered evidence — dApp (sub-ms sampled), Near-RT (per-decision), Non-RT (full counterfactual); inference-API extraction rate-limiting. | Evidence architecture + extraction defence |
| **M9–10** | Operator / testbed pilot — deploy alongside an SMO; 24-h soak; production PKCS#11 HSM custody; conformance dossier. | Pilot report + signed attestation packet |
| **M11–12** | Standardization + public benchmark — submit the AI/ML threat→control mapping as **candidate input to O-RAN WG11**; release the poisoning / robustness / DSA benchmark + dataset. | Candidate standards input + public benchmark |

---

## Expected deliverables and impact

**Deliverables**
- A production-grade, open-source **AI-for-RAN trust & audit rApp** (this repo).
- The **decision-level evidence / audit binding** — `SafetyCertificate` + hash
  chain + RFC-3161 anchor + counterfactual + provenance threading (the novel part).
- The **Decision Safety Shield** invariant projection (an engineering
  composition of prior-art shielding).
- **Robust + secure federated aggregation** baselines.
- **Benchmarking-ready code + committed results** (`benchmarks/`), with the
  federated-DSA demonstrator in build.
- A **poisoned-AI-PHY-decision / DSA dataset** for reproducible evaluation
  (`datasets/spectrum_dsa/`, in build).
- A **threat-model → control mapping** (`docs/THREAT_MODEL.md`) as a candidate
  WG11 input.

**Impact**
- **Safety:** out-of-spec air-interface emits from a 30%-poisoned decision stream
  reduced from **1,983 → 0** (plus 489 in-spec-but-harmful → 0; deterministic 10k-decision
  benchmark, `benchmarks/results/poisoning_shield.json`).
- **Poisoning resilience (honest):** robust aggregation only *partially* bounds a
  Byzantine pull and is comparable-or-worse than FedAvg under adaptive Fang/ALIE
  (Krum 12.2 vs FedAvg 8.23 under Fang-median); the guarantee is the deterministic
  Shield, with certified unlearning (`THREAT_MODEL.md` §8) as the repair layer.
- **Deployability:** gives operators and regulators (IMDA-style, NTN/LEO
  included) the replayable evidence layer their SMO lacks for federated AI-RAN
  agents on licensed spectrum.

### Expected outputs (mapping to the Call)

| Call output | What we provide |
|---|---|
| **Prototypes** | The runnable rApp + R1/A1/O1 adapters + `horizon-rapp` daemon; federated-DSA demonstrator in build. |
| **Algorithms or models** | The novel audit / certificate binding; the Shield invariant projection; Krum / median / trimmed-mean baselines; Shamir secure aggregation; RSA-PSS model-provenance. |
| **Benchmarking-ready code** | `benchmarks/poisoning_shield_benchmark.py` + committed `results/`; `benchmarks/secure_dsa_benchmark.py` in build. |
| **Datasets** | `datasets/spectrum_dsa/` + `DATASHEET.md` (in build); a synthetic poisoned-decision generator in the existing benchmark. |
| **Standardization contributions** | `docs/THREAT_MODEL.md` threat→control mapping as a **candidate input to O-RAN WG11** (not adopted). |

A transparent, weighted self-assessment of these outputs — including where we
score a **Gap** — is in [`docs/EVALUATION_CRITERIA.md`](docs/EVALUATION_CRITERIA.md).

---

## Honest status and gaps

Real, not mocked: the Shield, provenance signing / verify, robust / secure
aggregation, hash-chained evidence store, RFC-3161 TSA anchoring (real public
TSAs), and the R1/A1/O1 wire adapters are all working code with tests. The full
suite (490+ tests) is green in CI.

Known limitations (also in `docs/THREAT_MODEL.md`):

- **Aggregator robustness is baseline only — and we prove it, then repair it.**
  Krum / median / trimmed-mean are defeated by adaptive ALIE (Baruch, NeurIPS-19),
  Fang (USENIX Security-20), and Min-Max/Min-Sum (Shejwalkar, NDSS-21) attacks. A
  committed **five-family adversarial campaign** (`benchmarks/results/`, 75
  adversarial tests) demonstrates exactly where our defenses fail *and* where they
  hold — see `docs/THREAT_MODEL.md` §7. Robust aggregation is a *bound*, not a cure;
  when a poisoner is attributed, **certified federated unlearning** now *removes* it
  — driving a backdoor an undefended aggregator let through from success **1.0 → 0.04**
  (clean floor), bound to a signed certificate (`docs/THREAT_MODEL.md` §8). And the
  deterministic Shield + audit layer remains the guarantee — it bounds the emitted
  action to legal spectrum even from a fully compromised model (0 illegal emits
  from a backdoored DSA policy), and the integrity/audit perimeter is unbroken
  (8/8 probes blocked).
- **Secure aggregation is honest-but-curious only** (Bonawitz et al., CCS-17) and
  is in tension with robustness (secure-agg ⊥ robustness). Verifiable secret
  sharing for the malicious-server case is M3–4.
- **Invariant coverage is enumerated, not complete.** The Shield guarantees only
  the listed, independently-measured invariants; behaviours not expressible as an
  invariant are outside the guarantee.
- **Band numbers are calibrated, not byte-verified.** TS 38.104 EIRP / spectral-
  mask thresholds are pinned per licensed band in M1–2; we do not claim the spec
  numbers are byte-verified in this repo.
- **No DP accountant yet; no Sigstore transparency log yet;** inference-API
  extraction rate-limiting is roadmap (M7–8).
- **HSM custody** ships with in-memory (RSA-2048) and SoftHSM2 (PKCS#11)
  backends; production CloudHSM / Luna wire in through the same PKCS#11 path
  (M9–10).
- **The federated-DSA demonstrator** (`src/horizon_ric/spectrum/`,
  `benchmarks/secure_dsa_benchmark.py`, `datasets/spectrum_dsa/`) is in build by
  a parallel work-stream.

## AI-RAN Alliance positioning

An **AI-for-RAN** trust & audit contribution: an open, pre-competitive innovation
and benchmark for the trust gap that blocks federated AI-RAN agents on licensed
and NTN/LEO spectrum. It builds on the NTU/SCRIPTS federated-DSA research line
(see [`docs/RESEARCH_ALIGNMENT.md`](docs/RESEARCH_ALIGNMENT.md)) and is
complementary to — not competitive with — the vendors and groups shipping the AI
capability. The normative security specification is owned by O-RAN WG11 / 3GPP;
this project is a reference implementation, benchmark, and **candidate WG11
input**, not a competing standard.

---

Architecture, controls, and the full threat model: [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).
Research lineage and team: [`docs/RESEARCH_ALIGNMENT.md`](docs/RESEARCH_ALIGNMENT.md).
Evaluation rubric: [`docs/EVALUATION_CRITERIA.md`](docs/EVALUATION_CRITERIA.md).
License: Apache-2.0.
