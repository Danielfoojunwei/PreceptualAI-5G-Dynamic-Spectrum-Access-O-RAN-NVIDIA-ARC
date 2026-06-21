# Anticipated evaluation criteria — Horizon-RIC

> **These criteria are to be finalized and approved by the AI-RAN Alliance Board.
> This is our *anticipated* mapping**, derived from the Call for Innovation's six
> required proposal sections, its five expected outputs, and the Alliance's
> mission (innovation; benchmarks / blueprints; openness / pre-competitive
> contribution; AI-RAN relevance). We publish it so a reviewer can hold us to a
> transparent, falsifiable standard and see our own honest self-scores — including
> where we score a **Gap**.

Self-score key:

- **Strong** — built, tested, evidence committed in-repo, defensible today.
- **Adequate** — substantially built, but with a named limitation or in-build
  demonstrator dependency.
- **Gap** — known shortfall; the roadmap row that closes it is named.

---

## How the criteria were derived

The Call requires six proposal sections (problem / relevance; solution &
technical approach; deployment feasibility; timeline & milestones; deliverables &
impact; alliance fit) and expects five output types (prototypes; algorithms /
models; benchmarking-ready code; datasets; standardization contributions). The
Alliance mission adds innovation, openness / pre-competitive value, and AI-RAN
relevance. We collapse these into seven weighted criteria.

---

## Criterion 1 — AI-for-RAN relevance & track fit (weight 20%)

**What it measures.** Is this an AI-for-RAN capability — not generic compliance
plumbing? Does it make AI-RAN decisions better / safer / deployable?

**Our evidence.** Horizon-RIC is positioned as an AI-for-RAN **enabler and
benchmark**: it is the trust layer that makes a federated deep-RL
dynamic-spectrum-access agent (the NTU/SCRIPTS line — see
[`RESEARCH_ALIGNMENT.md`](RESEARCH_ALIGNMENT.md)) deployable on licensed / NTN
spectrum. The Decision Safety Shield (`src/horizon_ric/shield/`), robust /
secure aggregation (`src/horizon_ric/federated/`), and per-decision evidence
(`src/horizon_ric/evidence/`) all act on the *AI agent's RAN decisions*. The
in-build demonstrator (`src/horizon_ric/spectrum/`, `benchmarks/secure_dsa_benchmark.py`)
makes this an end-to-end AI-RAN story, not a sidecar.

**Self-score: Adequate.** The trust mechanisms are built and act on AI decisions;
the federated-DSA demonstrator that closes the "is this really AI-RAN?" loop is
in build by a parallel work-stream.

**Gap & closure.** Until `src/horizon_ric/spectrum/` and `secure_dsa.json` land,
the AI-RAN agent is referenced rather than shipped here. **M1–2 / M5–6** land the
demonstrator and a live AI-PHY integration, moving this to Strong.

---

## Criterion 2 — Genuine innovation / novelty, honestly scoped (weight 18%)

**What it measures.** Is there a real, defensible contribution to the state of
the art — and is the novelty claim honest about prior art?

**Our evidence.** The **only** claim of novelty is the **decision-level evidence /
audit binding**: a per-decision `SafetyCertificate`
(`src/horizon_ric/shield/certificate.py`) + SHA-256 hash chain + RFC-3161 anchor
(`src/horizon_ric/evidence/`) + counterfactual replay + model-provenance threading
(`src/horizon_ric/provenance/`). We do **not** claim novelty for the shield, the
aggregators, or the crypto. Honest prior art:

- **Safety shielding** is prior art: DeRAN (arXiv:2605.10648) does two-stage
  O-RAN action shielding; Alshiekh et al. (AAAI-18) introduced shielded RL;
  Kochdumper et al. (arXiv:2210.10691) do provably safe RL via set propagation.
- **Robust aggregators** (Krum / median / trimmed-mean) are pre-2019 and are
  **defeated** by Baruch et al. (NeurIPS-19, "A Little Is Enough") and Fang et
  al. (USENIX Security-20). We use them as a baseline, not a frontier.
- **The crypto** (SHA-256, RSA-PSS, Shamir, RFC-3161) is all standard.

**Self-score: Strong** — narrowly, on the audit / certificate binding only. We
explicitly do **not** claim SOTA anywhere else.

**Gap & closure.** The novelty is narrow by design. The risk is that "evidence
binding" reads as engineering, not research; **M11–12** publishes the
binding + benchmark so the contribution is externally citable. A Sigstore-style
transparency-log integration (currently absent) would strengthen the provenance
claim and is a roadmap item.

---

## Criterion 3 — Benchmarking-ready code & reproducible results (weight 16%)

**What it measures.** Can a third party run the benchmark and reproduce the
numbers? Are results committed?

**Our evidence.** `benchmarks/poisoning_shield_benchmark.py` with committed
results at `benchmarks/results/poisoning_shield.json`. Traceable numbers from
that file: on a 10,000-decision stream at 30% poison rate, the unguarded path
emits **2,366** illegal policies; the Shield emits **0** (every decision gets a
certificate). Under a Byzantine federation (20 honest / 8 Byzantine clients),
distance from the honest mean drops from **202** (FedAvg) to **8–12** (robust
aggregators). The DSA demonstrator benchmark
(`benchmarks/secure_dsa_benchmark.py` → `benchmarks/results/secure_dsa.json`)
runs a federated DSA agent under ALIE/Fang poisoning and shows the Shield keeps
every emission legal and the audit chain intact. The adversarial-robustness
benchmark (`benchmarks/neural_rx_pgd_benchmark.py` →
`benchmarks/results/neural_rx_pgd.json`) grades the Shield against an attack it
was **not** hand-coded for — a real white-box PGD attack on a real numpy neural
receiver (input gradient verified by finite difference): the neural receiver's
SER is ~22× the classical demapper's under attack, and the Shield's
independent-measurement fallback cuts the block-error impact ~12×.

Beyond these, a **five-family adversarial campaign** is committed (numpy, no mocks)
with ten result files under `benchmarks/results/` and **75 adversarial tests** in
CI: evasion (FGSM/BIM/MIM/transfer/boundary), physical-layer jamming +
imperfect-CSI, FL model-poisoning (sign-flip / scaling / Gaussian / Min-Max /
Min-Sum / ALIE / Fang) against every aggregator, FL data-poisoning + backdoor on
the DSA loop, and an integrity/audit/bypass battery. It reports honestly where our
defenses FAIL — adaptive poisoning beats Krum/median, a backdoor survives median at
its breakdown point, the white-box evasion gap collapses under realistic fading —
and where they hold (8/8 integrity probes; the Shield bounds the emitted action to
legal spectrum even from a compromised model). See `docs/THREAT_MODEL.md` §7. The
campaign now also *repairs* the worst FAIL: **certified federated unlearning**
(`federated_unlearning_suite.json`) removes a backdoor an undefended aggregator let
through (success 1.0 → 0.04) with a signed, audit-chainable certificate — bridging
the NTU/DTC federated-unlearning research (`docs/THREAT_MODEL.md` §8).

**Self-score: Strong.** The benchmarks are real, committed, reproducible, and
adversarially exhaustive — including attacks that defeat our own defenses, reported
openly. The remaining gap is live over-the-air I/Q, not attack coverage. The
original "the benchmark only tests what the
Shield was coded to catch" critique.

**Gap & closure.** The committed benchmark exercises a *synthetic* poisoned trace,
not live I/Q. The current robust-aggregation numbers are against a naive Byzantine
model — they would **not** survive an ALIE (Baruch) or Fang adaptive attack, and
we say so. **M3–4** adds those adaptive attackers to the benchmark; **M5–6**
adds live AI-PHY telemetry; **M11–12** releases the public benchmark + dataset.

---

## Criterion 4 — Datasets (weight 10%)

**What it measures.** Is there a committed, documented, reusable dataset?

**Our evidence.** A committed, versioned dataset has landed:
`datasets/spectrum_dsa/traces.jsonl` (1,620 federated-DSA decision traces, honest
+ ALIE/Fang-poisoned) with `manifest.json`, a full `DATASHEET.md` (provenance,
schema, synthetic-deterministic collection, intended use, honest limitations), and
a reproducible generator. The adversarial campaign additionally commits nine
`benchmarks/results/*.json` files that are themselves reusable evaluation artifacts.

**Self-score: Adequate.** A real, documented, regenerable dataset is committed; it
is synthetic-deterministic (not over-the-air), which the datasheet states plainly.

**Gap & closure.** The dataset is synthetic; **M5–6** captures real
neural-PHY / DSA traces on a testbed (srsRAN / NVIDIA Aerial) and **M11–12**
publishes the dataset + benchmark suite openly.

---

## Criterion 5 — Deployment feasibility & openness / pre-competitive value (weight 14%)

**What it measures.** Can an operator actually deploy this? Is it open and
pre-competitive (complementary to vendors, not a walled garden)?

**Our evidence.** Apache-2.0; torch-free (numpy + pydantic), installs on stock
Python 3.10+ with no accelerator. Real R1 / A1 / O1 adapters
(`src/horizon_ric/rapp/`, `src/horizon_ric/io/`) behind mTLS, OAuth2, circuit
breakers, and Casbin RBAC. The Shield validates the *data* an AI block emits, so
it never has to run the model — it is complementary to NVIDIA Aerial / Nokia
MantaRay / Ericsson, not competitive. The full test suite (300+ tests) is green
in CI (ruff + mypy + pytest + docker + yang-strict).

**Self-score: Strong** on openness and packaging; **Adequate** on field-proof.

**Gap & closure.** No operator/testbed pilot yet, and HSM custody ships with
in-memory + SoftHSM2 backends (production CloudHSM/Luna wire through the same
PKCS#11 path). **M9–10** runs an operator pilot with a 24-h soak and production
PKCS#11 custody.

---

## Criterion 6 — Standardization contribution (weight 12%)

**What it measures.** Does the work feed an open standard?

**Our evidence.** `docs/THREAT_MODEL.md` maps O-RAN WG11 / OWASP-ML /
3GPP TR 33.898 AI/ML threats to controls with an honest GAP register, positioned
as a **candidate input to O-RAN WG11** — not an adopted contribution and not a
competing standard.

**Self-score: Adequate.**

**Gap & closure.** We state honestly that band-specific TS 38.104 EIRP / spectral-
mask numbers are **calibrated per-deployment in M1–2**, not byte-verified against
the spec text here. The threat→control mapping is a *candidate* input; **M11–12**
submits it to WG11. The normative spec remains owned by O-RAN WG11 / 3GPP.

---

## Criterion 7 — Research lineage, team credibility & regulatory grounding (weight 10%)

**What it measures.** Is there a credible team and a real research base? Is the
regulatory framing matched to the funder?

**Our evidence.** [`RESEARCH_ALIGNMENT.md`](RESEARCH_ALIGNMENT.md) establishes
continuity with the NTU/SCRIPTS/DTC team — Prof. Kwok-Yan Lam (PI), Dr Li Feng,
Bowen Shen — and their published federated-DSA / LEO-power-control line, funded
under Singapore's FCP (IMDA + NRF). The regulatory spine is Singapore-first
(IMDA spectrum regulation; NTN/LEO PFD as a first-class scenario), with EU
(AI Act / NIS2 / Ofcom) retained as a secondary illustrative analogue.

**Self-score: Strong** on lineage and regulatory match.

**Gap & closure.** The team's published work is the *decision-maker*; the bridge
that runs their agent through this trust layer is the in-build demonstrator
(Criterion 1). **M1–2 / M5–6** close that bridge.

---

## Summary table

| # | Criterion | Weight | Self-score | Key gap → closing milestone |
|---|---|---|---|---|
| 1 | AI-for-RAN relevance & track fit | 20% | Adequate | DSA demonstrator in build → M1–2 / M5–6 |
| 2 | Genuine innovation (audit binding only) | 18% | Strong (narrow) | Externally citable + Sigstore log → M11–12 |
| 3 | Benchmarking-ready code & reproducibility | 16% | Strong | Live over-the-air I/Q (attack coverage now exhaustive: 5 families, 75 tests, + certified unlearning repair) → M5–6 |
| 4 | Datasets | 10% | Adequate | Over-the-air testbed traces → M5–6 |
| 5 | Deployment feasibility & openness | 14% | Strong / Adequate | Operator pilot + prod HSM → M9–10 |
| 6 | Standardization contribution | 12% | Adequate | Calibrate TS 38.104; submit to WG11 → M1–2 / M11–12 |
| 7 | Research lineage & regulatory grounding | 10% | Strong | Bridge demonstrator → M1–2 / M5–6 |

**Honesty note.** No criterion is a Gap after the adversarial campaign landed; the
softest scores are **Adequate**. The strongest honest claim is narrow: a
per-decision, tamper-evident, replayable evidence binding around an AI-RAN agent's
decisions, plus a deterministic output-shield that holds even when the model is
compromised. Everything else — the shield concept, the aggregators, the crypto — is
prior art we compose, not invent, and we **publish where adaptive attacks defeat
our ML-layer defenses** (Min-Max/Min-Sum/Fang vs the robust aggregators; a backdoor
surviving coordinate-median at its breakdown point). The remaining named gaps are a
Sigstore transparency log, an operator/testbed pilot, and live over-the-air I/Q —
milestones, not silent omissions.
