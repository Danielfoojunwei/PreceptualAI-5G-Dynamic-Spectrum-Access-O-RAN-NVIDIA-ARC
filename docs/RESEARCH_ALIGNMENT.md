# Research alignment — NTU / SCRIPTS / DTC and Singapore's FCP

> This document establishes the research lineage Horizon-RIC builds on, the
> team behind the capability base, and the honest division of labour between
> that team's published federated-RL spectrum work and the security / safety /
> audit **trust layer** this project contributes.

Horizon-RIC is not a standalone idea. It is the **trust layer** that makes a
specific, already-published line of federated deep-RL spectrum-access research
deployable on regulated, licensed, and satellite spectrum. This file names that
research, its authors, and its funder, and is explicit about what is *built*
versus *roadmap*.

---

## 1. Team and capability base

The federated / hierarchical deep-RL agents that **make the decisions**
Horizon-RIC guards come from the NTU Singapore group at **SCRIPTS** (Strategic
Centre for Research in Privacy-Preserving Technologies & Systems) and the
**Digital Trust Centre (DTC)**.

| Role | Person | Affiliation |
|---|---|---|
| Principal investigator | **Prof. Kwok-Yan Lam** | NTU; leads SCRIPTS, co-directs the Digital Trust Centre (DTC) |
| Researcher | **Dr Li Feng** (F Li) | NTU SCRIPTS |
| Researcher | **Bowen Shen** (B Shen) | NTU SCRIPTS |

### Publications forming the capability base

These are the group's real, published results on dynamic spectrum access (DSA),
federated / hierarchical deep RL, and privacy-preserving resource management.
They are the **decision-making** capability Horizon-RIC wraps:

1. F. Li, B. Shen, J. Guo, K.-Y. Lam, G. Wei, L. Wang, *"Dynamic spectrum access
   for IoT based on federated deep reinforcement learning,"* **IEEE Transactions
   on Vehicular Technology**, 71(7):7952–7956, 2022.
2. S. Zhang, K.-Y. Lam, B. Shen, L. Wang, F. Li, *"Dynamic spectrum access for
   IoT with hierarchical federated deep reinforcement learning,"* **Ad Hoc
   Networks**, 149:103257, 2023.
3. F. Li, J. Yang, K.-Y. Lam, B. Shen, G. Wei, *"Dynamic spectrum access for IoT
   with joint GNN and DQN,"* **Ad Hoc Networks**, 163:103596, 2024.
4. B. Shen, K.-Y. Lam, F. Li, L. Wang, *"Privacy-Aware Spectrum Pricing and Power
   Control Optimization for LEO Satellite IoT,"* **IEEE Transactions on Wireless
   Communications**, 2025.
5. F. Li, Y. Wang, K.-Y. Lam, B. Shen, L. Wang, *"A Secure Dynamic Spectrum
   Access Scheme for IoT With Swarm Learning,"* **IEEE Internet of Things
   Journal**, 2025.
6. F. Li, S. Shui, K.-Y. Lam, B. Shen, L. Wang, *"Secure Dynamic Spectrum Access
   in IoT Based on Machine Unlearning,"* 2025.
7. B. Shen, *"Privacy-preserving resource management in LEO satellite IoT,"* MSc
   thesis, NTU, 2025.

The throughline across these works: **federated and hierarchical deep-RL agents
(including a joint GNN+DQN formulation) that select channels and power under
privacy and adversarial constraints**, extended into LEO-satellite IoT power
control and pricing. That is the agent Horizon-RIC treats as the system under
trust.

**Machine-unlearning sub-line.** Publication 6 above — *Secure Dynamic Spectrum
Access in IoT Based on Machine Unlearning* — places **machine unlearning** squarely
inside this team's own DSA agenda, and it sits within Prof Lam's broader
federated-unlearning programme: *Privacy-Preserving Federated Unlearning with
Certified Client Removal* (Z. Liu, H. Ye, Y. Jiang, J. Shen, J. Guo, I. Tjuawinata
& K.-Y. Lam, arXiv:2404.09724, 2024) — the *Starfish* result that the unlearning
guarantee is a **bound on the distance to a model retrained from scratch** — and the
field survey *A Survey on Federated Unlearning: Challenges, Methods, and Future
Directions* (Z. Liu, Y. Jiang, J. Shen, M. Peng, K.-Y. Lam, X. Yuan & X. Liu, **ACM
Computing Surveys**, 2024). Horizon-RIC implements **certified federated unlearning**
on the DSA policy directly bridging this line (§3; `THREAT_MODEL.md` §8).

---

## 2. Funding lineage — Singapore's FCP

This research sits within Singapore's **Future Communications R&D Programme
(FCP)** — funded by **IMDA** and the **National Research Foundation (NRF)**,
roughly **S$70M**, led by **SUTD** with **NTU** as a partner. FCP funds the
exact surfaces this project touches: **non-terrestrial networks (NTN)**,
**network orchestration**, **mobile edge computing (MEC)**, and, critically,
**security**.

Horizon-RIC's contribution maps to the FCP **security** and **NTN /
orchestration** thrusts: it is the security and audit control that lets the
FCP-funded federated DSA / LEO power-control agents run on licensed and
satellite spectrum where a regulator (IMDA) requires demonstrable, replayable
proof of lawful behaviour.

---

## 3. Honest division of labour

The line between the prior work and this project is sharp, and we state it
plainly so reviewers can verify it.

**The NTU/SCRIPTS team built the decision-makers:**

- Federated and hierarchical deep-RL agents (joint GNN+DQN) that **make**
  dynamic-spectrum-access decisions — which channel, which power — under privacy
  and adversarial constraints (publications 1–3, 5–6).
- Privacy-preserving spectrum pricing and **power control for LEO satellite
  IoT** (publications 4, 7).

**Horizon-RIC adds the security / safety / audit trust layer** that makes those
federated decisions deployable on **regulated / licensed / satellite** spectrum:

| Trust-layer mechanism | What it does for the prior work | Code |
|---|---|---|
| **Robust aggregation** | Bounds a poisoning client's pull on the *shared DSA policy* during federated training (Krum / median / trimmed-mean). | `src/horizon_ric/federated/robust.py` |
| **Shamir secure aggregation** | Hides individual client updates — the successor to the team's privacy-preserving power-control line. | `src/horizon_ric/federated/secure.py` |
| **Certified federated unlearning** | *Removes* an attributed poisoning client's contribution from the shared DSA policy post-hoc — the **repair** layer for the backdoor robust aggregation only *bounds* — and binds the removal to a signed, audit-chainable `UnlearningCertificate` (Starfish-style distance-to-retrain bound + backdoor-probe). Bridges the team's own machine-unlearning-for-DSA work (pub. 6) and Lam's certified-client-removal line. | `src/horizon_ric/federated/unlearning.py` |
| **Decision Safety Shield** | A *model-independent* projection of any agent's chosen channel / power onto the TS 38.104 spectral mask + EIRP ceiling + an LEO/NTN power-flux-density (PFD) ceiling. | `src/horizon_ric/shield/` |
| **Hash-chained, RFC-3161-anchored decision record** | Per-decision evidence a regulator can replay (counterfactual + tamper-evident chain + timestamp anchor). | `src/horizon_ric/evidence/` |

The crucial honesty point: **the genuine novelty Horizon-RIC claims is only the
decision-level evidence / audit binding** (per-decision `SafetyCertificate` +
SHA-256 hash chain + RFC-3161 anchor + counterfactual replay + model-provenance
threading) — now extended to a signed, audit-chainable **unlearning certificate**.
The safety-shield concept, the aggregators, the crypto, **and the unlearning
algorithms themselves are prior art** (the latter Prof Lam's own federated-unlearning
line) — see the "Prior art & what's actually new" section in
[`README.md`](../README.md). What is ours is the *binding*: making the *NTU team's*
federated DSA / LEO decisions — and now the unlearning operations that repair them —
tamper-evidently auditable on licensed spectrum, not reinventing shielding,
aggregation, or unlearning.

---

## 4. Demonstrator — federated DSA on licensed / NTN spectrum

To make the division of labour concrete, the project ships a demonstrator that
takes a federated DSA agent in the spirit of the publications above and runs it
through the Horizon-RIC trust layer.

**Built / in build (the bridge, owned by a parallel work-stream):**

- A federated DSA agent — `src/horizon_ric/spectrum/`.
- A benchmark — `benchmarks/secure_dsa_benchmark.py`, with committed results at
  `benchmarks/results/secure_dsa.json`.
- A committed dataset — `datasets/spectrum_dsa/` with a `DATASHEET.md`.
- An **LEO PFD invariant** in the Shield — the power-flux-density ceiling that
  carries the satellite power-control line (publications 4, 7) into a
  regulator-checkable constraint.

**Already built (the trust mechanisms the demonstrator uses):**

- Robust aggregation (`src/horizon_ric/federated/robust.py`) and Shamir secure
  aggregation (`src/horizon_ric/federated/secure.py`).
- The Decision Safety Shield (`src/horizon_ric/shield/`) and its
  `SafetyCertificate` (`src/horizon_ric/shield/certificate.py`).
- The hash-chained, RFC-3161-anchored evidence store
  (`src/horizon_ric/evidence/`).

> Status note: the `spectrum/`, `secure_dsa_benchmark.py`, and
> `datasets/spectrum_dsa/` artefacts are under active construction by a parallel
> work-stream. Where this document references them, treat them as the
> **demonstrator-in-build**; the underlying trust mechanisms they depend on are
> already working code with tests.

---

## 5. Regulatory relevance — IMDA / Singapore first, NTN / LEO first-class

Because the funding and the team are Singaporean, the **primary** regulatory
spine is Singapore's:

- **IMDA spectrum regulation.** Singapore's Infocomm Media Development Authority
  licenses spectrum and sets emission / EIRP conditions; a federated DSA agent
  on licensed bands must demonstrably stay within those licence conditions. The
  Shield's per-decision projection plus the replayable evidence record is exactly
  the artefact an IMDA-style regulator would ask for.
- **NTN / LEO as a first-class scenario.** The LEO satellite IoT power-control
  line (publications 4, 7) drives the **PFD-ceiling invariant**: a non-terrestrial
  link must respect a power-flux-density limit at the Earth's surface to protect
  co-channel terrestrial services. Horizon-RIC treats this NTN/LEO scenario as a
  first-class invariant, not an afterthought.

The EU framing (EU AI Act high-risk classification, NIS2 incident reporting,
Ofcom explainability) is retained as a **secondary**, illustrative regulatory
analogue in [`README.md`](../README.md) and `docs/compliance/eu_ai_act.md` — it
shows the same trust gap appears in multiple jurisdictions, but it is not the
spine of this proposal.

---

## 6. Built vs roadmap — explicit

| Item | Status |
|---|---|
| Robust aggregation (Krum / median / trimmed-mean) | **Built**, tested |
| Shamir secure aggregation | **Built**, tested (honest-but-curious only — see README caveats) |
| Decision Safety Shield + `SafetyCertificate` | **Built**, tested |
| Hash-chained + RFC-3161-anchored evidence | **Built**, tested (real public TSAs) |
| Model-provenance threading | **Built**, tested |
| LEO PFD invariant | **In build** (demonstrator) |
| Federated DSA agent `src/horizon_ric/spectrum/` | **In build** (parallel work-stream) |
| `benchmarks/secure_dsa_benchmark.py` + results | **In build** (parallel work-stream) |
| `datasets/spectrum_dsa/` + `DATASHEET.md` | **In build** (parallel work-stream) |
| Malicious-server FL (verifiable secret sharing) | **Roadmap** (M3–4) |
| Differential-privacy accountant | **Roadmap** (M3–4) |

See [`docs/EVALUATION_CRITERIA.md`](EVALUATION_CRITERIA.md) for the criterion-by-
criterion self-assessment, including where the demonstrator's in-build status is
a scored gap.
