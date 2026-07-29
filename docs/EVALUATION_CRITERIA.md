# Self-assessment against the AI-RAN Alliance evaluation criteria

> **What this file is.** A self-assessment, not an external evaluation. A score
> here does not prove the claim it grades. The authoritative statement of what is
> reproducible is [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md), which names, for every
> quantitative claim, the CI job that re-executes it — and says plainly where
> nothing does.

> **Correction (29 July 2026).** Earlier revisions of this file presented a
> seven-factor rubric with invented percentage weights, described as our
> "anticipated" mapping. That was our own construction, not the Alliance's, and
> weighting criteria we do not own was misleading. The Call for Innovations
> publishes five evaluation criteria and six required content items; this file now
> maps to those and to nothing else. The numbers in earlier revisions were also
> stale — they predated the move onto ray-traced measurements and the
> introduction of CI gates.

## What the Call asks for

Six required content items:

1. Executive summary
2. Problem statement and market relevance
3. Innovative solution and technical approach
4. Deployment feasibility
5. Twelve-month timeline with milestones
6. Expected deliverables and impact

Five evaluation criteria:

1. Innovation and technical merit
2. Relevance to AI-RAN challenges
3. Commercial impact and feasibility
4. Strength of execution plan
5. Alignment with AI-RAN priorities

Expected output types: prototypes; algorithms or models; benchmarking-ready
code; datasets; standardization contributions.

Self-assessment key — **Strong**: built, CI-gated, defensible to a third party
today. **Adequate**: built and published, with a named limitation. **Gap**: known
shortfall, with the work package that closes it named.

---

## 1. Innovation and technical merit

**Self-assessment: Strong on the mechanism, narrow by design.**

The contribution is enforcement realised as a projection on the action space
*downstream* of the planner (`src/horizon_ric/shield/`): the planner emits a
proposal, only the projected action reaches the southbound adapter, and an
out-of-licence emission is unrepresentable rather than merely unlikely. The bound
is worst-case rather than distributional, holds for an arbitrary planner
including a hostile one, and is delivered as a per-decision `SafetyCertificate`
naming violated invariants, corrections and residual margins.

**Prior art we do not claim priority over.** Action projection for safe
reinforcement learning is established: Alshiekh et al. (AAAI 2018) introduced
shielded RL; Dalal et al. (arXiv:1801.08757) the safety layer; Kochdumper et al.
(arXiv:2210.10691) provably safe RL via reachable-set propagation. Robust
aggregators (Krum, coordinate median, trimmed mean) are pre-2019 and are
defeated by Baruch et al. (NeurIPS 2019) and Fang et al. (USENIX Security 2020);
we use them as baselines. The cryptography is off-the-shelf.

**What is new** is the realisation as a standards-facing enforcement plane:
invariants in the operator's regulatory vocabulary rather than in a reward
function, composed to a fixed point so satisfying one cannot silently violate
another; a fail-closed guard chain; emission over O-RAN A1/R1 as an ordinary
rApp; and evidence a third party can replay without trusting us.

**Technical merit is gated, not asserted.** Every headline number is
re-executed by CI on a clean runner and compared field by field —
`scripts/verify_poisoning_shield.py` compares the enforcement counts exactly,
with an independence check on the adjudicating oracle. Each gate is required to
fail under an injected regression; a gate that cannot fail is treated as no gate.

**Gap.** Global novelty is not established — that needs a systematic prior-art
and patent search plus peer review. Coverage is bounded by the enumerated
invariant set, which is a living register rather than a completeness claim.

## 2. Relevance to AI-RAN challenges

**Self-assessment: Strong.**

The blocker this addresses is the one that keeps AI advisory rather than
actuating: a transmission outside a licence is a regulatory breach, not a
degraded KPI, so an assurance argument that terminates in a probability is the
wrong shape for the liability. That question became live in 2026 — an AI
uplink-interference rApp was field-trialled across roughly 1500 5G and 1300 4G
cells and assessed at autonomy Level 3.86 toward Level 4, and the two largest RAN
vendors opened their SMO marketplaces to one another over R1. Operators will
actuate licensed spectrum through planners they did not write and cannot inspect
while retaining sole liability. See [`SIXG_READINESS.md`](SIXG_READINESS.md) and
[`ECOSYSTEM.md`](ECOSYSTEM.md).

The exposure is measured rather than argued: over 8000 decisions on ray-traced
ASU-campus propagation at 3.5 GHz, **4658** requested actions were out of licence
and **0** passed the emission boundary; **2442** of those were demanded by the
real geometry rather than by any attack, because 46% of measured receivers need
more than the licensed EIRP to close their link. A further **461** actions were
inside the emission mask yet harmful by adjacent-channel leakage; after
projection, **0**.

## 3. Commercial impact and feasibility

**Self-assessment: Adequate on feasibility, Gap on commercial validation.**
This is the weakest criterion and we do not dress it up.

**Feasibility.** Enforcement is transparent to the standard interface: 12/12
policies acknowledged `enforceStatus = ENFORCED` by the production Go
`ric-plt/a1` mediator with RMR 4.9.4 driving the official `ric-app/hw-python`
xApp, with no patch to either, confirmed by a three-witness proof and rebuilt
from source by `.github/workflows/xapp-e2e.yml`. It ships as an ordinary
non-real-time rApp: container images, a Helm chart whose rendered templates are
asserted by tests, EIAP and MantaRay onboarding descriptors, OAuth2 and mTLS
credential paths, and a TimescaleDB evidence backend. It is torch-free and
installs on stock Python with no accelerator.

**The control is not a performance tax.** Where the optimum is feasible,
enforcement surrenders **0.0 dB** — which is what makes it adoptable by a RAN
team rather than merely mandated at them.

**Gap.** No operator pilot, no vendor-platform onboarding, and no revenue. The
EIAP and MantaRay packages are wire-contract tests against publicly documented
surfaces, not interoperability results, and no vendor has onboarded this. The
"tens of microseconds per candidate action" enforcement cost is measured on one
host with no committed benchmark. WP1 and WP3 of the twelve-month plan are the
work that converts technical readiness into commercial validation.

## 4. Strength of execution plan

**Self-assessment: Adequate.**

Four work packages, each with a named owner, a falsifiable exit gate and a costed
allocation summing to the US$150,000 request: conformance profile with a second
independent planner shielded unmodified (G1); a second ray-traced scenario and
band reproducing every gate without retuning (G2); measured coexistence, exiting
on positive learning gain over a zero-data policy on measured data (G3);
audit-grade export and E2SM-RC closed loop, exiting on external replication from
the published archive (G4).

**Dominant risk.** WP3 depends on access to a measured interference source or
bench slot, which the project cannot supply for itself. Mitigation is sequencing
— WP1, WP2 and WP4 carry no external dependency — plus held contingency
and a documented fallback to a multi-cell ray-traced incumbent, published as the
weaker result it would be rather than presented as measured.

**Gap.** No written testbed-access commitment yet. Authorship and affiliation
sign-off from the named academic co-authors is outstanding and must precede any
submission that names them.

## 5. Alignment with AI-RAN priorities

**Self-assessment: Strong on openness, Adequate on standardization.**

Apache-2.0, developed in the open with public CI, and every published figure
reproducible from a checksum-pinned archive
([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)). Vendor-neutral by construction: the
layer never reads the planner, so it binds a member's model without that member
disclosing anything about it, and it is complementary to vendor AI stacks rather
than competitive with them. The invariant and certificate schema are offered as a
**candidate** assurance profile for enforceable service objectives, and the
protected-slice floor as a concrete realisation of capacity an AI cannot
reallocate. The threat→control mapping in
[`THREAT_MODEL.md`](THREAT_MODEL.md) is a candidate input to O-RAN WG11, not an
adopted contribution.

**Gap.** Nothing has been submitted to a working group yet, and band-specific
TS 38.104 EIRP and spectral-mask numbers are calibrated per deployment rather
than byte-verified against the specification text here. No 6G or IMT-2030
compliance is claimed or claimable: no such specification exists, and 3GPP
Rel-21 normative work is expected around end-2028.

---

## Expected output types

| Output type | Status |
|---|---|
| Prototypes | The enforcement layer runs as a non-real-time rApp; a live demonstration emits policy through the production A1 mediator into a real xApp, including a Shield-corrected over-power proposal refused rather than emitted. |
| Algorithms or models | Invariant algebra with fixed-point composition, protected-slice floor, certificate schema (`src/horizon_ric/shield/`). |
| Benchmarking-ready code | `benchmarks/` with `scripts/verify_*.py` gates in `.github/workflows/realdata.yml`; each gate proven to fail under injected regression. |
| Datasets | Checksum-pinned measurement manifests (`datasets/`) and per-decision evidence chains. Audit-grade export format is WP4, not yet built. |
| Standardization contributions | Candidate assurance profile; not yet submitted. |

## Where this assessment could be wrong

The honest summary is narrow: a projection-enforced control path whose safety
property does not depend on the planner, with per-decision replayable evidence,
gated by CI against measured propagation. Everything else — the shielding
concept, the aggregators, the cryptography — is prior art we compose rather
than invent. Three specific ways a reviewer should expect to find us short:
the propagation is site-specific ray tracing and not over-the-air capture; the
RAN is open-source software and not vendor equipment; and no third party has yet
reproduced any of it. The red rows of
[`CLAIM_LEDGER.md`](CLAIM_LEDGER.md) list every claim we have had to withdraw or
qualify, including one — "effective SER never exceeds the classical
baseline" — that our own gate caught and corrected to a 1 dB envelope with a
worst measured case of 0.370 dB.
