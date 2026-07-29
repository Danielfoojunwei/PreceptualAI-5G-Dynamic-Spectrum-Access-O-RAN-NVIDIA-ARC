# Claim ledger

Every quantitative claim made about this system, and exactly what backs it.

A claim is **green** only if a third party can reproduce it without asking us
anything: a CI job on a clean runner rebuilds the inputs from a checksum-pinned
source, re-runs the measurement, and a verifier fails the build if the number
moved. **Amber** means the command and the result are published and the run
reproduces locally, but nothing re-executes it automatically. **Red** means it
is not reproducible as stated and must be softened or withdrawn.

The distinction that matters: a committed result file is an assertion *about a
file*. Only a gate makes it evidence *about the system*.

Audited 29 July 2026. Numbers below are as published on this branch.

## Green — re-executed by CI on a clean runner

| Claim | Evidence | Gate |
|---|---|---|
| 12/12 policies `enforceStatus = ENFORCED` against the production Go `ric-plt/a1` mediator with RMR 4.9.4 driving `ric-app/hw-python`, three-witness | `deploy/xapp-e2e/results/xapp-e2e-proof.json` | `.github/workflows/xapp-e2e.yml` builds RMR, the mediator and the xApp **from source** on a clean runner, drives the pipeline under `HORIZON_ONCE_REQUIRE_ACCEPTED=1`, then asserts with `scripts/xapp_e2e_proof.py` |
| 12/12 accepted against the O-RAN SC A1 simulator | same run, dialect `osc_a1` | `osc-a1-simulator` job |
| 4096 receivers, six subbands, ASU campus 3.5 GHz, checksum-pinned | `datasets/deepmimo_asu_3p5/manifest.json` | `realdata.yml` rebuilds from the pinned archive; `scripts/verify_deepmimo_reproduction.py` compares field by field |
| 8000 decisions, **4658** illegal requests → **0** after projection; **2442** of them from real geometry alone; **461** in-mask-but-ACLR-harmful → **0** | `benchmarks/results/poisoning_shield.json` | `scripts/verify_poisoning_shield.py`, exact integer comparison, plus direct safety assertions and an independence check on the adjudicating oracle |
| FLTrust 11.97 dB vs clean FedAvg 11.92 dB; Krum 14.46 dB; Krum fails beyond 3 of 10 | `benchmarks/results/federated_coverage.json` → `byzantine_breakdown` | `scripts/verify_federated_coverage.py` |
| DP at fixed certified ε: clip calibration moves utility without changing the guarantee | same file → `dp_at_fixed_epsilon` | same verifier |
| Learning gain **+0.0096 to +0.0696** where the optimum is interior, **+0.000000** where it is at the cap; 0.0 dB utility surrendered | `benchmarks/results/projection_capacity_2x2.json` | `scripts/verify_committed_benchmark.py`; both capability gates asserted directly |
| 11/11 integrity probes detected or blocked, 0 bypassed (3 replay captured 5GAD traffic byte for byte) | `benchmarks/results/integrity_attack_suite.json` | same verifier, after `datasets/5gad_inl/build.py` fetches the captured corpus — without it the suite **skips** those three and honestly reports 8/8, which is how the gate first failed |
| Neural-receiver error under white-box attack stays inside the certified 1 dB envelope of the classical baseline, every attack, every regime | `benchmarks/results/evasion_suite.json` | same verifier, envelope check |
| Benchmarks provably consume the measured data (perturb the input, outputs must move or the benchmark must reject it) | — | `scripts/verify_data_dependence.py` |
| Anti-jam, beam-management, mobility-handover, credit-assignment, exploration-truncation, safety-utility-frontier | `benchmarks/results/*.json` | dedicated `scripts/verify_*.py`, all in `realdata.yml` |

## Amber — published and reproducible, not yet CI-gated

| Claim | Where | Why not green |
|---|---|---|
| Secure federated DSA policy quality and legality audit | `benchmarks/results/secure_dsa.json` | runs locally from a documented command; no verifier yet |
| Jamming siting / vulnerability map | `benchmarks/results/jamming_suite.json` | as above |
| Neural-RX PGD robustness | `benchmarks/results/neural_rx_pgd.json` | as above |
| PHY fading evaluation | `benchmarks/results/phy_fading.json` | as above |
| DP privacy suite ε audit (empirical lower bound vs analytic) | `benchmarks/results/dp_privacy.json` | long-running; the ε figure quoted externally comes from `federated_coverage.json`, which **is** gated |
| TimescaleDB SQL-level tamper detection | `deploy/EVIDENCE_BACKENDS.md` | needs a live TimescaleDB container; exercised manually, not in CI |
| Enforcement cost "tens of microseconds per candidate action" | — | measured on one host; no committed benchmark or gate |

## Red — must be stated with its limits, or not stated

| Claim | Status |
|---|---|
| "None reached the radio" | **Withdrawn.** There was no over-the-air capture and no vendor radio. Correct form: *no non-compliant action passed the Shield's software emission boundary in the tested O-RAN control path.* |
| "Structurally incapable of emitting an unlicensed action" | **Qualified.** Holds within the enumerated invariants, correctly configured limits, trusted telemetry and the guarded adapter path. It does not cover a missing invariant, a misconfigured licence limit, compromised telemetry, or a bypass path around the adapter. |
| "Effective SER never exceeds the classical baseline" | **Corrected.** The bound is `NeuralRxEnvelopeInvariant` with 1 dB tolerance. Worst measured case is 0.370 dB worse than classical, under fading — inside the envelope, but not "never worse". |
| OCUDU gNB joined to FlexRIC over E2 | **True but narrow.** Real E2 Setup, RIC admits it as `ngran_gNB`, E2SM-KPM and E2SM-RC registered — but only the CU-CP agent attaches, there is no UE so KPM indications would be zeros, transport is a disclosed UDP substitution (this kernel has no SCTP), and it is one machine with no CI. Proves E2AP/E2SM interoperability, not SCTP conformance or closed-loop control. |
| Ericsson EIAP / Nokia MantaRay interoperability | **Not claimed.** Descriptor packages and wire-contract tests against the publicly documented surface only. No vendor platform has onboarded this. |
| Any 6G / IMT-2030 compliance | **Not claimable.** No 6G specification exists; 3GPP Rel-21 normative work is expected ~end-2028. |

## Reproducing the green rows

```sh
python datasets/deepmimo_asu_3p5/build.py --cache-dir <cache> \
  --features features.jsonl --manifest manifest.json
python benchmarks/poisoning_shield_benchmark.py \
  --features features.jsonl --manifest manifest.json --out fresh.json
python scripts/verify_poisoning_shield.py --fresh fresh.json
```

`.github/workflows/realdata.yml` and `.github/workflows/xapp-e2e.yml` are the
authoritative recipes; they run on every change to the files they cover.

## Note on determinism

Several benchmarks draw from `numpy.random.Generator`, which carries no
cross-version bit-stream guarantee, so a numpy bump can act as a reseed. The
`realdata` job installs `numpy==2.2.6` (required by `deepmimo`) while the
runtime lock pins `2.4.6`. The poisoning→Shield counts were checked under both
and are identical, which is why that gate compares exactly; verifiers for float
aggregates use stated tolerances instead. Every result carries a `runtime`
block recording the numpy and Python versions it was produced under.
