---
title: Horizon-RIC Technical Validation and Benchmarking Dossier
subtitle: Test setup, proof objectives, measured results and assurance boundaries
date: 25 July 2026
---

# Executive proof statement

Horizon-RIC is a deterministic runtime-assurance layer placed after an AI
decision and before a radio-facing policy is emitted:

> **AI proposes. The Shield decides. Every emitted action carries replayable
> evidence.**

The measured evidence supports a specific proposition:

> Under the stated software, data and simulator conditions, Horizon-RIC
> prevented every independently graded unsafe emission in the poisoned
> benchmark, preserved legal operation across 16,384 external-data DSA
> decisions, produced replayable evidence, and completed an official
> open-source A1 policy lifecycle with local p99 decision-path latency of
> 40.399 microseconds.

“Verified” in this report always means verified under the declared input
domain, dependency versions, dataset, threat model and execution environment.
It does not mean certified, field-proven, carrier-grade or independently
validated.

## Measured anchors

| Measure | Result |
|---|---:|
| Poisoned decision stream | 10,000 decisions; 1,983 unsafe emissions reduced to 0 |
| In-spec but harmful decisions | 489 reduced to 0 |
| External DeepMIMO DSA | 16,384 decisions; 0 illegal post-Shield emissions |
| Best-subband gain | +1.730053 dB over fixed; +1.726286 dB over seeded random |
| Local decision path | p99 40.399 microseconds; 38,932 decisions/s |
| Named integrity attacks | 8/8 detected or blocked |
| AI-PHY PGD fallback | TBLER 0.7625 reduced to 0.0650 |
| Exact subject erasure | L2 distance to retrain 0.0 |

## Proof-objective map

| ID | What must be proven | Current result | Evidence class |
|---|---|---|---|
| P1 | Every emitted action satisfies every programmed invariant | Verified in the generated property domains | Property/unit test |
| P2 | Poisoned proposals cannot produce unsafe emissions under the configured oracle | 1,983 to 0 illegal; 489 to 0 harmful | Adversarial benchmark |
| P3 | DSA utility and safety reproduce on outside channel data | 16,384 decisions; 0 illegal; +1.730 dB vs fixed | External dataset |
| P4 | The local gate has measurable overhead | p99 40.399 microseconds; 38,932 decisions/s | Microbenchmark |
| P5 | Model and evidence tamper are detected | 8/8 named attacks detected or blocked | Integration/adversarial test |
| P6 | Decision legality remains independent of poisoned policy quality | 0 illegal emissions in the audited DSA attacks | Adversarial simulation |
| P7 | One Gaussian release has a reproducible privacy bound | One-round accountant verified; MIA remains diagnostic | Statistical simulation |
| P8 | Secure aggregation is correct and tamper-evident under its trust model | All honest cells verify; tamper/drop detection 1.0 | Functional/crypto benchmark |
| P9 | AI-PHY fallback limits neural degradation to a certified baseline | PGD TBLER 0.7625 to 0.0650 | Receiver simulation |
| P10 | Attributed removal can be certified against retraining | Exact recompute distance 0.0 | Retraining comparison |
| P11 | The adapter completes a real A1 lifecycle with an outside implementation | Register/create/list/status/delete passed | External simulator |
| P12 | The gate can enforce an NTN PFD ceiling | Projection reaches or stays below the configured ceiling | Software demonstrator |

# 1. System breakdown

The system follows one linear action path:

1. **Observe.** Receive the context needed by the configured controls: band,
   bandwidth, transmit power, antenna gain, receiver telemetry, confidence and,
   for NTN actions, slant range and PFD parameters.
2. **Propose.** Allow a learned or conventional policy to suggest an action.
   The policy remains responsible for quality and does not self-certify
   legality.
3. **Certify.** Evaluate numeric, spectral, EIRP, receiver-envelope,
   constellation, lawful-intercept and NTN controls.
4. **Dispose.** Pass a compliant action, project a repairable action, route to
   a certified fallback, or block if safety cannot be recovered.
5. **Emit.** Deliver only the disposed policy through the configured interface.
6. **Record.** Bind the proposal, checks, correction, selected action, model
   identity and context into a certificate and tenant-scoped evidence chain.

Robust aggregation, differential privacy, verifiable secure aggregation and
certified erasure govern how a future policy is built. They never bypass the
decision gate.

## Programmed invariant chain

| Control | Programmed proposition | Disposition |
|---|---|---|
| Numeric domain | Values are finite and physical; required bounds and NTN range are valid | Fail closed |
| Spectral mask | Occupied carrier and guard band stay inside the configured band | Project or block |
| Maximum EIRP | Transmit power plus antenna gain stays below the configured ceiling | Reduce power |
| Neural-RX envelope | Independent measured quality stays within the certified baseline tolerance | Fallback |
| Constellation legality | Order and PAPR remain inside the configured set/envelope | Project or block |
| Lawful-intercept gate | Required context or an audited deployment exception is present | Fail closed |
| NTN PFD ceiling | Downlink power-flux density stays below the configured surface ceiling | Reduce power |

This is an implemented control list, not a complete RF hazard analysis.

# 2. Validation design and environment

The validation program follows six rules:

- grade with an independent oracle whenever possible;
- pin outside datasets and simulators to identifiable versions;
- report distributions and confidence intervals instead of isolated means;
- use independent seeds as the statistical unit;
- state robust-aggregation breakdown and cryptographic trust assumptions; and
- separate a passing mechanism test from field, scale and certification claims.

## Recorded environment

| Field | Value |
|---|---|
| Operating system | Linux 6.12.13, x86-64 |
| Python | CPython 3.12.13 |
| Processor | AMD EPYC 9V74; 9 logical CPUs visible |
| Memory | 15 GiB; no swap |
| NumPy | 2.2.6 |
| pytest | 9.1.1 |
| Hypothesis | 6.161.2 |
| DeepMIMO | 4.0.0 |
| httpx | 0.28.1 |

Five cloud gates completed successfully for the reviewed snapshot: build, fast
tests, lint, outside-data reproduction and OSC A1 live integration.

A direct managed-workspace recheck selected 514 fast tests: 506 passed, 3
skipped and 1 was an expected failure. Four watchdog tests could not create
Unix datagram sockets because that operation is blocked by the workspace.
Excluding that environment-blocked module, the selected run completed with 503
passes, 3 skips and 1 expected failure.

# 3. P1 — Deterministic invariant enforcement

**Objective.** Show that the gate does not emit an action that its configured
invariant chain marks unsafe.

**Hypothesis.**

`emitted => certificate.safe AND every programmed invariant is satisfied`

**Setup.**

- 500 generated finite examples over frequency, bandwidth, transmit power,
  antenna gain, constellation order and PAPR;
- no deadline in the principal property test; and
- a second generator injects NaN, positive/negative infinity, strings and nulls
  into four RF fields.

**Acceptance.**

- zero emitted actions with a failed programmed invariant; and
- every malformed or non-finite case blocks and identifies the numeric-domain
  invariant.

**Observed result.** The tests passed. Emitted actions met the independently
asserted band, EIRP, constellation and PAPR conditions; malformed fields failed
closed.

**What this proves.** Implementation consistency over the generated domains.

**What it does not prove.** Completeness of the invariant list, correctness of
deployment telemetry or calibration for a particular licence.

# 4. P2 — Poisoned decision containment

**Objective.** Measure whether poisoned AI outputs reach emission as unsafe RF
actions.

**Setup.**

- 10,000 decisions;
- poison probability 30%;
- 2,995 poisoned inputs; and
- out-of-band, excessive EIRP, illegal constellation, high PAPR, receiver
  regression and in-band spectral-regrowth attacks.

The grading oracle integrates a carrier power spectral density and measures
out-of-band leakage, adjacent-channel leakage and EIRP. It is not a copy of the
Shield’s band-edge test.

**Acceptance.** Zero illegal post-Shield emissions and zero in-spec-but-harmful
post-Shield emissions.

| Metric | Unguarded | After Shield |
|---|---:|---:|
| Illegal emissions | 1,983 | 0 |
| In-spec but harmful | 489 | 0 |
| Blocked emissions | — | 0 |

**Result.** Pass for the configured adversary, oracle and RF parameters.

**Boundary.** The oracle is not every RF hazard and is not a calibrated
spectrum-analyser test.

# 5. P3 — External-data DSA

**Objective.** Reproduce DSA selection quality and post-Shield legality on an
outside site-specific channel dataset.

**Dataset.**

- DeepMIMO 4.0.0;
- ASU Campus 3.5 GHz Wireless InSite ray tracing;
- 131,931 candidate receivers;
- 85,157 receivers with finite paths;
- 4,096 deterministic samples;
- SISO isotropic antenna;
- 3.5 GHz carrier and 100 MHz bandwidth;
- 1,024 total and 60 selected OFDM subcarriers;
- 10 paths per receiver; and
- 6 subbands.

**Acceptance.**

- exact source archive and extracted-tree hashes;
- exact transform, structure, counts and safety outcomes;
- cross-host floating values within 0.0001 dB absolute tolerance; and
- zero illegal post-Shield emissions.

| Strategy | N | Mean gain (dBW) | Mean regret (dB) | p95 regret (dB) | Illegal |
|---|---:|---:|---:|---:|---:|
| Best subband | 4,096 | -136.520841 | 0.000000 | 0.000000 | 0 |
| Fixed subband 0 | 4,096 | -138.250894 | 1.730053 | 5.732700 | 0 |
| Seeded random | 4,096 | -138.247127 | 1.726286 | 5.484528 | 0 |
| Worst subband | 4,096 | -140.222845 | 3.702003 | 8.834999 | 0 |

Every decision was projected from an intentionally excessive EIRP proposal; no
decision was blocked and no post-Shield action was illegal.

**Lineage.**

- source archive SHA-256:
  `80e4a4983847023158ed61a5a489025d72f4edcdac89edf4ee1323699e1b7da3`
- extracted tree SHA-256:
  `42f9c6eb8f4b4c467fe04c63ae910ae26c75706b85c7decdbcb3a002cb5466b7`
- derived feature SHA-256:
  `ab4414a1dca5d1247f28908003003239033ab329ea04cf2f481a37132969e5af`

The scenario archive contained no separate dataset licence file. Raw and
row-level derived data are therefore not redistributed.

**Boundary.** Ray tracing is not OTA capture. This experiment contains no live
interference dynamics, scheduler, radio or operator traffic.

# 6. P4 — Local control-path performance

**Objective.** Establish a transparent local baseline for invariant evaluation
and certificate construction.

**Setup.**

- 1,000 warm-up and 50,000 timed decisions;
- deterministic valid and projectable terrestrial proposals;
- numeric, spectral, EIRP, receiver-envelope and constellation controls; and
- no network, database, SMO, RIC or radio I/O.

| Metric | Result |
|---|---:|
| Throughput | 38,932.350 decisions/s |
| Mean latency | 25.457 microseconds |
| p50 | 28.342 microseconds |
| p95 | 29.875 microseconds |
| p99 | 40.399 microseconds |
| Maximum | 2.930 milliseconds |
| Projected | 33,334 |
| Blocked | 0 |
| Maximum RSS | 77.6 MiB |

This is a descriptive baseline; no production threshold was pre-registered.
It excludes network/platform I/O, concurrency, failover, endurance and
availability.

# 7. P5 — Evidence integrity and provenance

**Objective.** Detect the named model, manifest, evidence and control-path
manipulations.

**Acceptance.** Eight of eight detected or blocked; zero bypasses.

| # | Attack | Primary control | Result |
|---:|---|---|---|
| 1 | Model swap / untrusted signer | Trusted-key pin | Detected |
| 2 | Weight-byte tamper | Vault integrity + SHA-256 | Detected |
| 3 | Training-manifest tamper | RSA-PSS coverage of manifest hash | Detected |
| 4 | Evidence field tamper | Previous-hash chain | Detected |
| 5 | Evidence reorder / replay | Ordered tenant chain | Detected |
| 6 | Shield self-report spoof | Independent measured receiver telemetry | Contained |
| 7 | Illegal emit attempt | Project/fallback/block disposition | Contained |
| 8 | Lawful-intercept bypass | Fail-closed context gate | Blocked |

Decision records use:

`H_i = SHA-256(H_(i-1) || canonicalJSON(R_i))`

A current host timing probe used one warm-up plus three measured iterations:

| Chain size | Append median | Verify median | Verify per record |
|---:|---:|---:|---:|
| 100 | 32.45 ms | 2.68 ms | 26.76 microseconds |
| 1,000 | 2,804.71 ms | 26.48 ms | 26.48 microseconds |

An internal hash chain detects mutation but does not stop its author from
rebuilding the whole chain. An RFC 3161 timestamp can externally anchor a chain
head.

# 8. P6 — Federated robustness and safety independence

**Objective.** Separate policy-quality resilience from decision legality.

**Setup.**

- tabular Q-learning DSA;
- 6 channels and 3 users;
- 12 clients, including 3 malicious clients (25%);
- FedAvg, coordinate median, trimmed mean and Krum; and
- q-table target, ALIE and Fang attacks.

Clean throughput was 1.0717 per slot. Under the blunt q-table target, FedAvg
throughput fell to 0.0. That proves a poisoned model can destroy utility.

The legality audit deliberately used the weakest quality defence, plain FedAvg:

| Attack | Decisions | Emitted | Illegal after Shield | Evidence chain |
|---|---:|---:|---:|---|
| Q-table target | 300 | 300 | 0 | Intact |
| ALIE | 300 | 295 | 0 | Intact |
| Fang-median | 300 | 290 | 0 | Intact |

**Result.** Legality remained independent of policy quality in the tested
attacks.

**Boundary.** Robust aggregation is conditional:

- FedAvg has a 0% breakdown point;
- Krum requires `n > 2f + 2` and is high variance;
- coordinate median needs fewer than roughly 50% Byzantine values per
  coordinate; and
- trimmed mean requires `beta >= f` and `n > 2beta`.

The eight-attack breakdown sweep used 12 honest clients, 80 dimensions, 8 seeds
and Byzantine fractions from 7.7% to 42.9%. Optimised attacks defeated robust
aggregators near their breakdown assumptions. No aggregator is a universal
defence.

# 9. P7 — Differential privacy and membership inference

**Objective.** Validate one declared Gaussian release and quantify its
privacy/utility trade-off.

**Setup.**

- 64 members and 64 non-members per seed;
- 10 independent seeds;
- 10,000 seed-level bootstrap resamples;
- 20 evaluation episodes;
- public clipping norm `C = 12`;
- replace-one client sensitivity `2C = 24`;
- `delta = 1e-5`; and
- noise multipliers `z in {0, 1, 4, 8}`.

| Release | One-round epsilon | MIA AUC | 95% seed CI | Throughput/slot |
|---|---:|---:|---:|---:|
| No DP | No guarantee | 0.5974 | [0.5481, 0.6420] | 1.2318 |
| z=1 | 5.3026 | 0.5507 | [0.5121, 0.5875] | 1.1000 |
| z=4 | 1.2675 | 0.5263 | [0.5005, 0.5513] | 0.4756 |
| z=8 | 0.6214 | 0.5207 | [0.4696, 0.5647] | 0.1885 |

At z=8, the interval crosses chance. This says that this attacker was not
reliably better than chance over the tested seeds. It does not prove privacy.
The mechanism and accountant carry the formal one-release claim.

The result excludes lifecycle composition, unaccounted releases and other
attackers.

# 10. P8 — Verifiable secure aggregation

**Objective.** Validate aggregation correctness, one-server masking,
tamper/drop detection, cost and the collusion boundary.

**Setup.**

- 2-of-2 additive sharing with Feldman commitments;
- 2,048-bit prime and 2,047-bit subgroup;
- generator 2;
- fixed-point scale 65,536; and
- correctness grid of 2/5/10 clients by 1/4/16 dimensions.

| Test | Scale | Result |
|---|---|---|
| Correctness | 9 client/dimension cells | All verified; max mean error 0.000006171 |
| One-server masking probe | 1,536 samples | 0 masking violations; top-bit chi-square z=1.653 |
| Integrity | 200 trials | Honest verify 1.0; tamper 1.0; drop 1.0 |
| Colluding-server local DP | 16 clients × 8 dimensions × 3 seeds | epsilon 1.2675; all aggregates verified |
| Collusion utility cost | Raw-to-view / aggregate distortion | Mean L2 258.2725 / 68.8738 |

Exact sharing does not survive server collusion: the two shares reconstruct the
submitted update. In the opt-in local-DP mode, clients clip and noise before
sharing, so colluding servers receive the stated private release instead of the
raw update.

The cost for five clients was 0.0395 s at one dimension, 0.3771 s at eight,
1.6267 s at 32 and 6.7227 s at 128. Modular exponentiations dominate and cost
grows approximately linearly with dimension.

# 11. P9 — AI-PHY fallback

**Objective.** Contain neural-receiver degradation without encoding individual
attack signatures.

**Primary setup.**

- 16-QAM;
- SNR 22 dB;
- white-box PGD with L-infinity epsilon 0.12;
- 100 symbols per block; and
- 20-block CRC window.

| Receiver route | PGD TBLER |
|---|---:|
| Neural receiver | 0.7625 |
| Classical baseline | 0.0650 |
| Shield effective | 0.0650 |

Fallback rate was 1.0 and attack reduction was 11.73 times.

An extended suite exercised FGSM, BIM, MIM, transfer and decision-based
boundary attacks across AWGN and four-tap Rayleigh fading with three seeds.
The effective receiver stayed within the configured baseline envelope at every
exercised point.

Under Rayleigh fading, the large AWGN neural/classical gap collapsed because
deep fades amplified perturbations for both receivers. A barrage jammer at JSR
+10 dB produced neural SER 0.897 and classical SER 0.892; routing could not
repair the degraded channel. The gate is fallback, not denoising.

# 12. P10 — Certified unlearning and subject erasure

**Objective.** Verify attributed removal against a gold-standard retrain.

**Subject-erasure setup.**

- 4 clients and 3 subjects per client;
- one subject removed;
- FedAvg and coordinate median;
- seeds 1–3; and
- 400 transitions erased.

| Aggregator | Certified L2 to retrain | Independent L2 | Cheap residual | Tamper rejected |
|---|---:|---:|---:|---|
| FedAvg | 0.0 | 0.0 | 0.0 | Yes |
| Median | 0.0 | 0.0 | 0.566558 mean; 0.587105 max | Yes |

For the non-linear median, exactness required a full recompute. The cheap linear
shortcut did not equal retraining.

In the client-unlearning experiment, a targeted FedAvg backdoor had success
1.0. Full retraining reduced it to 0.0365, while the zero-cost cached shortcut
remained at 1.0. The certificate correctly reported the shortcut’s large
distance to retraining instead of claiming removal.

Attribution remains a prerequisite; a stealthy update that is never attributed
cannot be unlearned.

# 13. P11 — O-RAN A1 interoperability

**Objective.** Exercise the adapter against an outside A1 implementation.

**Setup.**

- official O-RAN-SC sim-a1-interface;
- pinned revision `be2943f57211f62095dc5434099e331df237aafb`;
- `OSC_2.1.0`; and
- a hash-checked container build.

| Lifecycle stage | Acceptance | Result |
|---|---|---|
| Health | Live health endpoint responds | Pass |
| Register | Policy types 20001–20004 accepted | Pass |
| Create | QoS priority policy created | Pass |
| List | Created ID appears | Pass |
| Status | `enforceStatus` appears | Pass |
| Delete | ID disappears after deletion | Pass |

The O-RAN-SC project describes this component as a simulator used to test
Non-RT RIC services without deploying a Near-RT RIC. A passing result therefore
proves simulator REST interoperability, not a deployed OSC platform, operator
network, vendor stack or conformance.

Ericsson and Nokia contract tests use an in-process HTTP transport and are not
counted as vendor interoperability.

# 14. P12 — NTN PFD demonstrator

**Objective.** Validate NTN PFD computation and projection.

The invariant computes:

`PFD = EIRP_dBW - 10log10(4*pi*d^2) - 10log10(BW_MHz)`

**Acceptance.**

- analytic link-budget error below `1e-9`;
- high-power action reduced;
- post-projection PFD at or below `-146 dBW/m^2/MHz + 1e-6`; and
- final certificate safe.

Five tests cover a terrestrial no-op, analytic equality, high-power violation,
low-power compliance and full NTN-Shield disposition. All passed.

The ceiling is representative and must be calibrated to the applicable
service, allocation, geometry and jurisdiction. No satellite link, spectrum
analyser or regulator witnessed the test.

# 15. What is and is not proven

## Proven within the declared scope

- a functioning deterministic gate and machine-readable certificate;
- programmed invariants held across the exercised property and adversarial
  domains;
- DSA utility and safety reproduced on pinned outside ray-tracing data;
- eight named model, evidence and control manipulations were detected or
  contained;
- decision legality remained separate from federated policy quality; and
- the adapter completed a real lifecycle with the official O-RAN-SC simulator.

## Not established

- operation on a live O-RAN network, real radios or operator traffic;
- deployed Ericsson, Nokia or full OSC RIC/SMO interoperability;
- complete coverage of every unsafe RF behaviour;
- formal O-RAN, 3GPP, ITU-R, GDPR, NIS2 or telecom certification;
- security against unknown attacks or arbitrary compromised components;
- exact-share secrecy when both aggregation servers collude;
- carrier-scale latency, throughput, availability, failover or endurance;
- global novelty; or
- independent validation by an identifiable outside institution.

# 16. Next validation campaign

| Objective | Required setup | Required acceptance evidence |
|---|---|---|
| Live closed loop | Named O-RAN testbed, Near-RT RIC, RU/DU or SDR, calibrated RF path and representative traffic | Zero post-gate violations in shadow and controlled runs; retained spectrum and telemetry evidence |
| Commercial interoperability | Named product versions and endpoint/security configuration | Pre-registered lifecycle, rollback and negative tests pass on each target |
| Hazard completeness | Independent RF hazard analysis mapped to requirements and calibration sources | Every hazard controlled, accepted or explicitly excluded with sign-off |
| Carrier-scale performance | Representative topology, I/O, concurrency and hardware | Operator-agreed p99 budget, zero evidence loss, capacity margin, 24-hour soak and recovery |
| Security assurance | Independent red team including key/storage/collusion/wire-input scenarios | No unresolved critical finding; every residual risk has treatment |
| Privacy lifecycle | Enumerate every release, sampling rule and removal event | Full composition meets a pre-approved privacy budget |
| Regulatory/conformance | Qualified reviewers, jurisdiction and standards edition | Signed assessment or certification scope with traceable exceptions |
| Independent validation | Named institution, credentials, conflict disclosure and signed raw logs | Reproducible outside report with findings and limits |

Until this package exists, the defensible position is **validated software
assurance layer and field-pilot candidate**, not certified carrier-grade
control system.

# External sources

1. DeepMIMO, Getting Started:
   <https://deepmimo.net/docs/tutorials/1_getting_started.html>
2. DeepMIMO Quickstart: <https://deepmimo.net/docs/quickstart.html>
3. O-RAN Software Community, A1 Simulator Overview:
   <https://docs.o-ran-sc.org/projects/o-ran-sc-sim-a1-interface/en/latest/overview.html>
4. O-RAN Software Community, Simulator API:
   <https://docs.o-ran-sc.org/projects/o-ran-sc-sim-a1-interface/en/latest/simulator-api.html>
5. Hypothesis documentation: <https://hypothesis.readthedocs.io/>
6. pytest skip and xfail semantics:
   <https://docs.pytest.org/en/stable/how-to/skipping.html>
7. I. Mironov, “Rényi Differential Privacy,” IEEE CSF 2017:
   <https://doi.org/10.1109/CSF.2017.11>
8. RFC 3526: <https://www.rfc-editor.org/rfc/rfc3526>
