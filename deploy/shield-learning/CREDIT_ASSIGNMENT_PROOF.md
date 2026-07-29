# Shield projection bias on real DeepMIMO — credit-assignment proof

> **This document was rewritten after an adversarial post-mortem
> ([`ERRATA.md`](ERRATA.md)) found most of its learning claims to be artifacts of
> our own harness. The retractions are in [§ What we retract](#what-we-retract),
> stated before the results, with the measurement that killed each claim. The
> safety results stand unmodified and are now checked in both action-space arms.**

- **Runner:** [`../../benchmarks/credit_assignment_loop.py`](../../benchmarks/credit_assignment_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/credit_assignment.json`](../../benchmarks/results/credit_assignment.json)
- **Sample evidence record:** [`../../benchmarks/results/credit_assignment_sample_record.json`](../../benchmarks/results/credit_assignment_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_credit_assignment_evidence.py`](../../tests/test_credit_assignment_evidence.py)
- **Cross-run verifier:** [`../../scripts/verify_credit_assignment.py`](../../scripts/verify_credit_assignment.py)
- **Shared substrate:** [`../../src/horizon_ric/learning/shield_env.py`](../../src/horizon_ric/learning/shield_env.py)
- **Data:** DeepMIMO ASU Campus 3.5 GHz, **4096** ray-traced receivers (Wireless
  InSite), the exact checksum-pinned feature build the DSA and coverage
  benchmarks consume (`features_sha256 = ab4414a1…`).

## What we retract

Every retraction below is backed by a number in the committed result JSON,
produced by the command in [§ Reproduce](#reproduce-real-data-deterministic).

1. **RETRACTED — "Correct attribution fixes the epistemics; the penalty fixes the
   behaviour."** `CORRECTION_PENALTY` was **inert**. Sweeping it 0.0 → 5.0
   changed only `argmax_tie_count`, a reporting field; `illegal_proposals = 32`,
   `projection_rate_last_quarter = 0.016667` and `realised_mean_reward = 0.335835`
   were identical at every value **including zero**. The work was being done by
   an undocumented tie-break asymmetry: regime A broke exact ties by seeded
   random choice while B, C and D used `tied[0]` — the lowest EIRP, which is the
   legal cap. `tied[0]` was silently enforcing the constraint and the penalty was
   getting the credit. **Fixed, not deleted:** the tie-break rule is now uniform
   across all regimes (seeded random choice among exactly tied bins), which makes
   the penalty genuinely load-bearing, and the loop now *measures* that on every
   run — see [§ P2](#p2--the-correction-penalty-now-actually-bites-and-is-measured-doing-so).

2. **RETRACTED — "0.0 value error over the illegal bins" for regimes B and C, and
   the gate built on it.** Both regimes claim **zero** infeasible bins. The 0.0
   came from `_mae_over` scoring an **empty claimed set** as perfect, so the
   headline gate `naive.value_mae_infeasible > projection_aware.value_mae_infeasible`
   was satisfiable by **abstention**: a learner that claims nothing scored
   perfectly. `_mae_over` now returns `NaN` on an empty set, the JSON reports
   `null` plus an explicit `abstained_on_infeasible_bins: true`, and the gate has
   been replaced by one that cannot be met by refusing to answer.

3. **RETRACTED — "mean absolute value error 0.156 served fraction" as a measured
   property of the Shield.** It is exactly `mean_{b>cap} |R(cap) − R(b)|`, a pure
   function of where we put the top of the EIRP grid, and it has **zero** effect
   on realised reward. Measured on the real channels: **0.018311** at grid top
   34 dBm, **0.155697** at our chosen 52, **0.461023** at 110 — a **25×** span
   selected by one constant in the runner. Now published as
   `value_mae_infeasible_sensitivity`.

4. **RETRACTED — the "20-way argmax tie" as evidence about the Shield.** It is
   identically `grid_top − cap + 1 = 52 − 33 + 1`. Same free parameter as (3).
   It is still reported, but only as the *before* half of the masking
   measurement, and the verifier no longer asserts it equal to 20.

5. **RETRACTED by omission — we never compared against doing nothing.** Proposing
   a constant 33 dBm every step, with no data and no learning, realises
   **0.347412** — exactly `optimal_feasible_reward`, and exactly what the best
   deployable learner converges to. **Measured learning gain: 0.000000.** Over
   the full run every deployable learner is *strictly worse* than the zero-data
   policy. Now emitted per regime as `constant_cap_policy_reward`,
   `realised_regret_*` and `learning_gain_over_constant_cap_policy_*`.

6. **RETRACTED — the framing that this benchmark demonstrates a property of
   learning behind a projection operator.** It demonstrates a property of
   *withholding the constraint from the learner*. The Shield knows the feasible
   set analytically. Once the action space is masked with the Shield's own
   predicate, the entire pathology disappears — see [§ P1](#p1--the-feasibility-mask-is-the-shields-own-predicate).

**Not retracted, and re-proven in both arms:** the hard safety guarantee. No
regime, in either arm, ever executed an illegal action.

## The setup

A tabular epsilon-greedy bandit proposes an EIRP on a 1 dB grid **20..52 dBm**
that spans the 33 dBm legal cap. The Shield projects each proposal onto the legal
set; the environment executes the *projected* action and returns its **real**
reward — the served fraction of the 4096 receivers whose measured-channel SINR
clears 0 dB (noise floor −93.99 dBm at 20 MHz). The environment is deterministic
given (frequency, EIRP), so any error in a claimed value is pure mis-attribution.

Four credit regimes, 240 steps each, fixed seeds:

| regime | credit rule |
| --- | --- |
| A `naive_proposed_credit` | observed reward → **proposed** bin (the bug under test) |
| B `projection_aware` | observed reward → **executed** bin |
| C `correction_aware` | as B + a −0.2 behaviour penalty on a projected proposal |
| D `unconstrained_oracle` | **ORACLE, NOT DEPLOYABLE**: counterfactual no-Shield reward → proposed bin |

…now run in **two action-space arms**:

| arm | action space |
| --- | --- |
| `unmasked` | any bin on the 20..52 dBm grid; the learner must rediscover the cap from censored feedback. **Retained only because it is the sole arm in which the bias is observable — not as a recommendation.** |
| `shield_masked` | masked by `ShieldedSpectrumEnv.feasible_mask_grid` → `Shield.is_feasible`. Masked bins are never proposed, so they never enter the value table. |

## P1 — the feasibility mask is the Shield's own predicate

The Shield knows the constraint analytically; the benchmark used to make the
learner rediscover it. The mask now comes from
`ShieldedSpectrumEnv.feasible_mask_grid` → `Shield.is_feasible`, and the loop
verifies at run time that it agrees, bin for bin, with what the Shield actually
projects (`mask_agrees_with_projection_on_every_bin: true`). On this loop's grid
it yields **14 feasible / 19 infeasible** bins.

| regime | argmax tie | | illegal proposals | | infeasible bins claimed | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | **unmasked** | **masked** | **unmasked** | **masked** | **unmasked** | **masked** |
| A naive | **20** | **1** | **201** | **0** | **19** | **0** |
| B projection-aware | 1 | 1 | 37 | **0** | 0 | 0 |
| C correction-aware | 1 | 1 | 32 | **0** | 0 | 0 |
| D oracle | 1 | 1 | 208 | **0** | 19 | 0 |

**The pathology is gone.** With the mask on, no regime proposes an illegal
action, no regime claims a value for an infeasible bin, the 20-way tie collapses
to 1, and the projection rate is 0.0 in every quarter — so the credit rule stops
mattering entirely. What this benchmark measured was the cost of not telling the
learner what the Shield already knows.

**Masking is defence in depth, never a substitute.** `env.step` still disposes
every proposal through the Shield unconditionally and the guard chain still
refuses any uncorrected emit. The unmasked arm proposed 201 illegal actions and
executed **none** — the projection operator, not the mask, is the guarantee.

**An honest cost we did not expect.** Masking makes realised regret *worse*:
mean realised regret over the deployable regimes rises **0.013107 → 0.029593**
(`exploration_cost_of_masking = 0.016486`). The reason is another harness
artifact, now stated: unmasked, 20 of the 33 grid bins are above the cap and all
of them clip onto the single optimal executed action, so uniform exploration
lands on the optimum with probability **0.606**. Masking drops that to
**1/14 = 0.071**, and the learner pays the normal price of exploring. *The
projection operator was silently subsidising the unmasked arm's realised reward.*

## P2 — the correction penalty now actually bites, and is measured doing so

Every run re-trains regime C twice at the **same rng**, at penalty 0.0 and at
0.2, and compares the proposal trajectory and realised scalars
(`correction_penalty_probe`). This is the CI gate, and it is keyed on the
trajectory and on `projected_steps` — deliberately **not** on `argmax_tie_count`,
which moved 20 → 1 with the penalty even in the era when nothing else did.

| arm | penalty | proposal trajectory digest | projected steps | proj rate Q4 | realised reward | tie count |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| unmasked | **0.0** | sha256 `63f00c6800bbe875` | **208** | **0.933333** | 0.336973 | 20 |
| unmasked | **0.2** | sha256 `84a848719be8d301` | **32** | **0.016667** | 0.335835 | 1 |
| masked | 0.0 | sha256 `b8300fdf4577095b` | 0 | 0.0 | 0.321655 | 1 |
| masked | 0.2 | sha256 `b8300fdf4577095b` (**identical**) | 0 | 0.0 | 0.321655 | 1 |

Unmasked, switching the penalty off costs the Shield **6.5× the correction load**
(32 → 208 projections) and drives the last-quarter projection rate from 0.017 to
0.933. The penalty is load-bearing. **Under masking it is provably a strict
no-op** — nothing is ever projected, so it has nothing to penalise: identical
trajectory digest, identical realised reward. Regime C is therefore kept for the
unmasked arm and reported as redundant in the masked one.

That this is the *tie-break fix* doing the work is proven directly: restoring the
old asymmetric `tied[0]` rule in the runner and re-running makes the
`correction_penalty_bites_unmasked` gate **fail** (see
[§ Proof the gates bite](#proof-the-gates-bite)).

**Scope:** this concerns the credit loop only. The exploration-truncation loop's
`correction_aware` is a genuinely different learner that does real work; it is
not touched by this change.

## P3 — regret against a zero-data policy, and what the bias is measured against

### The number that matters most

| policy | realised (all) | realised (Q4) | learning gain vs zero-data (Q4) |
| --- | ---: | ---: | ---: |
| **constant 33 dBm, NO data, NO learning** | **0.347412** | **0.347412** | — |
| B `projection_aware` (best deployable, Q4) | 0.332300 | **0.347412** | **0.000000** |
| C `correction_aware` | 0.335835 | 0.341646 | −0.005766 |
| A `naive_proposed_credit` | 0.334781 | 0.345394 | −0.002018 |

**Measured learning gain of the best deployable learner over a zero-data constant
policy: exactly 0.000000.** Over the full run every deployable learner is
strictly *worse* than proposing 33 dBm blind (−0.0116 to −0.0151 served
fraction — the price of exploring). Nothing in this benchmark demonstrates that
learning beats not learning on this scenario. It never did; we simply never
measured the baseline.

### Unmasked arm, full table

| regime | MAE vs **R** (claimed) | MAE vs **R∘Π** | MAE legal bins | argmax EIRP | proj rate ¼→¼ | realised | Q4 regret | illegal proposals |
| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| A naive | **0.155697** (19/19) | **0.000000** | 0.0 | 33 (**20-way tie**) | 0.633 → **0.933** | 0.334781 | +0.002018 | 201 |
| B projection-aware | **`null` — abstained** (0/19) | `null` | 0.0 | 33 (unique) | 0.417 → 0.050 | 0.332300 | **0.000000** | 37 |
| C correction-aware | **`null` — abstained** (0/19) | `null` | 0.0 | 33 (unique) | 0.400 → **0.017** | 0.335835 | +0.005766 | 32 |
| D oracle (not deployable) | 0.000000 (19/19) | **0.155697** | 0.0 | **52** | 0.633 → 1.0¹ | 0.544978² | −0.252820 | 208 |

¹ D's proposals are still emitted only through the Shield.
² D's realised reward is its **no-Shield world** reward (Q4 0.600232); what it
actually emitted stayed capped (shielded execution mean 0.335696).

### The measuring stick, both ways round

`value_mae_infeasible` scores claims against **R** — the reward the *proposed*
action would earn with no Shield, a counterfactual **no deployed rApp can
observe**. Against **R∘Π** — what the proposal actually *causes* — the naive
learner's infeasible value function is **exactly right (0.000000)** and the
**oracle** is the one wrong by 0.155697. Both columns are now published side by
side. The "bias" is a statement about which counterfactual we chose to score
against, not about a defect the deployed system could detect.

### Grid-top sensitivity of the headline number

| grid top (dBm) | infeasible bins | `value_mae_infeasible` |
| ---: | ---: | ---: |
| 34 | 1 | 0.018311 |
| 40 | 7 | 0.070731 |
| **52 (published)** | 19 | **0.155697** |
| 80 | 47 | 0.342571 |
| 110 | 77 | 0.461023 |

A 25× range on the headline, chosen by `EIRP_MAX_DBM`. The verifier now asserts
that the published MAE is reproduced by this pure formula — if it ever is not,
something unexplained is happening and the run fails.

## The hard safety guarantee — re-proven in both arms

This is the claim the post-mortem could not break, and it is now checked
first-class over **all 8 regime-runs**, not just 4:

- `max_executed_eirp_dbm = 33.0` and `executed_out_of_band_steps = 0` for every
  regime in **both** arms — including the unmasked arm, whose naive regime
  proposed 201 illegal actions and executed none.
- Every `env.step` writes a DecisionRecord carrying **both** the proposed and
  executed actions plus projection flags to a hash-chained store. Chain length
  **1920**, `verify_first_broken_index = -1`.
- Training tables for A, B and C in both arms are **rebuilt exclusively from
  `load_transitions()`** — the verify-gated replay buffer — and match online
  training bit-for-bit (`replay_matches_online = true`). Regime D is deliberately
  *not* rebuildable: its oracle signal is not recorded evidence.
- After a tamper of record #1 on disk, `load_transitions()` raises
  `EvidenceIntegrityError` (`tampered_chain_refused = true`) and re-verification
  pinpoints the tamper at **index 1**.

### A note on legality predicates

Legality is judged by the **Shield's own predicate**, never by a benchmark-private
restatement. The shared substrate's `StepResult.illegal_without_shield` still uses
a *centre-frequency*-in-band test, which is wrong at the band edges. This loop
runs one interior subband (index 3, 3508.33 MHz), so the two predicates agree —
but the loop *measures* the agreement (`illegal_predicate_agrees_with_substrate`)
rather than assuming it, and the verifier fails if it ever breaks.

## Proof the gates bite

A gate that cannot fail is worse than no gate. Three independent injection
sweeps, all reproducible from the scripts described here:

| level | injections | caught |
| --- | ---: | ---: |
| verifier, result-level mutation | 36 | **36/36** |
| evidence tests, result-level mutation | 20 | **20/20** |
| **runner source-level regression, re-run on the real 4096-Rx data** | 5 | **5/5** |

The source-level ones are the load-bearing proof — each patches
`credit_assignment_loop.py`, re-runs the full loop on the real data, and checks
`gate()`:

| injected regression | gates that failed |
| --- | --- |
| masked arm silently unmasked | `mask_collapses_the_tie`, `mask_eliminates_illegal_proposals`, `mask_eliminates_infeasible_claims`, `correction_penalty_is_a_no_op_under_masking` |
| mask derived from a centre-frequency test instead of the Shield | `mask_agrees_with_projection`, `mask_matches_analytic_cap`, `mask_collapses_the_tie`, `mask_eliminates_illegal_proposals`, `bias_is_a_real_claim_not_abstention`, `correction_penalty_is_a_no_op_under_masking` |
| `CORRECTION_PENALTY = 0.0` | `correction_penalty_bites_unmasked` |
| **restore the old asymmetric `tied[0]` tie-break** | `correction_penalty_bites_unmasked` |
| `_mae_over` returns `0.0` on an empty set | `abstention_is_reported_as_abstention` |
| *(baseline, unpatched)* | *none — passes* |

## Reproduce (real data, deterministic)

```bash
# Build the licence-gated DeepMIMO feature file (checksum-pinned, deterministic).
python datasets/deepmimo_asu_3p5/build.py \
  --cache-dir /tmp/dm --features /tmp/dm/features.jsonl \
  --manifest /tmp/dm/manifest.json --samples 4096

# Run both arms x four credit regimes + the trust chain on the real channels.
python benchmarks/credit_assignment_loop.py \
  --features /tmp/dm/features.jsonl --manifest /tmp/dm/manifest.json \
  --out /tmp/dm/credit_assignment.json

# Compare against the committed run (tolerance bands, host-stable decisions).
python scripts/verify_credit_assignment.py \
  --committed-result benchmarks/results/credit_assignment.json \
  --actual-result /tmp/dm/credit_assignment.json \
  --actual-manifest /tmp/dm/manifest.json
```

The runner exits non-zero unless: the Shield binds; the mask is the Shield's own
predicate and agrees with what it projects; masking collapses the tie and
eliminates illegal proposals and infeasible claims; the bias claim is a real
claim rather than abstention, and abstention is reported as abstention; the
zero-data baseline is emitted and no positive learning gain is claimed over it;
`CORRECTION_PENALTY` demonstrably changes behaviour unmasked and demonstrably
does not when masked; nothing illegal was executed in any arm; and the evidence
chain verifies intact, replays exactly, then refuses the tamper.

## Scope and honesty

Real DeepMIMO ray tracing supplies every reward; the Shield, guard chain and
evidence chain are the shipped production code paths. This is **site-specific ray
tracing, not over-the-air capture**, and a **single scenario**. The learner is a
**tabular bandit** over proposed EIRP at one in-band subband — not a deep RL
agent — chosen so the value function is exactly inspectable.

What this artifact now supports, and nothing more:

1. **The hard safety guarantee holds**, under every credit rule, both action
   spaces, and adversarial replay: nothing illegal was ever executed, the
   evidence chain verified, and a tampered chain could not become training data.
2. **Feasibility must come from the Shield, not from a restatement of it.** When
   it does, the credit-assignment pathology this benchmark was built to exhibit
   does not occur.
3. **On this scenario, learning buys nothing.** A zero-data constant-cap policy
   ties the best learner exactly and beats every learner over the full run.
4. **The published bias magnitude was a free parameter of the EIRP grid**, and it
   is zero when scored against the reward the proposal actually causes.

The `unconstrained_oracle` regime is a measuring stick only and is **not
deployable**. The licence-gated raw feature rows are not redistributed; the
committed result JSON is bound to the canonical build by
`features_sha256`/`source_tree_sha256`, and the verifier re-checks a fresh
rebuild with tolerance bands rather than byte-comparing chaotic floats.
