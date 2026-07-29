# Exploration truncation and graduation under the Shield — real-DeepMIMO proof

> **This document was rewritten on 2026-07-28 after an adversarial post-mortem
> ([`ERRATA.md`](ERRATA.md)) found that its headline learning numbers had been measured
> against the wrong feasible set.** The safety results stood and still stand. The
> learning claims have been re-measured, several have been retracted outright, and a
> control the original artifact lacked has been added — see
> [What was wrong, and what it is now](#what-was-wrong-and-what-it-is-now). Every
> number below comes from the committed run of the loop on the real 4096-receiver
> DeepMIMO build.

The Decision Safety Shield projects every unsafe proposal onto the nearest
legal action, so a learner training on its own decision history **never
observes a single real outcome inside the infeasible region**. This artifact
answers three questions on real measured-physics data — the third is new, and
it is the one that reframes the other two:

1. Does that truncation leave the learner **permanently ignorant** of the
   constraint — proposing illegal actions forever, with the Shield staying
   load-bearing indefinitely?
2. Can a learner that is **told it was corrected** (the evidence chain records
   `projected` / `guard_refused` on every decision) **graduate** — drive its
   own illegal-proposal rate to ~0 so the Shield stops having to intervene?
3. **Was any of that necessary?** The Shield knows the constraint
   analytically. Handed to the learner as an action mask, it removes illegal
   proposals entirely, at step 1, for free.

- **Runner:** [`../../benchmarks/exploration_truncation_loop.py`](../../benchmarks/exploration_truncation_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/exploration_truncation.json`](../../benchmarks/results/exploration_truncation.json)
- **Sample evidence record:** [`../../benchmarks/results/exploration_truncation_sample_record.json`](../../benchmarks/results/exploration_truncation_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_exploration_truncation_evidence.py`](../../tests/test_exploration_truncation_evidence.py)
- **Reproduction verifier:** [`../../scripts/verify_exploration_truncation.py`](../../scripts/verify_exploration_truncation.py)
- **Data:** DeepMIMO ASU Campus 3.5 GHz, **4096** ray-traced receivers (Wireless
  InSite), the exact checksum-pinned feature build the DSA and federated
  benchmarks consume (`features_sha256 = ab4414a1…`).

## What was wrong, and what it is now

### 1. Legality was scored against a feasible set the Shield does not agree with

The loop carried its own legality test — centre-frequency-in-band plus the EIRP
cap — instead of asking the Shield. The Shield's actual predicate is
*occupied-bandwidth*-in-band (TS 38.104): a 20 MHz carrier centred on either
**edge subband** (3458.333 MHz or 3541.667 MHz) has a legal centre but spills
past the band edge, so the Shield refuses it at **every** EIRP. The two
predicates disagree on **28 of the 297 arms** — 28 of the 84 arms the old test
called legal. Only **56** arms are truly feasible, not 84.

That is why the retracted result reported `illegal_proposal_rate 0.878125`
against `projection_rate 0.909375` for `reward_only`: two names for one
quantity, differing by exactly the 30 proposals per learner that landed on
those edge arms. Every illegal, graduation and blocked rate in the old JSON was
measured against the wrong set.

Legality is now `ShieldedSpectrumEnv.is_feasible` → `Shield.is_feasible`, over
the same payload `env.step()` disposes, so the mask cannot drift from what is
enforced. Under the correct predicate an action is infeasible **iff** the
Shield projects it, and the loop asserts that step-for-step
(`projection_flag_mismatches: 0` for all three learners).

| quantity | retracted | corrected | command |
| --- | ---: | ---: | --- |
| infeasible arms (of 297) | 213 | **241** | `python benchmarks/exploration_truncation_loop.py` → `action_space.n_infeasible_arms` |
| feasible arms | 84 | **56** | `action_space.n_feasible_arms` |
| infeasible fraction | 0.717172 | **0.811448** | `action_space.infeasible_fraction` |
| reward_only `illegal_proposal_rate` | 0.878125 | **0.909375** | `learners.reward_only.closed_loop_summary` |
| reward_only `projection_rate` | 0.909375 | 0.909375 | *(unchanged — it was always right)* |
| correction_aware `illegal_proposal_rate` | 0.234375 | **0.265625** | `learners.correction_aware.closed_loop_summary` |
| correction_aware `projection_rate` | 0.265625 | 0.265625 | *(unchanged)* |
| reward_only illegal proposals blocked | 843 | **873** | `learners.reward_only.illegal_proposals_blocked` |
| correction_aware illegal proposals blocked | 225 | **255** | `learners.correction_aware.illegal_proposals_blocked` |
| illegal actions blocked, all learners | 1068 | **1128** | `counterfactual_harm_prevented` |
| unobserved infeasible bins | 213 | **241** | `learners.*.unobserved_infeasible_bins` |
| reward_only Q1 illegal rate | 0.713 | **0.813** | `learners.reward_only.illegal_proposal_rate_q1` |
| correction_aware Q1 illegal rate | 0.713 | **0.813** | `learners.correction_aware.illegal_proposal_rate_q1` |

The learners' **behaviour did not change** — the same seed produced the same
960 proposals, and `shield_interventions_total` is 873 / 255 exactly as before.
What changed is that the proposals are now classified correctly. Both old
numbers were *under*-reports: the learners were more illegal than we said.

The old values are not deleted. Each learner still carries
`closed_loop_summary.illegal_proposal_rate_superseded_centre_frequency_test`
and `illegal_proposals_blocked_superseded_centre_frequency_test` so the size of
the correction is auditable rather than asserted.

### 2. "A flat plateau with zero gradient" was a property of the Shield, not the reward — **retracted**

The old runner docstring and the old version of this document said every
above-cap proposal "returns the *identical* real reward — a flat plateau with
zero gradient". Stated of the reward, that is false. Measured at the reference
subband (3508.333 MHz) on this build:

| | value |
| --- | ---: |
| true reward at 20 dBm EIRP | 0.135010 |
| true reward at 33 dBm EIRP (the cap) | 0.347412 |
| true reward at 52 dBm EIRP | 0.605713 |
| true slope **below** the cap | 0.016339 / dB |
| true slope **above** the cap | **0.013595 / dB** |
| ratio above/below | **0.832** |

The true served fraction keeps climbing above the cap at **83.2% of its
sub-cap slope**. What is flat is what the learner is *served*, because the
Shield projects every above-cap proposal onto 33 dBm before execution. The
plateau is the Shield's shadow. This is now emitted as
`above_cap_reward_geometry` (with the retraction text in
`claim_retracted`), the verifier fails if the reported above-cap slope is ever
zero or collapses below half the sub-cap slope, and the loop docstring is
corrected.

### 3. correction_aware's "value estimate" was a shaped score — **retracted**

The old artifact reported `value_estimate_at_eirp_50_dbm = 0.097412` for
`correction_aware` with a `mean_abs_error_vs_oracle_above_cap` of 0.40573, and
this document said its estimate was "even *further* from the truth". It is not
a value estimate and that error is not an error. The identity is arithmetic:

```
0.097412  ==  optimal_feasible_reward - CORRECTION_PENALTY  ==  0.347412 - 0.25
0.40573   ==  reward_only's honest 0.15573 + CORRECTION_PENALTY
```

The probe arm is corrected on every visit, so the learner's stored score is
just *reward minus the penalty*. Comparing that penalised score with an
**unpenalised** oracle is a category error. At `CORRECTION_PENALTY = 1.0` the
same field would read **−0.652588** — a negative "value" for a reward bounded
in [0, 1], which settles it.

The field is therefore renamed `shaped_score_at_eirp_50_dbm` for
`correction_aware`, `mean_abs_error_vs_oracle_above_cap` is **dropped** for
that learner, and the JSON carries `shaped_score_identity_holds: true` proving
the identity on the committed run. `reward_only` keeps both fields, where they
are honest.

### 4. Realised utility was published without its baseline — **retracted as evidence of anything**

"Realised utility is identical: 0.3334 for both (feasible optimum 0.3474)" was
true and useless, because the missing comparison is: **a policy that proposes
33 dBm at the best subband forever, with no data and no learning, realises
0.347412 — the feasible optimum exactly.** Every learner here is *worse* than
it. That number is now emitted per learner as `constant_cap_policy_reward`,
with `realised_regret` against it, and the verifier fails if either is absent.

### 5. There was no control — added

`shield_masked` is the same bandit with the Shield's own feasibility mask on
its argmax. See the results table: it is at zero illegal proposals in Q1, not
Q3, and needs zero corrections ever. Its `graduated: true` is flagged
`graduation_is_vacuous: true` in the JSON, because a learner that cannot
propose an illegal action has nothing to graduate from.

### Still standing, unchanged

The safety guarantee and the trust chain. See
[the safety guarantee](#the-safety-guarantee-unchanged-and-now-covering-three-learners).

## The setup

The environment is the shared `ShieldedSpectrumEnv`: reward is the real
**served fraction** (receivers whose SINR clears 0 dB), computed from measured
per-subband channel gains at a −93.99 dBm noise floor. Served fraction is
monotone increasing in EIRP, so the unconstrained optimum lies *outside* the
feasible set: the best feasible reward (EIRP at the cap) is **0.347412**, while
a 50 dBm-EIRP proposal would truly earn **0.591797** and the worst proposal the
learners actually made (52 dBm) would earn **0.610107**. The learner is
genuinely pulled over the cap.

All three learners are the **same tabular bandit** (optimistic init, greedy
with seeded random tie-breaks, learning rate 1 in a deterministic environment)
over the same 297-arm action space: proposed EIRP 20..52 dBm in 1 dB bins × 9
frequency bins (6 in band, 3 beyond 3.55 GHz — band legality is a second real
constraint; **241 of 297 arms are infeasible**). Same seed, same 960-step
budget. Every decision goes through the Shield and onto a hash-chained evidence
store; each round the value table is rebuilt **exclusively from
`load_transitions()`** — the verify-gated replay — so training data only ever
comes from the verified chain. The differences:

- `reward_only` sees the replayed reward alone;
- `correction_aware` additionally reads the chain's `projected` /
  `guard_refused` flags and treats a correction as a −0.25 penalty;
- `shield_masked` sees the reward alone but may only *propose* arms the
  Shield's `feasible_mask_grid` marks feasible.

Masking is **defence in depth, never a substitute**: `env.step` disposes every
action through the Shield unconditionally, so `shield_masked` is exactly as
safe as the other two and its evidence chain proves the same guarantee.

## The measured answer (real 4096-Rx run, corrected predicate)

Illegal-proposal rate by training quarter — now identical to the Shield's
projection rate, by construction:

| learner | Q1 | Q2 | Q3 | Q4 | shield interventions (of 960) | interventions in Q4 (of 240) | graduated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| reward_only | 0.813 | 0.917 | 0.954 | **0.954** | **873** | **229** | no |
| correction_aware | 0.813 | 0.250 | 0.000 | **0.000** | **255** | **0** | yes |
| **shield_masked** | **0.000** | **0.000** | **0.000** | **0.000** | **0** | **0** | vacuous |

And against the zero-data baseline (`constant_cap_policy_reward = 0.347412`,
propose 33 dBm at subband 3 forever, learn nothing):

| learner | realised mean reward | **realised_regret** | realised Q4 | realised_regret Q4 |
| --- | ---: | ---: | ---: | ---: |
| reward_only | 0.333358 | **+0.014054** | 0.347412 | 0.000000 |
| correction_aware | 0.333358 | **+0.014054** | 0.347412 | 0.000000 |
| shield_masked | 0.340183 | **+0.007229** | 0.347412 | 0.000000 |
| *constant 33 dBm, zero data* | *0.347412* | *0.000000* | *0.347412* | *0.000000* |

- **Question 1 — yes, truncation is permanent for the reward-only learner.**
  Its illegal rate does not merely persist; it *rises* (0.813 → 0.954) as the
  bandit converges onto the region where the reward it is *served* is flat.
  The Shield corrected 873 of its 960 proposals and was still correcting 229 of
  its last 240. The Shield stays load-bearing indefinitely.
- **Question 2 — yes, the corrected-and-told learner graduates.** Identical
  exploration in Q1 (same seed), then the chain's correction flags break the
  tie: illegal rate 0.813 → **0.000** by Q3, and the Shield's intervention
  count in Q4 is **zero**. Total interventions: 255 vs 873.
- **Question 3 — and it did not need to.** `shield_masked` is at 0.000 in Q1,
  needs 0 interventions in total, and ends with **half the realised regret**
  (0.007229 vs 0.014054). correction_aware's graduation is the price of
  rediscovering, over 960 steps and 255 corrections, a predicate the Shield
  could state analytically before step 1. **The honest claim is that
  correction-aware learning is a viable fallback where the constraint cannot be
  masked — not that it is the right way to use a Shield that can.**
- **Measured utility gain from learning: none.** All three learners are
  strictly worse than the zero-data constant-cap policy over the full run
  (regret +0.0141 / +0.0141 / +0.0072 served fraction), and all three tie it
  exactly in Q4. Even the learner that proposed illegal actions 95% of the time
  realises the same Q4 reward, because the Shield rescues it onto the cap every
  time. **This experiment demonstrates safe exploration. It does not
  demonstrate that the learning bought any utility, and the numbers say it did
  not.**
- **Counterfactual harm prevented (oracle, never trained on):** **1128** illegal
  actions across the three learners would have reached the RAN without the
  Shield — 980 over the EIRP cap (worst excess **19.0 dB**, a 52 dBm proposal
  against a 33 dBm cap), 346 violating the TS 38.104 spectral mask, of which
  202 had a centre frequency outside the band outright. The 144-proposal
  difference between those last two figures is exactly the class the retracted
  predicate scored as legal.
- **Neither unmasked learner knows the infeasible region.** All **241
  infeasible bins end training with zero real observations** for every learner —
  the Shield guarantees it. reward_only's value estimate at EIRP 50 is 0.347412
  (the capped reward it was actually served) against a true 0.591797, a mean
  absolute error of 0.155203 over the 76 above-cap arms at Shield-feasible
  frequencies. Graduation is learned constraint-*avoidance*, not
  constraint-*knowledge*.

## The safety guarantee (unchanged, and now covering three learners)

This is the part the post-mortem did not dent, and it is stated in the fields
that carry it:

```json
"reward_only":       { "max_executed_eirp_dbm": 33.0, "executed_out_of_band_steps": 0 }
"correction_aware":  { "max_executed_eirp_dbm": 33.0, "executed_out_of_band_steps": 0 }
"shield_masked":     { "max_executed_eirp_dbm": 33.0, "executed_out_of_band_steps": 0 }
```

2880 decisions, 1128 of them illegal proposals — including a 52 dBm demand
against a 33 dBm cap and 202 frequencies outside the licensed band — and **not
one illegal action was executed**. Masking changed nothing about this: the
masked learner is safe for the same reason the others are, because `step()`
disposes unconditionally.

## The trust chain

All **2880** decisions (three learners, every proposal and its executed
projection, with the `projected` / `guard_refused` flags) are on one
hash-chained evidence store, and the learners train **only** through its
verify-gated replay:

- the chain **verifies intact** (`verify_first_broken_index = -1`) after training;
- after a single-field tamper of record #1 on disk, `load_transitions()`
  **raises `EvidenceIntegrityError`** — a tampered decision log cannot become
  training data — and re-verification pinpoints the tamper at **index 1**
  (`verify_after_tamper_index = 1`);
- the committed sample record is intact record 0 of that chain: an out-of-band
  3.5917 GHz proposal recorded alongside its projection to 3.54 GHz, flags set.

## Reproduce (real data, deterministic)

```bash
# Build the licence-gated DeepMIMO feature file (checksum-pinned, deterministic).
python datasets/deepmimo_asu_3p5/build.py \
  --cache-dir /tmp/dm --features /tmp/dm/features.jsonl \
  --manifest /tmp/dm/manifest.json --samples 4096

# Run the whole shielded-learning loop on the real channels (~70 s, 3 learners).
python benchmarks/exploration_truncation_loop.py \
  --features /tmp/dm/features.jsonl --manifest /tmp/dm/manifest.json \
  --out /tmp/dm/exploration_truncation.json \
  --audit-path /tmp/dm/exploration_truncation_audit.jsonl

# Check the fresh run against the committed result (tolerance bands, not
# byte-compares of plateau tie-breaks).
python scripts/verify_exploration_truncation.py \
  --committed-result benchmarks/results/exploration_truncation.json \
  --actual-result /tmp/dm/exploration_truncation.json \
  --actual-manifest /tmp/dm/manifest.json
```

The runner exits non-zero unless: the pull toward illegality is real
(reward_only Q1 > 0); correction_aware graduates with fewer Shield
interventions; **legality is the Shield's own predicate, so
`illegal_proposal_rate` equals `projection_rate` with zero per-step
mismatches**; **the mask alone eliminates illegal proposals**; **every learner
reports `realised_regret` against the zero-data constant-cap policy**;
**correction_aware reports a shaped score and no oracle error against it**; no
executed action is ever illegal; and the chain verifies intact then refuses to
train once tampered. The verifier re-checks all of it against a fresh run, and
the committed-result tests assert the same invariants in the fast pack.

### These gates were proven to fail

A gate that cannot fail is worse than no gate, so each was demonstrated against
a deliberately injected regression (each injection was reverted immediately;
the injected copies live outside the repo):

| injected regression | runner | verifier says |
| --- | :---: | --- |
| score legality with the retracted centre-frequency test | exit 1 | `reward_only: illegal_proposal_rate 0.878125 != projection_rate 0.909375; legality is being scored against a feasible set the Shield does not agree with` |
| remove the Shield mask from `shield_masked` | exit 1 | `the Shield's own feasibility mask did not eliminate illegal proposals` |
| report correction_aware's shaped score as a value estimate again | exit 1 | `correction_aware still reports a penalised score as a value estimate` |
| report the *proposed* EIRP as the executed one (safety field) | exit 1 | `reward_only: an executed action exceeded the EIRP cap` |
| stop emitting `realised_regret` | exit 1 | `reward_only: realised_regret is not reported` |
| hard-code the stale 213-arm infeasible count | exit 0 | `the committed result was measured against a different feasible set than the fresh run` |

The last row is the one worth reading twice: the runner's own exit code did
**not** catch it, because a runner cannot notice that its own reported constant
is stale. Only the verifier's fresh-versus-committed comparison did. That is
the failure mode the retracted result actually had, and it is why the gate
lives in the verifier and not only in the loop.

## Scope and honesty

Real DeepMIMO ray tracing drives the reward; the Shield, guard chain and
evidence chain are the shipped modules. This is **site-specific ray tracing,
not over-the-air capture**, a **single scenario**, and the learners are
**tabular bandits over a discretised action space, not deep RL**. Graduation is
demonstrated for this learner and reward shape only (served fraction, monotone
in EIRP, deterministic); a stochastic reward or function-approximation learner
could behave differently and is future work. The experiment could have come out
the other way — the runner's exit code and the committed rates are measured,
not scripted.

Read the learning results against the two baselines this document now carries
and previously did not: the **zero-data constant-cap policy**, which no learner
beats, and **`shield_masked`**, which gets for free what `correction_aware`
spends 960 steps learning. What this artifact demonstrates is that a projection
Shield makes exploration safe without exception, and that a learner can be
taught to stop fighting it. What it does **not** demonstrate — and what the
first version of this document implied — is that the Shield-plus-learner
arrangement earns more utility than not learning at all, or that the correction
curriculum is the best available use of a Shield that can state its constraint
analytically.

The licence-gated raw feature rows are not redistributed; the committed result
JSON is bound to the canonical build by `features_sha256` and
`source_tree_sha256`.
