# Exploration truncation and graduation under the Shield — real-DeepMIMO proof

> **Corrections:** an adversarial post-mortem found several headline claims in this
> document to be artifacts of the experiment harness rather than properties of the
> Shield. Read [`ERRATA.md`](ERRATA.md) alongside it. The safety results stand; most
> of the learning claims do not.

The Decision Safety Shield projects every unsafe proposal onto the nearest
legal action, so a learner training on its own decision history **never
observes a single real outcome inside the infeasible region**. This artifact
answers the two questions that follow, on real measured-physics data:

1. Does that truncation leave the learner **permanently ignorant** of the
   constraint — proposing illegal actions forever, with the Shield staying
   load-bearing indefinitely?
2. Can a learner that is **told it was corrected** (the evidence chain records
   `projected` / `guard_refused` on every decision) **graduate** — drive its
   own illegal-proposal rate to ~0 so the Shield stops having to intervene?

- **Runner:** [`../../benchmarks/exploration_truncation_loop.py`](../../benchmarks/exploration_truncation_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/exploration_truncation.json`](../../benchmarks/results/exploration_truncation.json)
- **Sample evidence record:** [`../../benchmarks/results/exploration_truncation_sample_record.json`](../../benchmarks/results/exploration_truncation_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_exploration_truncation_evidence.py`](../../tests/test_exploration_truncation_evidence.py)
- **Reproduction verifier:** [`../../scripts/verify_exploration_truncation.py`](../../scripts/verify_exploration_truncation.py)
- **Data:** DeepMIMO ASU Campus 3.5 GHz, **4096** ray-traced receivers (Wireless
  InSite), the exact checksum-pinned feature build the DSA and federated
  benchmarks consume (`features_sha256 = ab4414a1…`).

## The setup and why the pull toward illegality is real

The environment is the shared `ShieldedSpectrumEnv`: reward is the real
**served fraction** (receivers whose SINR clears 0 dB), computed from measured
per-subband channel gains at a −94.0 dBm noise floor. Served fraction is
**monotone increasing in EIRP**, so the unconstrained optimum lies *outside*
the feasible set: the best feasible reward (EIRP at the 33 dBm cap) is
**0.3474**, while a 50 dBm-EIRP proposal would truly earn **0.5918** and the
worst proposal the learners actually made (52 dBm) would earn **0.6101**. The
learner is genuinely pulled over the cap. But every above-cap proposal executes
at the cap and returns the *identical* real reward — a flat plateau with zero
gradient in the entire infeasible region.

Both learners are the **same tabular bandit** (optimistic init, greedy with
seeded random tie-breaks, learning rate 1 in a deterministic environment) over
the same 297-arm action space: proposed EIRP 20..52 dBm in 1 dB bins × 9
frequency bins (6 in band, 3 beyond 3.55 GHz — band legality is a second real
constraint, and 213 of 297 arms are infeasible). Same seed, same 960-step
budget. Every decision goes through the Shield and onto a hash-chained
evidence store; each round the value table is rebuilt **exclusively from
`load_transitions()`** — the verify-gated replay — so training data only ever
comes from the verified chain. The only difference: `reward_only` sees the
replayed reward alone; `correction_aware` additionally reads the chain's
`projected` / `guard_refused` flags and treats a correction as a −0.25 penalty.

## The measured answer (real 4096-Rx run)

Illegal-proposal rate by training quarter, and what the Shield had to do:

| learner | Q1 | Q2 | Q3 | Q4 | shield interventions (of 960) | interventions in Q4 (of 240) | graduated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| reward_only | 0.713 | 0.892 | 0.954 | **0.954** | **873** | **229** | no |
| correction_aware | 0.713 | 0.225 | 0.000 | **0.000** | **255** | **0** | **yes** |

- **Question 1 — yes, truncation is permanent for the reward-only learner.**
  Its illegal rate does not merely persist; it *rises* (0.713 → 0.954) as the
  bandit converges onto the zero-gradient plateau, where 19 of the 20
  reward-maximising arms (as it can see them) are over the cap. The Shield
  corrected 873 of its 960 proposals and was still correcting 229 of its last
  240. The Shield stays load-bearing indefinitely.
- **Question 2 — yes, the corrected-and-told learner graduates.** Identical
  exploration in Q1 (same seed), then the chain's correction flags break the
  plateau's tie: illegal rate 0.713 → **0.000** by Q3, and the Shield's
  intervention count in Q4 is **zero**. Total interventions: 255 vs 873.
- **Counterfactual harm prevented (oracle, never trained on):** 1068 illegal
  actions across both learners would have reached the RAN without the Shield —
  980 over the EIRP cap (worst excess **19.0 dB**, a 52 dBm proposal against a
  33 dBm cap) and 202 out of band. No executed action was ever illegal for
  either learner: max executed EIRP 33.0 dBm, zero out-of-band executions —
  safety held regardless of how badly the learner behaved.
- **The honest epistemic point: neither learner knows the infeasible region.**
  All **213 infeasible bins end training with zero real observations** for both
  learners — the Shield guarantees it. reward_only's value estimate at EIRP 50
  is 0.3474 (the capped reward it was actually served) against a true 0.5918;
  correction_aware's shaped value there (0.0974) is even *further* from the
  truth, deliberately — it is a policy device, not a world model. Graduation is
  learned constraint-*avoidance*, not constraint-*knowledge*.
- **Realised utility is identical: 0.3334 for both** (feasible optimum 0.3474).
  The Shield rescues every illegal proposal onto the cap, so the truncated
  learner loses no reward by staying illegal — which is exactly why reward
  alone can never teach it to stop. What the correction signal buys is not
  reward; it is the Shield's retirement from the loop.

## The trust chain

All 1920 decisions (both learners, every proposal and its executed projection,
with the `projected` / `guard_refused` flags) are on one hash-chained evidence
store, and the learners train **only** through its verify-gated replay:

- the chain **verifies intact** (`verify() = -1`) after training;
- after a single-field tamper of record #1 on disk, `load_transitions()`
  **raises `EvidenceIntegrityError`** — a tampered decision log cannot become
  training data — and re-verification pinpoints the tamper at **index 1**;
- the committed sample record is intact record 0 of that chain: an out-of-band
  3.5917 GHz proposal recorded alongside its projection to 3.54 GHz, flags set.

## Reproduce (real data, deterministic)

```bash
# Build the licence-gated DeepMIMO feature file (checksum-pinned, deterministic).
python datasets/deepmimo_asu_3p5/build.py \
  --cache-dir /tmp/dm --features /tmp/dm/features.jsonl \
  --manifest /tmp/dm/manifest.json --samples 4096

# Run the whole shielded-learning loop on the real channels.
python benchmarks/exploration_truncation_loop.py \
  --features /tmp/dm/features.jsonl --manifest /tmp/dm/manifest.json \
  --out /tmp/dm/exploration_truncation.json

# Check the fresh run against the committed result (tolerance bands, not
# byte-compares of plateau tie-breaks).
python scripts/verify_exploration_truncation.py \
  --committed-result benchmarks/results/exploration_truncation.json \
  --actual-result /tmp/dm/exploration_truncation.json \
  --actual-manifest /tmp/dm/manifest.json
```

The runner exits non-zero unless the pull toward illegality is real
(reward_only Q1 > 0), correction_aware graduates (Q4 below its own Q1 and below
reward_only's Q4) with fewer Shield interventions, no executed action is ever
illegal, and the chain verifies intact then refuses to train once tampered.
The committed-result tests assert the same invariants in the fast pack.

## Scope and honesty

Real DeepMIMO ray tracing drives the reward; the Shield, guard chain and
evidence chain are the shipped modules. This is **site-specific ray tracing,
not over-the-air capture**, a **single scenario**, and the learners are
**tabular bandits over a discretised action space, not deep RL**. Graduation is
demonstrated for this learner and reward shape only (served fraction, monotone
in EIRP, deterministic); a stochastic reward or function-approximation learner
could behave differently and is future work. The experiment could have come out
the other way — the runner's exit code and the committed rates are measured,
not scripted. The licence-gated raw feature rows are not redistributed; the
committed result JSON is bound to the canonical build by `features_sha256` and
`source_tree_sha256`.
