# Shield projection bias on real DeepMIMO — credit-assignment proof

> **Corrections:** an adversarial post-mortem found several headline claims in this
> document to be artifacts of the experiment harness rather than properties of the
> Shield. Read [`ERRATA.md`](ERRATA.md) alongside it. The safety results stand; most
> of the learning claims do not.

This is the Horizon artifact that answers a question the earlier benchmarks
could not: **does the Decision Safety Shield's projection operator bias what a
learner learns?** Every prior Shield exercise ran on tasks whose optimum was
already feasible, where the projection is provably free. Here the environment's
reward (served fraction of real ray-traced receivers) is monotone increasing in
EIRP, so the unconstrained optimum lies *outside* the legal set and the
projection genuinely binds — which is exactly what makes credit assignment
measurable. Same learner, same environment, same seeds; only the credit rule
changes.

- **Runner:** [`../../benchmarks/credit_assignment_loop.py`](../../benchmarks/credit_assignment_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/credit_assignment.json`](../../benchmarks/results/credit_assignment.json)
- **Sample evidence record:** [`../../benchmarks/results/credit_assignment_sample_record.json`](../../benchmarks/results/credit_assignment_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_credit_assignment_evidence.py`](../../tests/test_credit_assignment_evidence.py)
- **Cross-run verifier:** [`../../scripts/verify_credit_assignment.py`](../../scripts/verify_credit_assignment.py)
- **Shared substrate:** [`../../src/horizon_ric/learning/shield_env.py`](../../src/horizon_ric/learning/shield_env.py)
- **Data:** DeepMIMO ASU Campus 3.5 GHz, **4096** ray-traced receivers (Wireless
  InSite), the exact checksum-pinned feature build the DSA and coverage
  benchmarks consume (`features_sha256 = ab4414a1…`).

## The setup and why it is the honest test

A tabular epsilon-greedy bandit proposes an EIRP on a 1 dB grid **20..52 dBm**
that deliberately spans the 33 dBm legal cap (19 of 33 bins are illegal). The
Shield projects each proposal onto the legal set; the environment executes the
*projected* action and returns its **real** reward — the served fraction of the
4096 receivers whose measured-channel SINR clears 0 dB (noise floor −94.0 dBm at
20 MHz). Measured anchors on the real channels: the best feasible reward is
**0.3474** (EIRP at the cap), while proposing EIRP 50 executes at 33 → 0.3474
although its true unconstrained value is **0.5918**. The environment is
deterministic given (frequency, EIRP), so any error in a claimed value is pure
mis-attribution: the measurement isolates the credit-assignment bias exactly.

Four credit regimes, 240 steps each (full 33-bin sweep, then decaying
epsilon-greedy, fixed seeds):

| regime | credit rule |
| --- | --- |
| A `naive_proposed_credit` | observed reward → **proposed** bin (the bug under test) |
| B `projection_aware` | observed reward → **executed** bin |
| C `correction_aware` | as B + a −0.2 behaviour penalty on a projected proposal |
| D `unconstrained_oracle` | **ORACLE, NOT DEPLOYABLE**: counterfactual no-Shield reward → proposed bin |

## The result (real DeepMIMO, committed run)

Value error is |learned − true unconstrained value|, split over the 19 illegal
bins (EIRP 34..52) and the 14 legal bins; "proj rate" is the fraction of emitted
decisions the Shield had to correct, first vs last quarter of training.

| regime | MAE illegal bins (claimed) | MAE legal bins | argmax EIRP | proj rate ¼→¼ | realised reward | illegal proposals |
| --- | ---: | ---: | ---: | :---: | ---: | ---: |
| A naive | **0.1557** (19/19) | 0.0 | 33 (**20-way tie**) | 0.633 → **0.933** | 0.3348 | 201 |
| B projection-aware | 0.0 (0/19 claimed) | 0.0 | 33 (unique) | 0.417 → 0.050 | 0.3323 | 37 |
| C correction-aware | 0.0 (0/19 claimed) | 0.0 | 33 (unique) | 0.400 → **0.017** | 0.3358 | 32 |
| D oracle (not deployable) | 0.0 (19/19, exact) | 0.0 | **52** (grid top) | 0.633 → 1.0¹ | **0.5450**² | 208 |

¹ D's recorded rate: its proposals are still emitted only through the Shield.
² D's realised reward is its **no-Shield world** reward (last quarter 0.6002);
what it actually emitted stayed capped (shielded execution mean 0.3357).

Read honestly:

- **The bias is real, large, and attributable to naive credit.** Regime A books
  the capped reward **0.3474** as the value of *every* illegal EIRP, while the
  true unconstrained values run 0.3657 → 0.6057: mean absolute value error
  **0.156** served fraction over the illegal bins (up to 0.258 at the grid top),
  versus **0** for projection-aware credit. This is censored feedback: above the
  cap every proposal returns the same number, the gradient vanishes, and the
  learned value function flatlines at the wrong level.
- **The bias also poisons behaviour, not just beliefs.** A's terminal value
  table cannot distinguish the legal cap from any illegal EIRP — a **20-way
  argmax tie** spanning all 19 illegal bins — so it keeps proposing illegal
  actions to the very end (projection rate *rises* to 0.933; 201 illegal
  proposals at mean 9.9 dB over the cap, worth a mean counterfactual 0.50 had no
  Shield been present). Naive credit never graduates; it leans on the Shield
  forever.
- **Correct attribution fixes the epistemics; the penalty fixes the behaviour.**
  B and C claim nothing about the illegal region (0 claimed bins) and their
  legal-bin values are exact. C additionally learns to stop proposing illegal
  actions: projection rate falls 0.400 → 0.017 across training.
- **What the Shield hides — said plainly.** Because the projection is exact, the
  *realised* rewards of A, B and C are near-identical (0.332–0.336): with a
  perfect Shield, naive credit costs nothing in realised reward *on this task*.
  The measured harms are the wrong value function, the terminal indifference and
  the sustained correction load — the things that matter the moment such a value
  function is reused (transfer, planning, policy export) or the Shield is not
  exact.
- **Safety has a measurable price.** The oracle's converged no-Shield reward is
  0.6002 vs the best feasible 0.3474: **safety_utility_gap = 0.253** served
  fraction. That is what the 33 dBm cap costs on this scenario — and the oracle
  that measures it needs counterfactual access no deployable rApp has. Its
  argmax (52 dBm) sits at the top of the grid; the true unconstrained optimum
  lies even higher.

## The trust chain (all 960 decisions, replay-gated training)

- Every `env.step` of every regime writes a DecisionRecord carrying **both** the
  proposed and executed actions plus projection flags to a hash-chained store
  (`benchmarks/results/credit_assignment_audit.jsonl`, gitignored; intact
  record 0 is committed as the sample). Chain length **960**, `verify() = -1`.
- Training tables for A, B, C are **rebuilt exclusively from
  `load_transitions()`** — the verify-gated replay buffer — and match online
  training bit-for-bit (`replay_matches_online = true`). Regime D is
  deliberately *not* rebuildable: its oracle signal is not recorded evidence.
- After a tamper of record #1 on disk, `load_transitions()` **raises
  `EvidenceIntegrityError`** (`tampered_chain_refused = true`) — a tampered
  decision log cannot become training data — and re-verification pinpoints the
  tamper at **index 1**.
- **No regime ever emitted an illegal action to the RAN**: max executed EIRP is
  33.0 dBm in all four regimes, all executions in band.

## Reproduce (real data, deterministic)

```bash
# Build the licence-gated DeepMIMO feature file (checksum-pinned, deterministic).
python datasets/deepmimo_asu_3p5/build.py \
  --cache-dir /tmp/dm --features /tmp/dm/features.jsonl \
  --manifest /tmp/dm/manifest.json --samples 4096

# Run all four credit regimes + the trust chain on the real channels.
python benchmarks/credit_assignment_loop.py \
  --features /tmp/dm/features.jsonl --manifest /tmp/dm/manifest.json \
  --out /tmp/dm/credit_assignment.json

# Compare against the committed run (tolerance bands, host-stable decisions).
python scripts/verify_credit_assignment.py \
  --committed-result benchmarks/results/credit_assignment.json \
  --actual-result /tmp/dm/credit_assignment.json \
  --actual-manifest /tmp/dm/manifest.json
```

The runner exits non-zero unless the Shield binds (positive safety/utility gap),
naive credit's illegal-bin value error strictly exceeds projection-aware's,
correction-aware learning graduates, nothing illegal is ever emitted, and the
evidence chain verifies intact, replays exactly, then refuses the tamper.

## Scope and honesty

Real DeepMIMO ray tracing supplies every reward; the Shield, guard chain and
evidence chain are the shipped production code paths. This is **site-specific
ray tracing, not over-the-air capture**, and a **single scenario**. The learner
is a **tabular bandit** over proposed EIRP at one in-band subband — not a deep
RL agent — chosen so the value function is exactly inspectable and the bias
cannot hide in function-approximation error. The `unconstrained_oracle` regime
is a measuring stick only and is **not deployable**. The licence-gated raw
feature rows are not redistributed; the committed result JSON is bound to the
canonical build by `features_sha256`/`source_tree_sha256`, and the verifier
re-checks a fresh rebuild with tolerance bands rather than byte-comparing
chaotic floats.
