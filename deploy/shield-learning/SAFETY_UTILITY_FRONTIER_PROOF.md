# The safety–utility frontier on real DeepMIMO — what the Shield costs

> **Corrections applied.** An adversarial post-mortem ([`ERRATA.md`](ERRATA.md))
> found several headline claims in this document to be artifacts of the experiment
> harness rather than properties of the Shield. This revision **fixes the loop and
> restates the numbers**; the retractions are listed below and are not hedged. The
> safety results stand unchanged.

## What this document previously got wrong (retracted)

Every number below was re-measured on the same real 4096-receiver DeepMIMO build
after the fix; the command that produced each one is in
[Reproduce](#reproduce-real-data-deterministic).

1. **RETRACTED: two of the six `best_feasible_reward` figures were the reward of
   actions the Shield refuses.** The loop ranked subbands over **all six** centres.
   The Shield's in-band test is *occupied-bandwidth*-in-band (TS 38.104), not
   centre-frequency-in-band, so subbands 0 and 5 — whose 20 MHz allocation spills
   past a band edge — are **infeasible at every EIRP** (`spectral_mask_ts38104`).
   The ranking picked subband 0 at the 20 dBm cap and subband 5 at 52 dBm, and a
   silent ~1.7 MHz frequency projection then rescued the emit, which is why the
   error was invisible in the safety fields. A quantity named *best **feasible**
   reward* was being computed over infeasible actions.

   | cap | published | corrected | Δ |
   | ---: | ---: | ---: | ---: |
   | 20 dBm | 0.139160 (subband **0**, infeasible) | **0.138428** (subband 4) | −0.000732 (−3 Rx) |
   | 52 dBm | 0.610107 (subband **5**, infeasible) | **0.607422** (subband 4) | −0.002686 (−11 Rx) |

   Consequently `utility_forgone` at 20 dBm rises 0.860840 → **0.861572** and at
   52 dBm 0.389893 → **0.392578**; the 52 dBm reference anchor falls 0.610107 →
   **0.607422** and its forgone-vs-33 dBm 0.262695 → **0.260010**; the marginal
   price 26→20 dBm rises 0.017863 → **0.017985** and 52→46 dBm falls 0.010579 →
   **0.010132**. The four middle caps (26/33/40/46 dBm) are unchanged, so **the
   headline `utility_forgone_at_operational_cap = 0.652588` does not move.**
   Feasibility now comes from `Shield.is_feasible` via
   `ShieldedSpectrumEnv.is_feasible`, so this class of error cannot recur.

2. **RETRACTED: "the Shield nudges the centre frequency by ~1.7 MHz so the mask
   stays in band."** That sentence described the loop proposing an illegal subband
   and being corrected — presented as a benign implementation detail. The loop now
   proposes only on Shield-feasible subbands, so the **only** correction at any cap
   is the EIRP clamp and the executed centre is exactly a subband centre. The
   committed sample record shows it: `requested_frequency_hz` ==
   `safe_frequency_hz` == 3525000000.0, with only `tx_power_dBm` clamped
   105.3647 → 14.0.

3. **RETRACTED by omission: this loop's "optimiser" is worth nothing, and the
   document never said so.** There is no learner here — a fixed ~110 dBm demand is
   clamped by the Shield. Measured `realised_regret` against a **zero-data**
   policy that just transmits at the cap on the best feasible subband is
   **exactly 0.000000 at all six caps** (`max_realised_regret = 0.0`). Every cap
   now emits `constant_cap_policy_reward` and `realised_regret` so this is
   impossible to miss. The frontier is evidence that *the projection is exact*, not
   that anything was optimised.

4. **CORRECTED: the free-constraint case is much smaller than "safety is free"
   implies.** The whole subband decision is worth **15 receivers of 4096**
   (0.003662) at its own 32 dBm operating point, and **12 of 4096** (0.002930) over
   all six subbands at the 33 dBm cap — the figure in ERRATA item 12, which
   reproduces exactly. The EIRP axis spans **0.468994** over the swept caps, ~160×
   the frequency axis. And the free case forgoes **0.016357 (67 receivers)**
   against the zero-data constant-cap policy, because it deliberately runs 1 dB
   below the cap; `utility_cost = 0.0` is measured against its *own task*, not
   against the best legal action.

5. **CORRECTED: `best_subband` is not a reproducible identity and must never be
   exact-compared.** The argmax moves with power — over all six subbands it is 0 at
   20 dBm, 1 at 26, **5 at 32**, 3 at 33/40/46, 5 at 52; over the Shield-feasible
   four it is 4, 1, 2, 3, 3, 3, 4. A 12–15 receiver decision sits inside the
   last-ULP threshold-flip band the verifier already tolerates, so an exact index
   assertion would be a latent CI flake. *(No such assertion existed —
   `scripts/verify_safety_utility_frontier.py` never referenced `best_subband`, and
   the evidence test only asserted membership. The gate added here is preventive:
   the free case reproduces on its **achieved reward** within
   `SERVED_FRACTION_TOL`, plus structural checks that the winner is Shield-feasible
   and the candidate set is the Shield's.)*

**Not retracted.** The safety guarantee, the trust chain, the monotonicity of the
frontier, the positivity of the marginal price, and the zero-utility-cost result
for the free-constraint task all hold on the corrected run.

Every earlier Horizon benchmark exercised the Decision Safety Shield on tasks
whose optimum was already **inside** the feasible set — DSA subband selection
and the coverage power-fill — where the projection is provably free
(`benchmarks/results/deepmimo_dsa.json`: `illegal_emits_after_shield = 0`,
`mean_regret_db = 0.0` for the best-subband strategy over 4096 decisions). This
artifact measures the other regime, on real measured channels: a task whose
unconstrained optimum lies **outside** the feasible set, so the EIRP cap
genuinely binds and safety has a price. The purpose of this experiment is to
state that price plainly, not to soften it.

- **Runner:** [`../../benchmarks/safety_utility_frontier_loop.py`](../../benchmarks/safety_utility_frontier_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/safety_utility_frontier.json`](../../benchmarks/results/safety_utility_frontier.json)
- **Sample evidence record:** [`../../benchmarks/results/safety_utility_frontier_sample_record.json`](../../benchmarks/results/safety_utility_frontier_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_safety_utility_frontier_evidence.py`](../../tests/test_safety_utility_frontier_evidence.py)
- **Reproduction verifier:** [`../../scripts/verify_safety_utility_frontier.py`](../../scripts/verify_safety_utility_frontier.py)
- **Data:** DeepMIMO ASU Campus 3.5 GHz, **4096** ray-traced receivers (Wireless
  InSite), the exact checksum-pinned feature build the DSA and federated
  benchmarks consume (`features_sha256 = ab4414a1…`).

## The task and the definitions

The objective is the shared substrate's served fraction: a receiver is served
when `EIRP + measured_channel_gain − noise ≥ 0 dB` (noise −94.0 dBm over
20 MHz, NF 7). Served fraction is **monotone increasing in EIRP**, so the
unconstrained optimum is full service (reward 1.0) — attained, measured from
the real channel gains, at a full-service EIRP of **109.3 dBm** on the cheapest
subband (the weakest receiver sits ~110 dB below the strongest). That demand is
physically enormous, which is itself part of the honest picture: the
unconstrained optimum this frontier is measured against is an idealisation the
licence exists to forbid.

Per swept cap, a closed loop proposes that unconstrained optimum (best feasible
subband, full-service demand EIRP plus seeded jitter, 8 steps per cap); the
Shield must project **every** step onto the cap; the environment executes the
projected action and realises its **real** reward; every step lands on the
hash-chained evidence store.

- `best_feasible_reward(cap)` — best served fraction any legal action achieves,
  ranked over the **Shield-feasible** subbands `[1, 2, 3, 4]` only
  (`feasibility.source = Shield.is_feasible`; subbands 0 and 5 are refused with
  `spectral_mask_ts38104`).
- `utility_forgone(cap) = 1.0 − best_feasible_reward(cap)`.
- `constant_cap_policy_reward(cap)` — the **zero-data** baseline: transmit at the
  cap on the best feasible subband, no data, no learning.
- `realised_regret(cap) = constant_cap_policy_reward − realised_mean_reward`.
- **Marginal price of safety** — served fraction lost per dB of cap tightening
  between adjacent sweep points.

## The frontier (real measured channels, 4096 receivers)

| EIRP cap | subband | best feasible served fraction | realised (mean) | utility forgone | **realised regret vs zero-data** | projection rate | executed EIRP |
| ---: | :---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 20 dBm | 4 | 0.1384 (567 Rx) | 0.1384 | **0.8616** | **0.000000** | 1.0 | 20.0 dBm |
| 26 dBm | 1 | 0.2463 (1009 Rx) | 0.2463 | **0.7537** | **0.000000** | 1.0 | 26.0 dBm |
| **33 dBm** (operational) | 3 | **0.3474** (1423 Rx) | **0.3474** | **0.6526** | **0.000000** | 1.0 | 33.0 dBm |
| 40 dBm | 3 | 0.4778 (1957 Rx) | 0.4778 | **0.5222** | **0.000000** | 1.0 | 40.0 dBm |
| 46 dBm | 3 | 0.5466 (2239 Rx) | 0.5466 | **0.4534** | **0.000000** | 1.0 | 46.0 dBm |
| 52 dBm | 4 | 0.6074 (2488 Rx) | 0.6074 | **0.3926** | **0.000000** | 1.0 | 52.0 dBm |

Under every cap the realised reward equals the best feasible reward (the
projection lands exactly on the constrained optimum — the Shield forgoes
nothing *within* the feasible set), no executed action ever exceeds its own cap,
and every executed action stays in band.

**Read the regret column before the rest of the table.** It is zero everywhere,
and that is the honest summary of what this loop's decision procedure
contributes: **nothing measurable**. A policy with no data at all — transmit at
the cap on the best feasible subband, forever — realises the identical served
fraction at every one of the six caps. The frontier below is a statement about
the *physics of the campus under a power cap* and about the projection landing
exactly on the constrained optimum. It is not evidence of optimisation, and the
result JSON says so in `zero_data_baseline.note`.

## The marginal price of safety (the headline curve)

Served fraction lost per dB of cap tightening, between adjacent sweep points:

| tightening | served fraction lost / dB | receivers lost / dB |
| --- | ---: | ---: |
| 26 → 20 dBm | 0.017985 | ~74 |
| 33 → 26 dBm | 0.014439 | ~59 |
| 40 → 33 dBm | **0.018624** | **~76** |
| 46 → 40 dBm | 0.011475 | ~47 |
| 52 → 46 dBm | 0.010132 | ~42 |

Read honestly:

- **The constraint binds hardest right at the operational cap.** The steepest
  segment of the whole sweep is 33 → 40 dBm: each dB of the licence condition
  around 33 dBm costs ~1.9 % of the campus (~76 receivers per dB).
- **What the cap costs depends entirely on the baseline it is measured
  against, and the served-fraction objective saturates — so no single number
  is meaningful without its anchor.** Against the unconstrained *full-service*
  optimum (reward 1.0) the 33 dBm cap forgoes **0.6526**, leaving 2673 of 4096
  receivers unserved. But full service demands ~109 dBm EIRP (~85 MW), which is
  a physical idealisation, not a deployable alternative. Against deployable
  anchors the cost is much smaller (`reference_anchors` in the result JSON):

  | anchor EIRP | power | served | forgone vs the 33 dBm cap |
  | ---: | ---: | ---: | ---: |
  | 46 dBm | 40 W | 0.5466 | **0.1992** |
  | 52 dBm | 158 W | 0.6074 | **0.2600** |
  | 60 dBm | 1 kW | 0.6875 | 0.3401 |
  | 78 dBm | 63 kW | 0.9805 | 0.6331 |
  | ~109 dBm | 85 MW | 1.0000 | 0.6526 |

  Quote the anchor whenever quoting the cost. The honest one-line summary is
  that at the operational cap this campus serves ~35 % of measured receivers,
  versus ~55–61 % at a high-power-but-still-capped alternative — so the licence
  condition costs roughly **0.199–0.260 served fraction against a deployable
  baseline**, and only approaches 0.65 against an unbuildable one. Every anchor
  is itself a free parameter: the number is a pure function of which EIRP you
  choose to compare against, exactly as `unconstrained_optimum.eirp50_reference_reward`
  (0.5918 at 50 dBm) is a function of that 50, not a property of the Shield.
- **Cross-experiment consistency:** `benchmarks/results/credit_assignment.json`
  reports `safety_utility_gap = 0.25282`. That is the same quantity measured
  against its own 52 dBm action-grid top, and agrees with the 52 dBm anchor
  here (0.2600; the small difference is that experiment using its oracle's
  last-quarter mean rather than the exact best). The two results do not
  disagree — they use different anchors, now stated explicitly in both. Note
  what this agreement *is*: two benchmarks reporting the same
  grid-top-dependent free parameter, not two independent confirmations of a
  physical constant.
- **A looser cap never hurts** (best feasible served fraction is non-decreasing
  across the sweep) and **every adjacent pair loses served fraction when
  tightened** — the price is positive everywhere on this data, with
  diminishing marginal cost above ~40 dBm as the remaining unserved receivers
  get exponentially more expensive to reach.

## The free-constraint contrast: when safety costs nothing

The same loop then runs a task whose optimum **is** feasible: pick the best
subband at a fixed legal EIRP of 32 dBm (tx 26 dBm + 6 dBi < 33 dBm cap),
choosing among the subbands the **Shield** admits — `[1, 2, 3, 4]`, derived from
`Shield.is_feasible`, not from a local in-band rule this file could get wrong.
Result (6 closed-loop steps, same evidence chain):

| quantity | value |
| --- | ---: |
| candidate subbands (from `Shield.is_feasible`) | `[1, 2, 3, 4]` |
| refused by the Shield | `0`, `5` — `spectral_mask_ts38104` |
| task optimum (subband 2 @ EIRP 32) | 0.331055 |
| realised mean reward | 0.331055 |
| projection rate | **0.0** |
| corrected | **false** |
| guard refusals | 0 |
| **utility cost (vs its own task optimum)** | **0.0** |
| **realised regret vs zero-data constant-cap policy** | **0.016357 (67 Rx)** |

### How much this case is actually worth: 12–15 receivers

State this plainly, because "safety is free" is a much bigger-sounding claim
than the measurement supports:

- The entire subband decision is worth **0.003662 = 15 receivers of 4096** at
  this case's own 32 dBm (best minus worst Shield-feasible candidate), and
  **0.002930 = 12 of 4096** across all six subbands at the 33 dBm cap. That
  second figure is ERRATA item 12; it reproduces exactly.
- By contrast the EIRP axis spans **0.468994** across the swept caps — roughly
  **160×** the frequency axis. On this campus, *what power you are allowed*
  dominates *which subband you pick* by more than two orders of magnitude.
- `utility_cost = 0.0` is measured against this task's own optimum. Against the
  zero-data policy of simply sitting at the 33 dBm cap, the free case forgoes
  **0.016357 (67 receivers)** — four times the entire subband decision — purely
  because it runs 1 dB below the cap by construction.
- **The argmax is power-dependent and is not a reproducible identity.** Over all
  six subbands it is 0 at 20 dBm, 1 at 26, **5 at 32**, 3 at 33/40/46, 5 at 52;
  over the Shield-feasible four it is 4, 1, 2, 3, 3, 3, 4
  (`free_constraint_case.argmax_subband_by_eirp`). Nothing downstream may
  exact-compare `best_subband`; assert on the achieved reward with a tolerance.
  The result JSON carries `best_subband_is_reproducible_identity: false` so this
  is machine-visible, not just prose.

This reproduces, in the closed loop, exactly what the committed DSA benchmark
already showed over 4096 best-subband decisions
(`illegal_emits_after_shield = 0`, `mean_regret_db = 0.0`). So the honest
boundary, quantified on both sides with real data: **the Shield is free when the
task optimum is feasible — on a task worth 15 receivers — and costs a
measurable, explicitly reported amount when the constraint binds** — on this
campus, 0.199–0.260 served fraction at the operational cap measured against a
deployable higher-power baseline (0.6526 against the unbuildable full-service
ideal).

Whether that cost is acceptable is a **regulatory** question, not an
engineering one: the EIRP cap is a licence condition, and raising it is not
within the rApp's authority. Horizon's claim is that the cost of enforcing the
licence is explicit, bounded and auditable — not that it is zero.

## The trust chain (real, every step)

All 54 closed-loop steps (48 sweep + 6 free-case) are appended to a
hash-chained evidence store, each record carrying **both** the proposed and the
executed action:

- The chain **verifies intact** (`verify() = −1`) and the verify-gated replay
  buffer recovers all **54** transitions.
- After a single-byte tamper of record #1 on disk, `load_transitions()`
  **raises `EvidenceIntegrityError`** — a tampered decision log cannot become
  training data — and re-verification **pinpoints the tamper at index 1**.

## The safety guarantee, quoted from the regenerated result

Unchanged by every correction above. From
`benchmarks/results/safety_utility_frontier.json`:

```
frontier[*].max_executed_eirp_dbm  = 20.0, 26.0, 33.0, 40.0, 46.0, 52.0
                                     (each exactly its own eirp_cap_dbm, never above)
frontier[*].projection_rate        = 1.0   (every illegal proposal was projected)
frontier[*].executed_frequency_hz  = 3525000000.0, 3475000000.0, 3508333333.33,
                                     3508333333.33, 3508333333.33, 3525000000.0
                                     (all in [3.45e9, 3.55e9]; all exact subband centres)
free_constraint_case.max_executed_eirp_dbm = 32.0   (<= the 33.0 operational cap)
free_constraint_case.guard_refused_rate    = 0.0
trust_chain.verify_first_broken_index      = -1
trust_chain.evidence_chain_length          = 54
trust_chain.replayed_transitions_intact    = 54
trust_chain.tampered_chain_refused         = true
trust_chain.verify_after_tamper_index      = 1     (the exact tampered index)
features_sha256    = ab4414a1dca5d1247f28908003003239033ab329ea04cf2f481a37132969e5af
source_tree_sha256 = 42f9c6eb8f4b4c467fe04c63ae910ae26c75706b85c7decdbcb3a002cb5466b7
```

No illegal action was executed at any cap; the provenance binding is byte-identical
to the pre-correction result, so the corrected numbers come from the same data.
Masking the action space is **defence in depth only**: `ShieldedSpectrumEnv.step`
still disposes every action through the Shield unconditionally, and the 48
sweep steps are still projected 100 % of the time.

## Reproduce (real data, deterministic)

```bash
# Build the licence-gated DeepMIMO feature file (checksum-pinned, deterministic).
python datasets/deepmimo_asu_3p5/build.py \
  --cache-dir /tmp/dm --features /tmp/dm/features.jsonl \
  --manifest /tmp/dm/manifest.json --samples 4096

# Sweep the frontier on the real channels.
python benchmarks/safety_utility_frontier_loop.py \
  --features /tmp/dm/features.jsonl --manifest /tmp/dm/manifest.json \
  --out /tmp/dm/safety_utility_frontier.json

# Verify the fresh run against the committed result (tolerance bands, not bytes).
python scripts/verify_safety_utility_frontier.py \
  --committed-result benchmarks/results/safety_utility_frontier.json \
  --actual-result /tmp/dm/safety_utility_frontier.json \
  --actual-manifest /tmp/dm/manifest.json
```

The runner exits non-zero unless the frontier is monotone, the constraint
genuinely binds at 33 dBm, the marginal price is positive somewhere, the
free-constraint case costs exactly zero, **every ranked subband came from
`Shield.is_feasible` and every ranking winner lies inside that set**, **every cap
emits its zero-data `constant_cap_policy_reward` and a non-negative
`realised_regret`**, every executed action is legal under its own cap, and the
evidence chain verifies intact then refuses the tampered replay.

### The gates are proven to fail

A gate that cannot fail is worse than no gate, so each was run against a
deliberately regressed copy of the loop on the real data (regression injected in
`/tmp`, never in the repo). Both the runner's exit-0 contract
(`_invariants_hold`) and `scripts/verify_safety_utility_frontier.py` were checked:

| injected regression | runner | verifier | what caught it |
| --- | :---: | :---: | --- |
| rank subbands over all six again (the pre-fix bug) | FAIL | exit 1 | in-loop `RuntimeError: best_feasible selected a Shield-infeasible action` |
| …with the in-loop assert also removed | FAIL | exit 1 | `best_feasible_subband at cap 20.0/52.0 dBm is Shield-infeasible` |
| free case reverts to a local in-band rule that admits subband 0 | FAIL | exit 1 | `the free-constraint case ranked a candidate set that is not the Shield's` |
| drop `constant_cap_policy_reward` / `realised_regret` | FAIL | exit 1 | `cap N dBm did not report its zero-data constant-cap baseline` |
| claim a cap beat the zero-data policy (regret −0.01) | FAIL | exit 1 | `cap N dBm claims to beat a zero-data constant-cap policy` |
| report an emit 0.5 dB above its own cap | FAIL | exit 1 | `an executed action exceeded its own cap` |

The committed-result tests were proven to bite the same way, by mutating the
committed JSON in place and restoring it byte-identically (sha256 re-checked):
flipping a winner to an infeasible subband, forging a negative regret, inflating
the 12-receiver decision value, raising `max_executed_eirp_dbm` above its cap,
and rewriting `feasibility.source` away from the Shield each fail
`tests/test_safety_utility_frontier_evidence.py`.

## Scope and honesty

Real site-specific ray tracing (DeepMIMO ASU Campus 3.5 GHz), **not
over-the-air capture**, and a **single scenario**; served fraction with a 0 dB
SINR threshold is one utility proxy among several a licensee may care about,
and the full-service reference the frontier is measured against is a physical
idealisation (~109 dBm EIRP). **This benchmark contains no learner**, so nothing
here is evidence about learning behind a projection; its realised regret against
a zero-data constant-cap policy is 0.000000 at every cap. **Every "cost of
safety" figure is anchor-relative** — 0.6526 against full service, 0.199–0.260
against deployable anchors — and the anchor is a choice, not a measurement. The licence-gated raw feature rows are not
redistributed; the committed result is bound to the canonical build by
`features_sha256` / `source_tree_sha256`, the committed-result tests assert
every invariant in the fast pack, and the verifier script re-checks a fresh
rebuild with tolerance bands.
