# The safety–utility frontier on real DeepMIMO — what the Shield costs

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

- `best_feasible_reward(cap)` — best served fraction any legal action achieves.
- `utility_forgone(cap) = 1.0 − best_feasible_reward(cap)`.
- **Marginal price of safety** — served fraction lost per dB of cap tightening
  between adjacent sweep points.

## The frontier (real measured channels, 4096 receivers)

| EIRP cap | best feasible served fraction | realised (mean) | utility forgone | projection rate | executed EIRP |
| ---: | ---: | ---: | ---: | :---: | ---: |
| 20 dBm | 0.1392 (570 Rx) | 0.1392 | **0.8608** | 1.0 | 20.0 dBm |
| 26 dBm | 0.2463 (1009 Rx) | 0.2463 | **0.7537** | 1.0 | 26.0 dBm |
| **33 dBm** (operational) | **0.3474** (1423 Rx) | **0.3474** | **0.6526** | 1.0 | 33.0 dBm |
| 40 dBm | 0.4778 (1957 Rx) | 0.4778 | **0.5222** | 1.0 | 40.0 dBm |
| 46 dBm | 0.5466 (2239 Rx) | 0.5466 | **0.4534** | 1.0 | 46.0 dBm |
| 52 dBm | 0.6101 (2499 Rx) | 0.6101 | **0.3899** | 1.0 | 52.0 dBm |

Under every cap the realised reward equals the best feasible reward (the
projection lands exactly on the constrained optimum — the Shield forgoes
nothing *within* the feasible set), no executed action ever exceeds its own cap,
and every executed action stays in band.

## The marginal price of safety (the headline curve)

Served fraction lost per dB of cap tightening, between adjacent sweep points:

| tightening | served fraction lost / dB | receivers lost / dB |
| --- | ---: | ---: |
| 26 → 20 dBm | 0.017863 | ~73 |
| 33 → 26 dBm | 0.014439 | ~59 |
| 40 → 33 dBm | **0.018624** | **~76** |
| 46 → 40 dBm | 0.011475 | ~47 |
| 52 → 46 dBm | 0.010579 | ~43 |

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
  | 52 dBm | 158 W | 0.6101 | **0.2627** |
  | 60 dBm | 1 kW | 0.6875 | 0.3401 |
  | 78 dBm | 63 kW | 0.9805 | 0.6331 |
  | ~109 dBm | 85 MW | 1.0000 | 0.6526 |

  Quote the anchor whenever quoting the cost. The honest one-line summary is
  that at the operational cap this campus serves ~35 % of measured receivers,
  versus ~55–61 % at a high-power-but-still-capped alternative — so the licence
  condition costs roughly **0.20–0.26 served fraction against a deployable
  baseline**, and only approaches 0.65 against an unbuildable one.
- **Cross-experiment consistency:** `benchmarks/results/credit_assignment.json`
  reports `safety_utility_gap = 0.2528`. That is the same quantity measured
  against its own 52 dBm action-grid top, and agrees with the 52 dBm anchor
  here (0.2627; the small difference is that experiment using its oracle's
  last-quarter mean rather than the exact best). The two results do not
  disagree — they use different anchors, now stated explicitly in both.
- **A looser cap never hurts** (best feasible served fraction is non-decreasing
  across the sweep) and **every adjacent pair loses served fraction when
  tightened** — the price is positive everywhere on this data, with
  diminishing marginal cost above ~40 dBm as the remaining unserved receivers
  get exponentially more expensive to reach.

## The free-constraint contrast: when safety costs nothing

The same loop then runs a task whose optimum **is** feasible: pick the best
subband at a fixed legal EIRP of 32 dBm (tx 26 dBm + 6 dBi < 33 dBm cap),
choosing among subbands whose full 20 MHz reservation mask fits in band. Result
(6 closed-loop steps, same evidence chain):

| quantity | value |
| --- | ---: |
| task optimum (subband 2 @ EIRP 32) | 0.331055 |
| realised mean reward | 0.331055 |
| projection rate | **0.0** |
| corrected | **false** |
| guard refusals | 0 |
| **utility cost** | **0.0** |

This reproduces, in the closed loop, exactly what the committed DSA benchmark
already showed over 4096 best-subband decisions
(`illegal_emits_after_shield = 0`, `mean_regret_db = 0.0`). So the honest
boundary, quantified on both sides with real data: **the Shield is free when
the task optimum is feasible, and costs a measurable, explicitly reported
amount only when the constraint binds** — on this campus, ~0.20–0.26 served
fraction at the operational cap measured against a deployable higher-power
baseline (0.6526 against the unbuildable full-service ideal).

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
free-constraint case costs exactly zero, every executed action is legal under
its own cap, and the evidence chain verifies intact then refuses the tampered
replay.

## Scope and honesty

Real site-specific ray tracing (DeepMIMO ASU Campus 3.5 GHz), **not
over-the-air capture**, and a **single scenario**; served fraction with a 0 dB
SINR threshold is one utility proxy among several a licensee may care about,
and the full-service reference the frontier is measured against is a physical
idealisation (~109 dBm EIRP). The licence-gated raw feature rows are not
redistributed; the committed result is bound to the canonical build by
`features_sha256` / `source_tree_sha256`, the committed-result tests assert
every invariant in the fast pack, and the verifier script re-checks a fresh
rebuild with tolerance bands.
