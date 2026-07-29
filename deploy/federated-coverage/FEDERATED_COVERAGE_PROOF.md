# Federated + trust core on real DeepMIMO — coverage-map proof

> **This document has been rewritten.** An adversarial post-mortem
> ([`../shield-learning/ERRATA.md`](../shield-learning/ERRATA.md)) found that three of its headline claims were
> artifacts of a misconfigured experiment rather than properties of the system,
> and that two of its CI gates could be flipped by reseeding the client
> partition. The [retractions](#retractions) are listed explicitly below and the
> numbers throughout are the re-measured ones. **The safety results stand
> unmodified**; the security and privacy *rankings* did not.

This is the Horizon artifact that drives the **federated learning and trust
core** with real measured-physics data — no synthetic gradient vectors anywhere
in the loop.

- **Runner:** [`../../benchmarks/federated_coverage_loop.py`](../../benchmarks/federated_coverage_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/federated_coverage.json`](../../benchmarks/results/federated_coverage.json)
- **Sample evidence record:** [`../../benchmarks/results/federated_coverage_sample_record.json`](../../benchmarks/results/federated_coverage_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_federated_coverage_evidence.py`](../../tests/test_federated_coverage_evidence.py)
- **Reproduction verifier:** [`../../scripts/verify_federated_coverage.py`](../../scripts/verify_federated_coverage.py)
- **CI reproduction on real data:** [`../../.github/workflows/realdata.yml`](../../.github/workflows/realdata.yml)
- **Data:** DeepMIMO ASU Campus 3.5 GHz, **4096** ray-traced receivers (Wireless
  InSite), the exact checksum-pinned feature build the DSA benchmark consumes
  (`features_sha256 = ab4414a1…`).

## The task and why it fits the data

A real O-RAN **Coverage-and-Capacity-Optimization (CCO)** rApp function: learn a
map from receiver position to received channel gain (coverage, dBW), then pick
the worst-covered cell for a power-fill action. Each receiver contributes its
real mean per-subband gain `y = mean(g)` (dBW) and real position; the model is a
linear regressor over whitened quadratic position features `[x, y, x², y², xy]`.
Clients are 10 geographic, non-IID sites (ray tracing is site-specific). Each
runs a stable **FedProx proximal ridge** solve on its own receivers.

The received-gain field spans **103.7 dB** across the campus and is strongly
spatially structured (path loss), so a position→coverage model has real, large,
learnable signal:

| model | coverage RMSE |
| --- | ---: |
| predict-the-mean baseline (do nothing) | 20.11 dB |
| closed-form least squares (optimum) | 11.77 dB |
| **clean federated (FedAvg)** | **11.92 dB** |

### Server root set

512 receivers (every 8th) are held out as a **server-side root set** and given to
no client; the 10 clients partition the remaining 3584. In the deployment story
this is the operator's own drive-test data. It does two jobs, and having one
construct serve both is deliberate:

- it is the clean reference dataset **FLTrust** needs, and
- it is the **public calibration set** the DP clip norm is selected on, so the
  clip is never a function of the private cohort.

This is why clean FedAvg reads 11.92 dB here where the pre-correction document
reported 11.87 dB: clients now hold 3584 receivers rather than all 4096.

## The security result (real DeepMIMO, 3 of 10 clients Byzantine)

Every aggregator is now run **twice** — once with zero adversaries and once
under attack — because otherwise the cost of the *defence* is invisible and gets
misattributed to the *attack*. That misattribution is what the previous version
of this document did.

| aggregator | clean (f=0) | poisoned (f=3) | aggregation tax | poison tax |
| --- | ---: | ---: | ---: | ---: |
| FedAvg | 11.92 dB | **1.04·10⁷ dB** (diverged) | — | — |
| **FLTrust** | **11.91 dB** | **11.97 dB** | **−0.01 dB** | **+0.06 dB** |
| Krum | 14.46 dB | 14.46 dB | +2.53 dB | **+0.00 dB** |
| coordinate-median | 13.00 dB | 22.24 dB | +1.07 dB | +9.24 dB |
| trimmed-mean (β=3) | 12.03 dB | 34.04 dB | +0.10 dB | +22.01 dB |

Read honestly:

- **Naive FedAvg is catastrophically vulnerable on real data.** Three boosted
  clients *diverge* the coverage predictor — RMSE explodes to ~10⁷ dB and the
  power-fill it selects lands on a cell in the **51st percentile** of coverage,
  i.e. a perfectly ordinary cell that needs no fill at all.
- **Krum's cost is essentially *all* aggregation tax and *zero* poison tax.**
  Poisoned Krum scores 14.4572 dB and clean Krum scores 14.4572 dB — the same
  number to four decimals, because Krum selects one intact benign update and the
  colluders never win selection. The 2.53 dB it costs is the price of discarding
  K−1 updates under a non-IID split, paid whether or not anyone is attacking.
- **FLTrust is the right defence here.** It is clean-lossless (it very slightly
  *beats* FedAvg, −0.01 dB, by down-weighting the most off-distribution clients)
  and it holds clean parity under attack.

### FLTrust against more than the shipped attack

| attack | FedAvg | **FLTrust** | Krum | median | trimmed-mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| scaling / model-replacement | 1.04·10⁷ | **11.97** | 14.46 | 22.24 | 34.04 |
| sign-flip | 3433.11 | **11.97** | 14.46 | 22.24 | 33.97 |
| ALIE (Baruch NeurIPS'19) | 12.18 | **11.85** | 12.20 | 12.18 | 12.25 |
| Min-Max (Shejwalkar NDSS'21) | 14.59 | **11.88** | 14.46 | 18.56 | 21.03 |

FLTrust is the best entry in every column, and is the only aggregator that stays
within 0.2 dB of the clean optimum against all four.

### Breakdown behaviour: where Krum stops existing

Scaling attack, `f` of 10 clients malicious. `—` means the aggregator is
*mathematically undefined* at that adversary count (Krum requires `n > 2f+2`;
trimmed-mean requires `n > 2β`).

| f | **FLTrust** | Krum | median | trimmed-mean |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **11.93** | 14.46 | 14.76 | 12.53 |
| 2 | **11.93** | 14.46 | 15.92 | 15.77 |
| 3 | **11.97** | 14.46 | 22.24 | 34.04 |
| 4 | **11.98** | — | 323.66 | 323.66 |
| 5 | **12.13** | — | 2.0·10¹² | — |
| 6 | **12.75** | — | 2.6·10¹⁸ | — |
| 7 | **13.46** | — | 1.4·10²⁰ | — |

FLTrust degrades *smoothly* to 13.46 dB at 7 of 10 malicious — still below the
20.11 dB do-nothing baseline — because its robustness comes from the server's
own root direction rather than from a counting bound.

## Differential privacy: a defect, not a trade-off

`epsilon` is a function of `(noise multiplier, rounds, delta)` **only**. The clip
norm and the round count are therefore free utility knobs *at fixed ε*. Both were
set wrongly.

| configuration | clip | rounds | z | ε (δ=1e-5) | coverage RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| **legacy (as shipped)** | 1.0 | 40 | 8.0 | **4.1447** | **340.12 dB** |
| **tuned** | 0.075 | 12 | 4.3818 | **4.1447** | **14.34 dB** |

**Same ε, same δ = 1e-5, same K = 10, same accountant, same replace-one
adjacency.** A 24× utility difference at identical certified privacy. The tuned
result is below the 20.11 dB do-nothing baseline, which the legacy one missed by
17×. Two measured causes:

1. **The clip never bound.** Over 40 rounds × 10 clients on the *clean*
   (noise-free) trajectory, the maximum per-client update L2 is **0.24484** and
   the median is **0.09176** — **0 of 400** updates exceed the clip of 1.0. The
   noise was calibrated to a sensitivity ~4× larger than the data ever attains.
   (Measuring this on the *noisy* trajectory instead is circular: over-large
   noise blows the model up, which blows the gradients up, which makes any clip
   look like it binds. The loop measures the clean trajectory for this reason.)
2. **40 rounds for a model that converges in ~10.** At fixed ε, `z ∝ √R`. The
   sweep at the pinned ε, all at 4.1447:

   | rounds | 1 | 2 | 5 | **8** | **12** | 20 | 40 |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | z | 1.265 | 1.789 | 2.828 | 3.578 | 4.382 | 5.657 | 8.000 |
   | RMSE | 19.66 | 19.22 | 16.65 | **14.33** | **14.34** | 19.35 | 29.16 |

**The adjacency convention was not relaxed.** `dp.py` uses replace-one
sensitivity (`σ = z·2C`), which is the *stronger* statement. Switching to the
add/remove-one convention (`σ = z·C`) that most DP-FedAvg papers report against
would halve σ for free-looking gains, but it is a genuinely weaker guarantee, not
a fix, so it was left alone. `DPConfig.adjacency` now names the convention
explicitly instead of hard-coding a `2.0` in the middle of the mechanism.

**The clip is selected without touching the private cohort.**
`calibrate_clip_norm()` carves the server's *public root* receivers into
geographic pseudo-clients, runs the entire DP protocol on them at each candidate
clip, and scores on the root set. The shipped 0.075 is that sweep's argmin — a
rule a deployment can actually execute. Note that this is deliberately *not*
"clip at the largest observed update norm": that gives zero clipping bias but
scores **22.10 dB**, worse than doing nothing, because at fixed ε the noise
scales linearly in the clip. Trading a little clipping bias for a lot less noise
is the whole game, and it can be played entirely on public data.

Also corrected: the source comment `DP_NOISE = 8.0  # certified eps ~ 3.2` was
wrong by 30 %. The shipped accountant returns **4.1447** at that setting, which
the committed JSON in the same repository already said.

### One-shot sufficient-statistic release

The whitening is a *public* preprocessing, and it makes `AᵀA / N` exactly the
identity (max abs error **3.4·10⁻¹⁴**). The normal equations therefore collapse
to `w* = Aᵀb / N` — a single 6-dimensional sum over clients, releasable **once**
under the Gaussian mechanism instead of once per round. That is the `R = 1`
corner of the `z ∝ √R` law: `z = 1.2649`, the smallest any ε = 4.1447 permits.

| clients | 10 | 50 | 200 | 400 |
| --- | ---: | ---: | ---: | ---: |
| clip (public bound ρ/K) | 0.874 | 0.175 | 0.0437 | 0.0219 |
| RMSE @ ε=4.1447 | 89.77 dB | 21.61 dB | **12.84 dB** | **12.21 dB** |

At a 400-client federation the one-shot DP release lands **0.29 dB** from
non-private clean FedAvg (11.92 dB) and 0.43 dB from the closed-form optimum, at
a certified ε of 4.1447. Utility here is a function of *federation size*, not an
inherent privacy tax: at K = 10 the per-client sensitivity is too large a
fraction of the total and the one-shot release is worse than the iterative one.

## The power-fill target: gated as a predicate, not a cell id

The operational claim is "the fill lands on a genuinely badly covered cell". It
is now gated as exactly that — bottom decile of the clean model's own predicted
coverage — rather than as `robust_target_receiver == 9`.

| aggregator's fill target | coverage percentile | in bottom decile? |
| --- | ---: | :---: |
| clean FedAvg | 0.00 | yes |
| **FLTrust** | **0.00** | **yes** |
| Krum | 0.00 | yes |
| coordinate-median | 39.62 | no |
| trimmed-mean | 25.51 | no |
| **poisoned FedAvg** | **51.46** | **no** |

Across 12 partition seeds, the fill lands in the bottom decile at:

| FLTrust | Krum | median | trimmed-mean | poisoned FedAvg |
| ---: | ---: | ---: | ---: | ---: |
| **12/12** | 7/12 | 5/12 | 1/12 | 0/12 |

FLTrust recovers the clean model's own argmin cell at **12/12** seeds. Krum's
fill is out of the bottom decile at 5 of 12. **Recovering the right cell is a
property of FLTrust, not of robust aggregation in general** — which is why the
gate and the Shield/evidence record are both based on FLTrust.

## Re-gated CI: claims that survive a reseed

The old gates were fragile. Measured over 12 partition seeds:

| retired gate | held | why it was retired |
| --- | ---: | --- |
| `krum_rmse < baseline` | 9/12 | 0.65 dB margin against a 1.87 dB seed-induced sd |
| `robust_target_receiver == 9` | 5/12 | gated on a cell's identity, not on the claim |

The 17 gates now enforced hold at **12/12** partition seeds (and at 24/24 in a
wider sweep run for headroom). The load-bearing ones:

- **separation** — `robust_rmse < poisoned_fedavg_rmse / 10⁴` (measured
  separation is ~6 orders of magnitude, so the gate has 2 orders of headroom);
- **bounded absolute** — `robust_rmse ≤ 22.0 dB` (worst Krum over 24 seeds:
  21.19 dB);
- **FLTrust clean parity** — `fltrust_rmse ≤ clean_fedavg_rmse × 1.10`;
- **fill predicate** — bottom decile of clean predicted coverage;
- **DP** — `ε_tuned == ε_legacy == 4.1447` *and* `rmse_tuned < baseline`.

Seed-swept dispersion, 12 seeds:

| quantity | mean | sd | min | max |
| --- | ---: | ---: | ---: | ---: |
| clean FedAvg | 11.92 | 0.04 | 11.89 | 12.03 |
| FLTrust (poisoned) | 11.93 | 0.09 | 11.81 | 12.10 |
| Krum (poisoned) | 17.92 | 2.40 | 14.46 | 21.19 |
| median (poisoned) | 22.21 | 5.82 | 12.72 | 31.38 |
| DP tuned | 14.21 | 0.54 | 13.58 | 15.16 |

### The gates are proved to still bite

A gate nobody can fail is decoration. `--inject-regression` deliberately breaks a
defence; each mode must fail, and does:

| injected regression | gates it fails |
| --- | --- |
| `krum_off` (aggregate with the mean where Krum is expected) | `krum_separates_from_poisoned_mean`, `krum_bounded_absolute` (0/12 seeds) |
| `fltrust_abs_cos` (`\|cos\|` instead of `ReLU(cos)`) | `fltrust_bounded_absolute`, `fltrust_matches_clean_fedavg`, both fill gates |
| `fltrust_uniform_trust` (drop the trust score, keep the norm rescale) | `fltrust_matches_clean_fedavg` (0/12 seeds) |
| `dp_legacy_clip` (restore the never-binding clip) | `dp_tuned_beats_baseline` (0/12 seeds) |
| `fill_not_worst_cell` (aim at the median-coverage cell) | `robust_fill_targets_bottom_decile`, `robust_fill_recovers_clean_target` |

`scripts/verify_federated_coverage.py` rejects all of them, and rejects any
result whose `regression_injected` is not `none`.

## The trust chain (real, on the selected cell)

Unchanged, and it still holds. The Shield remains the **sole emit gate**:

- The robust-selected fill (in band, EIRP 32 dBm < 33) needs **no correction,
  passes the guard chain, and emits cleanly**, recorded.
- A coverage-fill that naively requests **EIRP 46 dBm** is **projected** onto the
  legal cap — **33 dBm** — and the raw over-power emit trips the
  `corrections_not_audited` guard, so **it never reaches the RAN**
  (`overpower_fill_reached_ran = false`).
- The evidence chain **verifies intact** (`verify() = -1`); after a single-byte
  tamper of record #1 on disk, re-verification **pinpoints the tamper at index
  1** and raises an X.733 `processingErrorAlarm`.

## Retractions

1. **RETRACTED — "The DP noise raises RMSE (350 dB) — an honest privacy/utility
   trade-off on this higher-dimensional model, not a defect."** It was a defect.
   At the identical certified ε = 4.1447, δ = 1e-5, replace-one adjacency and
   K = 10, tuning only the clip norm and the round count gives **14.34 dB** —
   below the do-nothing baseline. Nothing about the guarantee changed.

2. **RETRACTED — "Krum is the correct defense here."** 2.53 dB of Krum's 2.53 dB
   cost is present with **zero** adversaries; its poison tax is 0.00 dB. Krum is
   undefined past `f ≥ (K−2)/2`, so at K = 10 it cannot be run at all with 4 or
   more adversaries. FLTrust reaches clean-FedAvg parity under all four attacks
   in the battery and still scores 13.46 dB at 7 of 10 malicious. The related
   claim that Krum "recovers the same target cell (9)" holds at only 7/12 seeds
   for the bottom-decile predicate and 5/12 for exact cell identity.

3. **RETRACTED — "the six 100 MHz subbands differ by only ~0.3 dB in mean
   gain."** The per-subband **mean gain** spread on this canonical
   4096-receiver build is **0.0850 dB**. The 0.3 dB figure came from an earlier
   512-receiver cache and was carried forward incorrectly. (The **3.7020 dB**
   figure that also circulates is the *per-receiver across-subband* spread — a
   different quantity.) The conclusion the sentence was supporting is unaffected
   and in fact strengthened: a global default-subband task on this data is even
   more degenerate than stated, which is exactly why a strong-signal coverage
   task is the honest demonstration here.

4. **RETRACTED — the CI gates `krum_rmse < baseline` and
   `robust_target_receiver == 9`.** Measured across 12 partition seeds they hold
   at 9/12 and 5/12. Both are replaced, and the replacements are proved to hold
   12/12 *and* to fail under injected regressions.

## Reproduce (real data, deterministic)

```bash
# Build the licence-gated DeepMIMO feature file (checksum-pinned, deterministic).
python datasets/deepmimo_asu_3p5/build.py \
  --cache-dir /tmp/dm --features /tmp/dm/features.jsonl \
  --manifest /tmp/dm/manifest.json --samples 4096
# -> features_sha256 == ab4414a1…  (matches datasets/deepmimo_asu_3p5/manifest.json)

# Run the whole federated + trust loop on the real channels.
python benchmarks/federated_coverage_loop.py \
  --features /tmp/dm/features.jsonl --manifest /tmp/dm/manifest.json \
  --out benchmarks/results/federated_coverage.json

# Show the gate evidence across partition seeds (exit non-zero if any gate is
# seed-fragile). --seeds 24 was used for the headroom numbers above.
python benchmarks/federated_coverage_loop.py --seed-sweep --seeds 24

# Prove the gates still bite (each of these must exit non-zero).
for r in krum_off fltrust_abs_cos fltrust_uniform_trust dp_legacy_clip \
         fill_not_worst_cell; do
  python benchmarks/federated_coverage_loop.py --inject-regression "$r" \
    --out /tmp/inj.json
done
```

The runner exits non-zero unless all 17 gates pass **and** every gate holds at
every partition seed. CI runs exactly this on rebuilt real data
(`.github/workflows/realdata.yml`), and the committed-result tests assert the
same invariants in the fast pack.

## Scope and honesty

Real DeepMIMO ray tracing drives real federated learning (FedProx), real
Byzantine-robust aggregation (Krum / coordinate-median / trimmed-mean /
FLTrust), a real RDP privacy accountant, real Shield legality and a real
tamper-evident evidence chain. This is **site-specific ray tracing, not
over-the-air capture**, and a **single scenario**.

Remaining honest limitations:

- The DP **round count** (12) is the argmin of a sweep run on this scenario. The
  **clip norm** is selected by a public rule that only touches server-held root
  data, but the round count is not; a deployment would need to pre-register it or
  spend a small part of the budget selecting it.
- The one-shot release's near-parity result needs a **large federation**
  (K ≥ 200). At the shipped K = 10 it is worse than the iterative protocol.
- FLTrust's robustness is contingent on the server root set being **clean and
  representative**. That is a real trust assumption, traded against Krum's
  counting bound — not a free lunch.
- The licence-gated raw feature rows are not redistributed; the committed result
  JSON is bound to the canonical build by `features_sha256`, and the `realdata`
  workflow rebuilds and re-runs the loop to reproduce it.
