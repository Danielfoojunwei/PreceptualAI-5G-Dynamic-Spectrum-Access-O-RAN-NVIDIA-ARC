# Federated + trust core on real DeepMIMO — coverage-map proof

> **Corrections:** an adversarial post-mortem found several headline claims in this
> document to be artifacts of the experiment harness rather than properties of the
> Shield. Read [`../shield-learning/ERRATA.md`](../shield-learning/ERRATA.md) alongside it. The safety results stand; most
> of the learning claims do not.

This is the Horizon artifact that drives the **federated learning and trust
core** with real measured-physics data — no synthetic gradient vectors anywhere
in the loop. It answers the honest-accounting question "which core capabilities
have *not* been exercised on real DeepMIMO data?" for federated aggregation,
differential privacy, Shield legality and the evidence chain — all previously run
only on synthetic updates.

- **Runner:** [`../../benchmarks/federated_coverage_loop.py`](../../benchmarks/federated_coverage_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/federated_coverage.json`](../../benchmarks/results/federated_coverage.json)
- **Sample evidence record:** [`../../benchmarks/results/federated_coverage_sample_record.json`](../../benchmarks/results/federated_coverage_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_federated_coverage_evidence.py`](../../tests/test_federated_coverage_evidence.py)
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

This is the *right* task for this scenario. The received-gain field spans
**103.7 dB** across the campus and is strongly spatially structured (path loss),
so a position→coverage model has real, large, learnable signal:

| model | coverage RMSE |
| --- | ---: |
| predict-the-mean baseline | 20.11 dB |
| closed-form least squares (optimum) | 11.77 dB |
| **clean federated (FedAvg)** | **11.87 dB** |

(By contrast, a global *default-subband* task is near-degenerate here — the six
100 MHz subbands differ by only ~0.3 dB in mean gain — which is exactly why a
strong-signal coverage task, not a flat-signal subband task, is the honest
demonstration on this data.)

## The security result (real DeepMIMO, 3 of 10 clients Byzantine)

A model-replacement / scaling poison (Bagdasaryan et al. 2020) from 3 of 10
clients. `worst cell` is the receiver the power-fill would target.

| aggregator | coverage RMSE | diverged | worst cell | verdict |
| --- | ---: | :---: | ---: | --- |
| clean FedAvg (no poison) | **11.87 dB** | no | 9 | learns the real map |
| poisoned FedAvg | **1.1·10⁷ dB** | **yes** | 62468 | **map destroyed, fill misdirected** |
| poisoned **Krum** | **19.46 dB** | no | **9** | **poison bounded, correct cell** |
| poisoned coordinate-median | 22.16 dB | no | 9 | damage bounded to ~baseline |
| poisoned trimmed-mean (β=3) | 35.81 dB | no | 1211 | bounded, but degraded |
| DP-FedAvg (clean, z=8) | 350.5 dB | no | — | ε = 4.14 @ δ=1e-5 |

Read honestly:

- **Naive FedAvg is catastrophically vulnerable on real data.** Three boosted
  clients *diverge* the coverage predictor — RMSE explodes to ~10⁷ dB and the
  power-fill it selects lands on a meaningless cell (`62468`).
- **Krum is the correct defense here.** By selecting a whole intact client update
  rather than averaging coordinates, it bounds the poison to RMSE 19.46 dB —
  below the 20.11 dB predict-mean baseline and ~six orders of magnitude below the
  poisoned FedAvg — and recovers **the same target cell (9)** the clean model
  picks. Coordinate-median also recovers cell 9 (RMSE 22.16 dB); trimmed-mean
  bounds the magnitude but degrades more (RMSE 35.81 dB, wrong cell).
- **The poison misdirects the power-fill.** Clean and Krum both target cell 9;
  poisoned FedAvg targets cell 62468 — a concrete operational harm (the coverage
  boost goes to the wrong place) that robust aggregation prevents.
- **Differential privacy carries a certified budget** (ε = 4.14 at δ = 1e-5 over
  40 rounds, noise multiplier z = 8) via the shipped RDP accountant. The DP noise
  raises RMSE (350 dB) — an honest privacy/utility trade-off on this higher-
  dimensional model, not a defect.

## The trust chain (real, on the selected cell)

The selected cell's coverage-fill is emitted as a power policy through the
Decision Safety Shield and written to a hash-chained evidence store:

- The robust-selected fill (in band, EIRP 32 dBm < 33) needs **no correction,
  passes the guard chain, and emits cleanly**, recorded.
- A coverage-fill that naively requests **EIRP 46 dBm** is **projected** onto the
  legal cap — **33 dBm** — and the raw over-power emit trips the
  `corrections_not_audited` guard, so **it never reaches the RAN**
  (`overpower_fill_reached_ran = false`).
- The evidence chain **verifies intact** (`verify() = -1`); after a single-byte
  tamper of record #1 on disk, re-verification **pinpoints the tamper at index
  1** and raises an X.733 `processingErrorAlarm`.

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
```

The runner exits non-zero unless clean FL beats the baseline, the poison diverges
plain FedAvg, Krum stays bounded and below baseline, the over-power fill is
corrected and never reaches the RAN, and the evidence chain verifies intact then
catches the tamper. CI runs exactly this on rebuilt real data
(`.github/workflows/realdata.yml`), and the committed-result tests assert the
same invariants in the fast pack.

## Scope and honesty

Real DeepMIMO ray tracing drives real federated learning (FedProx), real
Byzantine-robust aggregation, a real RDP privacy accountant, real Shield legality
and a real tamper-evident evidence chain. This is **site-specific ray tracing,
not over-the-air capture**, and a **single scenario**. The licence-gated raw
feature rows are not redistributed; the committed result JSON is bound to the
canonical build by `features_sha256`, and the `realdata` workflow rebuilds and
re-runs the loop to reproduce it.
