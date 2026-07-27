# Federated + trust core on real DeepMIMO — proof

This is the first Horizon artifact that drives the **federated learning and trust
core** with real measured-physics data instead of synthetic gradient vectors. It
answers, on real channels, the honest-accounting question "which core
capabilities have *not* been exercised on real DeepMIMO data?" — the federated
aggregation, differential-privacy, Shield-legality and evidence-chain paths, all
of which previously ran only on synthetic updates.

- **Runner:** [`../../benchmarks/federated_dsa_loop.py`](../../benchmarks/federated_dsa_loop.py)
- **Result:** [`../../benchmarks/results/federated_dsa.json`](../../benchmarks/results/federated_dsa.json)
- **Sample evidence record:** [`../../benchmarks/results/federated_dsa_sample_record.json`](../../benchmarks/results/federated_dsa_sample_record.json)
- **CI-safe tests:** [`../../tests/test_federated_dsa_loop.py`](../../tests/test_federated_dsa_loop.py)
- **Data:** DeepMIMO ASU Campus 3.5 GHz, 512 ray-traced receivers (Wireless
  InSite), the same licence-gated feature file the DSA benchmark consumes.

## The loop

```
real ray-traced MIMO channels (DeepMIMO ASU 3.5 GHz, 512 receivers)
  -> geographic, non-IID federated clients (partitioned by real position)
  -> 25 rounds of FedAvg with real local SGD on each client's own channels
  -> Byzantine-robust aggregation (Krum / coordinate-median / trimmed-mean)
  -> differentially-private release (RDP accountant, certified epsilon)
  -> global subband-quality model  ->  DSA default-subband selection
  -> Decision Safety Shield (band + EIRP legality, projection operator)
  -> hash-chained, tamper-evident evidence record
```

The learning task is physics-grounded. Each receiver contributes its real
per-subband channel gain vector `g ∈ ℝ⁶` (dBW). Centering across the six subbands
removes large-scale path loss and leaves the *relative* frequency-selective
structure `s = g − mean(g)`. The federated model `w ∈ ℝ⁶` is trained by
minimising `L_c(w) = ½·E_r‖w − s_r‖²` on each client `c` with real SGD; its FedAvg
fixed point is the population subband-quality prior `μ*`. `argmax(w)` is the
network-wide default subband a cell uses when it must choose without per-UE CSI.

The 512 receivers split into **10 geographic clients** that are genuinely
non-IID — ray tracing is site-specific, so different campus areas prefer
different subbands. Measured per-site best-fixed subbands:
`[5, 1, 0, 3, 4, 4, 1, 4, 5, 4]`.

## Honest magnitude first

The ASU campus channel is **near frequency-flat on average**: the six 100 MHz
subbands differ by only **0.298 dB** in mean gain, so a *global-default* subband
is worth only a few tenths of a dB. We report that truthfully rather than inflate
it. Two real, modest DSA gains on this data:

| quantity | value | needs |
| --- | --- | --- |
| per-receiver oracle vs fixed subband | ~1.6 dB | real-time per-UE CSI (the DSA benchmark) |
| per-site personalization vs one global default | **+0.215 dB** (up to +0.57 dB) | the non-IID federated split |

The strong, unambiguous result on this data is not a dB headline — it is the
**security mechanism**, measured on the very same real channels.

## The security result (real DeepMIMO, 3 of 10 clients Byzantine)

A model-replacement / scaling poison (Bagdasaryan et al. 2020) from 3 of 10
clients. `cos→μ*` is cosine to the true population profile (scale-invariant, so a
sign inversion shows up even as magnitude diverges). `oracle_best_fixed = 4`.

| aggregator | selected subband | mean regret (dB) | cos→μ* | ‖w‖ | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| clean FedAvg (no poison) | **4** (best) | 1.604 | **+0.979** | 0.30 | recovers the true profile |
| poisoned FedAvg | **2** (worst) | 1.902 | **−0.879** | **1.4·10¹³** | **inverted + diverged** |
| poisoned **Krum** | **5** (good) | 1.606 | **+0.219** | 0.29 | **neutralised, near-optimal** |
| poisoned coordinate-median | 2 | 1.902 | −0.724 | 0.23 | magnitude bounded, ranking lost |
| poisoned trimmed-mean (β=3) | 2 | 1.902 | −0.691 | 0.26 | magnitude bounded, ranking lost |
| DP-FedAvg (clean, z=8) | 2 | 1.902 | −0.168 | 4.73 | ε = 3.21 @ δ=1e-5 |

Read honestly:

- **Naive FedAvg is catastrophically vulnerable on real data.** Three boosted
  clients *invert* the learned model (cosine goes negative) and make it diverge
  (‖w‖ → 10¹³), so it selects the globally-**worst** subband.
- **Krum is the correct defense here.** By selecting a whole intact client update
  rather than averaging coordinates, it stays aligned to the true profile
  (cos +0.219), keeps ‖w‖ bounded, and picks a near-optimal subband (regret
  1.606 dB ≈ the clean 1.604 dB).
- **Coordinate-median and trimmed-mean bound the *magnitude* but not the *fine
  ranking*** on this near-flat, heterogeneous signal: 3 identical extreme updates
  at 30% contamination still flip a 0.3 dB argmax. This is the documented
  "robust aggregation bounds damage but is not a silver bullet" behaviour — and
  it is exactly why the codebase ships Krum alongside them. On a channel with a
  larger real signal the coordinate operators recover the ranking too (see the
  clean row and the synthetic-fixture test, where all aggregators recover the
  true best subband).
- **Differential privacy carries a certified budget** (ε = 3.21 at δ = 1e-5 over
  25 rounds, noise multiplier z = 8) via the shipped RDP accountant. On a 0.3 dB
  signal the DP noise dominates the argmax — an honest illustration of the
  privacy/utility trade-off, not a defect.

## The trust chain (real, on the selected subbands)

Every selected subband is emitted as a spectrum-reservation policy through the
Decision Safety Shield and written to a hash-chained evidence store:

- The two model-selected subband policies are legal (in band, EIRP 32 dBm < 33):
  **no correction, guard chain passes, emitted cleanly**, recorded.
- An out-of-band, over-EIRP AI proposal (3.62 GHz, EIRP 46 dBm) is **projected**
  onto the nearest legal action — **3.54 GHz, EIRP 33 dBm** — and the raw unsafe
  emit trips the `corrections_not_audited` guard, so **the unsafe action never
  reaches the RAN** (`unsafe_oob_reached_ran = false`).
- The evidence chain **verifies intact** (`verify() = -1`), and after a
  single-byte tamper of record #1 on disk, re-verification **pinpoints the
  tamper at index 1** and raises an X.733 `processingErrorAlarm`.

## Reproduce

```bash
# Build the licence-gated DeepMIMO feature file (checksum-gated, ~133 MB fetch).
python datasets/deepmimo_asu_3p5/build.py \
  --cache-dir /tmp/dm --features /tmp/dm/features.jsonl \
  --manifest /tmp/dm/manifest.json --samples 512

# Run the whole federated + trust loop on the real channels.
python benchmarks/federated_dsa_loop.py \
  --features /tmp/dm/features.jsonl --manifest /tmp/dm/manifest.json \
  --out benchmarks/results/federated_dsa.json
```

The runner exits non-zero unless the poison inverts + diverges plain FedAvg,
Krum stays aligned and bounded, the unsafe proposal is corrected and never
reaches the RAN, and the evidence chain verifies intact then catches the tamper.

## Scope and honesty

Real DeepMIMO ray tracing drives real federated SGD, real Byzantine-robust
aggregation, a real RDP privacy accountant, real Shield legality and a real
tamper-evident evidence chain. This is **site-specific ray tracing, not
over-the-air capture**, and a **single scenario**. The global-default subband
signal is small (~0.3 dB) because the campus channel is near frequency-flat on
average; the poison-inversion-vs-robust-alignment contrast is the strong,
unambiguous result on this data. The CI-safe tests exercise the identical code
paths on a synthetic, schema-compatible fixture with a deliberately clearer
signal, so the invariants hold without the heavy DeepMIMO extras.
