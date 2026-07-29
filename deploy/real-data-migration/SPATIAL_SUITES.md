# Spatial / PHY suites: migration from synthetic draws to measured ray tracing

**Scope of this document.** Three benchmarks — `benchmarks/jamming_suite.py`,
`benchmarks/phy_fading_eval.py`, `benchmarks/secure_dsa_benchmark.py` — used to
be driven entirely by `numpy.random`. They are now driven by the licence-gated
DeepMIMO ASU Campus 3.5 GHz build that was already in the tree. No download was
needed and no dataset was rebuilt.

Everything below is a number produced by a command quoted next to it, run on
2026-07-28 with `/home/user/venv/bin/python` (numpy 2.4.6, CPython 3.11.15).

---

## 0. The data that made this possible

| File | What it actually is |
| --- | --- |
| `datasets/deepmimo_asu_3p5/generated/channel_features.jsonl` | 4096 rows: real 3D receiver position (`position_m`) + ray-traced gain in each of 6 subbands across 100 MHz |
| `datasets/deepmimo_asu_3p5/generated/angular_features.jsonl` | the same 4096 receivers: per-path `power_dbw`, `phase_deg`, `delay_ns`, AoD/AoA |
| `datasets/deepmimo_asu_3p5/manifest.json` | `features_sha256 = ab4414a1…69e5af`, `source_archive_sha256 = 80e4a498…7da3` |
| `datasets/deepmimo_asu_3p5/angular_manifest.json` | `features_sha256 = 13eb2f6b…a6884`, plus the real BS position `tx_position_m = [166.0, 104.0, 22.0]` |

Both files are licence-gated and gitignored; only the manifests and the derived
aggregate results in `benchmarks/results/` are committed. All three result JSONs
now carry `dataset`, `scenario`, `data_kind`, `source_archive_sha256`,
`source_tree_sha256`, `features_sha256` and a `scope_note`, copying the
convention in `benchmarks/deepmimo_dsa_benchmark.py`.

### The cross-check that makes the "this is really the ray-traced channel" claim testable

`angular_features.jsonl` and `channel_features.jsonl` were built by two separate
scripts (`build_angular.py`, `build.py`). `jamming_suite.py` and
`phy_fading_eval.py` re-synthesise the OFDM frequency response from the *angular*
per-path data,

```
H_r(f_k) = Σ_l 10**(P_l/20) · e^{j·φ_l} · e^{-j·2π·f_k·τ_l},   f_k = k·B/K,  K = 1024,  B = 100 MHz
```

then average `|H|²` over the same 60 selected subcarriers `build.py` uses, and
compare against the committed `subband_gain_dbw` after undoing DeepMIMO's `1/K`
OFDM scaling. Both benchmarks run this every time and abort above 1e-3 dB RMS:

```
$ /home/user/venv/bin/python benchmarks/phy_fading_eval.py
channel reconstruction vs committed features: 6.3e-05 dB RMS, max 0.002273 dB
```

6.3e-05 dB RMS over 4096 × 6 values. The channel these benchmarks now use is the
DeepMIMO channel, not an approximation of it. This also pins the absolute scale:
physical wideband channel gain = `subband_gain_dbw + 10·log10(1024)`.

---

## 1. `benchmarks/phy_fading_eval.py`

### What was synthetic (exact lines, from `git show HEAD:benchmarks/phy_fading_eval.py`)

```python
Xtr, ytr, _ = fading_dataset(40_000, M, snr_dB, rng)
Xte, yte, meta = fading_dataset(15_000, M, snr_dB, rng)
```

and inside `src/horizon_ric/phy/channel.py`, which `fading_dataset` calls:

```python
def rayleigh_taps(n_sym, n_taps, rng):
    pdp = np.exp(-np.arange(n_taps) / max(n_taps / 3.0, 1.0))
    ...
    taps = (rng.standard_normal((n_sym, n_taps)) + 1j * rng.standard_normal((n_sym, n_taps)))
```

The channel was four Rayleigh taps with an invented exponential power-delay
profile. Committed `setup` was `{"channel": "Rayleigh TDL (4 taps)"}` and the
result JSON had exactly three keys: `setup`, `points`, `honest_finding`. No
dataset, no hash, no scope note.

### What it is now

* The channel bank is the reconstructed ray-traced `H_r(f)` for all 4096 real
  receivers over all 1024 subcarriers.
* Each symbol draws a real receiver and a real subcarrier — a Monte-Carlo sample
  of the measured campus channel population.
* **Spatial train/test split**: the neural receiver trains on the 2044 receivers
  west of `x = -24.5511 m` and is evaluated on the 2052 receivers east of it, so
  every reported SER is generalisation to unseen campus geometry. The old version
  simply drew a fresh random channel for the test set.
* Measured dispersion statistics are reported rather than assumed:
  RMS delay spread median **109.5 ns**, p95 **324.0 ns**, max **677.4 ns**;
  BS-to-receiver distance median **253.0 m**, max **465.4 m**;
  measured link gain spans **104.2 dB** (`-174.1` to `-69.9` dBW).

### Before → after

```
$ /home/user/venv/bin/python benchmarks/phy_fading_eval.py
SNR 28.0 eps 0.07: PGD SER neural 0.0574±0.0011 vs classical 0.0495 (1.16x); Shield effective SER 0.0523, fallback 30%
SNR 34.0 eps 0.07: PGD SER neural 0.0444±0.0011 vs classical 0.0375 (1.18x); Shield effective SER 0.0393, fallback 36%
```

| Point | Synthetic Rayleigh (committed before) | Measured ray tracing (now) |
| --- | --- | --- |
| SNR 28 dB, PGD SER neural / LMMSE | 0.1295 / 0.1156 → **1.121x** | 0.0574 / 0.0495 → **1.161x** |
| SNR 34 dB, PGD SER neural / LMMSE | 0.1042 / 0.0904 → **1.153x** | 0.0444 / 0.0375 → **1.183x** |
| Shield fallback rate | 12% / 23% | 30% / 36% |

The headline claim survives and is now earned on measured propagation: the ~22x
neural-vs-classical adversarial gap of the single-symbol AWGN toy
(`neural_rx_pgd_benchmark.py`) does **not** survive multipath — it is 1.16–1.18x
here. Absolute SER is roughly halved versus the Rayleigh model, which is what you
would expect: a real ray-traced campus channel with a median 109 ns delay spread
has fewer catastrophically deep fades per symbol than a 4-tap Rayleigh draw. The
Shield falls back on 30–36% of windows instead of 12–23%.

### What this does NOT prove

Ray tracing is not over-the-air capture. The modulation, the AWGN, the neural
receiver and the PGD attacker are all still models sitting on top of measured
propagation. Two normalisations are disclosed in the JSON and repeated here:
each link is scaled to unit mean power (so `snr_dB` is a post-power-control
operating point, and the measured 104 dB large-scale spread is *reported* rather
than *applied*), and the setup is SISO isotropic, one BS, no mobility, no
Doppler, no inter-cell interference.

---

## 2. `benchmarks/jamming_suite.py`

### What was synthetic

```python
jsr_grid = [-10.0, -5.0, 0.0, 5.0, 10.0]
...
Xte, yte, meta = fading_dataset(15_000, M, snr_dB, rng)
Xj = barrage_jammer(Xte, jsr, rng)
```

The jammer-to-signal ratio was an invented grid of round numbers, and the victim
channel was the same synthetic Rayleigh draw as above. There was no jammer
position, no victim position, no path loss and no geometry anywhere in the file.
Committed keys: `setup`, four jammer sweeps, `imperfect_csi`, `honest_findings`.

### What it is now — two layers

**Layer A: jammer siting on measured propagation.** The jammer is placed at a
*real* DeepMIMO grid point, so its channel to the base station is measured. Ray
tracing is reciprocal and both ends are isotropic SISO, so the measured
BS→grid-point gain is used as the grid-point→BS gain; that reciprocity is the
only propagation assumption in Layer A. Victim uplink SINR at the BS is then

```
SINR_r,k = P_ue + G_r,k − 10·log10( 10^((P_j + G_j,k)/10) + 10^(N/10) )
```

with `G` measured throughout. Sweeping the jammer across 128 real grid points
produces a vulnerability map of the actual campus.

**Layer B: jammer waveforms on the measured channel.** Barrage / partial-band /
single-tone / pulsed / imperfect-CSI, run against the neural receiver, the LMMSE
receiver and the Shield's routed output over the reconstructed ray-traced
channel. The JSR operating points are no longer invented — they are the 10/50/90
percentiles of the JSR field that the representative real jammer site actually
produces: **[-25.299, -1.471, 25.955] dB**.

### Before → after

```
$ /home/user/venv/bin/python benchmarks/jamming_suite.py
channel reconstruction vs committed features: 6.3e-05 dB RMS
jammer siting (128 real sites, 23 dBm): campus outage@11dB min=0.472 median=0.649 max=1.000 (unjammed 0.472)
subband escape @ best   real site: aware-fixed +1.726 dB, aware-unaware +0.000 dB, 20 receivers re-tune
subband escape @ median real site: aware-fixed +2.010 dB, aware-unaware +0.097 dB, 1266 receivers re-tune
subband escape @ worst  real site: aware-fixed +1.952 dB, aware-unaware +0.193 dB, 1374 receivers re-tune
Shield: 12288 decisions over 3 real sites, illegal_emits=0
Layer B measured JSR operating points (dB): [-25.299, -1.471, 25.955]
```

| Quantity | Before | Now |
| --- | --- | --- |
| Jammer position | none | a real grid point; worst `[160.449, 56.831, 1.5]` m, 51.7 m from the BS; best `[-203.551, -156.169, 1.5]` m, 452.4 m from the BS |
| Victim positions | none | 4096 real, of which **2164** are servable at ≥11 dB when un-jammed |
| JSR | invented `[-10…10]` grid | measured field, p5 `-28.6` dB → p95 `+30.0` dB at the representative site |
| Campus denial | not measurable | a 23 dBm jammer denies **47.2%** (i.e. no worse than un-jammed) to **100%** of the campus depending purely on where it stands; median 64.9% |
| Shield legality | not evaluated | 12288 escape decisions, `shield_projected` on every one, **0** illegal emits |

Three findings that only exist because the data is real:

1. **Siting dominates, and gain beats distance as the predictor.** Across 128
   real sites, campus outage correlates with the jammer's *measured gain* to the
   BS at Pearson **r = 0.959**, versus **r = −0.746** for its straight-line
   distance. Distance is a usable proxy but not the thing; keep-out rules should
   be written against the ray-traced gain field.
2. **Jamming-aware subband selection is worth much less than it sounds.** At the
   representative real site, subband selection is worth **+2.010 dB** mean SINR
   over a fixed subband, but making that selection jamming-*aware* rather than
   just best-gain adds only **+0.097 dB** (+0.113 dB over the 2164 servable
   receivers) — even though **1266 of 4096** receivers do change which subband is
   optimal. The reason is visible in the measurement: the interference term is
   common to every victim and the jammer's ray-traced per-subband profile is only
   a few dB wide, so the victim's own gain variation dominates. Anyone selling
   jamming-aware DSA as a large win on this geometry is overselling it.
3. **Legality is orthogonal to the attack.** Every escape decision was submitted
   with a deliberately over-limit 44 dBm request; all 12288 were projected and
   none produced an illegal carrier.

The Layer B conclusions are unchanged in kind — a barrage jammer raises the noise
floor for both receivers so the Shield, which only *routes between* receivers,
cannot recover the loss — but they are now anchored to JSR values the real
geometry produces rather than to round numbers.

### What this does NOT prove

* **A modelled jammer on measured propagation is not a captured attack.** No
  jamming incident, no spectrum-monitoring trace and no adversary telemetry
  appears anywhere in this file.
* **The dataset has BS→grid-point channels only.** A jammer's channel to a victim
  *handset* does not exist in it at any price. Layer A is therefore deliberately
  an uplink / receiver-blocking geometry — the geometry the measurement actually
  supports. We did not invent a grid-point-to-grid-point path-loss model to
  manufacture a downlink jamming story.
* The transmit powers (23 dBm UE / jammer), noise figure (7 dB) and SINR
  thresholds (11 / 18 dB) are declared constants, not measurements. They are
  listed under `setup.link_budget` in the JSON so any reader can re-scale.
* Note the honest baseline: **47.2%** of the 4096 grid points are already below
  11 dB with no jammer at all, because the DeepMIMO grid extends well past this
  cell's useful range. Jamming figures should be read against that number, not
  against zero.

`benchmarks/antijam_loop.py` already used the real angular data for MVDR
null-steering; this suite is complementary (SINR geometry and link-level
receivers, not beamforming) and does not duplicate it.

---

## 3. `benchmarks/secure_dsa_benchmark.py` — only PARTIALLY made real

This one could not be fully migrated, and pretending otherwise would be the exact
failure mode this exercise exists to fix.

### What was synthetic, and what still is

The federated DSA loop is a Gilbert–Elliott primary-user MDP
(`src/horizon_ric/spectrum/dsa_env.py`) and the poisoning attacks are crafted
Q-tables and ALIE/Fang constructions (`src/horizon_ric/spectrum/attacks.py`).
**None of that can be made real from DeepMIMO**, because DeepMIMO contains
propagation, not spectrum occupancy and not federated-learning traffic. The
following remain synthetic and are listed as such in
`setup.synthetic_components` in the result JSON:

* primary-user occupancy (Gilbert–Elliott chain);
* energy-detection sensing error;
* the federated client partition — clients are seeded MDP replicas, not real
  geographic cohorts;
* **the poisoning attacks themselves.** These are not captured malicious updates
  from any deployed federated system.

### What was made real: the payoff, not the threat

The abstract MDP counts successful slots, which treats all six channels as
interchangeable. They are not. The benchmark now also scores every policy on the
measured spectrum: each secondary user is pinned per episode to one of the 4096
real ray-traced positions, and a successful transmission on channel `c` delivers
the spectral efficiency that receiver's *measured* subband gain supports (Shannon
rate, floored at an 11 dB scheduling threshold and capped at 7.4063 bit/s/Hz).
The six MDP channels map one-to-one onto the six real 3.5 GHz subbands.

Measured campus SNR under the declared link budget: median **12.1 dB**, p5
**−19.4 dB**, max **50.4 dB**; **2164 of 4096** receivers clear the 11 dB
threshold on their best subband.

The campus evaluator uses a *separate* RNG stream for the position draw so its
action sequence is bit-identical to the abstract evaluator's. That is verifiable:

```
$ /home/user/venv/bin/python -c "import json;d=json.load(open('benchmarks/results/secure_dsa.json'));print(d['policy_quality']['clean_reference']['throughput_per_slot'],d['policy_quality']['clean_reference_campus']['successes_per_slot'])"
1.0717 1.0717
```

The campus column is a pure re-weighting of the identical decision trace, not a
different experiment.

### Before → after

```
$ /home/user/venv/bin/python benchmarks/secure_dsa_benchmark.py
=== DSA policy quality: abstract slots | real campus bit/s/Hz ===
clean reference: 1.0717 | 4.002 (0.346 of genie)
  qtable_target  fedavg=0.00/0.00  median=0.68/2.52  krum=0.38/1.46
  alie           fedavg=1.01/3.78  median=0.94/3.52  krum=1.02/3.81
  fang_krum      fedavg=1.00/3.73  median=0.97/3.60  krum=1.00/3.74
  fang_median    fedavg=1.12/4.15  median=1.02/3.73  krum=0.38/1.46

LEGALITY GUARANTEE: HOLDS
```

The abstract numbers are unchanged from the committed result (clean reference
`throughput_per_slot = 1.0717`), which is the point: nothing about the security
experiment moved. What is new is the second column and one deflationary finding
it exposes —

> the clean, un-attacked policy reaches only **0.346 of the genie capacity**
> available at those measured positions.

The tabular policy's state is a PU sensing vector; it carries no information
about *which subband is good where*. On an abstract slot-count metric that
limitation is invisible. On the real campus it costs about two thirds of the
available capacity. That is a real limitation of the agent this repo ships, and
it was not visible before the data was attached.

### What this does NOT prove

The real component here is the utility metric and the spatial spread of the
secondary users. The threat model is unchanged and remains entirely synthetic. A
reader should not take "secure DSA validated on real data" from this file; the
correct reading is "the *cost* of a synthetic attack is now denominated in
bit/s/Hz deliverable on a real campus".

---

## 4. Handoffs (changes needed in files this work does not own)

1. **`benchmarks/results/SHA256SUMS`** lists digests for `jamming_suite.json`,
   `phy_fading.json` and `secure_dsa.json`. All three files changed, so those
   three lines are now stale and must be regenerated by whoever owns that file.
   Previous values: `16cf1500…99fc6` (jamming), `3d8fa327…77e9d` (phy_fading),
   `778686b8…8f8622aa1f` (secure_dsa).
2. **`src/horizon_ric/spectrum/federated_q.py`** — `federated_dsa_round()`
   constructs its clients internally from seeded `DSAEnv` replicas. To make the
   federated client partition *real* (each client = a geographic cohort of the
   measured campus, giving genuine non-IID spatial heterogeneity instead of a
   synthetic one) the function needs a way to inject per-client receiver
   populations. That is the single highest-value remaining step for
   `secure_dsa_benchmark.py` and it cannot be done from the benchmark file.
3. **`src/horizon_ric/phy/channel.py`** — `fading_dataset` / `rayleigh_taps`
   remain the synthetic path and are still used by other benchmarks
   (`neural_rx_pgd_benchmark.py`, `evasion_suite.py`). Nothing here changed them;
   a `RealChannelBank` equivalent living in `src/` would let those migrate too
   and would remove the duplicated reconstruction code now present in both
   `jamming_suite.py` and `phy_fading_eval.py`.
4. **Docs referencing these suites** — `README.md`, `docs/THREAT_MODEL.md`,
   `docs/EVALUATION_CRITERIA.md`, `docs/RESEARCH_ALIGNMENT.md` and
   `datasets/DATASETS_INDEX.md` mention these three benchmarks and may describe
   them as synthetic.

## 5. Verification

```
$ /home/user/venv/bin/python -m ruff check benchmarks/jamming_suite.py benchmarks/phy_fading_eval.py benchmarks/secure_dsa_benchmark.py
All checks passed!

$ for b in jamming_suite phy_fading_eval secure_dsa_benchmark; do /home/user/venv/bin/python benchmarks/$b.py >/dev/null; echo "$b exit=$?"; done
jamming_suite exit=0
phy_fading_eval exit=0
secure_dsa_benchmark exit=0
```

Wall-clock on this host: jamming_suite ~15 s, phy_fading_eval ~36 s,
secure_dsa_benchmark ~43 s.

All three benchmarks abort rather than fall back to synthetic data if the
DeepMIMO feature files are absent, if the row count disagrees with the manifest,
or (for the two that use it) if the ray reconstruction drifts more than 1e-3 dB
RMS from the committed subband gains. There is no silent synthetic path left in
them.
