# Poisoning suites: migration from synthetic gradients to a real federation

**Scope.** Four benchmarks —
`benchmarks/fl_poisoning_suite.py`, `benchmarks/poisoning_shield_benchmark.py`,
`benchmarks/dsa_poison_suite.py`, `benchmarks/neural_rx_pgd_benchmark.py` —
had no measurement anywhere in them. Their "clients", their "gradients", their
"decisions" and their "channel" were all `numpy.random`. They now run on the
licence-gated DeepMIMO ASU Campus 3.5 GHz build that was already in the tree. No
new dataset was downloaded and nothing was rebuilt.

Every number below was produced by the command quoted next to it, run on
2026-07-28 with `/home/user/venv/bin/python`.

**Read this first, because it is the honest core of the whole migration:**

> **A public corpus of real, captured malicious federated-learning client updates
> does not exist.** Two independent scouting passes looked for one and found only
> *method* papers — PoisonedFL, Fang et al., Shejwalkar & Houmansadr, ALIE,
> Bagdasaryan et al. — every one of which *synthesises* the malicious updates and
> evaluates on CIFAR-10 / Fashion-MNIST. Capturing real ones would require an
> actual adversary to attack an actual production federation *and* the operator to
> publish the raw gradient tensors. Nobody has done that.
>
> So these suites are **not** claimed to be attack-data-driven. The construction
> used everywhere below is:
>
> **REAL clients + REAL data + REAL benign updates + a PUBLISHED, CITED attack
> algorithm applied to them.**
>
> Each result JSON carries a `data_provenance` block that says which half is
> which, in those words, so a reader cannot mistake one for the other.

---

## 0. The real substrate

| File | What it actually is |
| --- | --- |
| `datasets/deepmimo_asu_3p5/generated/channel_features.jsonl` | 4096 rows: real 3D receiver position + Wireless InSite ray-traced gain in each of 6 subbands across 100 MHz at 3.5 GHz |
| `datasets/deepmimo_asu_3p5/generated/angular_features.jsonl` | the same 4096 receivers, per-path `power_dbw`, `phase_deg`, `delay_ns` |
| `datasets/deepmimo_asu_3p5/manifest.json` | `features_sha256 = ab4414a1…69e5af`, `source_archive_sha256 = 80e4a498…7da3` |

Raw rows stay gitignored and un-redistributed; only manifests and the derived
aggregates in `benchmarks/results/` are committed. All four result JSONs now
carry `dataset`, `scenario`, `data_kind`, `features_sha256`,
`source_archive_sha256`, `source_tree_sha256` and `data_provenance`.

The federated substrate all three FL-side suites share:

```
4096 measured receiver positions
  -> Lloyd partition on the REAL 2D coordinates  -> geographic, non-IID clients
  -> each client fits the measured coverage field (position -> 6 subband gains)
  -> its upload is a FedProx proximal ridge step on ITS OWN measurements
```

Defined once in `fl_poisoning_suite.CoverageFederation` and imported by
`poisoning_shield_benchmark.py`; `dsa_poison_suite.py` imports the same
`_partition_geographic`.

---

## 1. `benchmarks/fl_poisoning_suite.py`

### What was synthetic

```python
# git show 0df3dd9:benchmarks/fl_poisoning_suite.py
def honest_population(rng, n_honest, dim):
    mu = rng.normal(0.0, 1.0, size=dim)
    scale = rng.uniform(0.5, 1.5, size=dim)
    return [mu + scale * rng.normal(0.0, 1.0, size=dim) for _ in range(n_honest)]
```

12 clients × 80 coordinates of Gaussian noise. The reported metric was the L2
displacement of an aggregate of that noise. There was no model, no task, and no
unit anyone could interpret.

### What it is now

16 geographic client cohorts over 3584 client-held real receivers (512 held back
as the server root set FLTrust needs), each fitting a 36-parameter multi-output
coverage model (6 whitened quadratic position features × 6 measured subbands),
12 federated rounds, 6 partition seeds. Damage is reported as **coverage-map RMSE
in dB on the measured gains** — plus the old L2/cosine numbers, kept for
continuity.

The attacks are unchanged code (`horizon_ric.federated.poison_attacks`), now
computed from the *real* benign updates: sign_flip, scaling (Bagdasaryan et al.,
AISTATS'20), gaussian, min_max / min_sum (Shejwalkar & Houmansadr, NDSS'21),
alie (Baruch et al., NeurIPS'19), fang_krum / fang_median (Fang et al.,
USENIX-Sec'20). Defences: FedAvg, Krum, coordinate median, trimmed mean, and
**FLTrust** (Cao et al., NDSS'21), which the old suite did not grade at all.

### Before → after

```
$ /home/user/venv/bin/python benchmarks/fl_poisoning_suite.py --seeds 6 --rounds 12
```

| | before (0df3dd9) | after |
| --- | --- | --- |
| clients | 12 synthetic Gaussian draws | 16 geographic cohorts of 165–303 real receivers |
| update dimension | 80 (arbitrary) | 36 (the real model's parameters) |
| metric | L2 from the mean of noise | coverage RMSE in dB on measured gains |
| reference points | none | predict-the-mean 20.17 dB, centralised least squares 11.88 dB |
| defences graded | 4 | 5 (FLTrust added) |

Clean federation, no adversary (mean over 6 partition seeds):

```
clean RMSE dB: fedavg=13.19, krum=18.60, median=13.45, trimmed_mean=13.19, fltrust=12.67  | baseline 20.17
```

Attack at f = 8 of 16 clients (50%, the median/trimmed-mean breakdown point):

```
attack       |        fedavg |          krum |        median |  trimmed_mean |       fltrust
--------------------------------------------------------------------------------------------
sign_flip    |         20.17 |           N/A |         41.37 |           N/A |         12.65
scaling      |         52.16 |           N/A |        149.50 |           N/A |         12.65
gaussian     |        178.58 |           N/A |         13.52 |           N/A |         12.87
min_max      |         14.55 |           N/A |         20.06 |           N/A |         12.54
min_sum      |         14.65 |           N/A |         32.67 |           N/A |         12.51
alie         |         13.47 |           N/A |         20.92 |           N/A |         12.48
fang_krum    |         13.37 |           N/A |         13.37 |           N/A |         12.16
fang_median  |         81.28 |           N/A |        263.75 |           N/A |         12.66
```

(`N/A` = the aggregator's structural precondition `n > 2f + 2` / `n > 2β` is not
met at f = 8; the suite refuses to report a number rather than invent one.)

Findings that are now expressed in physical units:

* **Coordinate median is not a defence at its breakdown point.** Under `scaling`
  it lets the real coverage model degrade from 13.45 dB to **149.50 dB** RMSE, and
  under `fang_median` to **263.75 dB** — i.e. the learned coverage map becomes
  worse than useless (predict-the-mean is 20.17 dB).
* **FLTrust holds everywhere.** Across all 8 attacks × 4 Byzantine counts (1, 2,
  6, 8 of 16) its worst RMSE is **12.87 dB** against a 12.67 dB clean baseline.
  It is the only shipped defence that stays at clean quality at 50% adversaries.
* **Krum is bad before any attack.** At f = 6 (37.5%, its structural ceiling) its
  clean RMSE is 18.60 dB against FedAvg's 13.19 dB, and `min_max` forces it to
  *select* a malicious client in **43%** of rounds
  (`krum_byzantine_selection_rate`). This corroborates the earlier errata
  (`deploy/shield-learning/ERRATA.md`) on real data.

### What this does NOT prove

The malicious updates are algorithmic. They are optimised against the honest
clients' real updates under the full-knowledge threat model those papers assume,
which is the strongest published attacker, but they were not captured from an
adversary.

---

## 2. `benchmarks/poisoning_shield_benchmark.py`

### What was synthetic

```python
# git show 0df3dd9:benchmarks/poisoning_shield_benchmark.py
"frequency_hz": float(rng.uniform(BAND_LO + 15e6, BAND_HI - 15e6)),
"tx_power_dBm": float(rng.uniform(10, 24)),
"constellation_order": int(rng.choice([4, 16, 64, 256])),
...
benign = [rng.normal(0.0, 1.0, size=dim) for _ in range(n_honest)]
```

Both halves. "How often does the Shield's EIRP projection fire" was a property of
`rng.uniform(10, 24)`.

### What it is now

Each decision serves one **real receiver**:

* carrier = that receiver's **measured best subband** (the six subbands tile the
  licensed 100 MHz band inside a 2 MHz guard band, 16 MHz each);
* transmit power = **closed-loop power control** solving for a 20 dB SINR against
  that receiver's **measured** path gain, clipped to a 46 dBm RU ceiling;
* constellation order = link adaptation on the resulting SINR.

The federated half now uses `CoverageFederation`: 20 real geographic clients,
4 compromised, 36-dimensional real updates.

### Before → after

```
$ /home/user/venv/bin/python benchmarks/poisoning_shield_benchmark.py
decision stream: 4096 real receivers, measured best-subband gain -171.752..-67.37 dB;
  46.0% of receivers demand more EIRP than the licence allows
SHIELD (independent ACLR oracle): 4658 illegal emits unguarded -> 0 after the Shield  [PASS]
  in-spec-but-harmful caught: 461 unguarded -> 0 after Shield; EIRP projections 46.0% of decisions
FEDERATED ALIE/Fang on REAL updates (byz 20% of 20 real geographic clients, dim 36):
  no-attack krum dist 0.188; ALIE median 0.060; Fang-median median 0.180
```

| | before | after |
| --- | --- | --- |
| decisions | 10000 `rng.uniform` draws | 8000 decisions serving real receivers |
| unguarded illegal (independent ACLR oracle) | 1983 | 4658 |
| — of which caused by the poisoning | (not separated) | 2216 |
| — of which caused by **real geometry** | n/a | **2442** |
| shielded illegal | 0 | 0 |
| federated benign updates | `rng.normal`, dim 200 | real FedProx steps, dim 36 |

The new split is the interesting part, and it is a finding the synthetic version
could not have produced: **more than half the un-shielded licence violations here
have nothing to do with the attacker.** Closed-loop power control aiming at a
20 dB SINR for cell-edge receivers on this campus asks for more EIRP than the
licence permits on 46.0% of receivers. The Shield's EIRP projection is doing
routine regulatory work on real geometry before any adversary appears; the
poisoning then adds the qualitatively different failures (out-of-band carriers,
illegal modulation orders, band-edge-legal-but-ACLR-harmful spectral regrowth).
After the Shield the independent oracle finds zero violations in either class.

### What this does NOT prove

The *poisoning* of the decision stream is still modelled: `_poison()` synthesises
the six failure modes a poisoned neural-PHY model would exhibit. No capture of a
compromised RAN model's outputs is publicly available. PAPR is declared, not
measured.

---

## 3. `benchmarks/dsa_poison_suite.py`

### What was synthetic

Everything. Twelve federated clients that were seeded replicas of the same
abstract MDP — statistically identical, no geography, no propagation, no reason
to disagree — and damage counted in successful slots, which assumes every channel
is worth the same everywhere.

### What it is now

* **Real, non-IID clients.** 4096 real receivers → 12 geographic cohorts
  (225–420 receivers each).
* **Real per-client channel quality.** The 6 MDP channels *are* the 6 measured
  subbands. A client's reward for a successful transmission on channel `c` is
  scaled by the spectral efficiency its own receivers measurably support on
  subband `c` (Shannon, scheduled-MCS floor at 11 dB SINR, 256QAM ceiling at
  7.4063 bit/s/Hz). Clients disagree about which channel is worth using because
  the measured propagation says so.
* **Real damage units.** Policy quality is also reported as delivered bit/s/Hz at
  real receiver positions.
* **Real power requests** in the Shield section: closed-loop power control against
  the measured best-subband gain of the receiver each decision serves.
* Mean ± std over 3 MDP seeds (the geographic partition is real and fixed).

### Before → after

```
$ /home/user/venv/bin/python benchmarks/dsa_poison_suite.py
4096 real receivers -> 12 geographic clients (225-420 rx each); campus mean SE 4.0518 bit/s/Hz,
  41.9% of (receiver, subband) pairs below the scheduled-MCS floor

REWARD POISON (PU-clash/slot, mean+/-std over 3 seeds): clean 0.510 -> fedavg(nm=4) 0.662
  -> median(nm=4) 0.781 -> fedavg(nm=6) 0.812
  delivered bit/s/Hz per slot: clean 4.701 -> fedavg 4.941 -> median 5.213 (NOT a throughput attack)
BACKDOOR success: clean 0.135 | fedavg 1.000 | median(nm=4) 0.137 | median@breakdown(nm=6) 1.000
SHIELD: 0 illegal emits, 20 EIRP projections on real power requests, audit chain intact=True  [PASS]
```

| | before (0df3dd9) | after |
| --- | --- | --- |
| channels | 4 abstract | 6 = the 6 measured subbands |
| clients | 12 seeded MDP replicas | 12 geographic cohorts of real receivers |
| utility metric | successful slots | successful slots **and** delivered bit/s/Hz |
| statistics | 1 seed | mean ± std over 3 seeds |
| clean PU clash/slot | 0.2712 | 0.510 ± 0.023 |
| reward-poison fedavg PU clash (nm=4) | 0.6562 | 0.662 ± 0.029 |
| reward-poison median PU clash (nm=4) | 0.5900 | **0.781 ± 0.011** |
| backdoor SR fedavg / median(4) / median(6) | 1.0 / 0.1042 / 1.0 | 1.000 / 0.137 / 1.000 |

Two findings that only appear on real clients, both reported rather than dropped:

* **Reward poisoning is not a throughput attack here.** Delivered spectral
  efficiency does not fall (4.70 → 4.94 bit/s/Hz per slot); the inverted-reward
  clients push the aggregate towards transmitting *more*. The damage is
  **regulatory**: PU clashes per slot rise 0.510 → 0.662 (nm = 4) → 0.812
  (nm = 6).
* **Coordinate median is measurably worse than un-defended FedAvg on that
  metric** — 0.781 ± 0.011 vs 0.662 ± 0.029 at nm = 4, non-overlapping across the
  swept seeds. A reward-poisoned client's upload is a well-trained Q-table for an
  inverted objective, not a geometric outlier, so per-coordinate screening gives
  it full voting weight; and on genuinely non-IID geographic clients the
  coordinate median is not an estimate of the honest mean either, so it also
  discards the averaging that was helping.

**Scope of the Shield guarantee, stated in the JSON:** the Shield bounds
*emission* legality (band, EIRP, mask). It does not and cannot bound the PU-clash
rate, which is a policy-quality property. The clash rise above is residual damage
and belongs to the aggregator, not the Shield.

### What is still synthetic

* **Primary-user occupancy** (Gilbert–Elliott chain) and the sensing-error rate.
  The DeepMIMO dataset carries no occupancy trace. A real one exists — POWDER
  CBRS drive test, Zenodo 18272105, CC-BY-4.0, 2.2 MB, six real CBRS channels
  with per-channel RSRP/SINR/PCI and GPS — but wiring it in needs a
  `datasets/powder_cbrs_drivetest/build.py` that this file does not own (see §5).
* **The attacks**: reward poisoning, the single-row backdoor and free-riding are
  attacker algorithms, cited in `data_provenance.attack_citations`.

---

## 4. `benchmarks/neural_rx_pgd_benchmark.py` — and a claim that got smaller

### What was synthetic

```python
# git show 0df3dd9:benchmarks/neural_rx_pgd_benchmark.py
Xtr, ytr = make_dataset(80_000, M, SNR_dB, rng)   # flat AWGN at one assumed Es/N0 = 22 dB
```

### What it is now

The 4096 receivers' per-path rays are re-synthesised into the complex OFDM
frequency response and cross-checked against the independently built subband
gains every run:

```
reconstruction check 6.3e-05 dB RMS | measured gain span 104.227 dB over 4096 real receivers
```

Es/N0 per link is a **measured link budget** (33 dBm EIRP over 1024 subcarriers,
thermal noise + 7 dB NF applied to each receiver's own measured path gain), and
because that spans 104 dB there is no single operating point — the attack is run
over three measured-SINR cohorts and all three are reported. Amplitudes are
scaled by one cohort-wide AGC constant, **not** per link, so the measured SNR
spread survives into the symbol dataset.

### Before → after

```
$ /home/user/venv/bin/python benchmarks/neural_rx_pgd_benchmark.py
cohort            rx      cleanSER n/c        pgdSER n/c  n/c x
---------------------------------------------------------------
low_11_20dB      553  0.0935/0.0972     0.4068/0.3967      1.03
mid_20_30dB      612  0.0120/0.0123     0.2003/0.1785      1.12
high_30_45dB     972  0.0003/0.0002     0.1282/0.0968      1.32

Shield envelope tolerance sweep (fallback rate -> effective SER as a multiple of the certified LMMSE baseline):
  tol=1.0 dB (admits 1.259x): low:   0% -> 1.03x | mid:   0% -> 1.12x | high:   4% -> 1.31x
  tol=0.5 dB (admits 1.122x): low:   0% -> 1.03x | mid:   0% -> 1.12x | high:  17% -> 1.27x
  tol=0.2 dB (admits 1.047x): low:   0% -> 1.03x | mid:  17% -> 1.10x | high:  71% -> 1.09x
```

| | before (synthetic AWGN, 22 dB) | after (measured channels) |
| --- | --- | --- |
| PGD SER, neural vs classical | 0.01465 vs 0.000675 = **21.7x** | 0.1282 vs 0.0968 = **1.32x** (worst cohort) |
| clean SER | 0.0 / 0.0 | 0.0003 / 0.0002 (high cohort) |
| Shield fallback rate | 100% | 4% at the shipped 1.0 dB tolerance |
| attack reduction | 11.7x | 1.00–1.21x at 0.2 dB tolerance |

**Two corrections, both against us.**

1. **The 20x figure was an artefact of the AWGN model.** With flat AWGN an L∞
   budget of 0.12 sits below 16-QAM's decision half-margin (0.316), so the
   max-margin ML demapper is hard to flip while the learned boundary is not. On a
   *measured* frequency-selective channel the equaliser divides by H, so in the
   deep fades the same bounded perturbation on Y becomes an unbounded perturbation
   on the equalised symbol and hurts the certified receiver too. The
   `deep_fade_split` block in the result JSON shows the mechanism. On real
   channels the neural receiver's excess error is 1.03–1.32x, not ~20x.
2. **The shipped envelope tolerance is mis-calibrated for an attack this size.**
   `NeuralRxEnvelopeInvariant(tolerance_dB=1.0)` *admits* a receiver up to 1.259x
   worse than the certified baseline by construction, so against a 1.03–1.32x
   attack the fallback almost never fires. The tolerance sweep shows the mechanism
   itself is sound — at 0.2 dB it fires on 71% of blocks in the high-SINR cohort
   and pulls the effective error rate from 1.32x back to 1.09x of baseline — so
   this is a **calibration** finding, not a broken invariant. The synthetic attack
   was violent enough that the gap was invisible.

### What CANNOT be made real, and why no dataset fixes it

The PGD perturbation is a **function of this receiver's own weights**
(`NeuralReceiver.input_gradient`). An "adversarial RF dataset" would be
adversarial against somebody else's classifier and meaningless against ours. The
two canonical over-the-air adversarial-attack papers (arXiv 2002.02400, arXiv
2202.11197) release no data at all. The perturbation is synthetic **by
necessity**. The transmitted 16-QAM sequence is ours by construction. What the
real data buys is the channel the attack crosses and the SNR distribution it is
graded over — which is exactly what moved the headline number.

Rejected sources, for the record (from the scouting pass): RadioML/DeepSig is
CC BY-NC-SA (incompatible with this Apache-2.0 repo) *and* its download host's
TLS certificate expired on 2023-06-12; IEEE DataPort's jamming sets and CRAWDAD
`vanetjamming2012` are behind login walls with no declared redistribution
licence; the GENESYS CBRS radar+LTE corpus declares no licence and returns 403.

---

## 5. Handoffs (changes needed in files this work does not own)

1. **`benchmarks/results/SHA256SUMS` must be regenerated** for
   `fl_poisoning_suite.json`, `poisoning_shield.json`, `dsa_poison_suite.json`
   and `neural_rx_pgd.json`. Several agents rewrote result JSONs in parallel, so
   this file was deliberately not touched here — one pass at the end:
   `(cd benchmarks/results && sha256sum *.json > SHA256SUMS)`.
2. **`docs/THREAT_MODEL.md` §7 and `src/horizon_ric/federated/unlearning.py`'s
   module docstring** cite `benchmarks/results/dsa_poison_suite.json`; its schema
   changed (metrics are now `{"mean","std"}` objects and there is a new
   `delivered_bps_hz_per_slot`). Any prose quoting the old scalar numbers needs a
   re-read.
3. **Any doc repeating the "PGD makes the neural receiver ~20x worse" claim must
   be corrected to 1.03–1.32x on measured channels.** `benchmarks/phy_fading_eval.py`
   already says the synthetic claim does not survive multipath; the same
   correction belongs anywhere else it is quoted.
4. **Shield calibration follow-up (owner: whoever owns
   `src/horizon_ric/shield/invariants.py`).** `NeuralRxEnvelopeInvariant`'s
   default `tolerance_dB = 1.0` admits a 1.259x regression. The sweep in
   `benchmarks/results/neural_rx_pgd.json` shows 0.2 dB restores
   no-worse-than-baseline behaviour on real channels. This file did not change the
   default.
5. **Optional, to make the DSA occupancy real:** a
   `datasets/powder_cbrs_drivetest/build.py` pinning Zenodo record 18272105
   (CC-BY-4.0, 2.2 MB, `Save_oct22_cbrs_drivetestL.csv`
   sha256 `ce74219cb9dc281efc95806968ebfd5500ee82c833d4201ec5b6daad72e9a03f`,
   `GPS_oct22_cbrs_drivetest.csv`
   sha256 `5e8e2793dae2ec97d9df766cc17e7e08e652a1be715d896398e3ed5024b57d02`)
   would replace the Gilbert–Elliott occupancy chain in `dsa_poison_suite.py` with
   six real CBRS channels. Attribution: University of Utah POWDER, CC-BY-4.0.

---

## 6. Verification

```
$ /home/user/venv/bin/python -m ruff check benchmarks/fl_poisoning_suite.py \
    benchmarks/poisoning_shield_benchmark.py benchmarks/dsa_poison_suite.py \
    benchmarks/neural_rx_pgd_benchmark.py
All checks passed!

$ /home/user/venv/bin/python -m pytest tests/test_neural_rx_pgd.py \
    tests/test_dsa_data_poison.py tests/test_fl_poison_battery.py -q
25 passed

$ /home/user/venv/bin/python benchmarks/fl_poisoning_suite.py        # 11 s
$ /home/user/venv/bin/python benchmarks/poisoning_shield_benchmark.py # 12 s
$ /home/user/venv/bin/python benchmarks/dsa_poison_suite.py           # 23 s
$ /home/user/venv/bin/python benchmarks/neural_rx_pgd_benchmark.py    # 39 s
```

All four abort with an actionable message if the licence-gated feature file is
absent, rather than silently falling back to synthetic data:

```
missing real feature file datasets/deepmimo_asu_3p5/generated/channel_features.jsonl
This benchmark runs on real DeepMIMO ray-traced channels. Build them with:
python datasets/deepmimo_asu_3p5/build.py  (licence-gated, not redistributed in-repo)
```
