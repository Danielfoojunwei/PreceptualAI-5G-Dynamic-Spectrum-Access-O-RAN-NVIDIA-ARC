# Analog beam management on real DeepMIMO — beam-sweep proof

This is the Horizon artifact that drives **analog beam management** — the 5G NR
SSB-style beam sweep, serving-beam selection, and the gating of the resulting
beam policy — with real measured-physics data. No synthetic RF anywhere in the
loop: every per-path power, phase and angle-of-departure comes from the
Wireless InSite ray tracer.

- **Runner:** [`../../benchmarks/beam_management_loop.py`](../../benchmarks/beam_management_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/beam_management.json`](../../benchmarks/results/beam_management.json)
- **Sample evidence record:** [`../../benchmarks/results/beam_management_sample_record.json`](../../benchmarks/results/beam_management_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_beam_management_evidence.py`](../../tests/test_beam_management_evidence.py)
- **Verifier (fresh vs committed):** [`../../scripts/verify_beam_management.py`](../../scripts/verify_beam_management.py)
- **Data:** DeepMIMO ASU Campus 3.5 GHz, **4096** ray-traced receivers, the
  checksum-pinned *angular* feature build
  (`datasets/deepmimo_asu_3p5/generated/angular_features.jsonl`,
  `features_sha256 = 13eb2f6b…`, manifest
  [`angular_manifest.json`](../../datasets/deepmimo_asu_3p5/angular_manifest.json)).

## The task and why it fits the data

The real 5G NR beam-establishment procedure in miniature: the gNB's 8-element
half-wavelength ULA sweeps an **8-beam DFT codebook** (steering targets
`u_b = (2b+1)/8 − 1` tile sin-space `[-1, 1)` — an SSB P-1 sweep), each
receiver reports the strongest beam `argmax_b |w_b^H h|²`, and the selected
beam-activation policy is gated before it may touch the RAN. Each receiver's
array channel is built from its own real ray-traced paths:

    h = Σ_l sqrt(10^(P_l/10)) · e^{j·phase_l} · a(aod_az_l),
    a(φ)[n] = e^{j·2π·0.5·n·sin φ},  n = 0..7.

This is the *right* task for this scenario: receivers surround the site, so the
dominant angles of departure span essentially the full azimuth circle
(−179.9° … +179.7°). Beam choice is therefore a real decision — all 8 codebook
beams win somewhere and **77.2 %** of receivers are best served by a
non-boresight beam.

## The result (real geometry, 4096 receivers)

| figure | value | meaning |
| --- | ---: | --- |
| mean array gain | **7.05 dB** | best beam vs the single-element average |
| median array gain | 7.22 dB | |
| max array gain | 9.031 dB | at the physical `10·log10(8) = 9.031 dB` ceiling |
| mean misalignment recovery | **11.42 dB** | best beam vs a fixed broadside beam |
| median misalignment recovery | 12.77 dB | |
| distinct beams selected | **8 / 8** | histogram `[970, 1020, 750, 934, 104, 78, 75, 165]` |
| non-boresight fraction | 77.2 % | boresight = beam 3; modal beam = **1** |

Read honestly:

- **The array gain is real coherent combining, not an artefact.** Per-receiver
  gain is hard-bounded at `10·log10(8) ≈ 9.03 dB`; no receiver exceeds it, the
  best receivers sit exactly at it (near-single-path channels), and the 7.05 dB
  mean is what genuinely angle-spread multipath costs against the ceiling.
- **Sweeping is what pays.** A deployment stuck on the fixed broadside beam
  loses 11.4 dB on average against the swept-and-selected beam — the direct,
  physical value of the SSB sweep on this wide-AoD campus.
- **The codebook is fully exercised.** All 8 beams are the best choice for some
  receivers; the selection histogram is data, not a constant.

## The trust chain (real, on the selected beam)

The modal serving beam (beam 1, best for 1020 of 4096 receivers) is emitted as
a `horizon.beam.selection` policy through the Decision Safety Shield and
written to a hash-chained evidence store:

- The legal beam activation (in band, EIRP 32 dBm < 33) needs **no correction,
  passes the guard chain, and emits cleanly**, recorded.
- A beam activation that naively requests **EIRP 46 dBm** is **projected** onto
  the legal cap — **33 dBm** — and the raw over-power emit trips the
  `corrections_not_audited` guard, so **it never reaches the RAN**
  (`overpower_beam_reached_ran = false`).
- The evidence chain **verifies intact** (`verify() = -1`); after a single-byte
  tamper of record #1 on disk, re-verification **pinpoints the tamper at index
  1**. The intact record 0 is committed as the sample evidence record.

## Reproduce (real data, deterministic)

```bash
# Build the licence-gated DeepMIMO angular feature file (checksum-pinned).
python datasets/deepmimo_asu_3p5/build_angular.py

# Run the whole beam sweep + trust loop on the real ray geometry.
python benchmarks/beam_management_loop.py \
  --features datasets/deepmimo_asu_3p5/generated/angular_features.jsonl \
  --manifest datasets/deepmimo_asu_3p5/angular_manifest.json \
  --out benchmarks/results/beam_management.json

# Verify a fresh run against the committed result (host-stable checks only).
python scripts/verify_beam_management.py \
  --committed-result benchmarks/results/beam_management.json \
  --actual-result benchmarks/results/beam_management.json \
  --actual-manifest datasets/deepmimo_asu_3p5/angular_manifest.json
```

The runner exits non-zero unless the mean array gain lands in the real
coherent-combining range (> 6 dB, ≤ 9.03 dB), steering beats the fixed beam by
> 1 dB, at least 4 codebook beams are selected, the over-power activation is
corrected and never reaches the RAN, and the evidence chain verifies intact
then catches the tamper. The committed-result tests assert the same invariants
in the fast pack; the verifier compares a fresh run to the committed result on
host-stable decisions (modal beam, histogram within a boundary-flip tolerance,
means within a dB band), never on chaotic float bytes.

## Scope and honesty

Real DeepMIMO ray-traced angle-of-departure geometry drives a real DFT beam
sweep, real serving-beam selection, real Shield legality and a real
tamper-evident evidence chain. This is **site-specific ray tracing, not
over-the-air capture**; a **single scenario, single base station,
azimuth-plane ULA** (ray-traced elevations sit within ~2° of horizontal, so
azimuth carries the geometry). The 8-beam DFT sweep models analog SSB-style
beam management, not digital precoding or per-UE MU-MIMO. The licence-gated
raw feature rows are not redistributed; the committed result JSON is bound to
the canonical build by `features_sha256`/`source_tree_sha256`, and the
`realdata` workflow rebuilds the data and re-runs the loop to reproduce it.
