# Beam-level mobility / handover with Doppler ON — real-geometry proof

This is the Horizon artifact that drives the **beam-mobility decision path**
with real measured-physics ray geometry — no synthetic RF anywhere in the loop.
A UE drives a real ~1 km corridor across the ASU campus; its best serving beam
(from an 8-beam DFT codebook over the real angle-of-departure geometry) changes
as it moves; Doppler computed from the real per-path angle-of-arrival geometry
plus the UE velocity makes the channel time-varying; and the rApp's A3-event
handover decision (hysteresis + time-to-trigger) beats a greedy tracker on
ping-pong — then the target beam is gated through the Decision Safety Shield
and written to the hash-chained evidence store.

- **Runner:** [`../../benchmarks/mobility_handover_loop.py`](../../benchmarks/mobility_handover_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/mobility_handover.json`](../../benchmarks/results/mobility_handover.json)
- **Sample evidence record:** [`../../benchmarks/results/mobility_handover_sample_record.json`](../../benchmarks/results/mobility_handover_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_mobility_handover_evidence.py`](../../tests/test_mobility_handover_evidence.py)
- **Reproduction verifier:** [`../../scripts/verify_mobility_handover.py`](../../scripts/verify_mobility_handover.py)
- **Data:** DeepMIMO ASU Campus 3.5 GHz **angular** feature build — per-path
  power, phase, delay, AoD az/el and AoA az/el for **4096** ray-traced receivers
  (Wireless InSite), checksum-pinned by
  [`../../datasets/deepmimo_asu_3p5/angular_manifest.json`](../../datasets/deepmimo_asu_3p5/angular_manifest.json)
  and built by
  [`../../datasets/deepmimo_asu_3p5/build_angular.py`](../../datasets/deepmimo_asu_3p5/build_angular.py).

## The task and why it fits the data

A real O-RAN **Mobility-Robustness-Optimization (MRO)** rApp function at beam
level: track the best serving beam for a moving UE and hand over only on a
confirmed **A3 event** — candidate beam better than serving by HYST = 3 dB for
TTT = 3 consecutive samples — instead of greedily chasing the instantaneous
argmax beam.

Everything spatial is real:

- **Trajectory:** the 198 receivers with `y ∈ [75, 85]` m, ordered by `x` — a
  monotone **west-to-east 1049 m drive** (each receiver in the band has a
  unique `x`), passing ~24 m south of the BS at `[166, 104, 22]`. Heading is
  the finite-difference direction of travel; at 12 m/s the drive takes 87.4 s
  (mean Δt = 0.44 s between real samples).
- **Channel:** per receiver, `h = Σ_l α_l · a(aod_az_l)` on the 8-element
  half-wavelength ULA, with `α_l` from the real per-path power and phase.
- **Codebook:** 8 DFT beams with boresights ±7.2°, ±22.0°, ±38.7°, ±61.0°.
- **Doppler (ON):** per path, `f_d,l = (|v|/λ) · cos(heading − aoa_az_l)` from
  the path's real AoA azimuth; each path's phase advances by the accumulated
  `2π·f_d,l·Δt` along the drive. At 12 m/s: **max |f_d| = 140.0 Hz**,
  **coherence time ≈ 3.0 ms** — every 0.44 s sample is many coherence times
  apart, so the beam gains fade realistically hard between samples. (The
  per-sample phase advance is `2π·(step_m/λ)·cos(·)` — distance-driven, speed
  cancels — so the beam decisions depend only on the real geometry; the speed
  choice sets the reported Doppler shift and coherence time.)

## The mobility result (real geometry, Doppler-driven fading)

Greedy = serve the argmax beam every sample (0 dB hysteresis, TTT 1). A3 =
hysteresis 3 dB + TTT 3. Ping-pong = A→B→A within 5 samples. Outage = serving
beam > 3 dB below the best available beam.

| policy | handovers | ping-pongs | outage fraction |
| --- | ---: | ---: | ---: |
| greedy argmax | 29 | 19 | 0.000 |
| **A3 (3 dB, TTT 3)** | **3** | **0** | **0.101** |

Read honestly:

- **Greedy flaps.** In the Doppler-driven fading the instantaneous best beam
  flips between adjacent DFT beams near every beam boundary: 29 handovers, of
  which 19 are ping-pongs. Its outage is 0.000 *by construction* (it always
  serves the argmax) — the cost is signalling churn, not signal.
- **The A3 event is the right decision rule.** 3 dB of hysteresis plus a
  3-sample time-to-trigger cuts the drive to **3 confirmed handovers and zero
  ping-pongs** (greedy: 19), while staying within 3 dB of the best beam for
  **89.9%** of samples (outage 0.101 ≤ 0.25).
- **The mobility is real.** The A3 serving-beam sequence is 3 → 2 → 1 → 0,
  switching at x ≈ 101 m, 137 m and 154 m — exactly the departure-azimuth
  sweep as the UE closes on the BS at x = 166 m. Both policies visit the same
  4 beams; this is the geometry driving the decision, not a scripted schedule.

## The trust chain (real, on the selected target beam)

The confirmed A3 handover into the final serving beam (beam 0) is emitted as a
`horizon.mobility.handover` policy through the Decision Safety Shield and
written to a hash-chained evidence store:

- The legal handover emit (in band, EIRP 32 dBm < 33) needs **no correction,
  passes the guard chain, and emits cleanly**, recorded.
- A handover emit that naively requests **EIRP 46 dBm** is **projected** onto
  the legal cap — **33 dBm** — and the raw over-power emit trips the
  `corrections_not_audited` guard, so **it never reaches the RAN**
  (`overpower_handover_reached_ran = false`).
- The evidence chain **verifies intact** (`verify() = -1`); after a single-byte
  tamper of record #1 on disk, re-verification **pinpoints the tamper at index
  1** and raises an X.733 `processingErrorAlarm`.

## Reproduce (real data, deterministic)

```bash
# Build the licence-gated angular feature file (checksum-pinned, deterministic).
python datasets/deepmimo_asu_3p5/build_angular.py \
  --cache-dir /tmp/dm \
  --features /tmp/dm/angular_features.jsonl \
  --manifest /tmp/dm/angular_manifest.json --samples 4096

# Run the whole mobility + trust loop on the real ray geometry.
python benchmarks/mobility_handover_loop.py \
  --features /tmp/dm/angular_features.jsonl \
  --manifest /tmp/dm/angular_manifest.json \
  --out /tmp/dm/mobility_handover.json

# Check the fresh run against the committed result on host-stable invariants.
python scripts/verify_mobility_handover.py \
  --committed-result benchmarks/results/mobility_handover.json \
  --actual-result /tmp/dm/mobility_handover.json \
  --actual-manifest /tmp/dm/angular_manifest.json
```

The runner exits non-zero unless Doppler is genuinely on (max shift > 0), the
A3 policy cuts ping-pongs below greedy, keeps outage ≤ 0.25, makes ≥ 1 real
handover, the over-EIRP emit is corrected and never reaches the RAN, and the
evidence chain verifies intact then catches the tamper. The committed-result
tests assert the same invariants in the fast pack. The verifier deliberately
compares host-stable decisions (final serving beam, event counts within a small
fade tolerance) and **not** chaotic per-beam gain floats: the Doppler phase
accumulates ~60 wavelengths per trajectory step, so last-ULP rebuild
differences can flip a few boundary samples.

## Scope and honesty

- **Intra-cell BEAM handover, not inter-gNB handover.** The scenario has a
  single base station at `[166, 104, 22]` with an 8-element ULA; the
  "handover" is between beams of its 8-beam DFT codebook. No neighbour cell
  exists in this dataset.
- **Doppler is synthesized from real geometry.** The ray tracer is static; the
  time variation comes from the real per-path AoA azimuths combined with a
  *chosen* UE velocity (12 m/s along the real corridor). The trajectory,
  per-path powers/phases and AoD/AoA angles are unmodified Wireless InSite ray
  tracing.
- **Ray tracing, not over-the-air capture; one scenario.** The licence-gated
  raw feature rows are not redistributed; the committed result JSON is bound to
  the canonical build by `features_sha256`/`source_tree_sha256`, and
  `scripts/verify_mobility_handover.py` re-checks a fresh rebuild against it.
