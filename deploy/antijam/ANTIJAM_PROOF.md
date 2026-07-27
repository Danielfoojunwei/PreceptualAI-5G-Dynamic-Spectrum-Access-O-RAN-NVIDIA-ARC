# Anti-jam adaptive null-steering on real DeepMIMO — proof

This is the Horizon artifact that drives the **interference-mitigation + trust
core** with real angular ray geometry: the desired downlink channel per receiver
is its own real ray-traced path sum — no synthetic desired channel anywhere in
the loop. The jammer, honestly labelled, is a **modeled** interferer (this
single-BS scenario has no second real transmitter), and the mitigation is the
**practical adaptive nuller** (sample-matrix inversion from finite snapshots),
not an idealized perfect-covariance bound.

- **Runner:** [`../../benchmarks/antijam_loop.py`](../../benchmarks/antijam_loop.py)
- **Result (real 4096-Rx run):** [`../../benchmarks/results/antijam.json`](../../benchmarks/results/antijam.json)
- **Sample evidence record:** [`../../benchmarks/results/antijam_sample_record.json`](../../benchmarks/results/antijam_sample_record.json)
- **Tests (validate the real result):** [`../../tests/test_antijam_evidence.py`](../../tests/test_antijam_evidence.py)
- **Reproduction verifier:** [`../../scripts/verify_antijam.py`](../../scripts/verify_antijam.py)
- **Data:** DeepMIMO ASU Campus 3.5 GHz angular build, **4096** ray-traced
  receivers, up to 10 paths each with real per-path power / phase / AoD / AoA /
  delay (`datasets/deepmimo_asu_3p5/build_angular.py`, checksum-pinned).

## The task and why it fits the data

A real O-RAN **anti-jam** rApp function over the manifest's shared 8-element
half-wavelength ULA at 3.5 GHz:

1. **Real desired channel.** For each receiver, `h_d = Σ_l α_l · a(aod_az_l)`
   with `α_l = sqrt(10^(power_dbw/10)) · e^{j·phase}` — every path's power,
   phase and angle of departure is real Wireless InSite ray-tracing output.
2. **Jammer-blind serving beam.** The best of 8 DFT codebook beams
   (`u_b = (2b+1)/8 − 1`) for the real channel, picked *without* knowledge of
   the jammer — exactly what a beam-managed cell would be transmitting on.
3. **Modeled, angularly-spread jammer.** An interferer centred at a fixed,
   documented azimuth **−142°** (the rounded median strongest-path AoD of the
   build, i.e. inside the region where most real serving energy departs — the
   hardest honest placement), with jammer-to-signal ratio pinned **30 dB** above
   the mean serving-beam signal over a thermal 20 MHz noise floor (−131 dBW).
   The jammer is **not** a perfect point source: it is 7 equal-power sub-rays
   across a **4° angular spread** (local scattering / finite bandwidth), so the
   interference is not rank-1 and an 8-element array cannot null it infinitely.
4. **Practical SMI null-steering (not the idealized bound).** The array does
   **not** know the interference covariance. It estimates it from **24 finite
   jammer+noise training snapshots** (sample matrix inversion) and adds **10 dB
   diagonal loading**, giving `R̂`; the anti-jam weight is `w = R̂⁻¹ h_d`. Its
   SINR is scored against the **true** interference the beamformer actually
   faces — so finite-snapshot estimation error, loading, and the jammer spread
   are all paid for, exactly as a real adaptive array pays for them.

Because `h_d` is a real multipath ray sum — not a single plane wave — the
restored SINR is receiver-specific physics: receivers whose real ray geometry is
angularly close to the jammer recover less than receivers with separable
geometry.

## The result (real 4096-Rx run)

| metric | jammed (serving DFT beam) | anti-jam (SMI null-steer) |
| --- | ---: | ---: |
| mean SINR | **−44.08 dB** | **+15.76 dB** |
| median SINR | −40.69 dB | +14.82 dB |
| receivers usable (SINR > 0 dB) | **0.98 %** | **67.99 %** |

- **Operating point:** the SMI nuller restores a usable **+15.8 dB mean SINR**
  and takes the fleet from **1 % → 68 %** of receivers above 0 dB. This is the
  operationally meaningful, clearly-realistic number.
- **Mean SINR gain 59.83 dB** (median 60.16). The gain is large because the
  *baseline* is catastrophic: a jammer-blind serving beam under a 30 dB-JSR
  interferer in its own main lobe collapses to −44 dB. The swing reflects the
  strength of the jammer, not an inflated beamformer — the restored SINR itself
  is a sober +15.8 dB.
- **Mean effective jammer suppression 67.90 dB** (min 51.90) — the reduction in
  *total spread-jammer power* through the SMI weight versus the serving beam.
  Read honestly: this is an adaptive-array figure with **no hardware calibration
  error modeled**, so it is optimistic-but-achievable; real deployments with
  calibration and jammer motion typically see tens of dB. The invariant asserted
  is the conservative `> 10 dB`.
- **Not everything is restored.** ~32 % of receivers stay below 0 dB SINR even
  after nulling — their real channels are too weak or too aligned with the
  jammer. That residual is the honest cost of real geometry plus a spread jammer.

## The trust chain (real, on the computed mitigation)

The null-steer reconfiguration is emitted as a policy through the Decision Safety
Shield and written to a hash-chained evidence store:

- The legal null-steer (in band, EIRP 32 dBm < 33) needs **no correction, passes
  the guard chain, and emits cleanly**, recorded.
- A null-steer that naively requests **EIRP 46 dBm** is **projected** onto the
  legal cap — **33 dBm** — and the raw over-power emit trips the
  `corrections_not_audited` guard, so **it never reaches the RAN**
  (`overpower_nullsteer_reached_ran = false`).
- The evidence chain **verifies intact** (`verify() = −1`); after a single-byte
  tamper of record #1 on disk, re-verification **pinpoints the tamper at
  index 1**.

## Reproduce (real data, deterministic)

```bash
# Build the licence-gated angular feature file (checksum-pinned, deterministic).
python datasets/deepmimo_asu_3p5/build_angular.py \
  --cache-dir /tmp/dm --features /tmp/dm/angular_features.jsonl \
  --manifest /tmp/dm/angular_manifest.json --samples 4096

# Run the whole anti-jam + trust loop on the real channels.
python benchmarks/antijam_loop.py \
  --features /tmp/dm/angular_features.jsonl \
  --manifest /tmp/dm/angular_manifest.json \
  --out /tmp/dm/antijam.json

# Verify against the committed result (host-stable invariants, not raw floats).
python scripts/verify_antijam.py \
  --committed-result benchmarks/results/antijam.json \
  --actual-result /tmp/dm/antijam.json \
  --actual-manifest /tmp/dm/angular_manifest.json
```

The runner exits non-zero unless the mean SINR gain exceeds 5 dB, more receivers
are usable after mitigation than under the jammer, the mean effective jammer
suppression exceeds 10 dB, the over-EIRP null-steer is corrected and never
reaches the RAN, and the evidence chain verifies intact then catches the tamper.
The SMI training snapshots use a fixed seed, so the loop is reproducible;
cross-host reproduction binds on `source_tree_sha256` plus tolerance-banded
aggregates because the feature bytes themselves are host-dependent at the last
rounded decimal.

## Scope and honesty

The **desired channel and array response are real**: real ray-traced angle of
departure, per-path power and phase over the real 8-element half-wavelength ULA
the manifest publishes. The **jammer is a modeled interferer** — a single-BS
scenario has no second real transmitter, and this proof does not claim the
jammer itself is measured. The mitigation is a **practical SMI beamformer**
(finite training snapshots + diagonal loading against a spread jammer), not the
perfect-covariance MVDR bound; **no antenna calibration error is modeled**, so
the effective-suppression figure is optimistic. This is **site-specific ray
tracing, not over-the-air capture**, and a **single scenario**. The licence-gated
raw feature rows are not redistributed; the committed result JSON is bound to the
canonical build by `source_tree_sha256`.
