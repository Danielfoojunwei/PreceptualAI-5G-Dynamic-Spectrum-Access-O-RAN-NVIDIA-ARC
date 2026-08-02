# Gate-bearing constant inventory & swap harness (gate G2)

Gate **G2** asserts: *every published gate reproduces without retuning any
constant.* That assertion is only checkable if two things exist:

1. an explicit, **source-derived** list of the numeric/threshold constants whose
   value decides whether a published gate passes; and
2. evidence that each such constant is one a gate is genuinely **sensitive** to
   — otherwise "not retuned" is vacuous for a constant no gate can see.

This directory supplies both.

| File | Role |
| --- | --- |
| `inventory.py` | Enumerates the constants **from source by AST** (never transcribed) and can regenerate `inventory.json`. |
| `inventory.json` | The committed inventory the drift guard compares against. |
| `swap_harness.py` | Perturbs a constant and shows the dependent gate/check flips. |
| `../tests/test_constant_inventory.py` | Drift guard + value-equivalence + swap-flip tests. |

## How the inventory is built (no transcription)

Each constant's *provenance* (symbol, file, module, unit, dependent gates) is
declared in `CONSTANT_SPECS`; its **value and line are extracted from the
source file's AST** at that location. `generate_inventory()` rebuilds the whole
set, and the test re-runs it and diffs byte-for-byte against `inventory.json`.
A constant that moves without the inventory being regenerated fails the drift
guard — the same pattern as OCUDU's `extract_catalogue.py`.

Regenerate after any deliberate constant change:

```
PYTHONPATH=src:agentic/src:ocudu/src:audit \
  python audit/constants/inventory.py --out audit/constants/inventory.json
# or fail-if-stale in CI:
PYTHONPATH=src:agentic/src:ocudu/src:audit \
  python audit/constants/inventory.py --check
```

## `dependency` column

- **`sha256`** — the gate pins the file's SHA-256 (G1 pins `shield.py`,
  `invariants.py`, `ucb_spectrum.py`; G6/G8 re-derive and pin their sources).
  Editing the literal changes the digest and fails the pin.
- **`behavioural`** — the gate re-runs a check whose satisfied/blocked outcome
  the constant moves; the harness flips it directly.

## Inventory summary (22 constants)

| Constant | Value | Unit | Source | Dependent gate(s) | Dependency |
| --- | --- | --- | --- | --- | --- |
| `ShieldConfig.max_passes` | 8 | projection_passes | shield.py:39 | verify_g1_second_planner.py | sha256+behavioural |
| `default_terrestrial_shield.max_eirp_dBm` | 33.0 | dBm | shield.py:213 | verify_g1_second_planner.py, verify_poisoning_shield.py | sha256 |
| `SpectralMaskInvariant.guard_band_hz` | 0.0 | Hz | invariants.py:162 | verify_g1_second_planner.py | sha256+behavioural |
| `MaxEirpInvariant.max_eirp_dBm` | 33.0 | dBm | invariants.py:234 | verify_g1_second_planner.py, verify_poisoning_shield.py | sha256+behavioural |
| `ProtectedSliceFloorInvariant.floor` | 0.20 | fraction | invariants.py:297 | verify_g1_second_planner.py, verify_poisoning_shield.py | sha256+behavioural |
| `PfdCeilingInvariant.max_pfd_dBW_m2_MHz` | -146.0 | dBW/m^2/MHz | invariants.py:411 | verify_g1_second_planner.py | sha256+behavioural |
| `NeuralRxEnvelopeInvariant.tolerance_dB` | 1.0 | dB | invariants.py:505 | verify_g1_second_planner.py | sha256+behavioural |
| `NeuralRxEnvelopeInvariant.min_confidence` | 0.2 | fraction | invariants.py:506 | verify_g1_second_planner.py | sha256+behavioural |
| `ConstellationLegalityInvariant.max_papr_dB` | 8.5 | dB | invariants.py:595 | verify_g1_second_planner.py | sha256+behavioural |
| `UcbSpectrumPlanner.__init__.frequency_arms` | 9 | count | ucb_spectrum.py:142 | verify_g1_second_planner.py | sha256+behavioural |
| `UcbSpectrumPlanner.__init__.pa_floor_dBm` | 10.0 | dBm | ucb_spectrum.py:143 | verify_g1_second_planner.py | sha256 |
| `UcbSpectrumPlanner.__init__.pa_ceiling_dBm` | 35.0 | dBm | ucb_spectrum.py:144 | verify_g1_second_planner.py | sha256+behavioural |
| `UcbSpectrumPlanner.__init__.power_arms` | 6 | count | ucb_spectrum.py:145 | verify_g1_second_planner.py | sha256+behavioural |
| `AbsoluteSliceCapacityFloor.min_capacity_hz` | 20e6 | Hz | aggregate.py:175 | verify_g6_multi_agent.py | sha256+behavioural |
| `AggregateEirpBudget.max_total_eirp_dBm` | 36.0 | dBm | aggregate.py:257 | verify_g6_multi_agent.py | sha256+behavioural |
| `SpectralSeparation.guard_hz` | 1e6 | Hz | aggregate.py:326 | verify_g6_multi_agent.py | sha256+behavioural |
| `PrbConservation.tolerance` | 1e-9 | fraction | aggregate.py:400 | verify_g6_multi_agent.py | sha256+behavioural |
| `AggregatePfdCeiling.max_pfd_dBW_m2_MHz` | -146.0 | dBW/m^2/MHz | aggregate.py:450 | verify_g6_multi_agent.py | sha256+behavioural |
| `MIN_PRB_POLICY_RATIO` | 11 | ran_parameter_id | rc_slice_quota.py:88 | verify_g8_ocudu_conformance.py | sha256+behavioural |
| `MAX_PRB_POLICY_RATIO` | 12 | ran_parameter_id | rc_slice_quota.py:89 | verify_g8_ocudu_conformance.py | sha256+behavioural |
| `DEDICATED_PRB_POLICY_RATIO` | 13 | ran_parameter_id | rc_slice_quota.py:90 | verify_g8_ocudu_conformance.py | sha256+behavioural |
| `SliceQuota.max_ratio` | 100 | percent | rc_slice_quota.py:131 | **(none — see finding)** | — |

## The swap harness — proving sensitivity

`swap_harness.py` demonstrates, for each covered constant, that perturbing it
**flips** the gate/check. It never writes to `src/`: the `sha256` approach
perturbs the source text in memory and hashes it, after first confirming the
live file matches the committed G1 pin (so the demonstration is against the real
gate). The four named constants:

| Swap | Committed → perturbed | Flip observed |
| --- | --- | --- |
| slice floor `0.20` | alloc `= 0.20` satisfied → floor `0.25` **unsatisfied**; digest change fails G1 pin | ✔ |
| EIRP ceiling `33.0` | EIRP `= 33.0` admitted → ceiling `30.0` **blocked**; digest change fails G1 pin | ✔ |
| aggregate floor `20e6` | `0.20 × 100 MHz = 20 MHz` satisfied → floor `25e6` **unsatisfied**; digest change fails G6 pin | ✔ |
| `max_passes` `8` | fixable action safe → `max_passes=0` **blocked**; digest change fails G1 pin | ✔ |
| (extra) `MIN_PRB_POLICY_RATIO` `11` | id present in OCUDU catalogue → `999` **absent**, fails G8 id-match | ✔ |

Run it standalone:

```
PYTHONPATH=src:agentic/src:ocudu/src:audit python audit/constants/swap_harness.py
```

## Findings (honest posture)

- **`ShieldConfig.max_passes = 8` is conservative headroom, not a tuned
  threshold.** The default terrestrial/NTN chains converge in **≤ 1 projection
  pass** across the tested infeasible scenarios (`measure_projection_passes()`),
  so the *disposition* is insensitive to `max_passes` for any value ≥ 1. Its
  **exact value** is gate-bearing only through G1's SHA-256 pin of `shield.py`.
  The harness therefore flips the disposition by dropping to `0` and separately
  breaks the G1 pin — both are recorded rather than pretending the magnitude 8
  is behaviourally load-bearing.

- **`SliceQuota.max_ratio = 100` is sensitive to no published gate.** It is
  Horizon-side input validation, not asserted by G8 (which pins RAN Parameter
  *IDs* against OCUDU's catalogue, not the ratio default). It could be retuned
  freely without failing any gate. Recorded with `dependent_gates: []` and a
  note, exactly because a constant no gate is sensitive to is the one G2's claim
  must not silently cover.

## Reproduce

```
PYTHONPATH=src:agentic/src:ocudu/src:audit/src \
  python -m pytest audit/tests/test_constant_inventory.py -q
```
