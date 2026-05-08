# RUNBOOK — Day-1 demo on a single NVIDIA GB10 (DGX Spark / Project Digits)

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


**Audience: the demo engineer running the customer pilot kickoff next week.**
Zero cloud cost. Everything below runs on one GB10 (128 GB unified memory,
~1 PFLOPS fp4) sitting under your desk.

This is a smoke-checked sequence. Follow it top-to-bottom on the morning of
the demo and you will have:

1. Three model checkpoints + 3GPP TS 28.105 model cards.
2. A benchmark report.
3. A live-data E2E simulation log with real CelesTrak TLEs.
4. A real-constellation demo script ready to project.

If any step's expected runtime triples or any phase fails, stop, capture the
log, and triage — do **not** improvise data.

---

## 0. (One-time, if not already done) — Install the package

```bash
cd /home/<you>/horizon-ric
.venv/bin/pip install -e .
```

**Expected runtime:** 60-90 seconds.
**Expected output:** `Successfully installed horizon-ric-<version>` and the
egg-info directory is present at `src/horizon_ric.egg-info/`.

---

## 1. (One-time) — Download CelesTrak TLEs

```bash
.venv/bin/python scripts/download_tles.py
```

**Expected runtime:** 30-60 seconds (depends on CelesTrak HTTP latency).
**Expected output:** `data/tles/` populated with 19 `.txt` files, totalling
12,455 satellites. Console prints a per-file row count and a final
`[summary] 19 files, 12455 sats`.

---

## 2. (One-time) — Download ITU-R reference maps

```bash
.venv/bin/python scripts/download_itu_data.py
```

**Expected runtime:** 2-3 minutes.
**Expected output:** `data/itu_r/` contains the three extracted maps
(P.839-4 rain height, P.453-14 wet refractivity, P.1510-1 surface temp),
~928 MB total.

---

## 3. **Train on GB10** — single-shot, three phases

```bash
mkdir -p logs
.venv/bin/python scripts/train_on_gb10.py 2>&1 | tee logs/train_gb10.log
```

**Expected total wall-time on GB10:** **30-45 minutes** (estimated; not yet
measured on hardware).

Phase-by-phase budgets the script enforces:

| Phase | What it does                                           | Budget |
|-------|--------------------------------------------------------|--------|
| A     | SLA Risk Head v0.3 on full Aerial+DeepMIMO+UCC corpus  |  5 min |
| B     | Graph-JEPA encoder pretraining (action-free)           | 15 min |
| C     | Latent dynamics + risk head fine-tune on JEPA encoder  | 10 min |

If a phase blows past its budget the script stops the phase early and ships
what it has. The next phase still runs.

**Expected outputs (every line below should be present at the end):**

```
checkpoints/sla_head_v0.3_gb10.pt   (+ .md card)
checkpoints/jepa_encoder_v0.1.pt    (+ .md card)
checkpoints/world_model_v0.1.pt     (+ .md card)
```

**Verify the GB10 was actually used during the run** (in a second terminal,
while step 3 is still running):

```bash
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv -l 5
```

You should see `NVIDIA GB10` with non-zero `utilization.gpu` and growing
`memory.used`. If this shows 0% or the GPU name is anything else, the
script has fallen back to the reduced plan — check the very first log lines
for the loud `WARNING: GB10 NOT detected. Running REDUCED plan.` banner.

**Honest-data sanity check** — grep the log for:

```bash
grep -E '(real_data_used|synthetic_fallback)' logs/train_gb10.log
```

You must see `real_data_used=YES synthetic_fallback=NO`. If you see
anything else, halt and triage; do not show the customer numbers from a
synthetic fallback.

---

## 4. Run the benchmark suite

```bash
.venv/bin/python benchmarks/run_benchmarks.py
```

**Expected runtime:** 1-2 minutes on GB10.
**Expected output:** updated `benchmarks/RESULTS.md` with the latency,
calibration and constraint-coverage numbers for each loaded checkpoint.

---

## 5. End-to-end simulation against real CelesTrak constellation

```bash
.venv/bin/python scripts/e2e_simulation.py --real-tles 2>&1 | tee logs/e2e_real.log
```

**Expected runtime:** 2-5 minutes.
**Expected output:** a per-tick log of FAPI ingest → encoder → world-model
latent → risk head → planner emit → evidence write, terminating with the
hash-chained evidence file path and a final `CHAIN OK` line. The
real-TLE flag pulls satellites from `data/tles/` rather than the maritime
seed; you should see one of the 19 source files printed at startup.

---

## 6. Real-constellation visual demo (the customer-facing one)

```bash
.venv/bin/python scripts/demo_real_constellation.py
```

**Expected runtime:** 2-3 minutes (interactive; do not pipe to tee unless
you want to lose the live progress bar).
**Expected output:** the three magic-moment screens described in
`PILOT.md` — FAPI through encoder, AODT prior + DeepMIMO into world model,
counterfactual envelope on a real-data prediction, and the
`CHAIN BROKEN AT RECORD <N>` tamper test.

---

## Failure-mode quick reference

| Symptom | Where to look | First fix |
|---|---|---|
| Step 3 logs `GB10 NOT detected` | `nvidia-smi` shows no cuda device or wrong GPU | Reseat the unit; rerun |
| Step 3 logs `synthetic_fallback=YES` | Data paths broken | Re-run steps 1-2; verify `data/aerial`, `data/deepmimo`, `data/ucc_misl` exist |
| Step 3 phase B has loss > phase B init loss after 5 epochs | JEPA collapse | Re-run step 3 with seed=43 (edit the seed line in the script) |
| Step 5 prints `CHAIN BROKEN` *unexpectedly* (no tamper) | Evidence-store bug or stale checkpoint | Delete `evidence/store.jsonl` and rerun |
| Step 6 fails on missing `pyqt`/display | Demo machine has no X | Ssh tunnel + `xvfb-run` or use the headless flag if added |

---

## Data inventory the demo expects to see on disk

| Source | Path | Size |
|---|---|---|
| NVIDIA Aerial cuBB SDK FAPI/FH parquet | `data/aerial/` | 1.9 GB |
| DeepMIMO ASU campus 3.5 GHz | `data/deepmimo/` | 6.2 GB |
| UCC MISL 5G production traces | `data/ucc_misl/` | 28 MB |
| CelesTrak TLEs (19 files, 12,455 sats) | `data/tles/` | small |
| ITU-R maps (P.839-4, P.453-14, P.1510-1) | `data/itu_r/` | ~928 MB |
| **Trainable corpus total** | — | **~8 GB** |

8 GB fits in GB10 unified memory many times over; the corpus is loaded once
and reused across all three phases.

---

## When this runbook is "green"

You are ready to put the customer in front of the screen when:

1. `logs/train_gb10.log` ends with `DONE in <N>s` and `N <= 2700`.
2. All three checkpoints are present in `checkpoints/` with non-zero size.
3. `benchmarks/RESULTS.md` shows a row for each new checkpoint.
4. `logs/e2e_real.log` ends with `CHAIN OK`.
5. `nvidia-smi` during step 3 confirmed non-zero GB10 utilisation.

If any of those five fail, the demo is **red**. Fix the failing step before
the customer joins; do not ship a partial run.
