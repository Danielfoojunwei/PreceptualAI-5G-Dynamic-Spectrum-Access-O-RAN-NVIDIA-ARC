# PreceptualAI Production Benchmark Results

_Generated: 2026-05-06T15:50:19+00:00 · Canonical-to-v3-wave: 2026-05-08. See [`BENCHMARKS.md`](../BENCHMARKS.md) for the consolidated v3 numbers and [`README.md`](../README.md) Part III (sections 17–26) for empirical-evidence deep dive (24-h soak 99.93 % A1 success, EPFD 0.19 % violation rate, 18 µs/record audit verify, etc.)._

Methodology: 2 warm-up + 5+ measured iterations per bench; median + p95 reported. Status thresholds: PASS ≤ target, WATCH ≤ 1.5×target, FAIL > 1.5×target.

| name                              | median    | p95       | target            | status | notes                                              |
| --------------------------------- | --------- | --------- | ----------------- | ------ | -------------------------------------------------- |
| orbital_walker_22sat_propagate    | 1.5 µs    | 1.5 µs    | ≤ 20 µs/step      | PASS   | ns per sat-step (22 sats × 1000 steps)             |
| epfd_aggregate_22sat_snapshot     | 0.04 ms   | 0.04 ms   | ≤ 5 ms/call       | PASS   | per-call (500 calls/iter, 22 emitters)             |
| doppler_pass_envelope_600s        | 0.80 ms   | 0.80 ms   | ≤ 200 ms/envelope | PASS   | per-envelope (600s @ 5s step, 50 reps/iter)        |
| entity_tokenize_50assets          | 38.70 ms  | 45.49 ms  | ≤ 30 ms/call      | WATCH  | 50 mixed assets, 200 calls/iter                    |
| perceiver_fusion_forward_256x128  | 10.33 ms  | 10.95 ms  | ≤ 80 ms/forward   | PASS   | 200 input tokens, 50 fwds/iter                     |
| graph_jepa_loss_step              | 134.52 ms | 164.50 ms | ≤ 2000 ms/step    | PASS   | loss + backward + step + EMA, 20 iters             |
| latent_dynamics_rollout_H12_N256  | 64.38 ms  | 78.39 ms  | ≤ 250 ms/rollout  | PASS   | B=256, H=12 horizon                                |
| td_mpc_plan_full                  | 175.41 ms | 182.74 ms | ≤ 8000 ms         | PASS   | REDUCED: horizon=12, n_samples=64, n_iterations=3  |
| diffusion_tail_sample_32          | 35.75 ms  | 61.54 ms  | ≤ 1000 ms         | PASS   | REDUCED n_steps=20, n_samples=32                   |
| constraint_full_check_and_project | 0.06 ms   | 0.06 ms   | ≤ 50 ms/call      | PASS   | check + project, 22 emitters, 200 calls/iter       |
| e2e_stages_3_to_5_walltime        | 33.36 ms  | 33.36 ms  | ≤ 10000 ms        | PASS   | stage_2 + stage_3 + stage_4 (Stage 5 HTTP skipped) |
| epfd_time_cdf_500sat_1h           | 0.19 s    | 0.19 s    | ≤ 45 s            | PASS   | REDUCED: Walker 500/10/1, 1 h, 60 s step           |

**Summary:** 11/12 PASS · 1 WATCH · 0 FAIL
