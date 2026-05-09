"""Production benchmark suite for PreceptualAI.

12 micro/macro benchmarks covering orbital propagation, EPFD aggregation,
Doppler envelope, encoder pipeline, world-model rollout, planners,
constraint projection, and an E2E walltime check.

Methodology:
    - 2 warm-up iterations (discarded) followed by >=5 measured iterations.
    - Per benchmark: report median + p95 (sorted_values[int(0.95*(n-1))]).
    - Status: PASS (median <= target), WATCH (target < median <= 1.5x target),
      FAIL (median > 1.5x target).
    - Exits 0 if at least 8/12 PASS, else exits 1.
"""

from __future__ import annotations

import asyncio
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable

# Make src/ importable when run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from horizon_ric.encoder import (  # noqa: E402
    AssetAttributes,
    EntityTokenizer,
    EntityTokenizerConfig,
    EntityType,
    PerceiverConfig,
    PerceiverFusion,
)
from horizon_ric.encoder.graph_jepa import GraphJEPA, GraphJEPAConfig  # noqa: E402
from horizon_ric.core import LatentDynamics, LatentDynamicsConfig  # noqa: E402
from horizon_ric.policy import (  # noqa: E402
    GSOArcEntry,
    PreceptualAIConstraintConfig,
    PreceptualAIConstraintLayer,
    TDMPCConfig,
    TDMPCPlanner,
)
from horizon_ric.policy.diffusion_tail import (  # noqa: E402
    DiffusionConfig,
    DiffusionTailSampler,
)
from horizon_ric.planner.physics import (  # noqa: E402
    NGSOEmitter,
    epfd_down,
)
from horizon_ric.planner.physics.epfd import epfd_time_cdf  # noqa: E402
from horizon_ric.planner.physics.doppler import pass_envelope  # noqa: E402
from horizon_ric.planner.physics.orbital import (  # noqa: E402
    kepler_j2_step,
    keplerian_state,
    state_ecef_m,
    walker_delta_constellation,
)


SIM_T0 = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)
ES_LAT, ES_LON = 51.92, 4.48  # Rotterdam port


# ─── helpers ──────────────────────────────────────────────────────────────


def _measure(fn: Callable[[], Any], warmup: int = 2, iters: int = 5) -> list[int]:
    """Run fn warmup+iters times; return measured durations in ns."""
    for _ in range(warmup):
        fn()
    samples: list[int] = []
    for _ in range(iters):
        t0 = perf_counter_ns()
        fn()
        samples.append(perf_counter_ns() - t0)
    return samples


def _p95_ns(samples: list[int]) -> int:
    s = sorted(samples)
    return s[int(0.95 * (len(s) - 1))]


def _status(median: float, target: float) -> str:
    if median <= target:
        return "PASS"
    if median <= 1.5 * target:
        return "WATCH"
    return "FAIL"


def _fmt_us(ns: float) -> str:
    return f"{ns / 1e3:.1f} µs"


def _fmt_ms(ns: float) -> str:
    return f"{ns / 1e6:.2f} ms"


def _fmt_s(ns: float) -> str:
    return f"{ns / 1e9:.2f} s"


# ─── benchmarks ───────────────────────────────────────────────────────────


def bench_orbital_walker_22sat_propagate() -> dict:
    """Walker(53,22,2,1,550) → kepler_j2_step 1000× per sat → ns/sat-step."""
    elements = walker_delta_constellation(
        inclination_deg=53.0, total_sats=22, num_planes=2, phasing_F=1,
        altitude_km=550.0, epoch_utc=SIM_T0,
    )

    def run() -> None:
        for e in elements:
            cur = e
            for _ in range(1000):
                cur = kepler_j2_step(cur, 1.0)

    samples = _measure(run, warmup=2, iters=5)
    n_steps_per_iter = 22 * 1000
    per_step_ns = [s / n_steps_per_iter for s in samples]
    median_ns = statistics.median(per_step_ns)
    p95_ns = sorted(per_step_ns)[int(0.95 * (len(per_step_ns) - 1))]
    target_ns = 20_000.0  # 20 µs/step
    return {
        "name": "orbital_walker_22sat_propagate",
        "median": _fmt_us(median_ns),
        "p95": _fmt_us(p95_ns),
        "target": "≤ 20 µs/step",
        "status": _status(median_ns, target_ns),
        "notes": "ns per sat-step (22 sats × 1000 steps)",
    }


def bench_epfd_aggregate_22sat_snapshot() -> dict:
    elements = walker_delta_constellation(
        inclination_deg=53.0, total_sats=22, num_planes=2, phasing_F=1,
        altitude_km=550.0, epoch_utc=SIM_T0,
    )
    sat_ecef = [state_ecef_m(keplerian_state(e)) for e in elements]
    emitters = [
        NGSOEmitter(position_ecef=p, eirp_dBW=-15.0, name=f"sat-{i}")
        for i, p in enumerate(sat_ecef)
    ]

    def run() -> None:
        for _ in range(500):
            epfd_down(
                es_lat_deg=ES_LAT, es_lon_deg=ES_LON,
                wanted_gso_lon_deg=10.0,
                es_diameter_m=1.2, es_frequency_hz=12e9,
                ngso_satellites=emitters,
            )

    samples = _measure(run, warmup=2, iters=5)
    per_call_ns = [s / 500 for s in samples]
    median_ns = statistics.median(per_call_ns)
    p95_ns = sorted(per_call_ns)[int(0.95 * (len(per_call_ns) - 1))]
    target_ns = 5_000_000.0  # 5 ms
    return {
        "name": "epfd_aggregate_22sat_snapshot",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 5 ms/call",
        "status": _status(median_ns, target_ns),
        "notes": "per-call (500 calls/iter, 22 emitters)",
    }


def bench_doppler_pass_envelope_600s() -> dict:
    elements = walker_delta_constellation(
        inclination_deg=53.0, total_sats=22, num_planes=2, phasing_F=1,
        altitude_km=550.0, epoch_utc=SIM_T0,
    )
    s = keplerian_state(elements[0])

    def run() -> None:
        for _ in range(50):
            pass_envelope(s, ES_LAT, ES_LON, 28e9, 600.0, 5.0)

    samples = _measure(run, warmup=2, iters=5)
    per_call_ns = [t / 50 for t in samples]
    median_ns = statistics.median(per_call_ns)
    p95_ns = sorted(per_call_ns)[int(0.95 * (len(per_call_ns) - 1))]
    target_ns = 200_000_000.0  # 200 ms
    return {
        "name": "doppler_pass_envelope_600s",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 200 ms/envelope",
        "status": _status(median_ns, target_ns),
        "notes": "per-envelope (600s @ 5s step, 50 reps/iter)",
    }


def bench_entity_tokenize_50assets() -> dict:
    cfg = EntityTokenizerConfig()
    tok = EntityTokenizer(cfg)
    tok.eval()

    # 50 mixed assets across all 8 types.
    types_cycle = list(EntityType)
    assets: list[AssetAttributes] = []
    for i in range(50):
        t = types_cycle[i % len(types_cycle)]
        dim = cfg.attr_dims[t]
        assets.append(AssetAttributes(
            entity_type=t,
            attributes=torch.randn(dim),
            xyz_m=(1e6 + i * 1e3, 2e6 + i * 1e3, 3e6 + i * 1e3),
        ))

    def run() -> None:
        with torch.no_grad():
            for _ in range(200):
                tok(assets)

    samples = _measure(run, warmup=2, iters=5)
    per_call_ns = [s / 200 for s in samples]
    median_ns = statistics.median(per_call_ns)
    p95_ns = sorted(per_call_ns)[int(0.95 * (len(per_call_ns) - 1))]
    target_ns = 30_000_000.0  # 30 ms
    return {
        "name": "entity_tokenize_50assets",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 30 ms/call",
        "status": _status(median_ns, target_ns),
        "notes": "50 mixed assets, 200 calls/iter",
    }


def bench_perceiver_fusion_forward_256x128() -> dict:
    cfg = PerceiverConfig()  # default: n_latents=256, d_latent=128, d_input=128
    model = PerceiverFusion(cfg)
    model.eval()
    inputs = torch.randn(1, 200, cfg.d_input)

    def run() -> None:
        with torch.no_grad():
            for _ in range(50):
                model(inputs)

    samples = _measure(run, warmup=2, iters=5)
    per_call_ns = [s / 50 for s in samples]
    median_ns = statistics.median(per_call_ns)
    p95_ns = sorted(per_call_ns)[int(0.95 * (len(per_call_ns) - 1))]
    target_ns = 80_000_000.0  # 80 ms
    return {
        "name": "perceiver_fusion_forward_256x128",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 80 ms/forward",
        "status": _status(median_ns, target_ns),
        "notes": "200 input tokens, 50 fwds/iter",
    }


def bench_graph_jepa_loss_step() -> dict:
    perc_cfg = PerceiverConfig()
    context_encoder = PerceiverFusion(perc_cfg)
    jepa = GraphJEPA(context_encoder, GraphJEPAConfig(d_latent=perc_cfg.d_latent))
    optimizer = torch.optim.AdamW(
        list(jepa.context_encoder.parameters()) + list(jepa.predictor.parameters()),
        lr=1e-4,
    )
    inputs = torch.randn(2, 64, perc_cfg.d_input)

    def run() -> None:
        optimizer.zero_grad()
        out = jepa.loss(inputs)
        out["loss"].backward()
        optimizer.step()
        jepa.update_target()

    samples = _measure(run, warmup=2, iters=20)
    median_ns = statistics.median(samples)
    p95_ns = _p95_ns(samples)
    target_ns = 2_000_000_000.0  # 2000 ms
    return {
        "name": "graph_jepa_loss_step",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 2000 ms/step",
        "status": _status(median_ns, target_ns),
        "notes": "loss + backward + step + EMA, 20 iters",
    }


def bench_latent_dynamics_rollout_H12_N256() -> dict:
    cfg = LatentDynamicsConfig()  # d_latent=128, d_action=16, d_state=16
    ld = LatentDynamics(cfg)
    ld.eval()
    z0 = torch.randn(256, cfg.d_latent)
    actions = torch.randn(256, 12, cfg.d_action)

    def run() -> None:
        with torch.no_grad():
            ld.rollout(z0, actions)

    samples = _measure(run, warmup=2, iters=20)
    median_ns = statistics.median(samples)
    p95_ns = _p95_ns(samples)
    target_ns = 250_000_000.0  # 250 ms
    return {
        "name": "latent_dynamics_rollout_H12_N256",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 250 ms/rollout",
        "status": _status(median_ns, target_ns),
        "notes": "B=256, H=12 horizon",
    }


def bench_td_mpc_plan_full() -> dict:
    cfg = LatentDynamicsConfig()
    ld = LatentDynamics(cfg)
    ld.eval()
    # Must match LatentDynamicsConfig.d_action.
    action_dim = cfg.d_action

    def reward(traj, actions):
        return -traj.norm(dim=-1) - 0.01 * actions.norm(dim=-1)

    def cstr(traj, actions):
        return torch.relu(actions[..., 0].abs() - 2.5)

    # REDUCED config to keep within 60 s ceiling per audit guidance:
    #   horizon=12, n_samples=64, n_iterations=3 (defaults are 12/256/6).
    planner_cfg = TDMPCConfig(horizon=12, n_samples=64, n_iterations=3)
    planner = TDMPCPlanner(
        ld, reward, [cstr], planner_cfg, action_dim=action_dim,
    )
    z0 = torch.randn(cfg.d_latent)

    def run() -> None:
        with torch.no_grad():
            planner.plan(z0)

    samples = _measure(run, warmup=2, iters=5)
    median_ns = statistics.median(samples)
    p95_ns = _p95_ns(samples)
    target_ns = 8_000_000_000.0  # 8000 ms
    return {
        "name": "td_mpc_plan_full",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 8000 ms",
        "status": _status(median_ns, target_ns),
        "notes": "REDUCED: horizon=12, n_samples=64, n_iterations=3",
    }


def bench_diffusion_tail_sample_32() -> dict:
    # REDUCED n_steps=20 (default 50) per audit guidance.
    cfg = DiffusionConfig(n_steps=20)
    sampler = DiffusionTailSampler(cfg)
    sampler.eval()
    z0 = torch.randn(cfg.d_latent)
    action_seq = torch.randn(cfg.horizon, cfg.d_action)

    def run() -> None:
        sampler.sample(z0, action_seq, n_samples=32)

    samples = _measure(run, warmup=2, iters=5)
    median_ns = statistics.median(samples)
    p95_ns = _p95_ns(samples)
    target_ns = 1_000_000_000.0  # 1000 ms
    return {
        "name": "diffusion_tail_sample_32",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 1000 ms",
        "status": _status(median_ns, target_ns),
        "notes": "REDUCED n_steps=20, n_samples=32",
    }


def bench_constraint_full_check_and_project() -> dict:
    elements = walker_delta_constellation(
        inclination_deg=53.0, total_sats=22, num_planes=2, phasing_F=1,
        altitude_km=550.0, epoch_utc=SIM_T0,
    )
    sat_ecef = [state_ecef_m(keplerian_state(e)) for e in elements]
    emitters = [
        NGSOEmitter(position_ecef=p, eirp_dBW=-15.0, name=f"sat-{i}")
        for i, p in enumerate(sat_ecef)
    ]
    cl = PreceptualAIConstraintLayer(PreceptualAIConstraintConfig(
        gso_arcs=[GSOArcEntry(satellite_id="g1", longitude_deg=10.0)],
        epfd_floor_dBW_per_m2_per_ref_bw=-145.0,
    ))
    proposed_action = {
        "tx_power_dBm": 30.0,
        "antenna_gain_dBi": 14.0,
        "beam_azimuth_deg": 90.0,
        "beam_elevation_deg": 30.0,
        "frequency_hz": 3.7e9,
        "ai_workload_gpu_gb": 4.0,
    }
    ctx = {
        "earth_station_latitude_deg": ES_LAT,
        "earth_station_longitude_deg": ES_LON,
        "permitted_band_hz": (3.4e9, 4.2e9),
        "ngso_emitters": emitters,
        "wanted_gso_longitude_deg": 10.0,
    }

    def run() -> None:
        for _ in range(200):
            cl.check_feasibility(proposed_action, ctx)
            cl.project(proposed_action, ctx)

    samples = _measure(run, warmup=2, iters=5)
    per_call_ns = [s / 200 for s in samples]
    median_ns = statistics.median(per_call_ns)
    p95_ns = sorted(per_call_ns)[int(0.95 * (len(per_call_ns) - 1))]
    target_ns = 50_000_000.0  # 50 ms
    return {
        "name": "constraint_full_check_and_project",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 50 ms/call",
        "status": _status(median_ns, target_ns),
        "notes": "check + project, 22 emitters, 200 calls/iter",
    }


def bench_e2e_stages_3_to_5_walltime() -> dict:
    """Run the e2e_simulation stage_2 + stage_3 + stage_4 chain (no Stage 5
    HTTP). The brief says "stages 3 to 5" but explicitly says skip Stage 5
    HTTP — we run stages 2,3,4 in sequence which is the deterministic
    physics+encoder+decision chain (the slug `_3_to_5_` is preserved per
    the audit's name)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_e2e_sim", str(ROOT / "scripts" / "e2e_simulation.py"),
    )
    e2e = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(e2e)

    def run() -> None:
        physics_state = e2e.stage_2_physics_pack()
        latent = e2e.stage_3_encoder(physics_state)
        e2e.stage_4_decision(latent)

    samples = _measure(run, warmup=2, iters=3)
    median_ns = statistics.median(samples)
    p95_ns = _p95_ns(samples)
    target_ns = 10_000_000_000.0  # 10000 ms
    return {
        "name": "e2e_stages_3_to_5_walltime",
        "median": _fmt_ms(median_ns),
        "p95": _fmt_ms(p95_ns),
        "target": "≤ 10000 ms",
        "status": _status(median_ns, target_ns),
        "notes": "stage_2 + stage_3 + stage_4 (Stage 5 HTTP skipped)",
    }


def bench_epfd_time_cdf_500sat_1h() -> dict:
    """REDUCED: Walker 500/10/1, 1 hour @ 60s step (per audit)."""
    elements = walker_delta_constellation(
        inclination_deg=53.0, total_sats=500, num_planes=10, phasing_F=1,
        altitude_km=550.0, epoch_utc=SIM_T0,
    )

    def constellation_at(t: datetime) -> list[NGSOEmitter]:
        # Propagate every element to time t and convert to ECEF emitters.
        dt_s = (t - SIM_T0).total_seconds()
        ems: list[NGSOEmitter] = []
        for i, e in enumerate(elements):
            stepped = kepler_j2_step(e, dt_s)
            sat_state = keplerian_state(stepped)
            ems.append(NGSOEmitter(
                position_ecef=state_ecef_m(sat_state),
                eirp_dBW=-15.0,
                name=f"sat-{i}",
            ))
        return ems

    def run() -> None:
        epfd_time_cdf(
            es_lat_deg=ES_LAT, es_lon_deg=ES_LON,
            wanted_gso_lon_deg=10.0,
            es_diameter_m=1.2, es_frequency_hz=12e9,
            constellation_at=constellation_at,
            t_start_utc=SIM_T0,
            duration_s=3600.0,
            step_s=60.0,
        )

    # Reduce iters for the heaviest benchmark; still 5+ measurements.
    samples = _measure(run, warmup=2, iters=5)
    median_ns = statistics.median(samples)
    p95_ns = _p95_ns(samples)
    target_ns = 45_000_000_000.0  # 45 s
    return {
        "name": "epfd_time_cdf_500sat_1h",
        "median": _fmt_s(median_ns),
        "p95": _fmt_s(p95_ns),
        "target": "≤ 45 s",
        "status": _status(median_ns, target_ns),
        "notes": "REDUCED: Walker 500/10/1, 1 h, 60 s step",
    }


# ─── orchestrator ────────────────────────────────────────────────────────


BENCHMARKS: list[Callable[[], dict]] = [
    bench_orbital_walker_22sat_propagate,
    bench_epfd_aggregate_22sat_snapshot,
    bench_doppler_pass_envelope_600s,
    bench_entity_tokenize_50assets,
    bench_perceiver_fusion_forward_256x128,
    bench_graph_jepa_loss_step,
    bench_latent_dynamics_rollout_H12_N256,
    bench_td_mpc_plan_full,
    bench_diffusion_tail_sample_32,
    bench_constraint_full_check_and_project,
    bench_e2e_stages_3_to_5_walltime,
    bench_epfd_time_cdf_500sat_1h,
]


def render_table(rows: list[dict]) -> str:
    headers = ["name", "median", "p95", "target", "status", "notes"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    sep = "| " + " | ".join("-" * widths[h] for h in headers) + " |"
    head = "| " + " | ".join(h.ljust(widths[h]) for h in headers) + " |"
    body = [
        "| " + " | ".join(str(r[h]).ljust(widths[h]) for h in headers) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def main() -> int:
    torch.manual_seed(0)
    # Run on CPU to keep numbers stable / comparable across machines.
    # (Most ops here are CPU-bound; CUDA would require sync ops anyway.)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    print("# PreceptualAI benchmark suite — running 12 benchmarks…\n")
    rows: list[dict] = []
    for bench in BENCHMARKS:
        name = bench.__name__.replace("bench_", "")
        print(f"… {name}", flush=True)
        try:
            t0 = perf_counter_ns()
            row = bench()
            elapsed_s = (perf_counter_ns() - t0) / 1e9
            row.setdefault("notes", "")
            print(f"   → {row['status']}  median={row['median']}  "
                  f"(took {elapsed_s:.1f}s wall)")
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows.append({
                "name": name,
                "median": "n/a",
                "p95": "n/a",
                "target": "—",
                "status": "FAIL",
                "notes": f"exception: {type(exc).__name__}: {exc}",
            })

    table = render_table(rows)
    print()
    print(table)

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_watch = sum(1 for r in rows if r["status"] == "WATCH")
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    summary = (
        f"\n**Summary:** {n_pass}/{len(rows)} PASS · {n_watch} WATCH · {n_fail} FAIL\n"
    )
    print(summary)

    results_md = ROOT / "benchmarks" / "RESULTS.md"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results_md.write_text(
        f"# PreceptualAI Production Benchmark Results\n\n"
        f"_Generated: {timestamp}_\n\n"
        f"Methodology: 2 warm-up + 5+ measured iterations per bench; "
        f"median + p95 reported. Status thresholds: PASS ≤ target, "
        f"WATCH ≤ 1.5×target, FAIL > 1.5×target.\n\n"
        f"{table}\n"
        f"{summary}"
    )
    print(f"Wrote {results_md}")

    return 0 if n_pass >= 8 else 1


if __name__ == "__main__":
    sys.exit(main())
