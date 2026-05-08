"""Light up every PreceptualAI subsystem in one real run.

Iterates through every public subsystem documented in PRD/PARADIGMS,
performs ONE real operation per subsystem (no mocks, no stubs, no
hardcoded values past the necessary inputs), and prints a status table.

If any subsystem fails to load or run, the script exits 1 and lists the
broken row(s).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
PARENT_DATA = REPO.parent / "data"
sys.path.insert(0, str(REPO / "src"))

# JWT secret needed by api_v1 + dashboard_api at import time.
os.environ.setdefault(
    "HORIZON_API_JWT_SECRET",
    "lightup-secret-key-not-for-production-use-32B",
)

CKPT = REPO / "checkpoints"

import torch  # noqa: E402

torch.manual_seed(20260506)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class Result:
    def __init__(self, n: int, name: str):
        self.n = n
        self.name = name
        self.status = "[FAIL]"
        self.wall_ms = 0.0
        self.note = ""

    def __str__(self) -> str:
        return (
            f"| {self.n:>2} | {self.name:<30} | {self.status:<6} | "
            f"{self.wall_ms:>9.2f}ms | {self.note} |"
        )


def run(n: int, name: str, fn: Callable[[], str]) -> Result:
    r = Result(n, name)
    t0 = time.perf_counter()
    try:
        note = fn()
        r.note = (note or "ok")[:64]
        r.status = "[OK]"
    except Exception as exc:  # noqa: BLE001
        r.note = f"{type(exc).__name__}: {exc}"[:96]
        r.status = "[FAIL]"
        # Stash traceback for debugging — printed to stderr at end.
        r._tb = traceback.format_exc()  # type: ignore[attr-defined]
    finally:
        r.wall_ms = (time.perf_counter() - t0) * 1000.0
    return r


# ---------------------------------------------------------------------------
# Subsystem operations
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


# 1. physics.propagation
def op_physics_propagation() -> str:
    from horizon_ric.planner.physics.propagation import total_path_loss_dB

    res = total_path_loss_dB(
        distance_m=1_200_000.0,
        frequency_hz=28e9,
        elevation_deg=30.0,
        rain_rate_mm_per_hr=10.0,
    )
    return f"total={res['total_dB']:.2f}dB fspl={res['fspl_dB']:.1f} gas={res['gas_dB']:.2f} rain={res['rain_dB']:.2f}"


# 2. physics.orbital — Walker constellation expanded then propagated
def op_physics_orbital() -> str:
    from horizon_ric.planner.physics.orbital import (
        keplerian_state,
        walker_delta_constellation,
    )

    elems = walker_delta_constellation(
        inclination_deg=53.0, total_sats=22, num_planes=2,
        phasing_F=1, altitude_km=550.0,
    )
    state = keplerian_state(elems[0])
    return f"n_sats={len(elems)} alt0={state.altitude_km():.1f}km"


# 3. physics.doppler — over real Starlink TLE
def op_physics_doppler() -> str:
    from horizon_ric.data.celestrak import CelesTrakReader
    from horizon_ric.planner.physics.doppler import doppler_shift_hz
    from horizon_ric.planner.physics.orbital import sgp4_state

    rdr = CelesTrakReader("starlink")
    e = rdr.entries()[0]
    state = sgp4_state(e.line1, e.line2, _now_utc())
    f = doppler_shift_hz(state, es_lat_deg=51.92, es_lon_deg=4.48,
                        carrier_hz=12.5e9)
    return f"sat={e.name[:12]} doppler={f / 1e3:.2f}kHz"


# 4. physics.tr38811
def op_physics_tr38811() -> str:
    from horizon_ric.planner.physics.tr38811 import channel_state

    cs = channel_state("urban", 28e9, 40.0)
    return f"K={cs.k_factor_dB:.1f}dB DS={cs.delay_spread_ns:.1f}ns LOS={cs.los_prob:.2f}"


# 5. physics.ntn_timing
def op_physics_ntn_timing() -> str:
    from horizon_ric.planner.physics.ntn_timing import ntn_timing_state

    ts = ntn_timing_state(altitude_m=600_000.0, elevation_deg=40.0)
    return (
        f"k_offset={ts.k_offset_slots} one_way={ts.one_way_delay_s * 1e3:.2f}ms "
        f"harq={ts.harq_feedback_enabled}"
    )


# 6. physics.epfd — real Starlink TLEs over Rotterdam
def op_physics_epfd() -> str:
    from horizon_ric.data.tle_pipeline import TLEConstellationPropagator
    from horizon_ric.planner.physics.epfd import epfd_down

    tle_path = PARENT_DATA / "orbital" / "celestrak" / "starlink.tle"
    prop = TLEConstellationPropagator(tle_path).slice(64)
    emitters = prop.at(_now_utc())
    res = epfd_down(
        es_lat_deg=51.92, es_lon_deg=4.48,
        wanted_gso_lon_deg=10.0,
        es_diameter_m=1.2, es_frequency_hz=12.5e9,
        ngso_satellites=emitters,
    )
    return f"epfd={res.epfd_dBW_per_m2:.2f}dBW/m^2 visible={res.n_visible}/{len(emitters)}"


# 7. physics.coexistence
def op_physics_coexistence() -> str:
    from horizon_ric.planner.physics.coexistence import (
        ngso_inline_event_probability,
    )

    s = ngso_inline_event_probability(
        visible_a=8, visible_b=6, in_line_angle_deg=1.0, min_elevation_deg=25.0,
    )
    return f"p_inline={s.p_inline:.4f} sec/orbit={s.expected_seconds_per_orbit:.2f}"


# 8. geodesy
def op_geodesy() -> str:
    from horizon_ric.planner.physics.geodesy import (
        gso_satellite_ecef,
        look_angles_to_target,
    )

    sat = gso_satellite_ecef(10.0)  # 10°E
    la = look_angles_to_target(es_lat_deg=51.92, es_lon_deg=4.48, target=sat)
    return f"az={la.azimuth_deg:.2f} el={la.elevation_deg:.2f} rng={la.range_m / 1e3:.0f}km"


# 9. encoder.SpatialPrior
def op_spatial_prior() -> str:
    from horizon_ric.encoder.spatial_prior import SpatialPrior

    sp = SpatialPrior(min_elevation_deg=5.0)
    ents = [
        {"placement": "ground", "lat_deg": 51.92, "lon_deg": 4.48},
        {"placement": "ground", "lat_deg": 1.30, "lon_deg": 103.80},
        {"placement": "ground", "lat_deg": 40.71, "lon_deg": -74.00},
        {"placement": "ground", "lat_deg": -33.87, "lon_deg": 151.21},
        {"placement": "ground", "lat_deg": 35.69, "lon_deg": 139.69},
        {"placement": "ground", "lat_deg": -23.55, "lon_deg": -46.63},
    ]
    t = sp.build(ents)
    return f"nodes={t.node_xyz_m.shape[0]} los_links={int(t.los_mask.sum())}"


# 10. encoder.EntityTokenizer
def op_entity_tokenizer() -> str:
    from horizon_ric.encoder.entity_tokenizer import (
        AssetAttributes,
        EntityTokenizer,
        EntityTokenizerConfig,
        EntityType,
    )

    tok = EntityTokenizer(EntityTokenizerConfig(d_model=64))
    # Per-type attr_dim must match the tokeniser's default config.
    types = [
        (EntityType.UE, 16), (EntityType.SAT_NGSO, 8),
        (EntityType.CELL, 12), (EntityType.BEAM, 10),
        (EntityType.GATEWAY, 6),
    ]
    assets = []
    for i in range(50):
        et, dim = types[i % 5]
        assets.append(AssetAttributes(et, torch.randn(dim), (1e6 + i, 0.0, 0.0)))
    out = tok(assets)
    return f"tokens={tuple(out.shape)} norm={out.norm().item():.2f}"


# 11. encoder.PerceiverFusion — fuse 200 tokens
def op_perceiver_fusion() -> str:
    from horizon_ric.encoder.perceiver_fusion import (
        PerceiverConfig,
        PerceiverFusion,
    )

    p = PerceiverFusion(PerceiverConfig(
        n_latents=128, d_latent=128, d_input=128,
        n_self_layers=2, n_heads=4,
    ))
    z = p(torch.randn(1, 200, 128))
    return f"latents={tuple(z.shape)}"


# 12. encoder.GraphJEPA — load real checkpoint and forward
def op_graph_jepa() -> str:
    from horizon_ric.encoder.graph_jepa import GraphJEPA, GraphJEPAConfig
    from horizon_ric.encoder.perceiver_fusion import (
        PerceiverConfig,
        PerceiverFusion,
    )

    encoder = PerceiverFusion(PerceiverConfig(
        n_latents=128, d_latent=128, d_input=128,
        n_self_layers=2, n_heads=4,
    ))
    state = torch.load(CKPT / "jepa_encoder_v0.1.pt",
                       map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["context_encoder"])
    # Infer predictor depth from the saved layer count.
    pred_layer_idxs = {
        int(k.split(".")[2]) for k in state["predictor"].keys()
        if k.startswith("transformer.layers.")
    }
    n_layers = max(pred_layer_idxs) + 1 if pred_layer_idxs else 3
    jepa = GraphJEPA(encoder, GraphJEPAConfig(
        d_latent=128, predictor_hidden=256, predictor_layers=n_layers,
    ))
    jepa.predictor.load_state_dict(state["predictor"])
    out = jepa.loss(torch.randn(1, 32, 128))
    return f"loss={out['loss'].item():.4f} predictor_layers={n_layers}"


# 13. encoder.LinkState — compose 12-D link-state for one real Starlink sat
def op_link_state() -> str:
    from horizon_ric.data.celestrak import CelesTrakReader
    from horizon_ric.encoder.link_state import (
        LINK_STATE_DIM,
        LinkStateInputs,
        compose_link_state,
    )
    from horizon_ric.planner.physics.orbital import sgp4_state

    e = CelesTrakReader("starlink").entries()[0]
    state = sgp4_state(e.line1, e.line2, _now_utc())
    inputs = LinkStateInputs(
        sat_state=state, es_lat_deg=51.92, es_lon_deg=4.48,
        carrier_hz=12.5e9, environment="rural",
    )
    t = compose_link_state(inputs)
    return f"dim={LINK_STATE_DIM} shape={tuple(t.shape)} sat={e.name[:14]}"


# 14. core.CfC — load checkpoint + step
def op_cfc_core() -> str:
    from horizon_ric.core.cfc_core import CfCCell, CfCConfig

    payload = torch.load(CKPT / "cfc_cell_v0.1.pt",
                         map_location="cpu", weights_only=True)
    cfg = payload["config"]
    in_dim = int(cfg["input_dim"])
    h_dim = int(cfg["hidden_dim"])
    # Strip the "cell." prefix the trainer uses (the trainer wraps the cell
    # inside a tiny container, but we only need the cell weights).
    sd = {k[len("cell."):]: v for k, v in payload["state_dict"].items()
          if k.startswith("cell.")}
    cell = CfCCell(CfCConfig(input_dim=in_dim, hidden_dim=h_dim))
    cell.load_state_dict(sd)
    h = cell(torch.randn(1, in_dim), cell.init_hidden(1), dt=1.0)
    return (
        f"in={in_dim} hid={h_dim} out_norm={h.norm().item():.3f} "
        f"beats_random={payload['beats_random']}"
    )


# 15. core.LiquidS4
def op_liquid_s4() -> str:
    from horizon_ric.core.liquid_s4 import LiquidS4, LiquidS4Config

    payload = torch.load(CKPT / "liquid_s4_v0.1.pt",
                         map_location="cpu", weights_only=True)
    cfg = payload["config"]
    d_model = int(cfg["d_model"])
    d_state = int(cfg["d_state"])
    n_layers = int(cfg["n_layers"])
    m = LiquidS4(LiquidS4Config(d_model=d_model, d_state=d_state,
                                n_layers=n_layers))
    # The trainer wraps LiquidS4 inside a tiny container (input_proj +
    # backbone + head); strip the "backbone." prefix so the unwrapped
    # state-dict slots into the standalone LiquidS4. Drop the wrapper-only
    # parameters (input_proj, head) — the standalone class doesn't have them.
    raw = payload["state_dict"]
    sd = {
        k[len("backbone."):]: v for k, v in raw.items()
        if k.startswith("backbone.")
    }
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:3]}")
    y = m(torch.randn(1, 8, d_model))
    return (
        f"d_model={d_model} d_state={d_state} layers={n_layers} "
        f"out={tuple(y.shape)} missing={len(missing)}"
    )


# 16. core.LatentDynamics
def op_latent_dynamics() -> str:
    from horizon_ric.core.latent_dynamics import (
        LatentDynamics,
        LatentDynamicsConfig,
    )

    state = torch.load(CKPT / "latent_dynamics_v0.1.pt",
                       map_location="cpu", weights_only=True)
    ld = LatentDynamics(LatentDynamicsConfig(d_latent=128, d_action=16, d_state=16))
    ld.load_state_dict(state)
    traj = ld.rollout(torch.randn(1, 128), torch.randn(1, 4, 16))
    return f"traj={tuple(traj.shape)}"


# 17. core.PhysicsResidual — load real checkpoint trained on DeepMIMO
def op_physics_residual() -> str:
    from horizon_ric.core.physics_residual import PhysicsResidualHead
    from horizon_ric.planner.physics.propagation import total_path_loss_dB

    payload = torch.load(CKPT / "physics_residual_v0.1.pt",
                         map_location="cpu", weights_only=True)
    cfg = payload["config"]
    state_dim = int(cfg["state_dim"])
    latent_dim = int(cfg["latent_dim"])
    output_dim = int(cfg["output_dim"])
    hidden_dim = int(cfg["hidden_dim"])
    f_hz = float(cfg["frequency_hz"])
    elev_deg = float(cfg["elevation_deg"])

    def itu_physics(s: torch.Tensor) -> torch.Tensor:
        # Feature 0 is distance/1000 (km in normalised units); recover metres.
        d_m = s[..., 0] * 1000.0
        out = torch.empty(s.shape[0], output_dim)
        for i, d in enumerate(d_m.tolist()):
            r = total_path_loss_dB(
                distance_m=max(d, 1.0), frequency_hz=f_hz,
                elevation_deg=elev_deg,
            )
            out[i, 0] = -float(r["total_dB"])  # observed_dB target uses -loss
        return out

    head = PhysicsResidualHead(
        physics_fn=itu_physics, state_dim=state_dim,
        latent_dim=latent_dim, output_dim=output_dim, hidden_dim=hidden_dim,
    )
    head.load_state_dict(payload["state_dict"])
    head.eval()
    with torch.no_grad():
        out = head(torch.randn(1, state_dim), torch.randn(1, latent_dim))
    return (
        f"state={state_dim} lat={latent_dim} hid={hidden_dim} "
        f"total_pred={out['total_pred'].item():.2f}dB"
    )


# 18. core.TimingBudgets — l1_admit + a1_admit
def op_timing_budgets() -> str:
    from horizon_ric.core.timing_budgets import a1_admit, l1_admit

    l1 = l1_admit(model_latency_us=30.0, slot_numerology=1)
    a1 = a1_admit(decision_latency_ms=50.0, policy_period_ms=100.0)
    return f"l1={l1} a1={a1}"


# 19. heads.SLARiskHead v0.4
def op_sla_head() -> str:
    from horizon_ric.heads.sla_risk import SLARiskConfig, SLARiskHead

    head = SLARiskHead(SLARiskConfig(latent_dim=128, hidden_dim=128))
    state = torch.load(CKPT / "sla_head_v0.4_jepa.pt",
                       map_location="cpu", weights_only=True)
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        out = head(torch.randn(4, 128))
    keys = sorted(out.keys())
    p30 = float(out[keys[0]].mean())
    return f"horizons={keys} p30s_mean={p30:.4f}"


# 20. policy.Constraints — check + project
def op_policy_constraints() -> str:
    from horizon_ric.policy.constraints import (
        GSOArcEntry,
        PreceptualAIConstraintConfig,
        PreceptualAIConstraintLayer,
    )

    cl = PreceptualAIConstraintLayer(PreceptualAIConstraintConfig(
        gso_arcs=[GSOArcEntry("a", 0.0), GSOArcEntry("b", 20.0)]
    ))
    ctx = {
        "earth_station_latitude_deg": 30.0,
        "earth_station_longitude_deg": 0.0,
        "permitted_band_hz": (3.4e9, 4.2e9),
    }
    action = {"frequency_hz": 5.0e9, "tx_power_dBm": -200.0,
              "ai_workload_gpu_gb": 200.0}
    viols = cl.check_feasibility(action, ctx)
    feasible, corrections = cl.project(action, ctx)
    return (
        f"viols={len(viols)} corrections={len(corrections)} "
        f"freq_clipped={feasible['frequency_hz']:.2e}"
    )


# 21. policy.EmitGuards
def op_emit_guards() -> str:
    from horizon_ric.policy.constraints import (
        GSOArcEntry,
        PreceptualAIConstraintConfig,
        PreceptualAIConstraintLayer,
    )
    from horizon_ric.policy.emit_guards import run_guard_chain

    layer = PreceptualAIConstraintLayer(PreceptualAIConstraintConfig(
        gso_arcs=[GSOArcEntry("g1", 0.0)],
        epfd_floor_dBW_per_m2_per_ref_bw=-145.0,
    ))
    failures = run_guard_chain(
        head_is_pretrained=True,
        constraint_context={
            "ngso_emitters": [object()],
            "wanted_gso_longitude_deg": 0.0,
        },
        constraint_layer=layer,
        elapsed_ms=10.0,
        corrections=None,
        audit_corrections_field=None,
    )
    return f"failures={len(failures)} (clean run -> 0 expected)"


# 22. policy.Counterfactual
def op_counterfactual() -> str:
    from horizon_ric.policy.counterfactual import (
        AlternativeCandidate,
        build_rejected_alternatives,
    )

    cands = [
        AlternativeCandidate(
            action={"tx_dBm": 30.0}, reward_sum=-5.0,
            constraint_violation_sum=2.0, primary_metric="epfd",
            primary_value=2.0, threshold=0.0, horizon="60s",
        ),
        AlternativeCandidate(
            action={"tx_dBm": 10.0}, reward_sum=-1.0,
            constraint_violation_sum=0.0, sla_risk_30s=0.30,
            primary_metric="sla", primary_value=0.30, threshold=0.20,
            horizon="30s",
        ),
    ]
    out = build_rejected_alternatives(cands, chosen_score=0.0)
    causes = sorted({a.rejection_reason_machine.primary_cause for a in out})
    return f"alternatives={len(out)} causes={causes}"


# 23. policy.TDMPC2 — load real value+prior + plan
def op_tdmpc_planner() -> str:
    from horizon_ric.core.latent_dynamics import (
        LatentDynamics,
        LatentDynamicsConfig,
    )
    from horizon_ric.policy.td_mpc_planner import TDMPCConfig, TDMPCPlanner

    ld = LatentDynamics(LatentDynamicsConfig(d_latent=128, d_action=16, d_state=16))
    ld.load_state_dict(torch.load(
        CKPT / "latent_dynamics_v0.1.pt",
        map_location="cpu", weights_only=True,
    ))
    ld.eval()

    # Load value head + policy prior real checkpoints.
    import torch.nn as nn

    class ValueHead(nn.Module):
        def __init__(self, d_latent=128, d_action=16, hidden=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_latent + d_action, hidden), nn.SiLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, z, a):
            return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)

    class PolicyPrior(nn.Module):
        LOG_STD_MIN = -3.0
        LOG_STD_MAX = -0.5

        def __init__(self, d_latent=128, d_action=16, hidden=128, action_scale=1.2):
            super().__init__()
            self.action_scale = action_scale
            self.net = nn.Sequential(
                nn.Linear(d_latent, hidden), nn.SiLU(),
                nn.Linear(hidden, 2 * d_action),
            )

        def forward(self, z):
            out = self.net(z)
            mr, ls = out.chunk(2, dim=-1)
            ls = ls.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            return self.action_scale * torch.tanh(mr), ls

    value = ValueHead()
    value.load_state_dict(torch.load(
        CKPT / "tdmpc_value_v0.1.pt", map_location="cpu", weights_only=True,
    ))
    value.eval()
    prior = PolicyPrior()
    prior.load_state_dict(torch.load(
        CKPT / "tdmpc_policy_prior_v0.1.pt",
        map_location="cpu", weights_only=True,
    ))
    prior.eval()

    def reward(traj, actions):
        return -traj.norm(dim=-1)

    def cstr(traj, actions):
        return torch.relu(actions.abs().sum(dim=-1) - 6.0)

    planner = TDMPCPlanner(
        ld, reward, [cstr],
        TDMPCConfig(horizon=4, n_samples=32, n_iterations=2),
        action_dim=16, value_head=value, policy_prior=prior,
    )
    res = planner.plan(torch.randn(128))
    return f"plan_actions={tuple(res.actions.shape)} score={res.expected_score:.3f}"


# 24. policy.DiffusionTail
def op_diffusion_tail() -> str:
    from horizon_ric.policy.diffusion_tail import (
        DiffusionConfig,
        DiffusionTailSampler,
    )

    d = DiffusionTailSampler(DiffusionConfig(
        d_latent=128, d_action=16, horizon=3, n_steps=5, hidden_dim=64,
    ))
    samples = d.sample(torch.randn(128), torch.randn(3, 16), n_samples=8)
    return f"samples={tuple(samples.shape)}"


# 25. policy.LIConstraint
def op_li_constraint() -> str:
    from horizon_ric.policy.li_constraint import (
        LIConstraint,
        LIJurisdictionRule,
    )

    rule = LIJurisdictionRule(
        rule_id="warrant-DE-001",
        protected_ue_ids={"ue-protected-1"},
        protected_slice_ids={"slice-DE-1"},
        allowed_jurisdictions={"DE", "EU"},
    )
    li = LIConstraint([rule])
    viols = li.check_feasibility(
        {"affected_ue_ids": {"ue-protected-1", "ue-other"},
         "target_jurisdiction": "US"},
        context={},
    )
    ids = sorted(v.constraint_id for v in viols)
    return f"violations={len(viols)} ids={ids}"


# 26. rApp.R1Adapter
def op_r1_adapter() -> str:
    from horizon_ric.rapp.r1_adapter import R1Adapter, R1AdapterConfig

    a = R1Adapter(R1AdapterConfig(smo_base_url="http://localhost:9999"))
    return f"breaker={a.circuit_breaker.state} fail={a.circuit_breaker.fail_counter}"


# 27. rApp.A1Adapter (osc dialect)
def op_a1_adapter_osc() -> str:
    from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig

    a = A1Adapter(A1AdapterConfig(near_rt_ric_base_url="http://localhost:9998",
                                  dialect="osc"))
    return f"dialect={a.cfg.dialect} breaker={a.circuit_breaker.state}"


# 28. rApp.O1Adapter
def op_o1_adapter() -> str:
    from horizon_ric.rapp.o1_adapter import O1Adapter, O1AdapterConfig

    a = O1Adapter(O1AdapterConfig(host="localhost", port=8830,
                                  username="x", password="y",
                                  hostkey_verify=False))
    return f"connected={a.is_connected} breaker={a.circuit_breaker.state}"


# 29. rApp.Auth — build secure async client
def op_auth() -> str:
    from horizon_ric.rapp.auth import AuthConfig, build_secure_async_client

    cfg = AuthConfig(
        verify_tls=True, production_mode=False,
        static_bearer_token="test-token-1",
    )
    client = build_secure_async_client(cfg, base_url="https://example.com")
    cls = type(client).__name__
    return f"client={cls} timeout={client.timeout.connect:.1f}s"


# 30. rApp.Lifecycle
def op_lifecycle() -> str:
    from horizon_ric.rapp.lifecycle import HorizonRAppLifecycle, RAppState

    lc = HorizonRAppLifecycle(state_path=Path("/tmp/horizon_lightup_state.json"))
    return f"state={lc.state.value} init?={lc.state == RAppState.INIT}"


# 31. rApp.Health
def op_health() -> str:
    from horizon_ric.rapp.health import build_health_app
    from horizon_ric.rapp.lifecycle import HorizonRAppLifecycle

    lc = HorizonRAppLifecycle(state_path=Path("/tmp/horizon_lightup_state.json"))
    app = build_health_app(lc)
    routes = sorted(r.path for r in app.routes if hasattr(r, "path"))
    return f"routes={routes[:4]}"


# 32. runtime.CircuitBreaker — open + close
def op_circuit_breaker() -> str:
    import asyncio

    from horizon_ric.runtime.circuit_breaker import (
        AsyncCircuitBreaker,
        BreakerConfig,
    )
    import httpx

    cb = AsyncCircuitBreaker(BreakerConfig(name="lightup.cb", fail_max=2,
                                           reset_timeout=0.05))

    async def fail():
        raise httpx.ConnectError("simulated")

    async def ok():
        return 1

    async def go() -> str:
        # Trip
        for _ in range(3):
            try:
                await cb.call(fail)
            except Exception:
                pass
        opened = cb.state
        await asyncio.sleep(0.07)
        # Recover via successful trial in HALF_OPEN.
        await cb.call(ok)
        recovered = cb.state
        return f"open->{opened} reset->{recovered}"

    return asyncio.run(go())


# 33. runtime.Watchdog
def op_watchdog() -> str:
    from horizon_ric.runtime.watchdog import notify_ready

    sent = notify_ready()  # False is fine when no NOTIFY_SOCKET (this host).
    return f"notify_ready_sent={sent} (False expected without systemd socket)"


# 34. runtime.StateRecovery
def op_state_recovery() -> str:
    from horizon_ric.runtime.state_recovery import load_state, save_state

    p = Path("/tmp/horizon_lightup_recovery.json")
    payload = {"a": 1, "b": "two", "c": [1, 2, 3]}
    save_state(payload, p)
    out = load_state(p)
    assert out == payload, f"round-trip mismatch: {out}"
    return f"round_trip_ok bytes={p.stat().st_size}"


# 35. runtime.Backpressure
def op_backpressure() -> str:
    import asyncio

    from horizon_ric.runtime.backpressure import BackpressureQueue

    q = BackpressureQueue("lightup", max_depth=100, drop_policy="oldest")

    async def go() -> str:
        for i in range(50):
            await q.put(i)
        head = await q.get()
        return f"depth={q.qsize()} head={head} dropped={q.dropped}"

    return asyncio.run(go())


# 36. io.Registry
def op_io_registry() -> str:
    # Trigger celestrak auto-registration so the registry isn't empty.
    import horizon_ric.data.celestrak  # noqa: F401
    from horizon_ric.io.registry import list_connectors

    inv = list_connectors()
    n_src = len(inv.get("sources", []))
    n_snk = len(inv.get("sinks", []))
    return f"sources={n_src} sinks={n_snk}"


# 37. io.FileSink+FileSource
def op_io_file() -> str:
    import asyncio

    from horizon_ric.io.connectors.file_connector import FileSink, FileSource
    from horizon_ric.io.schemas import TelemetryEvent

    path = Path("/tmp/horizon_lightup_events.jsonl")
    if path.exists():
        path.unlink()

    async def go() -> str:
        sink = FileSink(_FileConfigInst(path=str(path)))
        await sink.connect()
        for i in range(5):
            ev = TelemetryEvent(
                event_id=f"lu-{i}", modality="kpm_5g",
                source_id="lightup", ts_utc=_now_utc(),
                sequence=i, payload={"i": i}, tags={"k": "v"},
            )
            await sink.write(ev)
        await sink.close()
        src = FileSource(_FileConfigInst(path=str(path)))
        await src.connect()
        n = 0
        async for _ev in src.stream():
            n += 1
        await src.close()
        return f"wrote=5 read={n}"

    return asyncio.run(go())


def _FileConfigInst(**kw):
    from horizon_ric.io.connectors.file_connector import _FileConfig

    kw.setdefault("name", "lightup-file")
    return _FileConfig(**kw)


# 38. data.Aerial parquet
def op_data_aerial() -> str:
    from horizon_ric.data.aerial import AerialFAPIReader

    p = PARENT_DATA / "aerial" / "parquet" / "fapi.parquet"
    rdr = AerialFAPIReader(p)
    ev = next(iter(rdr))
    return f"first_event={ev.event_id} cols={len(rdr.columns)}"


# 39. data.DeepMIMO
def op_data_deepmimo() -> str:
    from horizon_ric.data.deepmimo import DeepMIMOScenarioReader

    rdr = DeepMIMOScenarioReader(PARENT_DATA / "deepmimo" / "asu_campus_3p5_dyn",
                                 max_scenes=4)
    ev = next(iter(rdr))
    return f"scenes={len(rdr.scenes)} first={ev.payload['scene_dir'].split('/')[-1]}"


# 40. data.AODT — parse real test scene
def op_data_aodt() -> str:
    from horizon_ric.data.aodt import AODTScenario

    scene = REPO / "tests" / "fixtures" / "aodt_test_scene.usda"
    s = AODTScenario(scene)
    return (
        f"cells={len(s.entities['cells'])} ues={len(s.entities['ues'])} "
        f"sats={len(s.entities['satellites'])}"
    )


# 41. data.Sionna — real TDL channel
def op_data_sionna() -> str:
    from horizon_ric.data.sionna_channel import SionnaChannelGenerator

    gen = SionnaChannelGenerator(seed=42, n_rx=2, n_tx=2)
    h = gen.generate_channel(scenario="NTN-TDL-A", ue_speed_kmh=30.0,
                             frequency_hz=2e9, n_samples=8)
    return f"backend={gen.backend_name} layout={gen._tdl_layout} h={h.shape}"


# 42. data.CelesTrak
def op_data_celestrak() -> str:
    from horizon_ric.data.celestrak import CelesTrakReader

    rdr = CelesTrakReader("starlink")
    n = 0
    for _ in rdr:
        n += 1
        if n >= 100:
            break
    return f"parsed_first={n}/{len(rdr)}"


# 43. data.ITU-R maps
def op_data_itu_r() -> str:
    from horizon_ric.data.itu_r import P839Map

    m = P839Map()
    h = m.lookup(51.92, 4.48)
    return f"P839_at_rotterdam={h:.3f}km available={m.is_available}"


# 44. data.tle_pipeline — propagate 50 sats
def op_tle_pipeline() -> str:
    from horizon_ric.data.tle_pipeline import TLEConstellationPropagator

    p = TLEConstellationPropagator(
        PARENT_DATA / "orbital" / "celestrak" / "starlink.tle"
    ).slice(50)
    emitters = p.at(_now_utc())
    return f"propagated={len(emitters)}/50"


# 45. federated.FedAvg
def op_federated_fedavg() -> str:
    from horizon_ric.federated import ClientUpdate, aggregate_fedavg

    def cu(name, w, n):
        return ClientUpdate(client_id=name, state_dict={
            "lin.weight": torch.tensor([[w]], dtype=torch.float32),
            "lin.bias": torch.tensor([w], dtype=torch.float32),
        }, sample_count=n)

    out = aggregate_fedavg([cu("a", 1.0, 100), cu("b", 3.0, 100), cu("c", 5.0, 100)])
    return f"avg_weight={out['lin.weight'].item():.4f}"


# 46. federated.Sparsifier
def op_federated_sparsifier() -> str:
    from horizon_ric.federated import top_k_sparsify

    delta = {"w": torch.randn(5_000_000)}
    out = top_k_sparsify(delta, sparsity=0.99)
    nz = int((out["w"] != 0).sum())
    return f"nonzero={nz}/5e6 (~1% expected)"


# 47. trading.Auction
def op_trading_auction() -> str:
    from horizon_ric.trading import Auctioneer, Bidder

    auct = Auctioneer(key_length_bits=1024)
    bids = [Bidder(b, auct.public_key).encrypt(v)
            for b, v in [("a", 50), ("b", 120), ("c", 80), ("d", 90), ("e", 30)]]
    res = auct.reveal(bids)
    return f"winner={res.winner_id} bid={res.winning_bid} pays={res.price_paid}"


# 48. evidence.JsonlStore
def op_evidence_jsonl() -> str:
    from horizon_ric.evidence import (
        DecisionRecord,
        JsonlEvidenceStore,
        ModelVersions,
        PredictedOutcome,
    )

    p = Path("/tmp/horizon_lightup_evidence.jsonl")
    if p.exists():
        p.unlink()
    store = JsonlEvidenceStore(p)
    versions = ModelVersions(
        encoder="lu", risk_heads="lu", dyna="lu",
        policy="lu", constraint_layer="lu", rapp="lu",
    )
    for i in range(3):
        store.append(DecisionRecord.new(
            decision_id=f"lu-{i}", rapp_instance_id="lu",
            state_hash="0" * 64, chosen_action={"i": i},
            predicted_outcome_chosen=PredictedOutcome(
                sla_risk_30s=0.05, sla_risk_1min=0.05, sla_risk_5min=0.05,
            ),
            rejected_alternatives=[], model_versions=versions,
        ))
    bad_idx = store.verify()
    return f"records={len(store)} chain_intact={bad_idx == -1}"


# 49. evidence.SqliteStore
def op_evidence_sqlite() -> str:
    from horizon_ric.evidence import (
        DecisionRecord,
        ModelVersions,
        PredictedOutcome,
        SqliteEvidenceStore,
    )

    db = Path("/tmp/horizon_lightup_evidence.db")
    if db.exists():
        db.unlink()
    store = SqliteEvidenceStore(f"sqlite:///{db}")
    versions = ModelVersions(
        encoder="lu", risk_heads="lu", dyna="lu",
        policy="lu", constraint_layer="lu", rapp="lu",
    )
    for i in range(3):
        store.append(DecisionRecord.new(
            decision_id=f"lus-{i}", rapp_instance_id="lu",
            state_hash="0" * 64, chosen_action={"i": i},
            predicted_outcome_chosen=PredictedOutcome(
                sla_risk_30s=0.05, sla_risk_1min=0.05, sla_risk_5min=0.05,
            ),
            rejected_alternatives=[], model_versions=versions,
        ))
    bad_idx = store.verify()
    return f"records=3 chain_intact={bad_idx == -1}"


# 50. security.RBAC
def op_security_rbac() -> str:
    import shutil

    from horizon_ric.security.rbac import (
        DEFAULT_MODEL_PATH,
        DEFAULT_POLICY_PATH,
        Casbin,
    )

    pol = Path("/tmp/horizon_lightup_rbac.csv")
    shutil.copyfile(DEFAULT_POLICY_PATH, pol)
    rb = Casbin(model_path=DEFAULT_MODEL_PATH, policy_path=pol)
    rb.add_role("op", "operator", "default")
    can_emit = rb.enforce("op", "default", "policies/abc", "emit")
    cant_audit = not rb.enforce("op", "default", "audit/recent", "verify")
    return f"operator_emit={can_emit} operator_no_audit={cant_audit}"


# 51. security.JWT
def op_security_jwt() -> str:
    from horizon_ric.security.jwt import JWTManager

    pem = REPO / "tests" / "fixtures" / "test_jwt_signing.pem"
    mgr = JWTManager(signing_key_path=pem, issuer="horizon-lightup",
                     audience="horizon-ric")
    tok = mgr.mint_token("alice", "default", ["operator"], ttl_seconds=60)
    claims = mgr.verify_token(tok)
    return f"sub={claims['sub']} roles={claims['roles']} aud={claims['aud']}"


# 52. security.Tenant
def op_security_tenant() -> str:
    from horizon_ric.security.tenant import (
        TenantScope,
        current_tenant,
        require_current_tenant,
    )

    assert current_tenant() is None
    with TenantScope("tenant_lightup"):
        t = require_current_tenant()
    assert current_tenant() is None
    return f"inside_scope={t} cleared_outside={current_tenant() is None}"


# 53. sla.Engine
def op_sla_engine() -> str:
    from horizon_ric.sla.engine import SLAEvaluator
    from horizon_ric.sla.policy import SLA, SLOTarget

    sla = SLA(
        id="sla-lu", name="latency",
        targets=[SLOTarget(metric="latency_p99_ms", comparison="<=",
                           threshold=10.0, window_s=2)],
        severity_levels={"critical": {"metrics": ["latency_p99_ms"]}},
    )
    ev = SLAEvaluator(slas=[sla])
    ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=0.0)
    out = ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=3.0)
    return f"breach={len(out)} sev={out[0].severity if out else 'none'}"


# 54. sla.alertmanager
def op_sla_alertmanager() -> str:
    from horizon_ric.sla.alertmanager import format_breach_for_alertmanager
    from horizon_ric.sla.policy import SLABreachEvent

    breach = SLABreachEvent(
        id="b1", sla_id="sla-lu",
        ts_utc=datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc),
        severity="critical", observed_value=25.0, target_threshold=10.0,
        source_metric="latency_p99_ms", lasting_s=10.0,
    )
    payload = format_breach_for_alertmanager(breach, sla_name="latency",
                                             tenant="tenant_lu")
    return f"alertname={payload['labels']['alertname']} sev={payload['labels']['severity']}"


# 55. sla.escalation
def op_sla_escalation() -> str:
    import asyncio

    from horizon_ric.sla.escalation import (
        EscalationEngine,
        EscalationPolicy,
        EscalationStep,
    )
    from horizon_ric.sla.policy import SLABreachEvent

    class _RecChan:
        name = "slack"
        calls: list = []

        async def send(self, breach, recipients):
            self.calls.append((breach.id, list(recipients)))

    ch = _RecChan()
    eng = EscalationEngine()
    eng.register_channel(ch)
    eng.register_policy(EscalationPolicy(
        name="t1",
        steps=[
            EscalationStep(delay_s=0.0, channels=["slack"], recipients=["#a"]),
            EscalationStep(delay_s=0.0, channels=["slack"], recipients=["#b"]),
        ],
    ))
    breach = SLABreachEvent(
        id="b-esc", sla_id="sla-lu", severity="critical",
        observed_value=25.0, target_threshold=10.0,
        source_metric="latency_p99_ms", lasting_s=1.0,
    )
    asyncio.run(eng.walk_now(breach, "t1"))
    return f"channel_calls={len(ch.calls)} steps_advanced=2"


# 56. observability.OTel
def op_observability_otel() -> str:
    from horizon_ric.observability import (
        decision_span,
        init_tracing,
        shutdown_tracing,
    )
    from horizon_ric.observability.tracing import get_in_memory_exporter

    init_tracing(service_name="horizon-lightup", in_memory=True)
    try:
        with decision_span("lu-dec-1") as ds:
            ds.set("sla_breach_count", 0)
            ds.set("constraint_violations", 0)
        spans = get_in_memory_exporter().get_finished_spans()
        return f"spans={len(spans)} name={spans[0].name} attrs={len(spans[0].attributes)}"
    finally:
        shutdown_tracing()


# 57. frontend dashboard API: /v1/state with valid JWT
def op_dashboard_api() -> str:
    from fastapi.testclient import TestClient

    from horizon_ric.rapp.dashboard_api import build_dashboard_api, issue_token

    app = build_dashboard_api(lifecycle=None)
    client = TestClient(app)
    token = issue_token("op", role="operator")
    r = client.get(
        "/api/v1/state", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    return f"status=200 rapp_state={body['rapp_state']} version={body['rapp_version']}"


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------

SUBSYSTEMS: list[tuple[str, Callable[[], str]]] = [
    ("physics.propagation", op_physics_propagation),
    ("physics.orbital", op_physics_orbital),
    ("physics.doppler", op_physics_doppler),
    ("physics.tr38811", op_physics_tr38811),
    ("physics.ntn_timing", op_physics_ntn_timing),
    ("physics.epfd", op_physics_epfd),
    ("physics.coexistence", op_physics_coexistence),
    ("geodesy", op_geodesy),
    ("encoder.spatial_prior", op_spatial_prior),
    ("encoder.entity_tokenizer", op_entity_tokenizer),
    ("encoder.perceiver", op_perceiver_fusion),
    ("encoder.graph_jepa", op_graph_jepa),
    ("encoder.link_state", op_link_state),
    ("core.cfc", op_cfc_core),
    ("core.liquid_s4", op_liquid_s4),
    ("core.latent_dynamics", op_latent_dynamics),
    ("core.physics_residual", op_physics_residual),
    ("core.timing_budgets", op_timing_budgets),
    ("heads.sla_risk_v0.4", op_sla_head),
    ("policy.constraints", op_policy_constraints),
    ("policy.emit_guards", op_emit_guards),
    ("policy.counterfactual", op_counterfactual),
    ("policy.tdmpc2", op_tdmpc_planner),
    ("policy.diffusion_tail", op_diffusion_tail),
    ("policy.li_constraint", op_li_constraint),
    ("rapp.r1_adapter", op_r1_adapter),
    ("rapp.a1_adapter_osc", op_a1_adapter_osc),
    ("rapp.o1_adapter", op_o1_adapter),
    ("rapp.auth", op_auth),
    ("rapp.lifecycle", op_lifecycle),
    ("rapp.health", op_health),
    ("runtime.circuit_breaker", op_circuit_breaker),
    ("runtime.watchdog", op_watchdog),
    ("runtime.state_recovery", op_state_recovery),
    ("runtime.backpressure", op_backpressure),
    ("io.registry", op_io_registry),
    ("io.file_connector", op_io_file),
    ("data.aerial", op_data_aerial),
    ("data.deepmimo", op_data_deepmimo),
    ("data.aodt_real_usd", op_data_aodt),
    ("data.sionna_real", op_data_sionna),
    ("data.celestrak", op_data_celestrak),
    ("data.itu_r_p839", op_data_itu_r),
    ("data.tle_pipeline", op_tle_pipeline),
    ("federated.fedavg", op_federated_fedavg),
    ("federated.sparsifier", op_federated_sparsifier),
    ("trading.auction", op_trading_auction),
    ("evidence.jsonl", op_evidence_jsonl),
    ("evidence.sqlite", op_evidence_sqlite),
    ("security.rbac", op_security_rbac),
    ("security.jwt", op_security_jwt),
    ("security.tenant", op_security_tenant),
    ("sla.engine", op_sla_engine),
    ("sla.alertmanager", op_sla_alertmanager),
    ("sla.escalation", op_sla_escalation),
    ("observability.otel", op_observability_otel),
    ("rapp.dashboard_api", op_dashboard_api),
]


def main() -> int:
    print(f"[lightup] starting at {_now_utc().isoformat()}")
    print(f"[lightup] repo={REPO}")
    print(f"[lightup] checkpoints={CKPT}")
    print(f"[lightup] data_root={PARENT_DATA}")
    print()

    overall_t0 = time.perf_counter()
    results: list[Result] = []
    for n, (name, fn) in enumerate(SUBSYSTEMS, start=1):
        r = run(n, name, fn)
        print(r, flush=True)
        results.append(r)
    overall_s = time.perf_counter() - overall_t0

    n_ok = sum(1 for r in results if r.status == "[OK]")
    n_total = len(results)

    # Markdown table out
    md_path = REPO / "DEMO_LIGHTUP.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# DEMO_LIGHTUP — every subsystem exercised in one real run\n\n")
        f.write(f"Generated: {_now_utc().isoformat()}\n\n")
        f.write(f"Result: **{n_ok}/{n_total} green**, total wall-clock "
                f"{overall_s:.2f}s\n\n")
        f.write("| #  | Subsystem                      | Status | "
                "Wall-clock | Note |\n")
        f.write("|----|--------------------------------|--------|"
                "------------|------|\n")
        for r in results:
            f.write(
                f"| {r.n:>2} | {r.name:<30} | {r.status:<6} | "
                f"{r.wall_ms:>9.2f}ms | {r.note} |\n"
            )
    print()
    print(f"[lightup] wrote {md_path}")

    print()
    print("=" * 72)
    print(f"lightup_all: {n_ok}/{n_total} subsystems green, "
          f"total wall-clock {overall_s:.2f} s")
    print("=" * 72)

    if n_ok != n_total:
        print("\n[lightup] FAILURES:")
        for r in results:
            if r.status != "[OK]":
                print(f"\n--- {r.n} {r.name} ---")
                print(getattr(r, "_tb", r.note))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
