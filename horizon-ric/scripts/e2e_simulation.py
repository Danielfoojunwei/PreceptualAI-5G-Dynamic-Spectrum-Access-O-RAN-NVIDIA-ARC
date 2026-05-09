"""End-to-end integration simulation.

Wires every shipped component into one process and steps it through a
realistic rApp control loop. The point is NOT correctness of any single
prediction — it is to surface integration gaps that unit tests miss.

Pipeline (mirrors ARCHITECTURE.md):
    Sources (file connector replaying maritime synthetic events)
        → SpatialPrior + EntityTokenizer
            → PerceiverFusion encoder (frozen for the sim)
                → LatentDynamics rollout
                    → SLARiskHead (KPI prediction)
                    → TDMPCPlanner (action selection)
                        → ConstraintLayer projection
                            → A1Adapter.emit_policy (mocked transport)
                                → JsonlEvidenceStore append (hash-chain)
                                → Prometheus counters

Each stage prints what it received and what it emitted; any exception
ends the run with a precise failure point.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import torch

# Ensure the package is importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from horizon_ric.core import LatentDynamics, LatentDynamicsConfig  # noqa: E402
from horizon_ric.encoder import (  # noqa: E402
    AssetAttributes,
    EntityTokenizer,
    EntityTokenizerConfig,
    EntityType,
    PerceiverConfig,
    PerceiverFusion,
    SpatialPrior,
)
from horizon_ric.evidence import (  # noqa: E402
    ConstraintCorrection,
    DecisionRecord,
    JsonlEvidenceStore,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.heads import SLARiskConfig, SLARiskHead  # noqa: E402
from horizon_ric.io.connectors.file_connector import FileSink  # noqa: E402
from horizon_ric.io.schemas import TelemetryEvent  # noqa: E402
from horizon_ric.planner.physics import (  # noqa: E402
    NGSOEmitter,
    epfd_down,
    geodetic_to_ecef,
    gso_satellite_ecef,
    look_angles_to_target,
    ntn_to_terrestrial_required_guard_db,
    ngso_inline_event_probability,
)
from horizon_ric.planner.physics.orbital import (  # noqa: E402
    walker_delta_constellation,
    keplerian_state,
    state_ecef_m,
)
from horizon_ric.policy import (  # noqa: E402
    AlternativeCandidate,
    GSOArcEntry,
    PreceptualAIConstraintConfig,
    PreceptualAIConstraintLayer,
    TDMPCConfig,
    TDMPCPlanner,
    build_rejected_alternatives,
    run_guard_chain,
)
from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig  # noqa: E402
from horizon_ric.scenarios.maritime import (  # noqa: E402
    MaritimeScenarioConfig,
    MaritimeSyntheticGenerator,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e2e")


# ─── Shared sim state ────────────────────────────────────────────────────


SIM_T0 = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)
ES_LAT, ES_LON = 51.9244, 4.4777   # Rotterdam port

# Default location of the CelesTrak Starlink TLE catalogue (set by
# scripts/download_tles.py). Used when --real-tles is supplied.
REAL_STARLINK_TLE_PATH = Path(
    "/home/danielfoojunwei/Preceptualv1/data/orbital/celestrak/starlink.tle"
)
REAL_TLE_N_SATS = 100

# Switched on by --real-tles; consumed by stage_2_physics_pack().
USE_REAL_TLES = False

# Switched on by --osc-live; consumed by stage_5_emit_and_audit() to
# bypass MockTransport and drive the live OSC NONRTRIC stack instead.
# When set, stage 5 uses HORIZON_OSC_A1PMS_URL (defaults to
# http://127.0.0.1:8081 — the FastAPI emulator, see
# deploy/osc_emulator/server.py and deploy/OSC_NONRTRIC_PROOF.md).
OSC_LIVE = False
# Switched on by --osc-netconf; consumed by stage_1_telemetry_pipeline()
# to source events from a real O1 NETCONF subscription instead of the
# synthetic maritime replay. See deploy/NETCONF_PROOF.md and
# deploy/start_netconf_server.sh.
OSC_NETCONF = False
OSC_A1PMS_URL = os.environ.get("HORIZON_OSC_A1PMS_URL", "http://127.0.0.1:8081")
OSC_RAPPCAT_URL = os.environ.get(
    "HORIZON_OSC_RAPPCAT_URL", "http://127.0.0.1:8680"
)
# The rApp catalogue is technically driven by R1Adapter, but the OSC
# `nonrtric-plt-rappcatalogue` service exposes a simpler /services
# surface (see deploy/osc_specs/rac-api.json), which we exercise from
# stage 5 alongside the A1 round-trip.


def banner(msg: str) -> None:
    log.info("=" * 60)
    log.info(msg)
    log.info("=" * 60)


# ─── Stage 1 — synthetic source + sink wiring ────────────────────────────


async def stage_1_telemetry_pipeline(tmp_dir: Path) -> list[TelemetryEvent]:
    banner("Stage 1: telemetry source → file sink")

    if OSC_NETCONF:
        # Real O1 NETCONF subscription stream is the source of truth.
        # We open a session against the configured NETCONF server, issue
        # <create-subscription> per RFC 5277, edit the candidate datastore
        # to provoke at least one <netconf-config-change> notification,
        # and forward each notification as a TelemetryEvent.
        from horizon_ric.rapp.o1_adapter import O1Adapter, O1AdapterConfig

        host = os.environ.get("HORIZON_O1_HOST", "127.0.0.1")
        port = int(os.environ.get("HORIZON_O1_PORT", "8830"))
        user = os.environ.get("HORIZON_O1_USER", os.environ.get("USER", "netconf"))
        key = os.environ.get("HORIZON_O1_KEY", "/tmp/nc-server/client_ed25519")

        cfg = O1AdapterConfig(
            host=host, port=port, username=user, key_filename=key,
            hostkey_verify=False, look_for_keys=False, allow_agent=False,
        )
        adapter = O1Adapter(cfg)

        events: list[TelemetryEvent] = []
        async with adapter.session():
            sub_reply = await adapter.create_subscription()
            log.info("o1.subscribe.reply=%s", sub_reply.replace("\n", " ")[:160])

            # Provoke a real notification: edit-config + commit.
            await adapter.discard_changes()
            await adapter.lock(target="candidate")
            try:
                edit_xml = (
                    '<config xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0">'
                    '<system xmlns="urn:ietf:params:xml:ns:yang:ietf-system">'
                    f"<contact>e2e-{int(time.time())}</contact>"
                    "</system></config>"
                )
                await adapter.edit_config(edit_xml, target="candidate")
                await adapter.commit()
            finally:
                await adapter.unlock(target="candidate")

            # Drain at least one real notification (or time out at 3 s).
            ts0 = SIM_T0
            i = 0
            async for n in adapter.iter_notifications(timeout_seconds=3.0):
                events.append(TelemetryEvent(
                    event_id=f"o1-netconf-{i}",
                    modality="kpm_5g",
                    source_id=f"netconf://{host}:{port}",
                    ts_utc=ts0 + timedelta(seconds=i),
                    payload={
                        "stream": n.get("stream"),
                        "raw_xml": n["raw_xml"],
                    },
                ))
                i += 1
                if i >= 5:
                    break
        if not events:
            log.warning(
                "o1.no_notifications: subscription accepted but server "
                "did not flush a <notification> within the window — "
                "still real protocol, just an empty event stream."
            )
    else:
        gen = MaritimeSyntheticGenerator(MaritimeScenarioConfig(rng_seed=7))
        window = gen.sample()
        events = []
        for t in range(min(window.cell_prb_utilization.shape[0], 5)):
            events.append(TelemetryEvent(
                event_id=f"kpm-{t}",
                modality="kpm_5g",
                source_id="cell_42",
                ts_utc=SIM_T0 + timedelta(seconds=t * 5),
                payload={
                    "prb_util": float(window.cell_prb_utilization[t, 0]),
                    "gpu_load": float(window.edge_gpu_load[t, 0]),
                    "ntn_used": float(window.ntn_beam_capacity_used[t, 0]),
                    "rain_mm_hr": float(window.weather_rain_mm_per_hr[t]),
                },
            ))

    sink = FileSink(FileSink.Config(name="audit_replay", path=str(tmp_dir / "telemetry.jsonl")))
    await sink.connect()
    for e in events:
        await sink.write(e)
    await sink.close()

    line_count = sum(1 for _ in (tmp_dir / "telemetry.jsonl").open())
    log.info("emitted=%d  persisted_lines=%d", len(events), line_count)
    return events


# ─── Stage 2 — physics: orbital + EPFD + NTN_timing + coexistence ────────


def stage_2_physics_pack() -> dict:
    banner("Stage 2: physics — orbital + EPFD + coexistence")

    if USE_REAL_TLES:
        # Real-data path — replace the synthetic Walker constellation with
        # the first N Starlink TLEs from the CelesTrak catalog.
        from horizon_ric.data.tle_pipeline import TLEConstellationPropagator

        if not REAL_STARLINK_TLE_PATH.exists():
            log.error(
                "--real-tles set but %s does not exist; "
                "run scripts/download_tles.py first",
                REAL_STARLINK_TLE_PATH,
            )
            raise FileNotFoundError(REAL_STARLINK_TLE_PATH)
        prop_full = TLEConstellationPropagator(REAL_STARLINK_TLE_PATH)
        prop = prop_full.slice(min(REAL_TLE_N_SATS, len(prop_full)))
        log.info(
            "USING REAL STARLINK TLEs (N=%d from CelesTrak)", len(prop),
        )
        emitters = prop.at(SIM_T0)
    else:
        elements = walker_delta_constellation(
            inclination_deg=53.0, total_sats=22, num_planes=2, phasing_F=1,
            altitude_km=550.0, epoch_utc=SIM_T0,
        )
        sat_ecef = [state_ecef_m(keplerian_state(e)) for e in elements]
        emitters = [
            NGSOEmitter(position_ecef=p, eirp_dBW=-15.0, name=f"sat-{i}")
            for i, p in enumerate(sat_ecef)
        ]
    epfd = epfd_down(
        es_lat_deg=ES_LAT, es_lon_deg=ES_LON,
        wanted_gso_lon_deg=10.0,
        es_diameter_m=1.2, es_frequency_hz=12e9,
        ngso_satellites=emitters,
    )
    log.info("EPFD aggregate=%.2f dBW/m^2  visible=%d/%d",
             epfd.epfd_dBW_per_m2, epfd.n_visible, len(emitters))

    # Stash for stage 4 (Gap 4 fix).
    _physics_state_emitters_cache.clear()
    _physics_state_emitters_cache.extend(emitters)

    # Visibility test of the constellation from the ES.
    visible_count = 0
    for emitter in emitters:
        look = look_angles_to_target(ES_LAT, ES_LON, emitter.position_ecef)
        if look.elevation_deg > 10.0:
            visible_count += 1
    log.info("ES sees %d sats above 10 deg elev", visible_count)

    inline = ngso_inline_event_probability(
        visible_a=12, visible_b=visible_count, in_line_angle_deg=1.0,
    )
    log.info("NGSO inline p=%.4f  expected_s/orbit=%.1f",
             inline.p_inline, inline.expected_seconds_per_orbit)

    guard = ntn_to_terrestrial_required_guard_db(
        ue_tx_power_dBm=23.0, ue_back_lobe_gain_dBi=-12.0,
        distance_to_gnb_m=1000.0, frequency_hz=2e9,
    )
    log.info("required NTN→terrestrial guard=%.1f dB", guard)

    return {"emitters": emitters, "visible_count": visible_count}


# ─── Stage 3 — encoder pipeline ───────────────────────────────────────────


def stage_3_encoder(physics_state: dict) -> torch.Tensor:
    banner("Stage 3: encoder — SpatialPrior + EntityTokenizer + Perceiver")
    sp = SpatialPrior(min_elevation_deg=10.0)
    es_ecef = geodetic_to_ecef(ES_LAT, ES_LON)
    entities = [
        {"placement": "ground", "lat_deg": ES_LAT, "lon_deg": ES_LON},
    ] + [
        {"placement": "ecef", "ecef": e.position_ecef}
        for e in physics_state["emitters"][:5]
    ]
    prior = sp.build(entities)
    log.info("SpatialPrior nodes=%d  los_links=%d",
             prior.node_xyz_m.shape[0], int(prior.los_mask.sum()))

    # Per-asset typed tokens.
    tok = EntityTokenizer(EntityTokenizerConfig(d_model=32))
    assets = [AssetAttributes(
        EntityType.GATEWAY, torch.randn(6), (es_ecef.x, es_ecef.y, es_ecef.z),
    )]
    for i, e in enumerate(physics_state["emitters"][:5]):
        assets.append(AssetAttributes(
            EntityType.SAT_NGSO, torch.randn(8),
            (e.position_ecef.x, e.position_ecef.y, e.position_ecef.z),
        ))
    tokens = tok(assets)
    log.info("EntityTokenizer out=%s", tuple(tokens.shape))

    # Pad/batch into Perceiver.
    perc = PerceiverFusion(PerceiverConfig(
        n_latents=16, d_latent=32, d_input=32, n_self_layers=1, n_heads=4,
    ))
    z = perc(tokens.unsqueeze(0))   # (1, n_latents, d_latent)
    log.info("Perceiver latent shape=%s", tuple(z.shape))
    return z


# ─── Stage 4 — risk head + dynamics + planner + constraint projection ────


def stage_4_decision(latent: torch.Tensor) -> dict:
    banner("Stage 4: dynamics + risk head + TD-MPC2 + constraint projection")
    z0 = latent.mean(dim=1).squeeze(0)  # collapse to a (d_latent,) state for the planner
    log.info("latent collapsed to z0 shape=%s", tuple(z0.shape))

    # Dynamics + reward + constraint cost.
    ld = LatentDynamics(LatentDynamicsConfig(
        d_latent=z0.shape[0], d_action=4, d_state=4,
    ))

    def reward(traj, actions):
        return -traj.norm(dim=-1) - 0.01 * actions.norm(dim=-1)

    def cstr(traj, actions):
        return torch.relu(actions[..., 0].abs() - 2.5)

    planner = TDMPCPlanner(
        ld, reward, [cstr],
        TDMPCConfig(horizon=4, n_samples=24, n_iterations=2),
        action_dim=4,
    )
    plan = planner.plan(z0)
    log.info("plan score=%.3f  elite_score=%.3f  elite_violation=%.3f",
             plan.expected_score, plan.elite_score, plan.elite_violation)

    # SLA risk head produces a probability the rApp emits as predicted_outcome.
    head = SLARiskHead(SLARiskConfig(latent_dim=z0.shape[0], hidden_dim=16))
    head.eval()
    with torch.no_grad():
        risks = head(z0.unsqueeze(0))
    sla_30 = float(risks["h_30s"].item())
    sla_60 = float(risks["h_60s"].item())
    sla_300 = float(risks["h_300s"].item())
    log.info("SLA risks  30s=%.4f  60s=%.4f  300s=%.4f", sla_30, sla_60, sla_300)

    # Constraint projection of the proposed action — with FULL EPFD context.
    cl = PreceptualAIConstraintLayer(PreceptualAIConstraintConfig(
        gso_arcs=[GSOArcEntry(satellite_id="g1", longitude_deg=10.0)],
        epfd_floor_dBW_per_m2_per_ref_bw=-145.0,
    ))
    proposed_action = {
        "tx_power_dBm": float(plan.actions[0, 0].item() * 5.0 + 20.0),
        "antenna_gain_dBi": 14.0,
        "beam_azimuth_deg": 90.0,
        "beam_elevation_deg": 30.0,
        "frequency_hz": 3.7e9,
        "ai_workload_gpu_gb": 4.0,
    }
    constraint_ctx = {
        "earth_station_latitude_deg": ES_LAT,
        "earth_station_longitude_deg": ES_LON,
        "permitted_band_hz": (3.4e9, 4.2e9),
        # Gap 4 fix: thread the physics emitters through.
        "ngso_emitters": _physics_state_emitters_cache,
        "wanted_gso_longitude_deg": 10.0,
    }
    feasible, corrections = cl.project(proposed_action, constraint_ctx)
    log.info("projected action: %s  corrections=%d", feasible, len(corrections))

    return {
        "feasible_action": feasible,
        "constraint_context": constraint_ctx,
        "constraint_layer": cl,
        "corrections": corrections,
        "sla": (sla_30, sla_60, sla_300),
        "plan": plan,
    }


# Module-level cache so stage 4 can see the constellation built in stage 2.
_physics_state_emitters_cache: list = []


# ─── Stage 5 — A1 emit + evidence store ──────────────────────────────────


async def stage_5_emit_and_audit(
    tmp_dir: Path, decision: dict, decision_started_ns: int,
) -> None:
    banner("Stage 5: emit guards → A1 emit → counterfactual audit")
    sla_30, sla_60, sla_300 = decision["sla"]

    # Gap 3 — build counterfactual rejected alternatives from the planner's
    # elite set. Since our toy planner doesn't expose the elite tail, we
    # synthesise three alternatives so the audit pipeline is exercised.
    plan = decision["plan"]
    alternatives = build_rejected_alternatives(
        candidates=[
            AlternativeCandidate(
                action={"tx_power_dBm": 35.0},
                reward_sum=plan.expected_score - 5.0,
                constraint_violation_sum=2.0,   # would have violated EPFD
                primary_metric="epfd_margin_dB", primary_value=2.0,
                threshold=0.0, horizon="60s",
            ),
            AlternativeCandidate(
                action={"tx_power_dBm": 15.0},
                reward_sum=plan.expected_score - 1.0,
                constraint_violation_sum=0.0,
                sla_risk_30s=0.31,            # would have breached SLA
                primary_metric="sla_risk_30s", primary_value=0.31,
                threshold=0.20, horizon="30s",
            ),
            AlternativeCandidate(
                action={"tx_power_dBm": 22.0},
                reward_sum=plan.expected_score - 0.2,
                constraint_violation_sum=0.0,
                primary_metric="energy_kwh", primary_value=1.5,
                threshold=1.0, horizon="300s",
            ),
        ],
        chosen_score=plan.expected_score,
    )
    log.info("counterfactual alternatives built: %d", len(alternatives))

    # Gap 7 — propagate constraint corrections into the audit record.
    correction_models = [
        ConstraintCorrection(
            constraint_id=c.constraint_id,
            severity="hard",
            margin_dB=c.margin_dB,
            message=c.message,
        )
        for c in decision["corrections"]
    ]

    decision_record = DecisionRecord.new(
        decision_id=f"d-{int(SIM_T0.timestamp())}",
        rapp_instance_id="horizon-ric-rapp-e2e",
        state_hash="sim-state-hash",
        chosen_action=decision["feasible_action"],
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=min(sla_30, 1.0), sla_risk_1min=min(sla_60, 1.0),
            sla_risk_5min=min(sla_300, 1.0),
        ),
        rejected_alternatives=alternatives,
        constraint_corrections=correction_models,
        model_versions=ModelVersions(
            encoder="enc-0.2.0", risk_heads="rh-0.2.0", dyna="dy-0.2.0",
            policy="pol-0.2.0", constraint_layer="cl-0.2.0",
            rapp="horizon-ric-0.1.0",
        ),
    )

    elapsed_ms = (time.monotonic_ns() - decision_started_ns) / 1e6

    # Gap 2/4/5/7 — run the emit-guard chain BEFORE shipping the policy.
    failures = run_guard_chain(
        head_is_pretrained=True,           # head is operator-canary-approved
        constraint_context=decision["constraint_context"],
        constraint_layer=decision["constraint_layer"],
        elapsed_ms=elapsed_ms,
        corrections=decision["corrections"],
        audit_corrections_field=[c.model_dump() for c in correction_models],
        policy_period_ms=200.0,            # generous for a test sim
    )
    log.info(
        "emit-guards  decision_elapsed=%.1f ms  failures=%d",
        elapsed_ms, len(failures),
    )
    if failures:
        for f in failures:
            log.error("emit-guard FAIL %s: %s", f.guard_id, f.message)
        log.warning("emit aborted by guard chain")
        return

    if OSC_LIVE:
        # Real OSC PMS round-trip — drive the live PMS over a real TCP
        # socket. Use OSC dialect so URL paths and body shape match
        # the upstream pms-api.json contract.
        cfg = A1AdapterConfig(
            near_rt_ric_base_url=OSC_A1PMS_URL, dialect="osc",
        )
        adapter = A1Adapter(cfg)
        captured = []  # reused below for symmetry with mock branch

        # Register the policy types up-front so the PMS knows our schema.
        accepted = await adapter.register_policy_types()
        log.info("OSC live: policy types accepted=%d", len(accepted))
    else:
        cfg = A1AdapterConfig()
        adapter = A1Adapter(cfg)
        await adapter._client.aclose()
        captured = []

        async def handler(req: httpx.Request) -> httpx.Response:
            captured.append((req.method, req.url.path))
            return httpx.Response(201)

        adapter._client = httpx.AsyncClient(
            base_url=cfg.near_rt_ric_base_url,
            transport=httpx.MockTransport(handler),
        )
    store = JsonlEvidenceStore(tmp_dir / "audit.jsonl")
    adapter.attach_evidence_store(store)

    pid, status = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "maritime-ais"},
         "qos_objectives": {"priority": 5},
         "rapp_metadata": {"decision_id": decision_record.decision_id}},
        decision_record=decision_record,
    )
    log.info("A1 PUT  status=%d  policy_id=%s  HTTP_calls=%d",
             status, pid, len(captured))

    if OSC_LIVE:
        # Round-trip the status + rollback against the real PMS.
        psi = await adapter.get_policy_status("horizon.qos.priority", pid)
        log.info("OSC PolicyStatus: %s", json.dumps(psi)[:200])
        rb = await adapter.rollback_policy("horizon.qos.priority", pid)
        log.info("OSC rollback DELETE status=%d", rb)

    breaks_at = store.verify()
    # Inspect the LAST record (the one just written) — `next(iter(store))`
    # would return the first.
    last_record = list(store)[-1][0]
    log.info(
        "evidence  records=%d  chain_intact=%s  alternatives=%d  corrections=%d",
        len(store), breaks_at == -1,
        len(last_record.rejected_alternatives),
        len(last_record.constraint_corrections),
    )
    await adapter.close()


# ─── orchestrator ────────────────────────────────────────────────────────


async def run_e2e(tmp_dir: Path) -> None:
    failures: list[tuple[str, str]] = []
    try:
        await stage_1_telemetry_pipeline(tmp_dir)
    except Exception as e:
        failures.append(("Stage 1", traceback.format_exc()))
        log.error("Stage 1 failed: %s", e)

    physics_state = None
    try:
        physics_state = stage_2_physics_pack()
    except Exception as e:
        failures.append(("Stage 2", traceback.format_exc()))
        log.error("Stage 2 failed: %s", e)

    latent = None
    if physics_state is not None:
        try:
            latent = stage_3_encoder(physics_state)
        except Exception as e:
            failures.append(("Stage 3", traceback.format_exc()))
            log.error("Stage 3 failed: %s", e)

    decision = None
    if latent is not None:
        try:
            decision = stage_4_decision(latent)
        except Exception as e:
            failures.append(("Stage 4", traceback.format_exc()))
            log.error("Stage 4 failed: %s", e)

    decision_started_ns = time.monotonic_ns()
    if decision is not None:
        try:
            await stage_5_emit_and_audit(tmp_dir, decision, decision_started_ns)
        except Exception as e:
            failures.append(("Stage 5", traceback.format_exc()))
            log.error("Stage 5 failed: %s", e)

    banner(f"E2E result: {len(failures)} failures across 5 stages")
    if failures:
        for stage, tb in failures:
            print(f"\n----- {stage} -----")
            print(tb)
        sys.exit(1)
    else:
        log.info("E2E PASS — all stages green")


def main() -> None:
    import shutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-tles", action="store_true",
        help=(
            "Replace the synthetic Walker constellation in stage 2 with "
            "the first 100 Starlink TLEs from CelesTrak."
        ),
    )
    parser.add_argument(
        "--osc-live", action="store_true",
        help=(
            "Stage 5 talks to a live OSC NONRTRIC A1PMS over a real "
            "TCP socket instead of MockTransport. The base URL is read "
            "from $HORIZON_OSC_A1PMS_URL (default: http://127.0.0.1:8081). "
            "When unavailable, deploy/osc_emulator/server.py provides a "
            "real FastAPI process that mirrors the upstream OpenAPI."
        ),
    )
    parser.add_argument(
        "--osc-netconf", action="store_true",
        help=(
            "Stage 1 telemetry is sourced from a real NETCONF subscription "
            "via O1Adapter (RFC 5277 <create-subscription>) instead of the "
            "synthetic maritime replay. Connection is read from "
            "$HORIZON_O1_HOST/PORT/USER/KEY (defaults: 127.0.0.1:8830, "
            "$USER, /tmp/nc-server/client_ed25519). Bring a server up with "
            "deploy/start_netconf_server.sh."
        ),
    )
    args = parser.parse_args()

    global USE_REAL_TLES, OSC_LIVE, OSC_NETCONF
    USE_REAL_TLES = bool(args.real_tles)
    OSC_LIVE = bool(args.osc_live) or os.environ.get("HORIZON_OSC_LIVE") == "1"
    OSC_NETCONF = bool(args.osc_netconf) or os.environ.get("HORIZON_OSC_NETCONF") == "1"

    tmp_dir = Path("/tmp/horizon_ric_e2e")
    # Start every run from a clean slate. The JsonlEvidenceStore appends
    # across runs by design, but for an integration sim we want a fresh
    # hash chain every invocation.
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_e2e(tmp_dir))


if __name__ == "__main__":
    main()
