#!/usr/bin/env python3
"""End-to-end: real DeepMIMO MIMO channels → Sionna link-level → Horizon A1.

Pipeline demonstrated:

    DeepMIMO ASU 3.5 GHz ray tracing (real MIMO channel matrices per Rx site)
      -> Sionna coded MIMO-OFDM link-level sim (LDPC + QAM + LMMSE)  [real PHY KPIs]
      -> horizon_ric.phy.SionnaPhyBridge  -> TelemetryEvent (ue_qos)
      -> horizon_ric.rapp.DecisionPipeline (planner -> Shield -> guards -> A1)
      -> A1 policy PUT to a real near-RT RIC A1 endpoint -> DecisionRecord

Run with the ``phy`` + ``realdata`` extras (see deploy/sionna-phy/README.md):

    CUDA_VISIBLE_DEVICES="" .venv-phy/bin/python deploy/sionna-phy/run_phy_pipeline.py \
        --scenario-dir ~/oran-deps/deepmimo-cache/deepmimo_scenarios/asu_campus_3p5 \
        --out deploy/sionna-phy/results/phy-e2e-proof.json

Without ``--a1-base-url`` the script starts a real loopback A1 endpoint
(uvicorn, ephemeral port) so the full chain runs self-contained; pass a URL
to drive a live mediator/simulator instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _compute_mimo_channels(
    scenario_dir: Path, n_tx: int, n_rx: int, n_sc: int, sites: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Real ray-traced MIMO channel matrices for ``sites`` receiver locations.

    Returns ``(H, meta)`` where ``H`` is complex ``[sites, n_sc, n_rx, n_tx]``
    and ``meta`` carries each site's index, position, and wideband path gain.
    Sites are chosen evenly across the gain distribution (strong→weak) so the
    downstream decisions are diverse rather than all-healthy.
    """
    import deepmimo as dm

    prev = dm.config.get("scenarios_folder")
    dm.config.set("scenarios_folder", str(scenario_dir.parent))
    try:
        census = dm.load(str(scenario_dir), matrices=["rx_pos", "power"], max_paths=10)
        valid = np.flatnonzero(np.any(np.isfinite(census.power), axis=1))
        # Rank valid receivers by summed linear path power (channel strength),
        # then sample fractional ranks biased toward the strong end so the run
        # spans the whole SNR gradient — the near-BS sites that stay healthy
        # through to the coverage-limited tail — rather than only the far tail.
        pw = np.nansum(np.where(np.isfinite(census.power), census.power, 0.0), axis=1)
        strong_first = valid[np.argsort(pw[valid])[::-1]]           # descending gain
        frac = np.linspace(0.0, 1.0, sites) ** 2                    # denser near strong end
        picks = strong_first[(frac * (len(strong_first) - 1)).astype(np.int64)]

        params = dm.ChannelParameters()
        params.bs_antenna.shape = [n_tx, 1]
        params.ue_antenna.shape = [n_rx, 1]
        params.num_paths = 10
        params.ofdm.subcarriers = 1024
        params.ofdm.selected_subcarriers = np.linspace(0, 1023, n_sc, dtype=np.int64)
        params.ofdm.bandwidth = 100e6
        params.doppler = False
        params.freq_domain = True

        dataset = dm.load(
            str(scenario_dir), rx_sets={0: picks}, matrices=[
                "rx_pos", "tx_pos", "power", "phase", "delay",
                "aoa_az", "aoa_el", "aod_az", "aod_el",
            ], max_paths=10,
        )
        ch = dataset.compute_channels(params)  # [sites, n_rx, n_tx, n_sc]
        H = np.transpose(ch, (0, 3, 1, 2)).astype(np.complex64)  # [sites, n_sc, n_rx, n_tx]
        meta = []
        for i, idx in enumerate(picks):
            gain = 10.0 * np.log10(max(float(np.mean(np.abs(H[i]) ** 2)), 1e-30))
            meta.append({
                "receiver_index": int(idx),
                "position_m": [round(float(v), 3) for v in dataset.rx_pos[i]],
                "wideband_gain_dbw": round(gain, 3),
            })
    finally:
        dm.config.set("scenarios_folder", prev)
    return H, meta


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_a1_stub(port: int) -> Any:
    """Start a real loopback near-RT RIC A1AP (legacy) endpoint via uvicorn."""
    import uvicorn
    from fastapi import FastAPI, Request

    app = FastAPI()
    seen: dict[str, Any] = {"types": [], "policies": []}

    @app.get("/A1-P/v2/healthcheck")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.put("/A1-P/v2/policytypes/{ptid}")
    async def put_type(ptid: int) -> dict[str, int]:
        seen["types"].append(ptid)
        return {"policy_type_id": ptid}

    @app.put("/A1-P/v2/policytypes/{ptid}/policies/{pid}")
    async def put_policy(ptid: int, pid: str, request: Request):
        seen["policies"].append({"ptid": ptid, "pid": pid, "body": await request.json()})
        from fastapi import Response
        return Response(status_code=202)

    @app.get("/A1-P/v2/policytypes/{ptid}/policies/{pid}/status")
    def status(ptid: int, pid: str) -> dict[str, str]:
        return {"enforceStatus": "ENFORCED"}

    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    import time
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, seen


async def _drive(H: np.ndarray, meta: list[dict], args) -> dict[str, Any]:
    from horizon_ric.evidence.store import JsonlEvidenceStore
    from horizon_ric.phy.sionna_bridge import (
        PHY_RISK_MODEL,
        PhyMeasurement,
        SionnaPhyBridge,
        run_link_level_over_channels,
    )
    from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig
    from horizon_ric.rapp.pipeline import DecisionPipeline, PipelineConfig

    # 1a. Per-site link budget: fold each site's real ray-traced path gain into
    #     an operating Eb/N0. rx SNR = Ptx + channel_gain - noise; noise power =
    #     -174 dBm/Hz + 10log10(B) + NF. Eb/N0 = SNR - 10log10(bits*rate).
    noise_dbm = -174.0 + 10.0 * np.log10(100e6) + args.noise_figure_db
    import math as _math
    es_to_eb = 10.0 * _math.log10(max(args.qam_bits * args.code_rate, 1e-9))
    ebn0_per_site = []
    for m in meta:
        rx_snr_db = args.tx_power_dbm + m["wideband_gain_dbw"] - noise_dbm
        ebn0_per_site.append(rx_snr_db - es_to_eb)

    # 1b. Real coded MIMO-OFDM link-level simulation over the ray-traced
    #     channels (spatial structure) at each site's operating Eb/N0.
    kpis = run_link_level_over_channels(
        H, ebn0_db=ebn0_per_site, num_bits_per_symbol=args.qam_bits,
        code_rate=args.code_rate, num_codewords=args.codewords,
    )

    # 2. Assemble PHY measurements (KPIs + provenance per receiver site).
    measurements = []
    for m, k in zip(meta, kpis):
        prb = float(np.clip(100.0 * (1.0 - k["coded_bler"]) * 0.9 + 5.0, 0, 100))
        measurements.append(PhyMeasurement(
            receiver_index=m["receiver_index"],
            position_m=tuple(m["position_m"]),
            post_eq_sinr_db=k["post_eq_sinr_db"],
            coded_bler=k["coded_bler"],
            throughput_mbps=k["throughput_mbps"],
            spectral_eff_bps_hz=k["spectral_eff_bps_hz"],
            prb_util_pct=prb,
            mcs_bits_per_symbol=k["mcs_bits_per_symbol"],
            n_tx=k["n_tx"], n_rx=k["n_rx"], ebn0_db=k["ebn0_db"],
            extra={"wideband_gain_dbw": m["wideband_gain_dbw"]},
        ))

    # 3. Drive through the real Horizon pipeline against a real A1 endpoint.
    audit = Path(args.audit)
    audit.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(audit)
    adapter = A1Adapter(A1AdapterConfig(
        near_rt_ric_base_url=args.a1_base_url, dialect=args.dialect,
    ))
    accepted = await adapter.register_policy_types()
    adapter.attach_evidence_store(store)
    pipeline = DecisionPipeline(adapter, store, PipelineConfig())

    bridge = SionnaPhyBridge()
    decisions = []
    for meas in measurements:
        ev = bridge.to_telemetry(meas)
        res = await pipeline.process_event(ev)
        decisions.append({
            "receiver_index": meas.receiver_index,
            "wideband_gain_dbw": meas.extra["wideband_gain_dbw"],
            "ebn0_db": meas.ebn0_db,
            "post_eq_sinr_db": round(meas.post_eq_sinr_db, 3),
            "coded_bler": round(meas.coded_bler, 5),
            "throughput_mbps": round(meas.throughput_mbps, 2),
            "sla_risk_30s": ev.payload["sla_risk_30s"],
            "policy_type": res.policy_type,
            "accepted": res.accepted,
            "blocked": res.blocked,
            "enforcement_status": res.enforcement_status,
            "policy_id": res.policy_id,
        })
    await adapter.close()

    n_acc = sum(1 for d in decisions if d["accepted"])
    n_enf = sum(1 for d in decisions if d["enforcement_status"] in ("ENFORCED", "IN EFFECT"))
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "system_under_test": (
            "DeepMIMO ray-traced MIMO channels -> Sionna link-level PHY -> "
            "Horizon SionnaPhyBridge -> DecisionPipeline -> A1"
        ),
        "channel_source": {
            "dataset": "DeepMIMO ASU Campus 3.5 GHz (Wireless InSite ray tracing)",
            "mimo": f"{args.n_tx}x{args.n_rx}", "subcarriers": args.subcarriers,
            "sites": len(meta),
        },
        "phy_sim": {
            "engine": "NVIDIA Sionna (PyTorch)", "qam_bits": args.qam_bits,
            "code_rate": args.code_rate,
            "link_budget": {
                "tx_power_dbm": args.tx_power_dbm,
                "noise_figure_db": args.noise_figure_db,
                "ebn0_db_min": round(min(ebn0_per_site), 2),
                "ebn0_db_max": round(max(ebn0_per_site), 2),
            },
            "codewords_per_site": args.codewords, "risk_model": PHY_RISK_MODEL,
        },
        "a1": {"base_url": args.a1_base_url, "dialect": args.dialect,
               "registered_policy_types": accepted},
        "summary": {
            "sites": len(decisions), "accepted": n_acc, "enforced": n_enf,
            "blocked": sum(1 for d in decisions if d["blocked"]),
            "sinr_db_min": round(min(d["post_eq_sinr_db"] for d in decisions), 2),
            "sinr_db_max": round(max(d["post_eq_sinr_db"] for d in decisions), 2),
            "bler_min": round(min(d["coded_bler"] for d in decisions), 5),
            "bler_max": round(max(d["coded_bler"] for d in decisions), 5),
            "risk_min": min(d["sla_risk_30s"] for d in decisions),
            "risk_max": max(d["sla_risk_30s"] for d in decisions),
            "audit_chain_length": len(store),
            "audit_verify_first_broken_index": store.verify(),
        },
        "decisions": decisions,
        "result": "pass" if (n_acc >= 1 and store.verify() == -1) else "fail",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", type=Path, required=True)
    p.add_argument("--a1-base-url", default=None)
    p.add_argument("--dialect", default="legacy")
    # Default 1x4: single spatial layer with 4-antenna receive combining over
    # the real ray-traced channel — SNR-limited, the regime a scheduler assigns
    # coverage-limited UEs, giving a clean PHY-quality → decision gradient.
    # --n-tx 2 exercises 2-stream spatial multiplexing (surfaces the real MIMO
    # rank limitation of closely-spaced correlated BS antennas).
    p.add_argument("--n-tx", type=int, default=1)
    p.add_argument("--n-rx", type=int, default=4)
    p.add_argument("--subcarriers", type=int, default=60)
    p.add_argument("--sites", type=int, default=8)
    p.add_argument("--qam-bits", type=int, default=4)
    p.add_argument("--code-rate", type=float, default=0.5)
    p.add_argument("--codewords", type=int, default=12)
    p.add_argument("--tx-power-dbm", type=float, default=49.0,
                   help="Per-cell EIRP for the link budget (macro downlink).")
    p.add_argument("--noise-figure-db", type=float, default=7.0)
    p.add_argument("--audit", default="deploy/sionna-phy/results/phy-audit.jsonl")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    H, meta = _compute_mimo_channels(
        args.scenario_dir, args.n_tx, args.n_rx, args.subcarriers, args.sites
    )

    server = None
    if not args.a1_base_url:
        port = _free_port()
        server, _ = _start_a1_stub(port)
        args.a1_base_url = f"http://127.0.0.1:{port}"

    try:
        report = asyncio.run(_drive(H, meta, args))
    finally:
        if server is not None:
            server.should_exit = True

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
