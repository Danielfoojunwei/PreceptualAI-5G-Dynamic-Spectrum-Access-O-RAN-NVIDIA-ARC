"""Tests for the Sionna PHY → Horizon telemetry seam.

The pure-Python half (measurement → telemetry → risk → pipeline) runs
everywhere. The link-level simulation half needs the heavy ``phy`` extra
(sionna/tensorflow), so those tests ``importorskip`` and are skipped in the
default CI env — they run where the extra is installed.
"""

from __future__ import annotations

import asyncio

import pytest

from horizon_ric.phy.sionna_bridge import (
    PHY_RISK_MODEL,
    PhyMeasurement,
    SionnaPhyBridge,
    phy_measurements_to_pipeline,
)


def _measurement(**kw) -> PhyMeasurement:
    base = dict(
        receiver_index=1,
        position_m=(1.0, 2.0, 1.5),
        post_eq_sinr_db=25.0,
        coded_bler=0.0,
        throughput_mbps=400.0,
        spectral_eff_bps_hz=4.0,
        prb_util_pct=50.0,
        mcs_bits_per_symbol=4.0,
        n_tx=2,
        n_rx=2,
        ebn0_db=10.0,
    )
    base.update(kw)
    return PhyMeasurement(**base)


def test_healthy_link_is_low_risk() -> None:
    ev = SionnaPhyBridge().to_telemetry(_measurement())
    assert ev.modality == "ue_qos"
    assert ev.payload["risk_model"] == PHY_RISK_MODEL
    assert ev.payload["channel_source"] == "deepmimo_ray_tracing"
    assert ev.payload["mimo"] == "2x2"
    assert 0.0 <= ev.payload["sla_risk_30s"] < 0.35


def test_degraded_link_is_high_risk() -> None:
    ev = SionnaPhyBridge().to_telemetry(
        _measurement(post_eq_sinr_db=-2.0, coded_bler=0.8, spectral_eff_bps_hz=0.1)
    )
    assert ev.payload["sla_risk_30s"] > 0.7


def test_risk_is_monotonic_in_bler() -> None:
    b = SionnaPhyBridge()
    lo = b.to_telemetry(_measurement(coded_bler=0.0)).payload["sla_risk_30s"]
    mid = b.to_telemetry(_measurement(coded_bler=0.1)).payload["sla_risk_30s"]
    hi = b.to_telemetry(_measurement(coded_bler=0.9)).payload["sla_risk_30s"]
    assert lo <= mid <= hi


def test_risk_bounded_and_finite() -> None:
    b = SionnaPhyBridge()
    for sinr in (-50.0, 0.0, 50.0):
        for bler in (0.0, 0.5, 1.0):
            r = b.to_telemetry(
                _measurement(post_eq_sinr_db=sinr, coded_bler=bler)
            ).payload["sla_risk_30s"]
            assert 0.0 <= r <= 1.0


def test_sequence_increments() -> None:
    b = SionnaPhyBridge()
    e0 = b.to_telemetry(_measurement())
    e1 = b.to_telemetry(_measurement())
    assert e1.sequence == e0.sequence + 1


def test_phy_measurements_drive_pipeline_end_to_end() -> None:
    """A healthy and a degraded receiver both produce a guarded decision."""
    import httpx

    from horizon_ric.evidence.store import JsonlEvidenceStore
    from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig
    from horizon_ric.rapp.pipeline import DecisionPipeline, PipelineConfig

    async def _run(tmp_audit) -> list:
        captured: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(str(request.url))
            if request.url.path.endswith("/status"):
                return httpx.Response(200, json={"enforceStatus": "ENFORCED"})
            return httpx.Response(201)

        cfg = A1AdapterConfig(near_rt_ric_base_url="http://phy.test")
        adapter = A1Adapter(cfg)
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            base_url=cfg.near_rt_ric_base_url,
            transport=httpx.MockTransport(handler),
        )
        store = JsonlEvidenceStore(tmp_audit)
        pipeline = DecisionPipeline(adapter, store, PipelineConfig())
        measurements = [
            _measurement(receiver_index=1, post_eq_sinr_db=28.0, coded_bler=0.0),
            _measurement(
                receiver_index=2, post_eq_sinr_db=-1.0, coded_bler=0.7,
                spectral_eff_bps_hz=0.2,
            ),
        ]
        results = await phy_measurements_to_pipeline(measurements, pipeline)
        await adapter.close()
        return results

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        results = asyncio.run(_run(Path(d) / "audit.jsonl"))
    assert len(results) == 2
    # Both events were processed into DecisionRecords (accepted, blocked, or
    # dry-run) — the point is the PHY KPI reached a guarded A1 decision.
    assert all(r is not None for r in results)


# --- Link-level simulation (needs the heavy phy extra) ----------------------


def test_link_level_over_synthetic_mimo_channel_runs() -> None:
    """A tiny 2x2 MIMO link-level run must return finite, ordered KPIs.

    Skipped unless sionna/tensorflow are installed (the ``phy`` extra).
    """
    pytest.importorskip("sionna")
    pytest.importorskip("torch")
    import numpy as np

    from horizon_ric.phy.sionna_bridge import run_link_level_over_channels

    n_loc, n_sc, n_rx, n_tx = 2, 72, 2, 2
    rng = np.random.default_rng(0)
    # A strong, well-conditioned channel and a weak one — the strong one must
    # decode better (lower BLER) at the same Eb/N0.
    strong = (rng.normal(size=(1, n_sc, n_rx, n_tx))
              + 1j * rng.normal(size=(1, n_sc, n_rx, n_tx))).astype(np.complex64)
    weak = 0.05 * strong
    channels = np.concatenate([strong, weak], axis=0)
    kpis = run_link_level_over_channels(
        channels, ebn0_db=6.0, num_codewords=4, num_bits_per_symbol=2
    )
    assert len(kpis) == n_loc
    for k in kpis:
        assert all(
            k[f] == k[f] for f in ("post_eq_sinr_db", "coded_bler", "throughput_mbps")
        )  # not NaN
        assert 0.0 <= k["coded_bler"] <= 1.0
    assert kpis[0]["coded_bler"] <= kpis[1]["coded_bler"]
