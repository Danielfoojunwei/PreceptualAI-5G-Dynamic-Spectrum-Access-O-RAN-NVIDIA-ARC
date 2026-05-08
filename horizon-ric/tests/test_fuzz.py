"""Hypothesis-driven fuzz tests for PreceptualAI public API surface.

Each property test asserts a *real* invariant. Anything that fails here is
a real bug — not a flake. See tests/test_known_bugs.py for bugs that are
currently expected-to-fail until fixed.

Time-bounded: each test is capped at deadline=2000ms; the suite as a whole
is dominated by the property machinery, not by I/O.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

# ───────────────────────── propagation.py ─────────────────────────────


from horizon_ric.planner.physics.propagation import (
    free_space_path_loss_dB,
    gas_attenuation_dB,
    rain_attenuation_dB,
    total_path_loss_dB,
)


@settings(max_examples=200, deadline=2000)
@given(
    distance_m=st.floats(min_value=1.0, max_value=4e8),  # 1 m … GSO range
    frequency_hz=st.floats(min_value=1e6, max_value=3e11),  # 1 MHz … 300 GHz
)
def test_fspl_finite_and_monotone(distance_m: float, frequency_hz: float) -> None:
    """FSPL must be finite for every realistic (d, f); monotone in both args."""
    fspl = free_space_path_loss_dB(distance_m, frequency_hz)
    assert math.isfinite(fspl)
    # FSPL is strictly monotone in d for fixed f.
    fspl2 = free_space_path_loss_dB(distance_m * 2.0, frequency_hz)
    assert fspl2 > fspl
    # And in f for fixed d.
    fspl3 = free_space_path_loss_dB(distance_m, frequency_hz * 2.0)
    assert fspl3 > fspl


@settings(max_examples=100, deadline=2000)
@given(
    distance_m=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    frequency_hz=st.floats(min_value=1e6, max_value=3e11),
)
def test_fspl_rejects_nonpositive_distance(
    distance_m: float, frequency_hz: float
) -> None:
    with pytest.raises(ValueError):
        free_space_path_loss_dB(distance_m, frequency_hz)


@settings(max_examples=200, deadline=2000)
@given(
    elevation_deg=st.floats(min_value=0.5, max_value=90.0),
    frequency_ghz=st.floats(min_value=1.0, max_value=350.0),
    rho=st.floats(min_value=0.0, max_value=30.0),
    p_hpa=st.floats(min_value=500.0, max_value=1100.0),
)
def test_gas_attenuation_finite_nonnegative(
    elevation_deg: float, frequency_ghz: float, rho: float, p_hpa: float
) -> None:
    a = gas_attenuation_dB(elevation_deg, frequency_ghz, rho, p_hpa)
    assert math.isfinite(a), f"gas attn NaN/inf @ el={elevation_deg} f={frequency_ghz}"
    assert a >= 0.0, f"negative gas attn {a}"


@settings(max_examples=200, deadline=2000)
@given(
    rr=st.floats(min_value=0.0, max_value=200.0),
    el=st.floats(min_value=1.0, max_value=90.0),
    f=st.floats(min_value=1.0, max_value=100.0),
    pol=st.sampled_from(["horizontal", "vertical", "circular"]),
)
def test_rain_attenuation_finite_nonneg(
    rr: float, el: float, f: float, pol: str
) -> None:
    a = rain_attenuation_dB(rr, el, f, pol)
    assert math.isfinite(a)
    assert a >= 0.0


@settings(max_examples=80, deadline=2000)
@given(
    distance_m=st.floats(min_value=1e3, max_value=4e7),
    frequency_hz=st.floats(min_value=1e9, max_value=4e10),
    elevation_deg=st.floats(min_value=5.0, max_value=90.0),
    rain=st.floats(min_value=0.0, max_value=100.0),
)
def test_total_path_loss_components_consistent(
    distance_m: float,
    frequency_hz: float,
    elevation_deg: float,
    rain: float,
) -> None:
    out = total_path_loss_dB(distance_m, frequency_hz, elevation_deg, rain)
    for k in ("fspl_dB", "gas_dB", "rain_dB", "total_dB"):
        assert math.isfinite(out[k]), f"{k} not finite"
    # total = sum of components, exactly.
    assert abs(out["total_dB"] - (out["fspl_dB"] + out["gas_dB"] + out["rain_dB"])) < 1e-6


# ───────────────────────── policy/constraints.py ─────────────────────────────


from horizon_ric.policy.constraints import (
    GSOArcEntry,
    PreceptualAIConstraintConfig,
    PreceptualAIConstraintLayer,
)


@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
@given(
    tx_dBm=st.floats(min_value=-30.0, max_value=80.0),
    gain_dBi=st.floats(min_value=0.0, max_value=60.0),
    az=st.floats(min_value=0.0, max_value=360.0),
    el=st.floats(min_value=-30.0, max_value=90.0),
    freq=st.floats(min_value=1e9, max_value=40e9),
)
def test_constraint_check_never_silent_passes_violation(
    tx_dBm: float, gain_dBi: float, az: float, el: float, freq: float
) -> None:
    """If permitted band excludes freq, we must always observe a violation."""
    cfg = PreceptualAIConstraintConfig(
        gso_arcs=[GSOArcEntry("g1", 0.0)],
        edge_gpu_memory_gb_max=8.0,
    )
    layer = PreceptualAIConstraintLayer(cfg)
    # Force band that doesn't include `freq`.
    band_low = freq + 1e9
    band_high = freq + 2e9
    action = {
        "tx_power_dBm": tx_dBm,
        "antenna_gain_dBi": gain_dBi,
        "beam_azimuth_deg": az,
        "beam_elevation_deg": el,
        "frequency_hz": freq,
        "ai_workload_gpu_gb": 0.0,
    }
    ctx = {
        "earth_station_latitude_deg": 0.0,
        "earth_station_longitude_deg": 0.0,
        "permitted_band_hz": (band_low, band_high),
    }
    vios = layer.check_feasibility(action, ctx)
    ids = [v.constraint_id for v in vios]
    assert "itu_spectral_mask" in ids


@settings(max_examples=50, deadline=2000)
@given(
    gpu=st.floats(min_value=8.01, max_value=200.0),
)
def test_constraint_gpu_violation_is_real(gpu: float) -> None:
    cfg = PreceptualAIConstraintConfig(edge_gpu_memory_gb_max=8.0)
    layer = PreceptualAIConstraintLayer(cfg)
    action: dict[str, Any] = {"ai_workload_gpu_gb": gpu, "frequency_hz": 4e9}
    vios = layer.check_feasibility(action, {})
    ids = [v.constraint_id for v in vios]
    assert "edge_gpu_capacity" in ids


# ───────────────────────── evidence/store.py ─────────────────────────────


from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore


def _make_record(i: int) -> DecisionRecord:
    return DecisionRecord.new(
        decision_id=f"dec-{i:08d}",
        timestamp=datetime.now(timezone.utc),
        rapp_instance_id="r-test",
        state_hash="0" * 64,
        chosen_action={"action_type": "noop", "i": i},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.1, sla_risk_1min=0.1, sla_risk_5min=0.1
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="e", risk_heads="r", dyna="d", policy="p",
            constraint_layer="c", rapp="0",
        ),
    )


@settings(max_examples=20, deadline=4000, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(n=st.integers(min_value=1, max_value=20))
def test_evidence_chain_verify_intact(tmp_path_factory, n: int) -> None:
    """A clean chain of N records verifies as intact (-1)."""
    p: Path = tmp_path_factory.mktemp("ev") / "ev.jsonl"
    store = JsonlEvidenceStore(p)
    for i in range(n):
        store.append(_make_record(i))
    assert store.verify() == -1


@settings(max_examples=10, deadline=4000, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n=st.integers(min_value=2, max_value=10),
    tamper_idx=st.integers(min_value=0, max_value=9),
)
def test_evidence_chain_detects_tamper_at_exact_index(
    tmp_path_factory, n: int, tamper_idx: int
) -> None:
    """Tamper any line; verify() must point at it (or earlier)."""
    assume(tamper_idx < n)
    p: Path = tmp_path_factory.mktemp("ev") / "ev.jsonl"
    store = JsonlEvidenceStore(p)
    for i in range(n):
        store.append(_make_record(i))
    # Mutate one line's content.
    lines = p.read_text().splitlines()
    obj = lines[tamper_idx]
    # Flip a single character in the JSON record body to invalidate hash.
    if '"i": ' in obj:
        bad = obj.replace('"i": ', '"i": 99999, "x_": ')
        lines[tamper_idx] = bad
        p.write_text("\n".join(lines) + "\n")
        bad_idx = store.verify()
        assert bad_idx == tamper_idx, f"expected break at {tamper_idx}, got {bad_idx}"


# ───────────────────────── core/cfc_core.py ──────────────────────────────


import torch

from horizon_ric.core.cfc_core import CfCConfig
from horizon_ric.core.cfc_core import CfCCell


@settings(max_examples=20, deadline=4000)
@given(
    B=st.integers(min_value=1, max_value=4),
    in_dim=st.integers(min_value=1, max_value=8),
    hidden=st.integers(min_value=2, max_value=16),
    dt=st.floats(min_value=1e-3, max_value=1.0),
)
def test_cfc_forward_finite_and_grads(B: int, in_dim: int, hidden: int, dt: float) -> None:
    cfg = CfCConfig(input_dim=in_dim, hidden_dim=hidden, backbone_dim=hidden)
    cell = CfCCell(cfg)
    h = torch.zeros(B, hidden)
    x_in = torch.randn(B, in_dim, requires_grad=True)
    dt_t = torch.full((B, 1), dt)
    out = cell(x_in, h, dt_t)
    assert torch.isfinite(out).all(), "CfC output not finite"
    out.sum().backward()
    assert x_in.grad is not None and torch.isfinite(x_in.grad).all()


# ───────────────────────── io/schemas.py ─────────────────────────────────


from horizon_ric.io.schemas import TelemetryEvent


@settings(max_examples=200, deadline=2000)
@given(
    event_id=st.text(min_size=1, max_size=64),
    source=st.text(min_size=1, max_size=32),
    seq=st.one_of(st.none(), st.integers(min_value=0, max_value=10**9)),
    payload=st.dictionaries(st.text(max_size=8), st.integers(), max_size=4),
    tags=st.dictionaries(st.text(max_size=8), st.text(max_size=8), max_size=4),
)
def test_telemetry_event_roundtrip(
    event_id: str, source: str, seq: int | None, payload: dict, tags: dict
) -> None:
    e = TelemetryEvent(
        event_id=event_id,
        modality="kpm_5g",
        source_id=source,
        ts_utc=datetime.now(timezone.utc),
        sequence=seq,
        payload=payload,
        tags=tags,
    )
    raw = e.model_dump_json()
    e2 = TelemetryEvent.model_validate_json(raw)
    assert e2.event_id == event_id
    assert e2.source_id == source
    assert e2.sequence == seq
    assert e2.payload == payload
    assert e2.tags == tags


def test_telemetry_event_rejects_naive_ts() -> None:
    """Naive datetime must raise — no silent local-time interpretation."""
    with pytest.raises(Exception):  # pydantic ValidationError
        TelemetryEvent(
            event_id="e",
            modality="kpm_5g",
            source_id="s",
            ts_utc=datetime(2025, 1, 1, 0, 0, 0),  # naive
        )
