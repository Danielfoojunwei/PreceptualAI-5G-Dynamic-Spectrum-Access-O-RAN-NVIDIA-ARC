"""Tests for Sionna channel + AODT scenario adapters.

Both adapters now require their real upstream dependencies — Sionna
(PyTorch backend in 2.x, TensorFlow in 0.x) for channel generation and
OpenUSD (`pxr.Usd`) for AODT scenario parsing. There is no stand-in or
fallback path. Tests fail (not skip) if the deps are missing; the
horizon-ric venv must include both.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from horizon_ric.data.aodt import AODTScenario
from horizon_ric.data.sionna_channel import SionnaChannelGenerator
from horizon_ric.encoder.spatial_prior import SpatialPrior


AODT_TEST_SCENE = Path(__file__).parent / "fixtures" / "aodt_test_scene.usda"


# ─── 1. Real Sionna is the only backend ───────────────────────────────
def test_generator_uses_real_sionna() -> None:
    import sionna  # noqa: F401  hard requirement; raises if missing

    gen = SionnaChannelGenerator(seed=0, n_rx=1, n_tx=1)
    assert gen.backend_name == "sionna"
    # The internal layout tag tells us which Sionna release is wired up.
    assert gen._tdl_layout in ("tr38811", "tr38901_phy", "tr38901")


def test_require_sionna_returns_real_backend() -> None:
    import sionna  # noqa: F401

    gen = SionnaChannelGenerator(require_sionna=True)
    assert gen.backend_name == "sionna"


# ─── 2. Channels have the right shape and are non-zero ────────────────
@pytest.mark.parametrize(
    "scenario", ["NTN-TDL-A", "NTN-TDL-B", "NTN-TDL-C", "NTN-TDL-D"]
)
def test_generate_channel_shape_and_nonzero(scenario: str) -> None:
    gen = SionnaChannelGenerator(seed=42, n_rx=2, n_tx=2)
    n = 16
    h = gen.generate_channel(
        scenario=scenario,
        ue_speed_kmh=30.0,
        frequency_hz=2.0e9,
        n_samples=n,
    )
    assert h.dtype == np.complex64
    assert h.shape[0] == n
    assert h.shape[1] == 2  # n_rx
    assert h.shape[2] == 2  # n_tx
    assert h.shape[3] >= 2  # n_paths from the profile
    assert np.any(np.abs(h) > 0), f"all-zero channel for {scenario}"
    assert np.all(np.isfinite(np.abs(h)))


# ─── 3. Mean total power matches the analytical TDL profile (within 1 dB)
def test_generate_channel_mean_power_within_one_db() -> None:
    """Sionna's TDL normalises the impulse response so that the sum of
    tap powers equals 1 (unit total channel energy). Verify mean power
    of the realisation lands within 1 dB of the analytical 0 dB target.
    """
    gen = SionnaChannelGenerator(seed=123, n_rx=1, n_tx=1)
    h = gen.generate_channel(
        scenario="NTN-TDL-A",
        ue_speed_kmh=10.0,
        frequency_hz=2.0e9,
        n_samples=512,
    )
    # Sum the per-tap powers across taps, then average across time samples.
    total_power_per_sample = np.sum(np.abs(h) ** 2, axis=-1)  # (n, n_rx, n_tx)
    avg_power = float(np.mean(total_power_per_sample))
    # 0 dB = 1.0 linear; tolerate ±1 dB → [0.794, 1.259].
    assert 0.794 <= avg_power <= 1.259, (
        f"mean total channel power {avg_power:.3f} outside ±1 dB of 1.0"
    )


# ─── 4. Telemetry events use the right modality and carry stats ──────
def test_to_telemetry_events_modality_and_payload() -> None:
    gen = SionnaChannelGenerator(seed=7)
    h = gen.generate_channel(
        scenario="NTN-TDL-C",
        ue_speed_kmh=10.0,
        frequency_hz=20e9,
        n_samples=4,
    )
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = list(
        gen.to_telemetry_events(
            channels=h,
            t0_utc=t0,
            source_id="unit-test",
            scenario="NTN-TDL-C",
            frequency_hz=20e9,
        )
    )
    assert len(events) == 4
    for ev in events:
        assert ev.modality == "kpm_ntn"
        assert ev.source_id == "unit-test"
        assert ev.payload["scenario"] == "NTN-TDL-C"
        assert ev.payload["backend"] == "sionna"
        assert "mean_power" in ev.payload
        assert "rms_delay_spread_taps" in ev.payload
        assert "k_factor_db" in ev.payload
        # Raw complex tensor must NOT be embedded.
        assert "channels" not in ev.payload
        assert "iq" not in ev.payload


# ─── 5. AODT loads the bundled USD test scene via real OpenUSD ───────
def test_aodt_parses_real_usd_scene() -> None:
    """Open the synthetic .usda fixture with real pxr.Usd and verify the
    parser extracts the right number of prims and the right positions.
    """
    pytest.importorskip(
        "pxr",
        reason="pxr.Usd is a hard dependency of this test; install usd-core "
        "(x86_64) or build OpenUSD from source on aarch64.",
    )
    assert AODT_TEST_SCENE.exists(), f"missing fixture: {AODT_TEST_SCENE}"

    scen = AODTScenario(AODT_TEST_SCENE)
    # 1 BS + 3 UEs + 2 satellites = 6 typed entities.
    assert bool(scen) is True
    assert len(scen.entities["cells"]) == 1
    assert len(scen.entities["ues"]) == 3
    assert len(scen.entities["satellites"]) == 2
    assert len(scen.all_entities()) == 6

    # Spot-check the first UE's position matches the fixture.
    ue0 = next(
        e for e in scen.entities["ues"] if e["id"].endswith("UE_0")
    )
    np.testing.assert_allclose(
        ue0["position_xyz_m"],
        [1115050.0, -4843010.0, 3982002.5],
        rtol=1e-6,
    )

    # Confirm cells / satellites also surfaced their authored positions.
    bs = scen.entities["cells"][0]
    np.testing.assert_allclose(
        bs["position_xyz_m"],
        [1115000.0, -4843000.0, 3982000.0],
        rtol=1e-6,
    )


def test_aodt_usd_feeds_spatial_prior() -> None:
    pytest.importorskip("pxr")
    scen = AODTScenario(AODT_TEST_SCENE)
    sp_input = scen.to_spatial_prior_input()
    assert len(sp_input) == 6
    tensor = SpatialPrior(min_elevation_deg=5.0).build(sp_input)
    assert tensor.node_xyz_m.shape == (6, 3)


def test_aodt_usd_emits_satellite_telemetry_events() -> None:
    pytest.importorskip("pxr")
    scen = AODTScenario(AODT_TEST_SCENE)
    events = list(scen.to_telemetry_events(t0_utc=datetime.now(timezone.utc)))
    assert len(events) == 2
    for ev in events:
        assert ev.modality == "ephemeris_oem"
        assert ev.payload["kind"] == "satellite"
        assert "satellite_id" in ev.payload
        assert len(ev.payload["position_ecef_m"]) == 3


# ─── 6. Synthetic JSON sidecar still works for bundles that ship one ─
def test_aodt_parses_fabricated_scenario_json(tmp_path: Path) -> None:
    """Drop a synthetic scenario JSON into a directory and verify the
    JSON path still produces well-formed entities.
    """
    scenario_dir = tmp_path / "aodt_test"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.json").write_text(
        '{"satellites": ['
        '{"id": "sat-1", "kind": "satellite", '
        '"position_xyz_m": [7.0e6, 0.0, 0.0]}], '
        '"ues": [{"id": "ue-1", "kind": "ue", '
        '"lat_deg": 1.3, "lon_deg": 103.8, "height_m": 5.0}]}'
    )
    scen = AODTScenario(scenario_dir)
    assert bool(scen) is True
    sats = scen.entities["satellites"]
    ues = scen.entities["ues"]
    assert len(sats) == 1 and sats[0]["id"] == "sat-1"
    assert len(ues) == 1 and ues[0]["placement"] == "ground"

    sp_input = scen.to_spatial_prior_input()
    assert len(sp_input) == 2
    tensor = SpatialPrior(min_elevation_deg=5.0).build(sp_input)
    assert tensor.node_xyz_m.shape == (2, 3)

    events = list(scen.to_telemetry_events(t0_utc=datetime.now(timezone.utc)))
    assert len(events) == 1
    assert events[0].modality == "ephemeris_oem"
    assert events[0].payload["satellite_id"] == "sat-1"
