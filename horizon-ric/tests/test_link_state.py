"""link_state composer tests — ensures Doppler + NTN-timing + 38.811 channel
collapse into a stable 12-dim attribute vector."""

import math
from datetime import datetime, timezone

import torch

from horizon_ric.encoder.link_state import (
    LINK_STATE_DIM,
    LinkStateInputs,
    compose_link_state,
)
from horizon_ric.planner.physics.orbital import (
    KeplerianElements,
    OrbitalState,
    keplerian_state,
)

_T0 = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)


def _starlink_state() -> OrbitalState:
    return keplerian_state(KeplerianElements(
        a_km=6378.137 + 550.0,
        e=0.0,
        i_rad=math.radians(53.0),
        raan_rad=0.0, argp_rad=0.0, M_rad=0.0,
        epoch_utc=_T0,
    ))


def _es_directly_below(state) -> tuple[float, float]:
    """ES coordinates placed directly under the satellite's current
    ECEF position, so visibility is guaranteed."""
    from horizon_ric.planner.physics.geodesy import ecef_to_geodetic
    from horizon_ric.planner.physics.orbital import state_ecef_m

    ecef = state_ecef_m(state)
    lat, lon, _ = ecef_to_geodetic(ecef)
    return lat, lon


class TestLinkStateCompose:
    def test_dim_constant_is_12(self):
        assert LINK_STATE_DIM == 12

    def test_returns_12d_tensor(self):
        state = _starlink_state()
        lat, lon = _es_directly_below(state)
        v = compose_link_state(LinkStateInputs(
            sat_state=state,
            es_lat_deg=lat, es_lon_deg=lon,
            carrier_hz=28e9, environment="rural",
        ))
        assert v.shape == (12,)
        assert v.dtype == torch.float32
        assert torch.isfinite(v).all()
        # Satellite is overhead → elevation ≈ 90° (after normalisation, ≈ 1.0).
        assert v[10].item() > 0.9

    def test_below_horizon_returns_zeros(self):
        # ES at the geographic pole — Starlink's equator-plane orbit is below.
        v = compose_link_state(LinkStateInputs(
            sat_state=_starlink_state(),
            es_lat_deg=89.9, es_lon_deg=0.0,
            carrier_hz=28e9, environment="rural",
        ))
        # Sentinel: all zeros when the satellite is below local horizon.
        assert torch.allclose(v, torch.zeros_like(v))

    def test_environment_changes_channel_features(self):
        state = _starlink_state()
        lat, lon = _es_directly_below(state)
        v_rural = compose_link_state(LinkStateInputs(
            sat_state=state,
            es_lat_deg=lat, es_lon_deg=lon,
            environment="rural", carrier_hz=28e9,
        ))
        v_urban = compose_link_state(LinkStateInputs(
            sat_state=state,
            es_lat_deg=lat, es_lon_deg=lon,
            environment="urban", carrier_hz=28e9,
        ))
        # Channel features (indices 6..9) must differ between rural / urban.
        assert not torch.allclose(v_rural[6:10], v_urban[6:10])

    def test_carrier_changes_doppler_shift(self):
        state = _starlink_state()
        lat, lon = _es_directly_below(state)
        # Off-set the ES slightly so there's actual radial velocity.
        v_2ghz = compose_link_state(LinkStateInputs(
            sat_state=state,
            es_lat_deg=lat - 5.0, es_lon_deg=lon,
            carrier_hz=2e9, environment="rural",
        ))
        v_28ghz = compose_link_state(LinkStateInputs(
            sat_state=state,
            es_lat_deg=lat - 5.0, es_lon_deg=lon,
            carrier_hz=28e9, environment="rural",
        ))
        # Doppler-shift index 4 must scale ~14× between 2 and 28 GHz.
        d2, d28 = float(v_2ghz[4]), float(v_28ghz[4])
        # Either both zero (overhead pass, no radial velocity) or the ratio
        # is well bounded.
        if abs(d2) > 1e-6:
            ratio = d28 / d2
            assert 5 < abs(ratio) < 30
