"""Property-based checks for the enumerated terrestrial Shield contract."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from horizon_ric.shield import default_terrestrial_shield

BAND_LO = 3.40e9
BAND_HI = 3.50e9
MAX_EIRP = 33.0

finite = st.floats(
    min_value=-1.0e10,
    max_value=1.0e10,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@settings(max_examples=500, deadline=None)
@given(
    frequency_hz=finite,
    bandwidth_hz=finite,
    tx_power_dBm=st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ),
    antenna_gain_dBi=st.floats(
        min_value=-100,
        max_value=100,
        allow_nan=False,
        allow_infinity=False,
    ),
    constellation_order=st.integers(min_value=-1024, max_value=2048),
    papr_dB=st.floats(
        min_value=-100,
        max_value=100,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_any_emitted_random_action_satisfies_every_programmed_invariant(
    frequency_hz: float,
    bandwidth_hz: float,
    tx_power_dBm: float,
    antenna_gain_dBi: float,
    constellation_order: int,
    papr_dB: float,
):
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO,
        band_hi_hz=BAND_HI,
        max_eirp_dBm=MAX_EIRP,
    )
    disposition = shield.dispose(
        {
            "block": "learned_constellation",
            "frequency_hz": frequency_hz,
            "bandwidth_hz": bandwidth_hz,
            "tx_power_dBm": tx_power_dBm,
            "antenna_gain_dBi": antenna_gain_dBi,
            "constellation_order": constellation_order,
            "papr_dB": papr_dB,
        }
    )
    if disposition.certificate.emit_blocked:
        return

    safe = disposition.safe_action
    assert disposition.certificate.safe
    assert all(check.satisfied for check in disposition.certificate.invariants)
    assert all(
        math.isfinite(float(safe[field]))
        for field in (
            "frequency_hz",
            "bandwidth_hz",
            "tx_power_dBm",
            "antenna_gain_dBi",
            "papr_dB",
        )
    )
    assert safe["bandwidth_hz"] > 0
    assert safe["frequency_hz"] - safe["bandwidth_hz"] / 2 >= BAND_LO
    assert safe["frequency_hz"] + safe["bandwidth_hz"] / 2 <= BAND_HI
    assert safe["tx_power_dBm"] + safe["antenna_gain_dBi"] <= MAX_EIRP + 1e-9
    assert safe["constellation_order"] in (4, 16, 64, 256)
    assert 0.0 <= safe["papr_dB"] <= 8.5


@given(
    field=st.sampled_from(
        [
            "frequency_hz",
            "bandwidth_hz",
            "tx_power_dBm",
            "antenna_gain_dBi",
        ]
    ),
    bad=st.sampled_from([math.nan, math.inf, -math.inf, "not-a-number", None]),
)
def test_nonfinite_or_malformed_rf_values_fail_closed(field: str, bad: object):
    action = {
        "block": "ric_policy",
        "frequency_hz": 3.45e9,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": 6.0,
    }
    action[field] = bad
    disposition = default_terrestrial_shield(
        band_lo_hz=BAND_LO,
        band_hi_hz=BAND_HI,
        max_eirp_dBm=MAX_EIRP,
    ).dispose(action)
    assert disposition.certificate.emit_blocked
    assert not disposition.certificate.safe
    assert "numeric_domain_sanity" in disposition.certificate.violated_ids
