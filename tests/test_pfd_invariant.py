"""Tests for the LEO/NTN power-flux-density ceiling invariant (ITU-R-style)."""

from __future__ import annotations

import math

from horizon_ric.shield import (
    PfdCeilingInvariant,
    Shield,
    default_ntn_shield,
)
from horizon_ric.shield.invariants import PfdCeilingInvariant as PfdInv


def test_pfd_is_noop_for_terrestrial_actions():
    inv = PfdCeilingInvariant()
    action = {"tx_power_dBm": 40.0, "sat_antenna_gain_dBi": 30.0}  # no ntn flag
    chk = inv.evaluate(action, {})
    assert chk.satisfied is True
    safe, corr = inv.project(action, {})
    assert corr == []
    assert safe == action


def test_pfd_computation_matches_link_budget():
    # EIRP = 10 dBm + 20 dBi = 30 dBm = 0 dBW. d = 550 km. BW = 1 MHz.
    action = {
        "ntn": True,
        "tx_power_dBm": 10.0,
        "sat_antenna_gain_dBi": 20.0,
        "slant_range_m": 5.5e5,
        "bandwidth_hz": 1e6,
    }
    pfd = PfdCeilingInvariant.compute_pfd(action)
    d = 5.5e5
    expected = 0.0 - 10 * math.log10(4 * math.pi * d * d) - 10 * math.log10(1.0)
    assert abs(pfd - expected) < 1e-9


def test_pfd_violation_reduces_satellite_power_to_ceiling():
    inv = PfdCeilingInvariant(max_pfd_dBW_m2_MHz=-146.0)
    # High EIRP at close range → over the PFD ceiling.
    action = {
        "ntn": True,
        "tx_power_dBm": 33.0,
        "sat_antenna_gain_dBi": 30.0,
        "slant_range_m": 5.5e5,
        "bandwidth_hz": 1e6,
    }
    chk = inv.evaluate(action, {})
    assert chk.satisfied is False  # this regime breaches the ceiling

    safe, corr = inv.project(action, {})
    assert corr
    # After projection PFD must sit on (or below) the ceiling.
    pfd_after = PfdCeilingInvariant.compute_pfd(safe)
    assert pfd_after <= -146.0 + 1e-6
    # Power was reduced, not increased.
    assert safe["tx_power_dBm"] < action["tx_power_dBm"]


def test_pfd_satisfied_when_low_power():
    inv = PfdCeilingInvariant(max_pfd_dBW_m2_MHz=-146.0)
    action = {
        "ntn": True,
        "tx_power_dBm": -10.0,
        "sat_antenna_gain_dBi": 5.0,
        "slant_range_m": 1.2e6,
        "bandwidth_hz": 5e6,
    }
    assert inv.evaluate(action, {}).satisfied is True


def test_default_ntn_shield_enforces_pfd_and_keeps_action_legal():
    shield = default_ntn_shield(
        band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0,
        max_pfd_dBW_m2_MHz=-146.0,
    )
    assert isinstance(shield, Shield)
    assert "pfd_ceiling_ntn" in shield.invariant_ids

    # A LEO downlink the agent commanded too hot.
    action = {
        "block": "leo_power_control",
        "ntn": True,
        "frequency_hz": 3.45e9,
        "bandwidth_hz": 1e6,
        "tx_power_dBm": 33.0,
        "antenna_gain_dBi": 6.0,
        "sat_antenna_gain_dBi": 30.0,
        "slant_range_m": 5.5e5,
    }
    disp = shield.dispose(action, decision_id="leo-1")
    assert disp.certificate.safe is True
    safe = disp.safe_action
    # PFD now within ceiling.
    assert PfdInv.compute_pfd(safe) <= -146.0 + 1e-6
