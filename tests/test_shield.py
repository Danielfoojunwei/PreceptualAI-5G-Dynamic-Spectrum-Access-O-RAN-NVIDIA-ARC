"""Tests for the Decision Safety Shield."""

from __future__ import annotations

import pytest

from horizon_ric.policy.li_constraint import LIConstraint
from horizon_ric.shield import (
    ConstellationLegalityInvariant,
    MaxEirpInvariant,
    NeuralRxEnvelopeInvariant,
    Shield,
    SpectralMaskInvariant,
    default_terrestrial_shield,
)


def _lab_li() -> LIConstraint:
    return LIConstraint(rules=[], fail_closed=False, deployment_audit_record="lab")


def _shield() -> Shield:
    return default_terrestrial_shield(
        band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0, li_constraint=_lab_li()
    )


def test_clean_decision_passes_untouched():
    s = _shield()
    clean = {
        "block": "ric_policy",
        "frequency_hz": 3.45e9,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": 6.0,
    }
    disp = s.dispose(clean, decision_id="clean")
    assert disp.certificate.safe is True
    assert disp.certificate.projected is False
    assert disp.certificate.emit_blocked is False
    assert disp.certificate.violated_ids == []


def test_poisoned_decision_is_made_safe_and_certified():
    s = _shield()
    poisoned = {
        "block": "neural_rx",
        "frequency_hz": 3.55e9,  # outside band
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 40.0,
        "antenna_gain_dBi": 6.0,  # EIRP 46 dBm > 33
        "constellation_order": 1024,  # illegal
        "papr_dB": 12.0,  # over ceiling
        "predicted_tbler": 0.4,
        "baseline_tbler": 0.05,
        "demap_confidence": 0.1,
    }
    disp = s.dispose(poisoned, decision_id="poisoned", rng_seed=7)
    cert = disp.certificate
    # The shield must make it safe — a poisoned model cannot emit illegally.
    assert cert.safe is True
    assert cert.emit_blocked is False
    assert cert.projected is True
    assert cert.fallback_used is True
    # In-band, within EIRP ceiling, legal order, bounded PAPR.
    a = disp.safe_action
    assert 3.40e9 <= a["frequency_hz"] <= 3.50e9
    assert a["tx_power_dBm"] + a["antenna_gain_dBi"] <= 33.0 + 1e-6
    assert a["constellation_order"] in (4, 16, 64, 256)
    assert a["papr_dB"] <= 8.5 + 1e-6
    # Corrections are recorded for the audit chain.
    assert {c.constraint_id for c in cert.corrections} >= {
        "spectral_mask_ts38104",
        "max_eirp",
        "constellation_legality",
    }


def test_certificate_carries_provenance_and_seed():
    s = _shield()
    prov = {"weights_sha256": "abc", "signed": True}
    disp = s.dispose(
        {"block": "neural_rx", "frequency_hz": 3.45e9, "bandwidth_hz": 20e6,
         "tx_power_dBm": 20.0, "antenna_gain_dBi": 6.0,
         "predicted_tbler": 0.05, "baseline_tbler": 0.05, "demap_confidence": 0.9},
        decision_id="d", rng_seed=42, model_provenance=prov, loop_tier="near_rt",
    )
    assert disp.certificate.rng_seed == 42
    assert disp.certificate.model_provenance == prov
    assert disp.certificate.loop_tier == "near_rt"
    d = disp.certificate.to_dict()
    assert d["decision_id"] == "d" and d["rng_seed"] == 42


def test_li_fail_closed_blocks_emit():
    # Fail-closed LI with no rules must block every action.
    li = LIConstraint(rules=[], fail_closed=True)
    s = default_terrestrial_shield(band_lo_hz=3.4e9, band_hi_hz=3.5e9, li_constraint=li)
    disp = s.dispose(
        {"block": "ric_policy", "frequency_hz": 3.45e9, "bandwidth_hz": 20e6,
         "tx_power_dBm": 20.0, "antenna_gain_dBi": 6.0},
        decision_id="li",
    )
    assert disp.certificate.emit_blocked is True
    assert disp.certificate.safe is False


def test_neural_rx_envelope_falls_back_to_classical():
    inv = NeuralRxEnvelopeInvariant(tolerance_dB=1.0)
    bad = {"block": "neural_rx", "predicted_tbler": 0.5, "baseline_tbler": 0.01,
           "demap_confidence": 0.9}
    assert inv.evaluate(bad, {}).satisfied is False
    safe, corr = inv.project(bad, {})
    assert safe["block"] == "classical_lmmse"
    assert corr


def test_spectral_mask_blocks_when_carrier_wider_than_channel():
    inv = SpectralMaskInvariant(band_lo_hz=3.40e9, band_hi_hz=3.41e9)  # 10 MHz channel
    action = {"frequency_hz": 3.405e9, "bandwidth_hz": 20e6}  # 20 MHz carrier
    safe, corr = inv.project(action, {})
    assert safe.get("emit_blocked") is True
    assert corr


def test_max_eirp_reduces_power():
    inv = MaxEirpInvariant(max_eirp_dBm=30.0)
    safe, corr = inv.project({"tx_power_dBm": 40.0, "antenna_gain_dBi": 5.0}, {})
    assert safe["tx_power_dBm"] + 5.0 == pytest.approx(30.0)
    assert corr
