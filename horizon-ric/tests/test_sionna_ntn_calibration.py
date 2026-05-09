"""Sionna NTN-TDL calibration evidence (closes Devil-C Finding #19).

The project surfaces Sionna NTN-TDL channels via `SionnaChannelGenerator`
which currently maps NTN-TDL-A/B/C/D requests to the TR 38.901 TDL-A/B/C/D
backbones (see `src/horizon_ric/data/sionna_channel.py`). Devil-C #19
asks: prove that the Sionna realisations actually carry the spec'd tap
statistics so the world model is being fit on real channel statistics
and not on Sionna's defaults having drifted from spec.

We compare Sionna's per-tap mean power, RMS delay spread, and (LoS
profile) Rician K-factor to the TR 38.901 §7.7.2 reference values.

Tolerance:
    * Mean total tap power         : ±1 dB    (spec normalises to 1.0)
    * RMS delay spread             : ±10 % of requested DS
    * K-factor (TDL-D / TDL-E only): ±1 dB

If Sionna is not importable, the whole module is skipped — we never
fake the calibration, we surface the missing dependency honestly.

References:
    * 3GPP TR 38.901 v17.0 §7.7.2 — TDL-A..E tap powers, delays, K-factor.
    * 3GPP TR 38.811 Annex C       — NTN-TDL profiles (A/B NLOS, C/D LOS).
    * Sionna PyTorch backend      :
        sionna.phy.channel.tr38901.TDL  (Sionna 2.x).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

sionna = pytest.importorskip("sionna")
try:
    from sionna.phy.channel.tr38901 import TDL  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - import-time skip
    pytest.skip(f"sionna.phy.channel.tr38901.TDL not importable: {exc}", allow_module_level=True)


# TR 38.901 §7.7.2 reference K-factors for the LoS TDL profiles. These are
# the spec values that any TR 38.901 TDL implementation MUST produce when
# normalised to the requested delay-spread.
TR38901_K_FACTOR_DB = {
    "D": 13.3,   # TR 38.901 Table 7.7.2-4
    "E": 22.0,   # TR 38.901 Table 7.7.2-5
}

# Profiles to calibrate. NTN-TDL-A/B map to TR 38.901 TDL-A/B (NLOS);
# NTN-TDL-C/D map to TDL-C/D (LOS). The mapping lives in
# horizon_ric.data.sionna_channel._NTN_TO_TDL_MODEL.
PROFILES = ["A", "B", "C", "D"]

REQUESTED_DS_NS = 100.0  # spec-typical S-band scaling factor
CARRIER_HZ = 2.0e9
N_RX_ANT = 1
N_TX_ANT = 1


def _stats(model: str) -> dict[str, float]:
    """Return per-profile total power (linear), RMS delay spread (ns), and
    K-factor (dB; nan for NLOS)."""
    tdl = TDL(
        model=model,
        delay_spread=REQUESTED_DS_NS * 1e-9,
        carrier_frequency=CARRIER_HZ,
        num_rx_ant=N_RX_ANT,
        num_tx_ant=N_TX_ANT,
    )
    p = tdl.mean_powers.detach().cpu().numpy().astype(np.float64)
    d = tdl.delays.detach().cpu().numpy().astype(np.float64)
    total_p_lin = float(p.sum())
    p_norm = p / total_p_lin
    mean_d = float((d * p_norm).sum())
    rms_ds_s = float(np.sqrt((((d - mean_d) ** 2) * p_norm).sum()))
    rms_ds_ns = rms_ds_s * 1e9
    if bool(tdl.los):
        k_db = float(10.0 * math.log10(float(tdl.k_factor.item())))
    else:
        k_db = float("nan")
    return {
        "total_p_lin": total_p_lin,
        "rms_ds_ns": rms_ds_ns,
        "k_factor_db": k_db,
    }


@pytest.mark.parametrize("model", PROFILES)
def test_sionna_tdl_total_power_within_1db(model: str) -> None:
    """Total tap power must equal 1.0 (linear) within ±1 dB.

    TR 38.901 normalises every TDL profile so Σ p_i = 1. Sionna inherits
    that normalisation. ±1 dB == factor in [10**-0.1, 10**0.1] ≈ [0.794, 1.259].
    """
    s = _stats(model)
    total_db = 10.0 * math.log10(s["total_p_lin"])
    assert abs(total_db) <= 1.0, (
        f"TDL-{model} total tap power = {total_db:+.2f} dB "
        f"(expected 0.0 dB ±1 dB; spec TR 38.901 §7.7.2)"
    )


@pytest.mark.parametrize("model", PROFILES)
def test_sionna_tdl_rms_delay_spread_within_10pct(model: str) -> None:
    """RMS delay spread of the realised taps must equal the requested DS
    within ±10 %.

    TR 38.901 §7.7.3 ("Scaling factors for delay spread") guarantees the
    profile's RMS delay spread equals the requested ``delay_spread``
    after applying the Σ p_i τ_i normalisation.
    """
    s = _stats(model)
    rel = abs(s["rms_ds_ns"] - REQUESTED_DS_NS) / REQUESTED_DS_NS
    assert rel <= 0.10, (
        f"TDL-{model} realised RMS DS = {s['rms_ds_ns']:.2f} ns; "
        f"requested {REQUESTED_DS_NS:.2f} ns (deviation {rel*100:.1f}%, "
        f"tolerance 10%)"
    )


@pytest.mark.parametrize("model", ["D"])  # only LoS profile in the project's NTN map
def test_sionna_tdl_k_factor_within_1db(model: str) -> None:
    """LoS TDL profiles must reproduce the TR 38.901 K-factor within ±1 dB.

    Only TDL-D and TDL-E are LoS in TR 38.901; the project's NTN map
    surfaces TDL-D so we calibrate that one. The spec value is 13.3 dB
    (TR 38.901 Table 7.7.2-4).
    """
    expected = TR38901_K_FACTOR_DB[model]
    s = _stats(model)
    assert math.isfinite(s["k_factor_db"]), (
        f"TDL-{model} should be a LoS profile but K-factor is non-finite"
    )
    diff = abs(s["k_factor_db"] - expected)
    assert diff <= 1.0, (
        f"TDL-{model} K-factor = {s['k_factor_db']:.2f} dB "
        f"(spec {expected:.2f} dB ±1 dB; deviation {diff:.2f} dB)"
    )
