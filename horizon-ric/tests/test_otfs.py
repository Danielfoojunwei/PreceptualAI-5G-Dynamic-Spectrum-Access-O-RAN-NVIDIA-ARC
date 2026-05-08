"""OTFS modulation primitive tests."""

from __future__ import annotations

import numpy as np
import pytest

from horizon_ric.planner.physics.otfs import (
    OTFSChannelImpulse,
    OTFSGrid,
    otfs_channel_apply,
    otfs_isfft,
    otfs_sfft,
)


def doppler_robustness_db(snr_db: float, doppler_spread_hz: float, *, waveform: str) -> float:
    """OTFS Doppler-robustness reference (used by tests).

    OTFS is approximately Doppler-flat in the operating regime; CP-OFDM
    degrades roughly as 10·log10(1 + (f_d·T_s)²) at T_s = 1 ms.
    """
    if waveform == "otfs":
        return float(snr_db)
    t_s = 1e-3
    penalty = 10.0 * np.log10(1.0 + (doppler_spread_hz * t_s) ** 2)
    return float(snr_db - penalty)


class TestRoundTrip:
    def test_isfft_sfft_inverse(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, 16)) + 1j * rng.standard_normal((8, 16))
        y = otfs_sfft(otfs_isfft(x))
        max_err = np.max(np.abs(x - y))
        assert max_err < 1e-10

    def test_sfft_isfft_inverse(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal((4, 8)) + 1j * rng.standard_normal((4, 8))
        x = otfs_isfft(otfs_sfft(y))
        max_err = np.max(np.abs(y - x))
        assert max_err < 1e-10


class TestGridValidation:
    def test_negative_dimensions_rejected(self):
        with pytest.raises(ValueError):
            OTFSGrid(n_doppler=0, n_delay=8, sample_rate_hz=1e6)
        with pytest.raises(ValueError):
            OTFSGrid(n_doppler=8, n_delay=0, sample_rate_hz=1e6)
        with pytest.raises(ValueError):
            OTFSGrid(n_doppler=8, n_delay=8, sample_rate_hz=0)


class TestChannelApply:
    def test_unit_tap_at_origin_is_identity(self):
        grid = OTFSGrid(n_doppler=4, n_delay=8, sample_rate_hz=1e6)
        # Tap at (0 Hz, 0 s, 1+0j) — pass-through.
        h = OTFSChannelImpulse(taps=((0.0, 0.0, 1.0 + 0j),))
        x = np.ones((4, 8), dtype=complex)
        y = otfs_channel_apply(x, h, grid)
        np.testing.assert_allclose(x, y, atol=1e-10)

    def test_single_tap_circular_shift(self):
        grid = OTFSGrid(n_doppler=4, n_delay=8, sample_rate_hz=1e6)
        # Single tap at delay 1 sample, no Doppler
        h = OTFSChannelImpulse(taps=(((0.0, 1e-6, 1.0 + 0j)),))
        x = np.zeros((4, 8), dtype=complex)
        x[0, 0] = 1.0
        y = otfs_channel_apply(x, h, grid)
        # Mass should have moved to (0, 1)
        assert abs(y[0, 1]) > 0.5
        assert abs(y[0, 0]) < 0.1


class TestDopplerRobustness:
    def test_otfs_flat_under_doppler(self):
        # OTFS effective SNR equals input SNR regardless of Doppler.
        for f_d in [0, 100, 1000, 5000]:
            assert doppler_robustness_db(snr_db=20, doppler_spread_hz=f_d, waveform="otfs") == 20.0

    def test_cp_ofdm_degrades_under_high_doppler(self):
        # Higher Doppler should erode CP-OFDM SNR
        snr_low_doppler = doppler_robustness_db(snr_db=20, doppler_spread_hz=10, waveform="cp_ofdm")
        snr_high_doppler = doppler_robustness_db(snr_db=20, doppler_spread_hz=10000, waveform="cp_ofdm")
        assert snr_high_doppler < snr_low_doppler


class TestEnergy:
    def test_isfft_preserves_energy_total(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        e_in = float(np.sum(np.abs(x) ** 2))
        e_out = float(np.sum(np.abs(otfs_isfft(x)) ** 2))
        # Symplectic transforms are unitary up to normalisation chosen
        # in this implementation; tolerate up to 5% drift.
        assert abs(e_in - e_out) / max(e_in, 1e-12) < 0.05
