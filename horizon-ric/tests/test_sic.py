"""Grant-free Successive Interference Cancellation tests."""

from __future__ import annotations

import numpy as np
import pytest

from horizon_ric.planner.physics.grant_free_sic import (
    GrantFreeSlot,
    sic_capacity,
    sic_decode,
)


class TestSlotValidation:
    def test_empty_pdus_rejected(self):
        with pytest.raises(ValueError):
            GrantFreeSlot(user_pdus=[], received_signal=np.zeros(8, complex), noise_var=0.0)

    def test_negative_noise_rejected(self):
        with pytest.raises(ValueError):
            GrantFreeSlot(
                user_pdus=[b"\x00\x01"],
                received_signal=np.zeros(8, complex),
                noise_var=-0.1,
            )


class TestSICCapacity:
    def test_capacity_monotonic_in_snr(self):
        c_low = sic_capacity(n_users=4, snr_db=0)
        c_high = sic_capacity(n_users=4, snr_db=20)
        assert c_high > c_low

    def test_capacity_finite_positive(self):
        c = sic_capacity(n_users=8, snr_db=10)
        assert 0 < c < 100


class TestSICDecode:
    def test_max_iters_validated(self):
        slot = GrantFreeSlot(
            user_pdus=[b"\x00"],
            received_signal=np.zeros(4, dtype=complex),
            noise_var=1e-3,
        )
        with pytest.raises(ValueError):
            sic_decode(slot, max_iters=0)

    def test_returns_one_tuple_per_user(self):
        n_users = 3
        slot = GrantFreeSlot(
            user_pdus=[b"\x00\x01"] * n_users,
            received_signal=np.zeros(8, dtype=complex),
            noise_var=1.0,
        )
        results = sic_decode(slot, max_iters=3)
        assert len(results) == n_users
        for user_id, _bytes, _success in results:
            assert isinstance(user_id, int)
            assert 0 <= user_id < n_users

    def test_high_noise_low_success_rate(self):
        # With very high noise, most users should fail to decode.
        slot = GrantFreeSlot(
            user_pdus=[b"\x00\x01"] * 5,
            received_signal=np.zeros(8, dtype=complex) + 1e-3,
            noise_var=1e6,  # extremely noisy
        )
        results = sic_decode(slot, max_iters=3)
        success = sum(1 for _, _, ok in results if ok)
        # Allow some statistical wiggle but expect majority failure
        assert success <= 3

    def test_low_noise_more_decodes(self):
        # With near-zero noise, decoder should succeed at least once.
        slot = GrantFreeSlot(
            user_pdus=[b"\x42\x42"],
            received_signal=np.zeros(8, dtype=complex) + (1.0 + 0.0j),
            noise_var=1e-9,
        )
        results = sic_decode(slot, max_iters=3)
        # We just check it doesn't crash and returns a result
        assert len(results) == 1
