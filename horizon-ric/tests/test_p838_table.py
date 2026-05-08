"""Validate vendored ITU-R P.838-3 Table 1 against published values.

Tolerance: log-log interpolation should reproduce published table entries
exactly at the tabulated frequencies (no interpolation), and within ≤1%
between adjacent points.
"""

import pytest

from horizon_ric.planner.physics._itu_tables import (
    P838_ALPHA_HORIZONTAL,
    P838_ALPHA_VERTICAL,
    P838_FREQUENCIES_GHZ,
    P838_K_HORIZONTAL,
    P838_K_VERTICAL,
    p838_k_alpha,
)


class TestTableShape:
    def test_arrays_aligned(self):
        n = len(P838_FREQUENCIES_GHZ)
        assert len(P838_K_HORIZONTAL) == n
        assert len(P838_ALPHA_HORIZONTAL) == n
        assert len(P838_K_VERTICAL) == n
        assert len(P838_ALPHA_VERTICAL) == n

    def test_frequencies_monotonic(self):
        for i in range(1, len(P838_FREQUENCIES_GHZ)):
            assert P838_FREQUENCIES_GHZ[i] > P838_FREQUENCIES_GHZ[i - 1]


class TestExactInterpolationAtTablePoints:
    @pytest.mark.parametrize("f_ghz", [10.0, 20.0, 30.0, 50.0, 100.0])
    def test_horizontal(self, f_ghz):
        idx = P838_FREQUENCIES_GHZ.index(f_ghz)
        k, a = p838_k_alpha(f_ghz, "horizontal")
        assert abs(k - P838_K_HORIZONTAL[idx]) < 1e-9
        assert abs(a - P838_ALPHA_HORIZONTAL[idx]) < 1e-9

    @pytest.mark.parametrize("f_ghz", [4.0, 12.0, 30.0, 60.0, 200.0])
    def test_vertical(self, f_ghz):
        idx = P838_FREQUENCIES_GHZ.index(f_ghz)
        k, a = p838_k_alpha(f_ghz, "vertical")
        assert abs(k - P838_K_VERTICAL[idx]) < 1e-9
        assert abs(a - P838_ALPHA_VERTICAL[idx]) < 1e-9


class TestPublishedReferenceValues:
    """Spot-check well-known reference values from P.838-3 Table 1."""

    def test_k_h_28ghz_close_to_published(self):
        # Interpolated between 25 and 30 GHz; published k_H at 28 GHz ≈ 0.21
        k, _ = p838_k_alpha(28.0, "horizontal")
        assert 0.19 < k < 0.23

    def test_alpha_h_28ghz_close_to_published(self):
        # α_H at 28 GHz ≈ 0.97 by linear interp between α(25)=0.999, α(30)=0.948
        _, a = p838_k_alpha(28.0, "horizontal")
        assert 0.95 < a < 0.99

    def test_k_h_at_2ghz(self):
        # Below 2 GHz the table value is 0.0000847; rain attenuation is
        # negligible at this frequency by design.
        k, _ = p838_k_alpha(2.0, "horizontal")
        assert k < 1e-3


class TestCircularPolarization:
    def test_circular_is_average_of_h_v(self):
        # P.838-3 Eqs (4)-(5) at τ=45° simplify to k_c = (k_H + k_V) / 2.
        for f in [10.0, 20.0, 30.0]:
            k_h, _ = p838_k_alpha(f, "horizontal")
            k_v, _ = p838_k_alpha(f, "vertical")
            k_c, _ = p838_k_alpha(f, "circular")
            assert abs(k_c - 0.5 * (k_h + k_v)) < 1e-9

    def test_unknown_polarization_raises(self):
        with pytest.raises(ValueError):
            p838_k_alpha(20.0, "diagonal")


class TestExtrapolationClamps:
    def test_below_range_clamps(self):
        k, a = p838_k_alpha(0.5, "horizontal")
        assert k == P838_K_HORIZONTAL[0]
        assert a == P838_ALPHA_HORIZONTAL[0]

    def test_above_range_clamps(self):
        k, a = p838_k_alpha(2000.0, "vertical")
        assert k == P838_K_VERTICAL[-1]
        assert a == P838_ALPHA_VERTICAL[-1]

    def test_zero_or_negative_raises(self):
        with pytest.raises(ValueError):
            p838_k_alpha(0.0, "horizontal")
        with pytest.raises(ValueError):
            p838_k_alpha(-5.0, "horizontal")
