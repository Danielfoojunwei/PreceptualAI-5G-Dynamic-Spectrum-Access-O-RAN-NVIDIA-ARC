"""Beam pattern tests — 3GPP TR 38.901 + ITU-R F.1336."""

from horizon_ric.planner.physics.beam_pattern import (
    angle_between_vectors_deg,
    beam_gain_dB,
    omni_pattern_dB,
)


class TestBeamPattern:
    def test_boresight_max_gain(self):
        gain = beam_gain_dB(
            azimuth_deg=0.0, elevation_deg=90.0,
            boresight_az_deg=0.0, boresight_el_deg=90.0,
            max_gain_dBi=14.0,
        )
        assert gain == 14.0

    def test_off_boresight_drops(self):
        boresight = beam_gain_dB(0.0, 90.0, 0.0, 90.0, max_gain_dBi=14.0)
        off = beam_gain_dB(30.0, 90.0, 0.0, 90.0, max_gain_dBi=14.0)
        assert off < boresight

    def test_far_off_axis_hits_floor(self):
        # 90° off boresight should land at sidelobe floor
        gain = beam_gain_dB(
            azimuth_deg=180.0, elevation_deg=90.0,
            boresight_az_deg=0.0, boresight_el_deg=90.0,
            max_gain_dBi=14.0, sidelobe_floor_dBi=-25.0,
        )
        assert gain <= -15  # well below max

    def test_omni_pattern_smooth(self):
        # Omni at 0° elevation = peak; gradual drop above
        peak = omni_pattern_dB(0.0, max_gain_dBi=8.0)
        higher = omni_pattern_dB(45.0, max_gain_dBi=8.0)
        assert peak >= higher
        assert higher > peak - 5  # smooth roll-off, not a cliff


class TestAngleBetweenVectors:
    def test_same_direction_zero(self):
        assert angle_between_vectors_deg(45.0, 30.0, 45.0, 30.0) < 1e-6

    def test_opposite_directions_180(self):
        # Opposite az + 180° apart = ~180°
        a = angle_between_vectors_deg(0.0, 0.0, 180.0, 0.0)
        assert abs(a - 180.0) < 1e-3

    def test_orthogonal_90(self):
        # Two horizons 90° apart
        a = angle_between_vectors_deg(0.0, 0.0, 90.0, 0.0)
        assert abs(a - 90.0) < 1e-3
