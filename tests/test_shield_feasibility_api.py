"""The Shield's public feasibility API — the masking predicate (P1).

The Shield knows its constraint set analytically. Before this API existed each
benchmark restated that constraint in its own words, and the restatements did
not agree with the Shield: ``exploration_truncation_loop._arm_illegal`` tests
*centre-frequency*-in-band while :class:`SpectralMaskInvariant` tests
*occupied-bandwidth*-in-band, so the two edge subbands (whose 20 MHz allocation
spills past a band edge) are "legal" to the benchmark and infeasible to the
Shield. Every legality rate measured against the benchmark's own predicate was
therefore measured against the wrong feasible set.

These tests pin the contract that fixes it:

* ``Shield.is_feasible(a)`` is *exactly* "the Shield would not have to correct
  ``a``" — proven by enumerating the benchmarks' action grids and asserting
  ``is_feasible(a) == (dispose(a).safe_action == a)`` cell by cell.
* the boundary action exactly **at** the 33 dBm cap is FEASIBLE (no off-by-one);
* the vectorised mask fast path is equivalent to running the Shield per action;
* masking is *defence in depth*: it never changes what ``dispose`` projects, and
  every projected action is feasible afterwards regardless.

Feasibility is a pure function of the regulatory constraint (band edges,
occupied bandwidth, EIRP ceiling) and never reads the channel matrix, so the
grid enumerations below run on a zero-gain placeholder environment; the
real-data test at the bottom re-runs the same agreement proof against the
committed 4096-receiver DeepMIMO build when it is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from horizon_ric.learning.shield_env import (
    ANTENNA_GAIN_DBI,
    BAND_HI_HZ,
    BAND_LO_HZ,
    MAX_EIRP_DBM,
    N_SUBBANDS,
    ShieldedSpectrumEnv,
    SpectrumAction,
    load_gain_matrix,
    subband_center_hz,
)
from horizon_ric.shield import default_terrestrial_shield

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "datasets/deepmimo_asu_3p5/generated/channel_features.jsonl"

# Geometry of the two published action spaces, restated here verbatim so a
# silent change to either benchmark grid shows up as a test failure.
EIRP_MIN_DBM = 20.0
EIRP_MAX_DBM = 52.0
OOB_FREQ_BINS = 3


def _grid() -> tuple[np.ndarray, np.ndarray]:
    """exploration_truncation_loop's 9 frequency bins x 33 EIRP bins."""
    width = (BAND_HI_HZ - BAND_LO_HZ) / N_SUBBANDS
    in_band = [subband_center_hz(b) for b in range(N_SUBBANDS)]
    oob = [BAND_HI_HZ + (k + 0.5) * width for k in range(OOB_FREQ_BINS)]
    eirps = np.arange(EIRP_MIN_DBM, EIRP_MAX_DBM + 0.5, 1.0)
    return np.asarray(in_band + oob, dtype=np.float64), eirps


@pytest.fixture(scope="module")
def env() -> ShieldedSpectrumEnv:
    # Feasibility never reads ``gains``; a placeholder keeps this module runnable
    # without the licence-gated feature build.
    return ShieldedSpectrumEnv(np.zeros((8, N_SUBBANDS), dtype=np.float64))


# --------------------------------------------------------------------------
# 1. The public API exists and is well formed.
# --------------------------------------------------------------------------
def test_check_returns_one_result_per_invariant() -> None:
    shield = default_terrestrial_shield(band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ)
    action = {
        "block": "policy_emit",
        "frequency_hz": subband_center_hz(3),
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": ANTENNA_GAIN_DBI,
    }
    checks = shield.check(action)
    assert [c.invariant_id for c in checks] == shield.invariant_ids
    assert shield.is_feasible(action) is True
    assert shield.violations(action) == []
    # context is optional and defaults to empty.
    assert shield.check(action, {}) == checks
    # The private spelling still works for the projection loop.
    assert shield._evaluate(action, {}) == checks


def test_is_feasible_fails_closed_on_a_malformed_action() -> None:
    """A NaN payload must be refused, not raise, not sneak through."""
    shield = default_terrestrial_shield(band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ)
    bad = {
        "block": "policy_emit",
        "frequency_hz": float("nan"),
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": ANTENNA_GAIN_DBI,
    }
    assert shield.is_feasible(bad) is False
    violated = [c.invariant_id for c in shield.violations(bad)]
    assert violated[0] == "numeric_domain_sanity"


# --------------------------------------------------------------------------
# 2. is_feasible == "the Shield would not have to correct this action".
# --------------------------------------------------------------------------
def _agreement_count(env: ShieldedSpectrumEnv, actions: list[SpectrumAction]) -> int:
    shield = env._shield
    for action in actions:
        payload = env.action_payload(action)
        disposition = shield.dispose(dict(payload), {}, decision_id="agreement")
        unchanged = disposition.safe_action == payload
        not_projected = not disposition.certificate.projected
        assert env.is_feasible(action) == unchanged == not_projected, action
    return len(actions)


def test_is_feasible_agrees_with_projection_on_the_exploration_grid(
    env: ShieldedSpectrumEnv,
) -> None:
    freqs, eirps = _grid()
    actions = [
        SpectrumAction(float(f), float(e) - ANTENNA_GAIN_DBI) for f in freqs for e in eirps
    ]
    assert len(actions) == 297
    assert _agreement_count(env, actions) == 297


def test_is_feasible_agrees_with_projection_on_the_credit_grid(
    env: ShieldedSpectrumEnv,
) -> None:
    eirps = np.arange(EIRP_MIN_DBM, EIRP_MAX_DBM + 0.5, 1.0)
    actions = [
        SpectrumAction(subband_center_hz(b), float(e) - ANTENNA_GAIN_DBI)
        for b in range(N_SUBBANDS)
        for e in eirps
    ]
    assert len(actions) == 198
    assert _agreement_count(env, actions) == 198


def test_is_feasible_agrees_with_projection_off_grid(env: ShieldedSpectrumEnv) -> None:
    """Half-dB EIRPs and frequencies straddling both band edges, not just bins."""
    freqs = np.linspace(BAND_LO_HZ - 3e7, BAND_HI_HZ + 3e7, 25)
    eirps = np.arange(20.0, 110.5, 2.5)
    actions = [
        SpectrumAction(float(f), float(e) - ANTENNA_GAIN_DBI) for f in freqs for e in eirps
    ]
    assert _agreement_count(env, actions) == len(actions)


# --------------------------------------------------------------------------
# 3. Known-illegal is rejected; the boundary AT the cap is feasible.
# --------------------------------------------------------------------------
def test_known_illegal_actions_are_infeasible(env: ShieldedSpectrumEnv) -> None:
    over_cap = SpectrumAction(subband_center_hz(3), 44.0)  # EIRP 50 dBm
    out_of_band = SpectrumAction(3.62e9, 26.0 - ANTENNA_GAIN_DBI)
    assert over_cap.eirp_dbm > MAX_EIRP_DBM
    assert env.is_feasible(over_cap) is False
    assert env.infeasibility_reasons(over_cap) == ["max_eirp"]
    assert env.is_feasible(out_of_band) is False
    assert env.infeasibility_reasons(out_of_band) == ["spectral_mask_ts38104"]


def test_action_exactly_at_the_cap_is_feasible(env: ShieldedSpectrumEnv) -> None:
    """33.0 dBm is legal. An off-by-one here would silently shrink the action set."""
    at_cap = SpectrumAction(subband_center_hz(3), MAX_EIRP_DBM - ANTENNA_GAIN_DBI)
    assert at_cap.eirp_dbm == pytest.approx(MAX_EIRP_DBM)
    assert env.is_feasible(at_cap) is True
    # …and one ten-thousandth of a dB above it is not.
    assert env.is_feasible(SpectrumAction(at_cap.frequency_hz, at_cap.tx_power_dbm + 1e-4)) is False


def test_edge_subbands_are_infeasible_even_though_their_centre_is_in_band(
    env: ShieldedSpectrumEnv,
) -> None:
    """The exact disagreement the benchmarks' private predicate got wrong.

    Subbands 0 and 5 have a legal *centre* frequency but their 20 MHz occupied
    bandwidth spills past a band edge, so the Shield must correct them.
    """
    feasible = [
        env.is_feasible(SpectrumAction(subband_center_hz(b), 26.0 - ANTENNA_GAIN_DBI))
        for b in range(N_SUBBANDS)
    ]
    assert feasible == [False, True, True, True, True, False]
    for b in (0, 5):
        centre = subband_center_hz(b)
        assert BAND_LO_HZ <= centre <= BAND_HI_HZ  # centre-in-band says "legal"
        assert env.infeasibility_reasons(
            SpectrumAction(centre, 26.0 - ANTENNA_GAIN_DBI)
        ) == ["spectral_mask_ts38104"]


def test_shield_disagrees_with_centre_frequency_legality_on_exactly_28_arms(
    env: ShieldedSpectrumEnv,
) -> None:
    """Pins the measured disagreement so it cannot silently drift back."""
    freqs, eirps = _grid()

    def centre_freq_illegal(f: float, e: float) -> bool:
        return e > MAX_EIRP_DBM + 1e-9 or f < BAND_LO_HZ or f > BAND_HI_HZ

    disagreements = [
        (f, e)
        for f in freqs
        for e in eirps
        if centre_freq_illegal(float(f), float(e))
        != (not env.is_feasible(SpectrumAction(float(f), float(e) - ANTENNA_GAIN_DBI)))
    ]
    assert len(disagreements) == 28
    # All 28 are arms the old predicate called legal and the Shield calls
    # infeasible: the two edge subbands x the 14 at-or-under-cap EIRP bins.
    assert {round(float(f)) for f, _ in disagreements} == {
        round(subband_center_hz(0)),
        round(subband_center_hz(5)),
    }
    assert {float(e) for _, e in disagreements} == set(np.arange(20.0, 34.0, 1.0))
    mask = env.feasible_mask_grid(freqs, eirps)
    assert int(mask.sum()) == 56  # 84 centre-frequency-legal arms minus the 28


# --------------------------------------------------------------------------
# 4. The mask helper.
# --------------------------------------------------------------------------
def test_mask_grid_matches_per_action_shield_evaluation(env: ShieldedSpectrumEnv) -> None:
    """The vectorised fast path is equivalent to running the Shield per cell."""
    for freqs, eirps in (
        _grid(),
        (np.asarray([subband_center_hz(b) for b in range(N_SUBBANDS)]), np.arange(20.0, 53.0)),
        (np.linspace(BAND_LO_HZ - 2e7, BAND_HI_HZ + 2e7, 31), np.arange(20.0, 110.5, 0.5)),
    ):
        fast = env.feasible_mask_grid(freqs, eirps, fast=True)
        slow = env.feasible_mask_grid(freqs, eirps, fast=False)
        assert fast.shape == (len(freqs), len(eirps))
        assert np.array_equal(fast, slow)


def test_mask_grid_ravel_matches_the_flat_arm_index(env: ShieldedSpectrumEnv) -> None:
    """``arm = f_idx * len(eirp_bins) + e_idx`` — the exploration loop's ordering."""
    freqs, eirps = _grid()
    flat = env.feasible_mask_grid(freqs, eirps).ravel()
    for arm in range(len(freqs) * len(eirps)):
        f = float(freqs[arm // len(eirps)])
        e = float(eirps[arm % len(eirps)])
        assert bool(flat[arm]) == env.is_feasible(SpectrumAction(f, e - ANTENNA_GAIN_DBI))


@pytest.mark.parametrize(
    ("cap_dbm", "expected_feasible"), [(26.0, 28), (33.0, 56), (40.0, 84), (46.0, 108), (52.0, 132)]
)
def test_mask_tracks_the_swept_eirp_cap(cap_dbm: float, expected_feasible: int) -> None:
    """safety_utility_frontier_loop sweeps the cap; the mask must follow it.

    A mask hard-coded to 33 dBm would silently mis-report every non-default
    operating point on the frontier.
    """
    swept = ShieldedSpectrumEnv(
        np.zeros((8, N_SUBBANDS), dtype=np.float64), max_eirp_dbm=cap_dbm
    )
    freqs = np.asarray([subband_center_hz(b) for b in range(N_SUBBANDS)])
    eirps = np.arange(EIRP_MIN_DBM, EIRP_MAX_DBM + 0.5, 1.0)
    fast = swept.feasible_mask_grid(freqs, eirps, fast=True)
    assert np.array_equal(fast, swept.feasible_mask_grid(freqs, eirps, fast=False))
    assert int(fast.sum()) == expected_feasible
    # 4 in-band subbands x the EIRP bins at or under the swept cap.
    assert expected_feasible == 4 * int(cap_dbm - EIRP_MIN_DBM + 1)


def test_mask_tracks_the_carrier_bandwidth() -> None:
    """A 5 MHz carrier fits inside the edge subbands; a 20 MHz one does not."""
    narrow = ShieldedSpectrumEnv(np.zeros((8, N_SUBBANDS), dtype=np.float64), bandwidth_hz=5e6)
    freqs = np.asarray([subband_center_hz(b) for b in range(N_SUBBANDS)])
    eirps = np.asarray([26.0])
    fast = narrow.feasible_mask_grid(freqs, eirps, fast=True)
    assert np.array_equal(fast, narrow.feasible_mask_grid(freqs, eirps, fast=False))
    assert fast[:, 0].tolist() == [True] * N_SUBBANDS


def test_feasible_mask_over_a_sequence(env: ShieldedSpectrumEnv) -> None:
    actions = [
        SpectrumAction(subband_center_hz(3), 26.0 - ANTENNA_GAIN_DBI),
        SpectrumAction(subband_center_hz(3), 52.0 - ANTENNA_GAIN_DBI),
        SpectrumAction(subband_center_hz(0), 26.0 - ANTENNA_GAIN_DBI),
    ]
    assert env.feasible_mask(actions).tolist() == [True, False, False]


# --------------------------------------------------------------------------
# 5. Masking is defence in depth, never a substitute for projection.
# --------------------------------------------------------------------------
def test_masking_never_changes_what_the_shield_projects(env: ShieldedSpectrumEnv) -> None:
    """Consulting the mask must not perturb the enforcement path, and every
    infeasible action must still be projected onto a feasible one."""
    freqs, eirps = _grid()
    shield = env._shield
    mask = env.feasible_mask_grid(freqs, eirps).ravel()
    for arm in range(len(freqs) * len(eirps)):
        f = float(freqs[arm // len(eirps)])
        e = float(eirps[arm % len(eirps)])
        action = SpectrumAction(f, e - ANTENNA_GAIN_DBI)
        payload = env.action_payload(action)

        before = shield.dispose(dict(payload), {}, decision_id="a").safe_action
        env.feasible_mask_grid(freqs, eirps)  # mask consulted in between
        after = shield.dispose(dict(payload), {}, decision_id="a").safe_action
        assert before == after

        executed = SpectrumAction(float(after["frequency_hz"]), float(after["tx_power_dBm"]))
        # Unconditional guarantee: post-projection the action is always feasible,
        # whether or not the caller bothered to look at the mask.
        assert env.is_feasible(executed) is True
        assert executed.eirp_dbm <= MAX_EIRP_DBM + 1e-9
        if bool(mask[arm]):
            assert after == payload  # feasible arms pass through untouched


# --------------------------------------------------------------------------
# 6. Same proof, on the real committed DeepMIMO build.
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not FEATURES.is_file(),
    reason="licence-gated real DeepMIMO features absent; built by the realdata workflow",
)
def test_agreement_holds_on_the_real_data_environment() -> None:
    real = ShieldedSpectrumEnv(load_gain_matrix(FEATURES))
    freqs, eirps = _grid()
    actions = [
        SpectrumAction(float(f), float(e) - ANTENNA_GAIN_DBI) for f in freqs for e in eirps
    ]
    assert _agreement_count(real, actions) == 297
    assert int(real.feasible_mask_grid(freqs, eirps).sum()) == 56
