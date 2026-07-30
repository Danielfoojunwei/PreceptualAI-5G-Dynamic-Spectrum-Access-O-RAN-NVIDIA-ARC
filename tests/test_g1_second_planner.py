"""WP1 gate G1 — a second, independently written planner is shielded unmodified.

These tests pin the four properties that make the G1 claim mean something:

1. **Independence.** ``horizon_ric.planners.ucb_spectrum`` imports nothing from
   ``horizon_ric.shield`` (and no numpy). Asserted twice: on the module source,
   and by executing the module standalone in a clean interpreter and checking
   that neither ``horizon_ric`` nor ``numpy`` reaches ``sys.modules``. The
   in-process ``sys.modules`` cannot answer this on its own, because this test
   file imports the Shield itself.
2. **The adapter is a translation layer only.** Its output is JSON-primitive
   throughout — numpy scalars would break certificate signing — and survives a
   ``json.dumps`` round trip unchanged.
3. **Enforcement.** Over a few hundred decisions from the independent planner,
   every disposition the Shield did not block satisfies EIRP <= ceiling and sits
   inside the licensed channel less its guard band.
4. **Non-vacuity.** The planner genuinely requests illegal emissions. Property 3
   is trivially true of a planner that only ever asks for legal things, so this
   test fails if the arm grid is ever quietly made safe.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from horizon_ric.planners import (
    UcbSpectrumPlanner,
    frequency_arm_grid,
    spectrum_choice_to_action,
    ucb_spectrum,
)
from horizon_ric.shield import default_terrestrial_shield

BAND_LO_HZ = 3.40e9
BAND_HI_HZ = 3.50e9
GUARD_BAND_HZ = 2.0e6
MAX_EIRP_DBM = 33.0
ANTENNA_GAIN_DBI = 6.0
CARRIER_BW_HZ = 20.0e6

TUNING_CENTRE_HZ = 3.445e9
TUNING_SPAN_HZ = 110.0e6
FREQUENCY_ARMS = 9
PA_FLOOR_DBM = 10.0
PA_CEILING_DBM = 35.0  # above MAX_EIRP_DBM - ANTENNA_GAIN_DBI (= 27.0 dBm)
POWER_ARMS = 6

DECISIONS = 400
SEED = 7
EIRP_TOL_DB = 1e-6
BAND_TOL_HZ = 1e-6
USABLE_LO_HZ = BAND_LO_HZ + GUARD_BAND_HZ
USABLE_HI_HZ = BAND_HI_HZ - GUARD_BAND_HZ

JSON_PRIMITIVES = (str, int, float, bool, type(None))
FORBIDDEN_IMPORTS = ("horizon_ric.shield", "numpy")


def _reward(centre_hz: float, tx_power_dBm: float) -> float:
    """Plain throughput-seeking reward: nothing in it mentions the licence."""
    sinr_dB = tx_power_dBm + ANTENNA_GAIN_DBI - 30.0 - (centre_hz % 7.0e6) / 1.0e6
    return min(1.0, max(0.0, math.log2(1.0 + 10.0 ** (sinr_dB / 10.0)) / 4.0))


def _planner() -> UcbSpectrumPlanner:
    return UcbSpectrumPlanner(
        reward=_reward,
        tuning_centre_hz=TUNING_CENTRE_HZ,
        tuning_span_hz=TUNING_SPAN_HZ,
        frequency_arms=FREQUENCY_ARMS,
        pa_floor_dBm=PA_FLOOR_DBM,
        pa_ceiling_dBm=PA_CEILING_DBM,
        power_arms=POWER_ARMS,
        seed=SEED,
    )


def _shield():
    return default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ,
        band_hi_hz=BAND_HI_HZ,
        max_eirp_dBm=MAX_EIRP_DBM,
        guard_band_hz=GUARD_BAND_HZ,
    )


def _is_illegal(action: dict) -> bool:
    """Independent legality check — first principles, no Shield constants."""
    eirp_dBm = float(action["tx_power_dBm"]) + float(action.get("antenna_gain_dBi", 0.0))
    half_bw_hz = float(action["bandwidth_hz"]) / 2.0
    lower_edge_hz = float(action["frequency_hz"]) - half_bw_hz
    upper_edge_hz = float(action["frequency_hz"]) + half_bw_hz
    return (
        eirp_dBm > MAX_EIRP_DBM + EIRP_TOL_DB
        or lower_edge_hz < USABLE_LO_HZ - BAND_TOL_HZ
        or upper_edge_hz > USABLE_HI_HZ + BAND_TOL_HZ
    )


# 1. Independence ------------------------------------------------------------
def test_planner_source_does_not_reference_the_shield_or_numpy() -> None:
    source = Path(ucb_spectrum.__file__).read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_IMPORTS:
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


def test_planner_loads_with_no_shield_and_no_numpy_in_a_clean_interpreter() -> None:
    """Execute the planner module standalone: nothing of horizon_ric, no numpy.

    Importing it by its dotted path would drag in ``horizon_ric/__init__.py``
    (which does re-export the Shield), so the module is loaded straight from its
    file. That isolates the question to this one module's own imports.
    """
    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('ucb_probe', {ucb_spectrum.__file__!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        # dataclasses resolves annotations through sys.modules, so the standalone
        # module has to be registered under its own throwaway name first.
        "sys.modules['ucb_probe'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "leaked = sorted(k for k in sys.modules if k.split('.')[0] in ('horizon_ric', 'numpy'))\n"
        "print(mod.UcbSpectrumPlanner.planner_id, leaked)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "ucb1_spectrum_v1 []"


def test_planner_returns_its_own_dataclass_not_a_dict() -> None:
    choice = _planner().step()
    assert isinstance(choice, ucb_spectrum.SpectrumChoice) is True
    assert isinstance(choice, dict) is False
    with pytest.raises((AttributeError, TypeError)):
        choice.centre_hz = 1.0  # type: ignore[misc]


# 2. The adapter is a translation layer only ---------------------------------
def test_adapter_output_is_json_primitive_throughout() -> None:
    action = spectrum_choice_to_action(
        _planner().step(), bandwidth_hz=CARRIER_BW_HZ, antenna_gain_dBi=ANTENNA_GAIN_DBI
    )
    for key, value in action.items():
        assert isinstance(key, str) is True
        assert isinstance(value, JSON_PRIMITIVES) is True
        # A numpy scalar passes the isinstance check above via float subclassing
        # in some versions; pin the exact types the certificate signer accepts.
        assert type(value) in (str, float, int, bool, type(None))
    assert json.loads(json.dumps(action)) == action


def test_adapter_does_not_forge_the_shields_output_sentinels() -> None:
    action = spectrum_choice_to_action(
        _planner().step(), bandwidth_hz=CARRIER_BW_HZ, antenna_gain_dBi=ANTENNA_GAIN_DBI
    )
    assert ("emit_blocked" in action) is False
    assert ("shield_fallback_to" in action) is False


def test_adapter_applies_no_safety_logic() -> None:
    """An illegal choice must arrive at the Shield exactly as illegal as it left."""
    choice = ucb_spectrum.SpectrumChoice(
        centre_hz=3.39e9, tx_power_dBm=35.0, arm_index=0, ucb_score=math.inf
    )
    action = spectrum_choice_to_action(
        choice, bandwidth_hz=CARRIER_BW_HZ, antenna_gain_dBi=ANTENNA_GAIN_DBI
    )
    assert action["frequency_hz"] == pytest.approx(3.39e9)
    assert action["tx_power_dBm"] == pytest.approx(35.0)
    assert _is_illegal(action) is True


# 3. Enforcement + 4. Non-vacuity -------------------------------------------
def test_every_emitted_disposition_is_legal_and_the_planner_is_not_vacuously_safe() -> None:
    planner = _planner()
    shield = _shield()
    illegal_requests = 0
    emitted = 0
    peak_emitted_eirp_dBm = -math.inf

    for d in range(DECISIONS):
        proposed = spectrum_choice_to_action(
            planner.step(), bandwidth_hz=CARRIER_BW_HZ, antenna_gain_dBi=ANTENNA_GAIN_DBI
        )
        illegal_requests += int(_is_illegal(proposed))

        disposition = shield.dispose(proposed, decision_id=f"t-{d:04d}")
        if disposition.certificate.emit_blocked:
            continue

        emitted += 1
        safe = disposition.safe_action
        eirp_dBm = float(safe["tx_power_dBm"]) + float(safe["antenna_gain_dBi"])
        half_bw_hz = float(safe["bandwidth_hz"]) / 2.0
        assert eirp_dBm <= MAX_EIRP_DBM + EIRP_TOL_DB
        assert float(safe["frequency_hz"]) - half_bw_hz >= USABLE_LO_HZ - BAND_TOL_HZ
        assert float(safe["frequency_hz"]) + half_bw_hz <= USABLE_HI_HZ + BAND_TOL_HZ
        assert _is_illegal(safe) is False
        assert disposition.certificate.safe is True
        peak_emitted_eirp_dBm = max(peak_emitted_eirp_dBm, eirp_dBm)

    # Non-vacuity: the unshielded planner really did demand illegal emissions.
    assert illegal_requests > 0
    assert emitted > 0
    # The ceiling is actually reached, so the bound above is tight rather than
    # merely satisfied by a planner that stayed well inside it.
    assert peak_emitted_eirp_dBm == pytest.approx(MAX_EIRP_DBM, abs=EIRP_TOL_DB)


def test_the_arm_grid_itself_contains_illegal_arms() -> None:
    """Pins the source of the non-vacuity: the grid overhangs the licence.

    If someone narrows the tuning span or drops the PA ceiling to 27 dBm, the
    Shield would have nothing to correct and the gate above would pass while
    proving nothing. This test fails first, and says why.
    """
    planner = _planner()
    illegal_arms = sum(
        1
        for i in range(planner.n_arms)
        for centre_hz, tx_power_dBm in (planner.arm(i),)
        if _is_illegal(
            {
                "frequency_hz": centre_hz,
                "bandwidth_hz": CARRIER_BW_HZ,
                "tx_power_dBm": tx_power_dBm,
                "antenna_gain_dBi": ANTENNA_GAIN_DBI,
            }
        )
    )
    assert illegal_arms > 0
    assert max(planner.power_grid_dBm) > MAX_EIRP_DBM - ANTENNA_GAIN_DBI
    assert min(planner.centre_grid_hz) - CARRIER_BW_HZ / 2.0 < USABLE_LO_HZ
    assert max(planner.centre_grid_hz) + CARRIER_BW_HZ / 2.0 > USABLE_HI_HZ


def test_the_arm_grid_is_derived_from_the_same_helper_the_caller_sees() -> None:
    """A benchmark declaring a per-centre channel must get the planner's own grid."""
    planner = _planner()
    assert planner.centre_grid_hz == frequency_arm_grid(
        centre_hz=TUNING_CENTRE_HZ, span_hz=TUNING_SPAN_HZ, arms=FREQUENCY_ARMS
    )


def test_the_planner_is_deterministic_for_a_given_seed() -> None:
    """Committed counts are only reproducible if the decision stream is."""

    def trace(planner: UcbSpectrumPlanner) -> list[tuple[int, float, float]]:
        return [
            (c.arm_index, c.centre_hz, c.tx_power_dBm)
            for c in (planner.step() for _ in range(120))
        ]

    assert trace(_planner()) == trace(_planner())
