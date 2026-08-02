"""The WP1 planner contract is checked by signature, not by ``isinstance``.

This module exists because of a defect found after WP1's two halves were first
committed: the second planner did not satisfy the interface WP1 had just
defined, and nothing detected it. ``UcbSpectrumPlanner.propose`` takes no
observation and returns a private dataclass, yet
``isinstance(planner, Planner)`` returned ``True`` — ``runtime_checkable``
compares attribute *presence* and never signatures, so the mismatch would have
surfaced as a ``TypeError`` deep inside ``shielded()``.

What is pinned here:

* the bare planner is REJECTED by ``planner_contract_problems``, with a message
  that names the missing observation argument;
* ``isinstance`` still says ``True`` for it, so nobody is tempted to go back to
  using ``isinstance`` as a conformance check;
* the explicit ``UcbSpectrumActionPlanner`` wrapper conforms and drives
  ``shielded()`` end to end, with the Shield still enforcing;
* the wrapper does not smuggle safety logic in — an over-ceiling proposal
  arrives at the Shield as illegal as the planner made it, and is projected
  rather than pre-clamped.
"""

from __future__ import annotations

import inspect

import pytest

from horizon_ric.assurance.planner import (
    Planner,
    conforms_to_planner,
    planner_contract_problems,
    shielded,
)
from horizon_ric.planners import (
    UcbSpectrumActionPlanner,
    UcbSpectrumPlanner,
    spectrum_choice_to_action,
)
from horizon_ric.shield import default_terrestrial_shield

BAND_LO = 3.40e9
BAND_HI = 3.50e9
MAX_EIRP = 33.0
BANDWIDTH = 20e6
ANTENNA_GAIN = 6.0

# Deliberately reaches above the regulatory allowance (MAX_EIRP - ANTENNA_GAIN
# = 27.0 dBm) so the planner really does propose illegal actions.
PA_CEILING = 35.0
EIRP_TOL_DB = 1e-6


def _bare() -> UcbSpectrumPlanner:
    return UcbSpectrumPlanner(
        reward=lambda choice: choice.tx_power_dBm / PA_CEILING,
        tuning_centre_hz=(BAND_LO + BAND_HI) / 2.0,
        tuning_span_hz=110e6,
        pa_ceiling_dBm=PA_CEILING,
        seed=0,
    )


def _wrapped() -> UcbSpectrumActionPlanner:
    return UcbSpectrumActionPlanner(
        _bare(), bandwidth_hz=BANDWIDTH, antenna_gain_dBi=ANTENNA_GAIN
    )


def _shield():
    return default_terrestrial_shield(
        band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP
    )


def test_isinstance_is_not_a_conformance_check() -> None:
    """The trap, pinned so it stays visible rather than being rediscovered."""
    bare = _bare()
    assert isinstance(bare, Planner) is True
    assert conforms_to_planner(bare) is False


def test_bare_planner_is_rejected_by_signature() -> None:
    problems = planner_contract_problems(_bare())
    assert len(problems) == 1
    assert "observation" in problems[0]
    # The real signature is quoted back, so the message is actionable.
    assert "propose(observation) -> Action" in problems[0]


def test_wrapper_conforms() -> None:
    wrapped = _wrapped()
    assert planner_contract_problems(wrapped) == []
    assert conforms_to_planner(wrapped) is True
    sig = inspect.signature(wrapped.propose)
    assert len(sig.parameters) == 1


def test_missing_planner_id_is_reported() -> None:
    class NoId:
        def propose(self, observation):  # noqa: ANN001, ANN201
            return {}

    problems = planner_contract_problems(NoId())
    assert any("planner_id" in p for p in problems)


def test_too_many_required_arguments_is_reported() -> None:
    class TwoArgs:
        planner_id = "two_args"

        def propose(self, observation, extra):  # noqa: ANN001, ANN201
            return {}

    problems = planner_contract_problems(TwoArgs())
    assert any("positional" in p for p in problems)


def test_wrapper_drives_shielded_end_to_end() -> None:
    shield = _shield()
    wrapped = _wrapped()
    for i in range(200):
        disposition = shielded(wrapped, shield, {}, decision_id=f"d-{i}")
        cert = disposition.certificate
        if cert.emit_blocked:
            continue
        action = disposition.safe_action
        eirp = action["tx_power_dBm"] + action["antenna_gain_dBi"]
        assert eirp <= MAX_EIRP + EIRP_TOL_DB, action
        half = action["bandwidth_hz"] / 2.0
        assert action["frequency_hz"] - half >= BAND_LO - EIRP_TOL_DB, action
        assert action["frequency_hz"] + half <= BAND_HI + EIRP_TOL_DB, action


def test_wrapper_adds_no_safety_logic() -> None:
    """An over-ceiling choice must reach the Shield still over the ceiling.

    If the wrapper clamped, the G1 result would be measuring the wrapper rather
    than the Shield.
    """
    bare = _bare()
    wrapped = UcbSpectrumActionPlanner(
        bare, bandwidth_hz=BANDWIDTH, antenna_gain_dBi=ANTENNA_GAIN
    )
    proposed_eirps = []
    for _ in range(120):
        action = wrapped.propose({})
        proposed_eirps.append(action["tx_power_dBm"] + action["antenna_gain_dBi"])
    assert max(proposed_eirps) > MAX_EIRP, (
        "the wrapper never let an illegal EIRP through, so this suite would "
        "pass vacuously"
    )


def test_wrapper_output_matches_the_bare_adapter() -> None:
    """The wrapper is exactly inner.propose() piped through the adapter."""
    a = _bare()
    b = _bare()  # same seed, so the same arm sequence
    wrapped = UcbSpectrumActionPlanner(
        a, bandwidth_hz=BANDWIDTH, antenna_gain_dBi=ANTENNA_GAIN
    )
    for _ in range(25):
        via_wrapper = wrapped.propose({})
        via_adapter = spectrum_choice_to_action(
            b.propose(), bandwidth_hz=BANDWIDTH, antenna_gain_dBi=ANTENNA_GAIN
        )
        assert via_wrapper == via_adapter


def test_observe_reaches_the_inner_planner() -> None:
    wrapped = _wrapped()
    before = wrapped.inner.rounds_played
    choice = wrapped.inner.propose()
    wrapped.observe(choice, 1.0)
    assert wrapped.inner.rounds_played > before


def test_no_output_sentinel_is_ever_proposed() -> None:
    wrapped = _wrapped()
    for _ in range(50):
        action = wrapped.propose({})
        assert "emit_blocked" not in action
        assert "shield_fallback_to" not in action


@pytest.mark.parametrize("bad", [None, object(), 42, "planner"])
def test_non_planners_are_rejected(bad: object) -> None:
    assert conforms_to_planner(bad) is False
