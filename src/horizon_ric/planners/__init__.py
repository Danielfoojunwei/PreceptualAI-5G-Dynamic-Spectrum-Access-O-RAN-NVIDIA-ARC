"""Independently written planners, plus the one adapter that hands them to the Shield.

This package exists for WP1 deliverable C, gate G1: *a second, independently
written planner is shielded without modification to either the planner or the
Shield.* :mod:`horizon_ric.planners.ucb_spectrum` is that second planner (UCB1
over a frequency x power arm grid, pure stdlib, no knowledge of the Shield);
:func:`spectrum_choice_to_action` below is the entire seam between it and
:mod:`horizon_ric.shield`.

Keeping the adapter here rather than inside the planner module keeps the planner
importable — and testable — with no part of ``horizon_ric.shield`` anywhere in
sight.
"""

from typing import Any

from horizon_ric.planners.ucb_spectrum import (
    RewardFn,
    SpectrumChoice,
    UcbSpectrumPlanner,
    frequency_arm_grid,
    power_arm_grid,
)

# ---------------------------------------------------------------------------
# ADAPTER — the only seam between an independent planner and the Shield.
# ---------------------------------------------------------------------------


def spectrum_choice_to_action(
    choice: SpectrumChoice,
    *,
    bandwidth_hz: float,
    antenna_gain_dBi: float,
    block: str = "policy_emit",
) -> dict[str, Any]:
    """ADAPTER: translate a :class:`SpectrumChoice` into the Shield's action dict.

    This is a translation layer and it contains **no safety logic**. Read the
    body: it renames three numbers, attaches the two the Shield's action contract
    requires but a spectrum planner has no opinion about (the carrier bandwidth
    and the antenna gain, both properties of the radio the caller owns), casts
    everything to JSON primitives, and returns. There is no clamping, no
    band-edge test, no EIRP arithmetic, no fallback. An illegal choice arrives at
    the Shield exactly as illegal as the planner made it.

    That matters because of what this function is evidence for. The planner was
    **not modified for the Shield**: ``ucb_spectrum.py`` imports nothing from
    ``horizon_ric.shield`` and does not know that an EIRP ceiling or a licensed
    band exists. The Shield was **not modified for the planner**: it is called
    through its ordinary public ``dispose`` contract with the default terrestrial
    invariant chain, and ``scripts/verify_g1_second_planner.py`` pins the SHA-256
    of ``shield.py``, ``invariants.py`` and ``ucb_spectrum.py`` so the gate fails
    if either side is edited to make the other pass. Everything that reconciles
    the two lives in these few lines.

    Deliberately absent from the returned dict: ``emit_blocked`` and
    ``shield_fallback_to``. Those are the Shield's *output* sentinels; an adapter
    that set them would be forging a disposition. Provenance about which planner
    proposed the action belongs on ``dispose(..., model_provenance=...)``, not in
    the action payload.
    """
    return {
        "block": str(block),
        "frequency_hz": float(choice.centre_hz),
        "bandwidth_hz": float(bandwidth_hz),
        "tx_power_dBm": float(choice.tx_power_dBm),
        "antenna_gain_dBi": float(antenna_gain_dBi),
    }


__all__ = [
    "RewardFn",
    "SpectrumChoice",
    "UcbSpectrumPlanner",
    "frequency_arm_grid",
    "power_arm_grid",
    "spectrum_choice_to_action",
]
