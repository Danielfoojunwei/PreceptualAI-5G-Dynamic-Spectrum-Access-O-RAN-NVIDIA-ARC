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


class UcbSpectrumActionPlanner:
    """Presents :class:`UcbSpectrumPlanner` as a WP1 :class:`Planner`.

    Why this class exists, plainly: the bare
    :class:`~horizon_ric.planners.ucb_spectrum.UcbSpectrumPlanner` does **not**
    satisfy the WP1 planner contract, and nothing about that is accidental. Its
    method is ``propose(self) -> SpectrumChoice`` — no observation argument, and
    a private dataclass rather than an action dict — because it was written
    against the radio and knows nothing of the Shield. The contract is
    ``propose(observation) -> Action``.

    The trap is that ``isinstance(bare_planner, Planner)`` returns **True**
    anyway. ``runtime_checkable`` compares attribute presence, never signatures,
    so the arity mismatch would surface as a ``TypeError`` inside
    :func:`~horizon_ric.assurance.planner.shielded` — at the call site, looking
    for all the world like conformance until the moment it is used. Use
    :func:`~horizon_ric.assurance.planner.planner_contract_problems` to get a
    real answer; it checks the signature and rejects the bare planner by name.

    So the reconciliation is explicit and lives here rather than being smuggled
    into either side. This wrapper holds the radio constants the planner has no
    opinion about, ignores the observation (UCB1 carries its own state and needs
    no features — recorded rather than hidden, since a planner that ignores its
    input is a fact a reviewer should be told), and delegates translation to
    :func:`spectrum_choice_to_action`. It adds no safety logic, exactly as the
    adapter adds none.
    """

    def __init__(
        self,
        inner: UcbSpectrumPlanner,
        *,
        bandwidth_hz: float,
        antenna_gain_dBi: float,
        block: str = "policy_emit",
        planner_id: str | None = None,
    ) -> None:
        self._inner = inner
        self._bandwidth_hz = float(bandwidth_hz)
        self._antenna_gain_dBi = float(antenna_gain_dBi)
        self._block = str(block)
        self.planner_id = planner_id or getattr(
            inner, "planner_id", "ucb1_spectrum_v1"
        )

    @property
    def inner(self) -> UcbSpectrumPlanner:
        """The unmodified planner, for tests that assert its independence."""
        return self._inner

    def propose(self, observation: Any) -> dict[str, Any]:
        """Satisfy the WP1 contract. ``observation`` is accepted and unused."""
        del observation  # UCB1 is stateful; it needs no per-step features.
        return spectrum_choice_to_action(
            self._inner.propose(),
            bandwidth_hz=self._bandwidth_hz,
            antenna_gain_dBi=self._antenna_gain_dBi,
            block=self._block,
        )

    def observe(self, choice: SpectrumChoice, reward: float) -> None:
        """Forward a realised reward to the inner planner's UCB1 statistics.

        Note what is *not* fed back: the Shield's projection. The planner learns
        from the reward of the action it proposed, not from the action that was
        actually emitted, so its arm statistics never learn that the licence
        exists. That is the honest arrangement for G1 — a planner that adapted to
        the enforcement boundary would no longer be independent of it — and it is
        why the illegal-request count stays high across the whole run instead of
        decaying.
        """
        self._inner.observe(choice, reward)


__all__ = [
    "RewardFn",
    "SpectrumChoice",
    "UcbSpectrumActionPlanner",
    "UcbSpectrumPlanner",
    "frequency_arm_grid",
    "power_arm_grid",
    "spectrum_choice_to_action",
]
