"""UCB1 spectrum planner — a second, independently written planner (WP1 G1).

WP1 deliverable C gates one claim: *a second, independently written planner is
shielded without modification to either the planner or the Shield.* This module
is that second planner. It was written against the radio, not against the
Shield.

Independence from what already exists in this tree, concretely:

* the production planner (:mod:`horizon_ric.rapp.pipeline`) is a **deterministic
  risk-band rules** planner: it maps a telemetry event's ``sla_risk_30s`` onto
  one of three A1 policy types and emits a schema-conformant payload dict;
* the existing learner (:mod:`horizon_ric.spectrum.federated_q`) is **tabular
  epsilon-greedy Q-learning** over a DSA state/action enumeration, and it
  returns a bare ``numpy`` Q-table;
* this planner is **UCB1** — deterministic optimism-in-the-face-of-uncertainty
  over a discretised (centre-frequency x transmit-power) arm grid, driven by a
  ``reward`` callback supplied by the caller — and it returns its own frozen
  :class:`SpectrumChoice` dataclass: not a dict, not an array. It is pure
  stdlib ``math`` + ``random``; **numpy is deliberately not imported**, so the
  algorithmic independence is visible in the import block rather than merely
  asserted in prose.

What this module deliberately does NOT know:

* it imports nothing from :mod:`horizon_ric.shield` — no invariant, no
  certificate, no ``dispose``. Do not add such an import: the G1 gate
  (``scripts/verify_g1_second_planner.py``) pins this file's SHA-256 and
  ``tests/test_g1_second_planner.py`` re-loads it in a clean interpreter and
  asserts that neither ``horizon_ric`` nor ``numpy`` ends up in ``sys.modules``;
* it has never heard of an EIRP ceiling, a licensed band edge, a guard band, or
  the Shield's action-dict contract. It emits a *radio intent*; translating that
  intent into the Shield's action dict is the adapter's job
  (:func:`horizon_ric.planners.spectrum_choice_to_action`), and that adapter is
  the only seam between the two halves;
* consequently its arm grid is the **hardware** envelope it is handed — the
  power amplifier's range and the front end's tuning range — not the regulatory
  envelope. Given a reward callback that pays for throughput (i.e. any ordinary
  one), UCB1 will converge on the highest-power arm it has, and it will keep
  sampling arms whose carrier sits outside the licence. So this planner *does*
  request illegal emissions, on purpose and by construction. That is load
  bearing: if the unguarded planner never asked for anything illegal, "the
  Shield stopped it" would be an empty claim.

Reward convention: UCB1's confidence radius assumes rewards in ``[0, 1]``. The
caller's callback is expected to honour that; nothing here rescales it, because
rescaling would be the planner quietly reinterpreting the caller's objective.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

# A caller-supplied reward: (centre_hz, tx_power_dBm) -> reward in [0, 1].
RewardFn = Callable[[float, float], float]


@dataclass(frozen=True)
class SpectrumChoice:
    """One arm this planner has decided to play — the planner's whole output.

    This is deliberately *not* the Shield's action dict, and it is deliberately
    not a dict at all: a typed frozen record is this planner's native interface
    style, and keeping it that way is what makes the adapter a visible, auditable
    seam rather than an accident of two components happening to agree on keys.

    ``ucb_score`` is the upper confidence bound that won the selection
    (``inf`` for an arm that has never been played, which is UCB1's cold-start
    rule). It is diagnostic only.
    """

    centre_hz: float
    tx_power_dBm: float
    arm_index: int
    ucb_score: float


def frequency_arm_grid(*, centre_hz: float, span_hz: float, arms: int) -> tuple[float, ...]:
    """``arms`` evenly spaced centre frequencies covering the front end's tuning span.

    Exposed as a module-level function (not buried in ``__init__``) so a caller
    that needs to describe the arm grid — a benchmark declaring its propagation
    model per centre frequency, say — derives it from the *same* code the planner
    plays, instead of re-deriving a subtly different grid.
    """
    if arms < 1:
        raise ValueError("arms must be >= 1")
    if arms == 1:
        return (float(centre_hz),)
    lo = float(centre_hz) - float(span_hz) / 2.0
    step = float(span_hz) / float(arms - 1)
    return tuple(lo + i * step for i in range(arms))


def power_arm_grid(*, floor_dBm: float, ceiling_dBm: float, arms: int) -> tuple[float, ...]:
    """``arms`` evenly spaced transmit powers spanning the amplifier's range.

    ``ceiling_dBm`` is a **hardware** limit (what the PA can produce), not a
    regulatory one. This planner has no way to express a regulatory limit and no
    business enforcing one.
    """
    if arms < 1:
        raise ValueError("arms must be >= 1")
    if arms == 1:
        return (float(floor_dBm),)
    step = (float(ceiling_dBm) - float(floor_dBm)) / float(arms - 1)
    return tuple(float(floor_dBm) + i * step for i in range(arms))


class UcbSpectrumPlanner:
    """UCB1 bandit over a (centre-frequency x transmit-power) arm grid.

    Usage is two lines and needs nothing from the rest of this package::

        planner = UcbSpectrumPlanner(reward=my_reward, tuning_centre_hz=3.445e9,
                                     tuning_span_hz=110e6, pa_ceiling_dBm=35.0)
        choice = planner.step()          # -> SpectrumChoice

    ``step`` plays one round: select the arm with the largest upper confidence
    bound, ask the caller's ``reward`` callback what it paid, fold that into the
    arm's running mean, and return the :class:`SpectrumChoice` that was played.
    ``propose`` and ``observe`` are available separately for callers that want to
    interleave the measurement themselves.

    Deterministic given ``seed``: ``random.Random`` is used only to break ties
    between equal upper confidence bounds and to order the cold-start sweep, and
    Mersenne Twister's stream for these calls is stable across CPython releases
    (unlike ``numpy.random.Generator`` — see :mod:`horizon_ric.runtime_env`).
    """

    planner_id = "ucb1_spectrum_v1"

    def __init__(
        self,
        *,
        reward: RewardFn,
        tuning_centre_hz: float,
        tuning_span_hz: float,
        frequency_arms: int = 9,
        pa_floor_dBm: float = 10.0,
        pa_ceiling_dBm: float = 35.0,
        power_arms: int = 6,
        exploration_c: float = math.sqrt(2.0),
        seed: int = 0,
    ) -> None:
        self._reward = reward
        self._centres = frequency_arm_grid(
            centre_hz=tuning_centre_hz, span_hz=tuning_span_hz, arms=frequency_arms
        )
        self._powers = power_arm_grid(
            floor_dBm=pa_floor_dBm, ceiling_dBm=pa_ceiling_dBm, arms=power_arms
        )
        self._c = float(exploration_c)
        self._rng = random.Random(seed)
        n = len(self._centres) * len(self._powers)
        self._pulls = [0] * n
        self._mean = [0.0] * n
        self._t = 0

    # ── arm grid (read-only introspection) ───────────────────────────────
    @property
    def centre_grid_hz(self) -> tuple[float, ...]:
        return self._centres

    @property
    def power_grid_dBm(self) -> tuple[float, ...]:
        return self._powers

    @property
    def n_arms(self) -> int:
        return len(self._pulls)

    @property
    def rounds_played(self) -> int:
        return self._t

    def arm(self, index: int) -> tuple[float, float]:
        """The ``(centre_hz, tx_power_dBm)`` pair at ``index``."""
        n_power = len(self._powers)
        return self._centres[index // n_power], self._powers[index % n_power]

    def pulls(self, index: int) -> int:
        return self._pulls[index]

    def mean_reward(self, index: int) -> float:
        return self._mean[index]

    # ── UCB1 ─────────────────────────────────────────────────────────────
    def _ucb(self, index: int) -> float:
        pulls = self._pulls[index]
        if pulls == 0:
            return math.inf
        radius = self._c * math.sqrt(math.log(max(self._t, 1)) / pulls)
        return self._mean[index] + radius

    def propose(self) -> SpectrumChoice:
        """Select an arm by UCB1 and return it. Does not consume a reward."""
        cold = [i for i, p in enumerate(self._pulls) if p == 0]
        if cold:
            # UCB1 cold start: every arm is played once, in a seeded order.
            index = self._rng.choice(cold)
            score = math.inf
        else:
            scores = [self._ucb(i) for i in range(self.n_arms)]
            best = max(scores)
            index = self._rng.choice([i for i, s in enumerate(scores) if s == best])
            score = best
        centre_hz, tx_power_dBm = self.arm(index)
        return SpectrumChoice(
            centre_hz=float(centre_hz),
            tx_power_dBm=float(tx_power_dBm),
            arm_index=int(index),
            ucb_score=float(score),
        )

    def observe(self, choice: SpectrumChoice, reward: float) -> None:
        """Fold ``reward`` into the running mean of the arm ``choice`` played."""
        index = choice.arm_index
        self._pulls[index] += 1
        self._t += 1
        k = self._pulls[index]
        self._mean[index] += (float(reward) - self._mean[index]) / k

    def step(self) -> SpectrumChoice:
        """Play one round: propose, measure via the caller's callback, learn."""
        choice = self.propose()
        self.observe(choice, self._reward(choice.centre_hz, choice.tx_power_dBm))
        return choice


__all__ = [
    "RewardFn",
    "SpectrumChoice",
    "UcbSpectrumPlanner",
    "frequency_arm_grid",
    "power_arm_grid",
]
