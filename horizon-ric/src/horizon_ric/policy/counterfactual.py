"""Counterfactual builder (paradigm H1, E2E_DEBUG gap 3).

Convert a TD-MPC2 `PlanResult` (or any candidate-action set) into the
DecisionRecord's `rejected_alternatives` field, with structured
rejection reasons keyed off the per-rollout reward + constraint
violation.

The signature mirrors how the planner already returns its elite tail —
caller passes the elite candidates, their rewards, their per-constraint
violations, plus a metric → rejection-cause mapping.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal, Optional

import torch

from horizon_ric.evidence.explanation import generate_human_explanation
from horizon_ric.evidence.schema import (
    PredictedOutcome,
    RejectedAlternative,
    RejectionReasonMachine,
)

if TYPE_CHECKING:
    from horizon_ric.integrations.viavi_digital_twin import (
        CounterfactualRollout,
        ViaviDigitalTwin,
    )

_HORIZON = Literal["30s", "60s", "300s"]


@dataclass
class AlternativeCandidate:
    """One non-chosen rollout from the planner with its scoring inputs."""

    action: dict
    """Serialisable action dict that would have been emitted."""
    reward_sum: float
    constraint_violation_sum: float
    sla_risk_30s: float = 0.0
    sla_risk_1min: float = 0.0
    sla_risk_5min: float = 0.0
    primary_metric: str = "score"
    primary_value: float | None = None
    threshold: float = 0.0
    horizon: _HORIZON = "30s"


def build_rejected_alternatives(
    candidates: Iterable[AlternativeCandidate],
    chosen_score: float,
    soft_constraint_threshold: float = 1e-6,
    random_seed: int = 0,
    external_twin: Optional["ViaviDigitalTwin"] = None,
    state: Optional[dict] = None,
    chosen_action: Optional[dict] = None,
    twin_n_steps: int = 100,
) -> list[RejectedAlternative]:
    """Materialise an audit-ready list of `RejectedAlternative`s.

    Cause classification:
        * violation > threshold      → "constraint_violation_hard"
        * sla_risk_30s   > 0.20      → "sla_breach_predicted"
        * reward_sum < chosen − 0    → "energy_cost"  (catch-all)
        * else                        → "policy_oscillation"

    Each cause maps to a human explanation via `generate_human_explanation`.

    `random_seed` (Devil A/C #9, #26): the RNG seed pinned during the
    planner call that produced these candidates. Stamped onto every
    emitted ``RejectedAlternative`` so a regulator replaying with the
    same (state, action_space, planner_config, random_seed) sees an
    identical list. Caller-provided so the seed of record is the seed
    actually used by the planner; `0` is a safe default for fully-
    deterministic decision paths.

    ``external_twin`` (M3 part 2): optional :class:`ViaviDigitalTwin`
    hook. When set, each candidate's predicted KPIs are additionally
    cross-checked by replaying ``(state, candidate.action)`` against
    the VIAVI Pipeline 2 digital twin. The cause classification and
    record schema are unchanged — the twin's predicted KPIs are
    reflected back into ``PredictedOutcome.sla_risk_*`` when the
    backend returns an ``"sla_risk"`` channel, otherwise the local
    planner's values are kept. ``state`` and ``chosen_action`` are
    only consulted when ``external_twin`` is provided. If ``None``,
    behaviour is byte-identical to the pre-M3 path so existing
    counterfactual tests stay green.
    """
    twin_results: dict[int, "CounterfactualRollout"] = {}
    if external_twin is not None:
        if state is None or chosen_action is None:
            raise ValueError(
                "external_twin requires `state` and `chosen_action` "
                "to be provided so the twin can replay each candidate.",
            )

    out: list[RejectedAlternative] = []
    sorted_cands = sorted(
        candidates, key=lambda c: c.reward_sum, reverse=True,
    )
    if external_twin is not None and sorted_cands:
        twin_results = _run_twin_for_candidates(
            external_twin,
            state or {},
            chosen_action or {},
            sorted_cands,
            twin_n_steps,
        )

    for rank, cand in enumerate(sorted_cands, start=1):
        if cand.constraint_violation_sum > soft_constraint_threshold:
            cause = "constraint_violation_hard"
        elif cand.sla_risk_30s > 0.20:
            cause = "sla_breach_predicted"
        elif cand.reward_sum < chosen_score:
            cause = "energy_cost"
        else:
            cause = "policy_oscillation"

        primary_value = (
            cand.primary_value
            if cand.primary_value is not None
            else cand.constraint_violation_sum
            if cause == "constraint_violation_hard"
            else cand.sla_risk_30s
        )
        reason = RejectionReasonMachine(
            primary_cause=cause,
            primary_metric=cand.primary_metric,
            predicted_value=float(primary_value),
            threshold=float(cand.threshold),
            horizon=cand.horizon,
        )
        sla30 = float(cand.sla_risk_30s)
        sla60 = float(cand.sla_risk_1min)
        sla300 = float(cand.sla_risk_5min)
        twin_rollout = twin_results.get(rank - 1)
        if twin_rollout is not None:
            twin_sla = twin_rollout.predicted_kpis_alternative.get("sla_risk")
            if twin_sla is not None:
                # The twin gives one number; we mirror it across all
                # three horizons in the absence of horizon-resolved
                # twin output. Keeps existing schema valid.
                sla30 = sla60 = sla300 = float(twin_sla)

        out.append(RejectedAlternative(
            rank=rank,
            action=cand.action,
            predicted_outcome=PredictedOutcome(
                sla_risk_30s=sla30,
                sla_risk_1min=sla60,
                sla_risk_5min=sla300,
            ),
            rejection_reason_machine=reason,
            rejection_reason_human=generate_human_explanation(reason),
            random_seed=int(random_seed),
        ))
    return out


def _run_twin_for_candidates(
    twin: "ViaviDigitalTwin",
    state: dict,
    chosen_action: dict,
    sorted_cands: list[AlternativeCandidate],
    n_steps: int,
) -> dict[int, "CounterfactualRollout"]:
    """Replay each candidate through the twin and return rollouts by index.

    Synchronously invokes the (async) twin via ``asyncio.run`` when no
    loop is running. If a caller is already inside a running loop they
    should drive ``twin.rollout`` directly and post-process — this
    helper exists so the synchronous ``build_rejected_alternatives``
    contract is preserved.
    """
    async def _gather() -> list["CounterfactualRollout"]:
        results: list["CounterfactualRollout"] = []
        for cand in sorted_cands:
            roll = await twin.rollout(
                state,
                chosen_action,
                cand.action,
                n_steps=n_steps,
            )
            results.append(roll)
        return results

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to spin one up.
        rollouts = asyncio.run(_gather())
        return dict(enumerate(rollouts))
    # We are already inside a loop — refuse to nest.
    raise RuntimeError(
        "build_rejected_alternatives(external_twin=...) cannot be "
        "called from inside a running event loop; use the async "
        "twin.rollout API directly and post-process.",
    )


def candidates_from_planner_elite(
    elite_actions: torch.Tensor,
    elite_rewards: torch.Tensor,
    elite_violations: torch.Tensor,
    action_keys: list[str],
    sla_per_candidate: torch.Tensor | None = None,
) -> list[AlternativeCandidate]:
    """Convert the planner's elite top-K rollouts to AlternativeCandidates.

    elite_actions:     (E, A) — first action of each elite rollout.
    elite_rewards:     (E,)
    elite_violations:  (E,)   — summed across constraints + horizon.
    action_keys:       names for the elements of an action vector.
    sla_per_candidate: optional (E, 3) tensor of (sla_30, sla_60, sla_300).
    """
    if elite_actions.ndim != 2:
        raise ValueError(
            f"elite_actions must be (E, A); got {tuple(elite_actions.shape)}"
        )
    E, A = elite_actions.shape
    if A != len(action_keys):
        raise ValueError(
            f"action_keys has {len(action_keys)} names, action dim is {A}"
        )
    if elite_rewards.shape[0] != E or elite_violations.shape[0] != E:
        raise ValueError("rewards/violations must have one entry per elite candidate")

    out: list[AlternativeCandidate] = []
    for i in range(E):
        action = {
            key: float(elite_actions[i, j].item())
            for j, key in enumerate(action_keys)
        }
        if sla_per_candidate is not None:
            s30, s60, s300 = (float(x) for x in sla_per_candidate[i].tolist())
        else:
            s30 = s60 = s300 = 0.0
        out.append(AlternativeCandidate(
            action=action,
            reward_sum=float(elite_rewards[i].item()),
            constraint_violation_sum=float(elite_violations[i].item()),
            sla_risk_30s=s30, sla_risk_1min=s60, sla_risk_5min=s300,
            primary_metric="score",
            primary_value=float(elite_rewards[i].item()),
            threshold=0.0,
            horizon="30s",
        ))
    return out


__all__ = [
    "AlternativeCandidate",
    "build_rejected_alternatives",
    "candidates_from_planner_elite",
]
