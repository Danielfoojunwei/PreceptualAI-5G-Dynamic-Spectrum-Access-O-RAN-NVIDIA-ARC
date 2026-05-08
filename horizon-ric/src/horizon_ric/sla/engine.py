"""SLAEvaluator — evaluates observations against active SLAs.

Behaviour:
  - For each (SLA, target), compares observed metric to threshold.
  - A target is "in-breach" the moment the comparison fails. The
    evaluator tracks first-breach-time per (sla_id, source_metric);
    a breach event is emitted only after `window_s` of sustained breach.
  - On sustained breach, ONE event is created and updated with an ever-
    increasing `lasting_s`. No flood.
  - On recovery (observation passes), the breach state is cleared so
    the next breach is a NEW event with a fresh `id`.
  - Severity is derived from the SLA's `severity_levels` mapping; when
    not specified the engine defaults to "warning". Targets may opt
    into "critical" by inclusion in `severity_levels["critical"]["metrics"]`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from horizon_ric.sla.policy import (
    SLA,
    Severity,
    SLABreachEvent,
    SLOComparison,
    SLOTarget,
)

if TYPE_CHECKING:
    from horizon_ric.evidence.schema import DecisionRecord

logger = structlog.get_logger(__name__)


def _passes(value: float, comparison: SLOComparison, threshold: float) -> bool:
    """Return True iff the observation SATISFIES the SLO target."""
    if comparison == ">=":
        return value >= threshold
    if comparison == "<=":
        return value <= threshold
    if comparison == "==":
        return value == threshold
    raise ValueError(f"unknown comparison: {comparison}")


@dataclass
class _BreachState:
    """In-memory tracking for one (sla_id, source_metric) pair."""

    first_breach_mono: float
    breach_event: SLABreachEvent | None = None  # populated once window_s elapses


def _resolve_severity(sla: SLA, target: SLOTarget) -> Severity:
    """Pick a severity for a breach of `target` under `sla`."""
    levels = sla.severity_levels or {}
    crit = levels.get("critical") or {}
    crit_metrics = crit.get("metrics", []) if isinstance(crit, dict) else []
    if target.metric in crit_metrics:
        return "critical"
    warn = levels.get("warning") or {}
    warn_metrics = warn.get("metrics", []) if isinstance(warn, dict) else []
    if target.metric in warn_metrics:
        return "warning"
    info = levels.get("info") or {}
    info_metrics = info.get("metrics", []) if isinstance(info, dict) else []
    if target.metric in info_metrics:
        return "info"
    # Default — warning is the operator-friendly default.
    return "warning"


@dataclass
class SLAEvaluator:
    """Stateful SLA evaluator. Use `evaluate()` from the rApp main loop."""

    slas: list[SLA]
    _states: dict[tuple[str, str], _BreachState] = field(default_factory=dict)

    @classmethod
    def from_db(cls, engine) -> "SLAEvaluator":
        """Build an evaluator from the SQLite DB."""
        from horizon_ric.sla.policy import load_slas

        return cls(slas=load_slas(engine))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        observation: dict[str, float],
        decision_record: "DecisionRecord | None" = None,
        *,
        now_mono: float | None = None,
    ) -> list[SLABreachEvent]:
        """Evaluate one observation tick. Returns the active breach events.

        The returned list reflects the CURRENT breach state. A sustained
        breach yields the same event id every tick (with `lasting_s`
        updated). A recovery removes the breach from the returned list.
        """
        if now_mono is None:
            now_mono = time.monotonic()
        now_utc = datetime.now(timezone.utc)
        decision_id = decision_record.decision_id if decision_record is not None else None

        active: list[SLABreachEvent] = []
        seen_keys: set[tuple[str, str]] = set()

        for sla in self.slas:
            for target in sla.targets:
                key = (sla.id, target.metric)
                obs_val = observation.get(target.metric)
                if obs_val is None:
                    # Metric not present this tick — leave any existing state alone.
                    continue
                seen_keys.add(key)
                ok = _passes(obs_val, target.comparison, target.threshold)
                state = self._states.get(key)

                if ok:
                    # Recovered (or never breached). Clear state.
                    if state is not None:
                        logger.info(
                            "horizon.sla.recovered",
                            sla_id=sla.id,
                            metric=target.metric,
                            observed=obs_val,
                            threshold=target.threshold,
                        )
                        del self._states[key]
                    continue

                # Breach condition holds.
                if state is None:
                    state = _BreachState(first_breach_mono=now_mono)
                    self._states[key] = state

                elapsed = now_mono - state.first_breach_mono
                if elapsed < target.window_s:
                    # Pending — not yet a breach event.
                    continue

                if state.breach_event is None:
                    state.breach_event = SLABreachEvent(
                        sla_id=sla.id,
                        ts_utc=now_utc,
                        severity=_resolve_severity(sla, target),
                        observed_value=float(obs_val),
                        target_threshold=float(target.threshold),
                        decision_id_at_breach=decision_id,
                        source_metric=target.metric,
                        lasting_s=elapsed,
                    )
                    logger.warning(
                        "horizon.sla.breach",
                        sla_id=sla.id,
                        breach_id=state.breach_event.id,
                        metric=target.metric,
                        severity=state.breach_event.severity,
                        observed=obs_val,
                        threshold=target.threshold,
                        decision_id=decision_id,
                        lasting_s=elapsed,
                    )
                else:
                    # Sustained breach — update lasting_s + observed_value.
                    state.breach_event.lasting_s = elapsed
                    state.breach_event.observed_value = float(obs_val)
                    logger.debug(
                        "horizon.sla.breach.sustained",
                        sla_id=sla.id,
                        breach_id=state.breach_event.id,
                        metric=target.metric,
                        lasting_s=elapsed,
                    )

                active.append(state.breach_event)

        return active

    def active_breaches(self) -> list[SLABreachEvent]:
        """Return all currently-active (post-window) breach events."""
        return [s.breach_event for s in self._states.values() if s.breach_event is not None]

    def reset(self) -> None:
        """Clear all in-memory breach state (used by tests)."""
        self._states.clear()


__all__ = ["SLAEvaluator"]
