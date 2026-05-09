"""Guard registry — pluggable pre-emit guard chain.

The :func:`horizon_ric.policy.emit_guards.run_guard_chain` historically
hard-coded the four production guards. This module makes the chain
pluggable by registry: external code can register additional guards
(e.g. an operator-specific O-RAN WG11 attestation) without touching the
emit_guards module.

Each registered guard is a callable returning ``GuardFailure | None``.
The function signature is intentionally flexible — guards take a single
``GuardContext`` dict so adding a new guard does not force a signature
change to ``run_guard_chain``.

Built-in registrations (mirror the original four pre-emit guards):

    head_pretrained               — refuse if SLA head is untrained.
    constraint_context_complete   — refuse if EPFD context is incomplete.
    decision_within_a1_budget     — refuse if decision exceeded budget.
    corrections_recorded          — refuse if corrections were applied
                                     but the audit field is empty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, ClassVar

from horizon_ric.policy.constraints import PreceptualAIConstraintLayer
from horizon_ric.policy.emit_guards import (
    GuardFailure,
    guard_constraint_context_complete,
    guard_corrections_recorded,
    guard_decision_within_a1_budget,
    guard_predicted_outcome_pretrained,
)

logger = logging.getLogger(__name__)


# ─── GuardContext: single bundle passed to every guard ──────────────────


@dataclass
class GuardContext:
    """Bundle of inputs passed to every registered guard.

    Adding a new guard does not change ``run_guard_chain``'s signature —
    it just reads whatever fields it needs from this dataclass.
    """

    head_is_pretrained: bool = True
    constraint_context: dict[str, Any] | None = None
    constraint_layer: PreceptualAIConstraintLayer | None = None
    elapsed_ms: float = 0.0
    policy_period_ms: float = 100.0
    corrections: list[Any] | None = None
    audit_corrections_field: list[dict[str, Any]] | None = None
    extras: dict[str, Any] | None = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}


GuardFn = Callable[[GuardContext], "GuardFailure | None"]


# ─── Built-in adapters ──────────────────────────────────────────────────


def _g_head_pretrained(ctx: GuardContext) -> GuardFailure | None:
    return guard_predicted_outcome_pretrained(ctx.head_is_pretrained)


def _g_constraint_context_complete(ctx: GuardContext) -> GuardFailure | None:
    if ctx.constraint_layer is None:
        return None
    return guard_constraint_context_complete(
        ctx.constraint_context or {}, ctx.constraint_layer
    )


def _g_decision_within_a1_budget(ctx: GuardContext) -> GuardFailure | None:
    return guard_decision_within_a1_budget(ctx.elapsed_ms, ctx.policy_period_ms)


def _g_corrections_recorded(ctx: GuardContext) -> GuardFailure | None:
    return guard_corrections_recorded(ctx.corrections, ctx.audit_corrections_field)


# ─── Registry implementation ────────────────────────────────────────────


@dataclass(frozen=True)
class GuardEntry:
    name: str
    fn: GuardFn


class GuardRegistry:
    """name → callable lookup for emit-guards."""

    _ENTRY_POINT_GROUP: ClassVar[str] = "horizon_ric.guards"

    def __init__(self):
        self._entries: dict[str, GuardEntry] = {}
        self._loaded_entry_points = False

    def register(self, name: str, fn: GuardFn) -> None:
        if not callable(fn):
            raise TypeError(f"guard {name!r} must be callable, got {fn!r}")
        if name in self._entries and self._entries[name].fn is not fn:
            raise ValueError(
                f"guard {name!r} already registered to {self._entries[name].fn!r}"
            )
        self._entries[name] = GuardEntry(name=name, fn=fn)
        logger.debug("registered guard %s -> %s", name, fn)

    def get(self, name: str) -> GuardFn:
        self._maybe_load_entry_points()
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(
                f"unknown guard {name!r}; registered: {sorted(self._entries)}"
            )
        return entry.fn

    def get_guards(self) -> dict[str, GuardFn]:
        """All registered (name → fn) pairs (ordered)."""
        self._maybe_load_entry_points()
        return {n: e.fn for n, e in self._entries.items()}

    def list_guards(self) -> list[str]:
        self._maybe_load_entry_points()
        return list(self._entries)

    def _maybe_load_entry_points(self) -> None:
        if self._loaded_entry_points:
            return
        self._loaded_entry_points = True
        try:
            eps = metadata.entry_points(group=self._ENTRY_POINT_GROUP)
        except Exception:  # pragma: no cover
            return
        for ep in eps:
            try:
                fn = ep.load()
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to load guard entry point %s: %s", ep.name, exc)
                continue
            if callable(fn):
                self.register(ep.name, fn)


registry = GuardRegistry()


def register_guard(name: str, fn: GuardFn) -> None:
    registry.register(name, fn)


def get_guards() -> dict[str, GuardFn]:
    return registry.get_guards()


def list_guards() -> list[str]:
    return registry.list_guards()


# ─── Built-ins registration ─────────────────────────────────────────────


register_guard("head_pretrained", _g_head_pretrained)
register_guard("constraint_context_complete", _g_constraint_context_complete)
register_guard("decision_within_a1_budget", _g_decision_within_a1_budget)
register_guard("corrections_recorded", _g_corrections_recorded)


# ─── Refactored guard-chain runner ──────────────────────────────────────


def run_guard_chain(
    ctx: GuardContext,
    *,
    only: list[str] | None = None,
) -> list[GuardFailure]:
    """Run every registered guard in registration order.

    Args:
        ctx: The guard context (see :class:`GuardContext`).
        only: Optional whitelist of guard names to run.

    Returns:
        Ordered list of (possibly empty) ``GuardFailure``\\ s.
    """
    failures: list[GuardFailure] = []
    for name, fn in get_guards().items():
        if only is not None and name not in only:
            continue
        try:
            f = fn(ctx)
        except Exception as exc:  # defensive — never let a guard crash the rApp
            failures.append(
                GuardFailure(
                    guard_id=f"_guard_internal_error::{name}",
                    message=f"guard {name!r} raised {type(exc).__name__}: {exc}",
                )
            )
            continue
        if f is not None:
            failures.append(f)
    return failures


__all__ = [
    "GuardContext",
    "GuardEntry",
    "GuardFn",
    "GuardRegistry",
    "get_guards",
    "list_guards",
    "register_guard",
    "registry",
    "run_guard_chain",
]
