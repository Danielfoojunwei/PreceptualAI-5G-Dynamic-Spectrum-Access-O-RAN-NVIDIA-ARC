"""Verifier registry — pluggable hard-constraint verifiers.

A *verifier* is a tiny ABC with two operations:

  * ``check(action, ctx) -> list[Violation]`` — feasibility predicate.
  * ``project(action, ctx) -> (action, list[Correction])`` — projection
    onto the feasible set.

The composition pattern mirrors :mod:`horizon_ric.io.registry` and
:mod:`horizon_ric.core.encoder_registry`: register by name, retrieve by
name, swap by config flip. Built-in verifiers wrap the existing physics
modules so the production code path is unchanged but newcomers can:

  1. Register their own verifier (e.g. a national radio-licensing rule)
     by calling ``register_verifier("my_rule", MyVerifier)``.
  2. Compose any subset by listing names in
     :class:`PreceptualAIConstraintLayer`'s ``verifier_chain`` config.

Built-in registrations:

    gso_pfd          — GSO arc PFD floor (ITU-R Article 22 pre-check).
    itu_spectral_mask — Permitted-band edge check.
    edge_gpu         — Jetson Orin Nano memory + bandwidth ceiling.
    epfd             — Aggregate EPFD-down (ITU-R S.1503-3).
    li_jurisdiction  — ETSI TS 103 221 LI non-interference.

Each verifier receives the *whole* PreceptualAIConstraintConfig at
construction time so it can pluck out only the fields it needs. This
keeps the registry callable form simple: ``cls(config)`` returns a
ready-to-run verifier.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, ClassVar

from horizon_ric.contracts.constraint_layer import ConstraintViolation

logger = logging.getLogger(__name__)


# ─── Verifier ABC ────────────────────────────────────────────────────────


class Verifier(ABC):
    """Tiny verifier contract.

    Implementations are typically thin wrappers around an existing
    constraint module (e.g. ``planner.physics.epfd``). The check / project
    pair is sufficient for both training-time Lagrangian construction
    (``check`` returns margins) and inference-time projection.
    """

    constraint_id: ClassVar[str]

    @abstractmethod
    def check(
        self, action: dict[str, Any], ctx: dict[str, Any]
    ) -> list[ConstraintViolation]:
        """Return a (possibly empty) list of violations for the action."""

    @abstractmethod
    def project(
        self, action: dict[str, Any], ctx: dict[str, Any]
    ) -> tuple[dict[str, Any], list[ConstraintViolation]]:
        """Project an infeasible action onto the feasible set.

        Returns:
            (feasible_action, applied_corrections).
        """


# Type alias for the registered factory: callable taking a config object
# (or dict) and returning a Verifier instance.
VerifierFactory = Callable[..., Verifier]


# ─── Built-in adapters around the existing modules ───────────────────────


class _PFDVerifier(Verifier):
    """GSO arc PFD pre-check. Wraps ``PreceptualAIConstraintLayer._max_pfd_at_any_gso``."""

    constraint_id = "gso_pfd_floor"

    def __init__(self, config=None):
        from horizon_ric.policy.constraints import (
            PreceptualAIConstraintConfig,
            PreceptualAIConstraintLayer,
        )

        self.cfg = config or PreceptualAIConstraintConfig()
        self._impl = PreceptualAIConstraintLayer(self.cfg)

    def check(self, action, ctx):
        # Filter the parent layer's violations to our id.
        return [
            v for v in self._impl.check_feasibility(action, ctx)
            if v.constraint_id == self.constraint_id
        ]

    def project(self, action, ctx):
        # Step the projection on a single-id projection (drop other ids
        # to keep the verifier contract clean).
        feasible, all_corr = self._impl.project(action, ctx)
        ours = [c for c in all_corr if c.constraint_id == self.constraint_id]
        return feasible, ours


class _SpectralMaskVerifier(Verifier):
    """ITU spectral-mask band-edge check."""

    constraint_id = "itu_spectral_mask"

    def __init__(self, config=None):
        from horizon_ric.policy.constraints import (
            PreceptualAIConstraintConfig,
            PreceptualAIConstraintLayer,
        )

        self.cfg = config or PreceptualAIConstraintConfig()
        self._impl = PreceptualAIConstraintLayer(self.cfg)

    def check(self, action, ctx):
        return [
            v for v in self._impl.check_feasibility(action, ctx)
            if v.constraint_id == self.constraint_id
        ]

    def project(self, action, ctx):
        feasible, all_corr = self._impl.project(action, ctx)
        ours = [c for c in all_corr if c.constraint_id == self.constraint_id]
        return feasible, ours


class _EdgeGPUVerifier(Verifier):
    """Jetson Orin Nano edge-GPU memory + bandwidth ceiling."""

    constraint_id = "edge_gpu_capacity"

    def __init__(self, config=None):
        from horizon_ric.policy.constraints import (
            PreceptualAIConstraintConfig,
            PreceptualAIConstraintLayer,
        )

        self.cfg = config or PreceptualAIConstraintConfig()
        self._impl = PreceptualAIConstraintLayer(self.cfg)

    def check(self, action, ctx):
        return [
            v for v in self._impl.check_feasibility(action, ctx)
            if v.constraint_id == self.constraint_id
        ]

    def project(self, action, ctx):
        feasible, all_corr = self._impl.project(action, ctx)
        ours = [c for c in all_corr if c.constraint_id == self.constraint_id]
        return feasible, ours


class _EPFDVerifier(Verifier):
    """ITU-R S.1503-3 aggregate EPFD-down."""

    constraint_id = "itu_epfd_down"

    def __init__(self, config=None):
        from horizon_ric.policy.constraints import (
            PreceptualAIConstraintConfig,
            PreceptualAIConstraintLayer,
        )

        self.cfg = config or PreceptualAIConstraintConfig()
        self._impl = PreceptualAIConstraintLayer(self.cfg)

    def check(self, action, ctx):
        return [
            v for v in self._impl.check_feasibility(action, ctx)
            if v.constraint_id == self.constraint_id
        ]

    def project(self, action, ctx):
        # EPFD has no projection step in the existing layer (it's a refusal,
        # not a projection): re-run check and return any violations as
        # uncorrectable corrections.
        feasible = dict(action)
        corrections = self.check(feasible, ctx)
        return feasible, corrections


class _LIVerifier(Verifier):
    """Lawful Intercept jurisdiction guard. Wraps ``LIConstraint``."""

    constraint_id = "li_jurisdiction"

    def __init__(self, config=None):
        from horizon_ric.policy.li_constraint import LIConstraint

        # `config` for this verifier is a list of LIJurisdictionRule.
        rules = config if config is not None else []
        self._impl = LIConstraint(rules)

    def check(self, action, ctx):
        return self._impl.check_feasibility(action, ctx)

    def project(self, action, ctx):
        return self._impl.project(action, ctx)


# ─── Registry implementation ─────────────────────────────────────────────


@dataclass(frozen=True)
class VerifierEntry:
    """A registered verifier factory."""

    name: str
    factory: VerifierFactory


class VerifierRegistry:
    """name → factory lookup for verifiers.

    Mirrors the encoder / connector registries: in-process registration is
    simplest, plus opt-in entry-point loading under
    ``horizon_ric.verifiers``.
    """

    _ENTRY_POINT_GROUP: ClassVar[str] = "horizon_ric.verifiers"

    def __init__(self):
        self._entries: dict[str, VerifierEntry] = {}
        self._loaded_entry_points = False

    def register(self, name: str, factory: VerifierFactory) -> None:
        if not callable(factory):
            raise TypeError(
                f"factory for verifier {name!r} must be callable, got {factory!r}"
            )
        if name in self._entries and self._entries[name].factory is not factory:
            raise ValueError(
                f"verifier {name!r} already registered to "
                f"{self._entries[name].factory!r}"
            )
        self._entries[name] = VerifierEntry(name=name, factory=factory)
        logger.debug("registered verifier %s -> %s", name, factory)

    def get(self, name: str, config: Any = None) -> Verifier:
        self._maybe_load_entry_points()
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(
                f"unknown verifier {name!r}; registered: {sorted(self._entries)}"
            )
        instance = entry.factory(config)
        if not isinstance(instance, Verifier):
            raise TypeError(
                f"verifier factory for {name!r} returned non-Verifier "
                f"{type(instance).__name__}"
            )
        return instance

    def list_verifiers(self) -> list[str]:
        self._maybe_load_entry_points()
        return sorted(self._entries)

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
                obj = ep.load()
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "failed to load verifier entry point %s: %s", ep.name, exc
                )
                continue
            if callable(obj):
                self.register(ep.name, obj)


registry = VerifierRegistry()


def register_verifier(name: str, factory: VerifierFactory) -> None:
    registry.register(name, factory)


def get_verifier(name: str, config: Any = None) -> Verifier:
    return registry.get(name, config)


def list_verifiers() -> list[str]:
    return registry.list_verifiers()


# ─── Built-ins registration ─────────────────────────────────────────────


register_verifier("gso_pfd", _PFDVerifier)
register_verifier("itu_spectral_mask", _SpectralMaskVerifier)
register_verifier("edge_gpu", _EdgeGPUVerifier)
register_verifier("epfd", _EPFDVerifier)
register_verifier("li_jurisdiction", _LIVerifier)


# ─── Composed-chain helper ───────────────────────────────────────────────


def run_verifier_chain(
    names: list[str],
    action: dict[str, Any],
    ctx: dict[str, Any],
    *,
    configs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[ConstraintViolation], list[ConstraintViolation]]:
    """Run a sequence of verifiers in order: project → check → project …

    Returns:
        (feasible_action, all_corrections_applied, residual_violations).
    """
    configs = configs or {}
    feasible = dict(action)
    all_corrections: list[ConstraintViolation] = []
    for name in names:
        v = get_verifier(name, configs.get(name))
        feasible, corrections = v.project(feasible, ctx)
        all_corrections.extend(corrections)
    # Final residual check across the full chain.
    residual: list[ConstraintViolation] = []
    for name in names:
        v = get_verifier(name, configs.get(name))
        residual.extend(v.check(feasible, ctx))
    return feasible, all_corrections, residual


__all__ = [
    "Verifier",
    "VerifierEntry",
    "VerifierFactory",
    "VerifierRegistry",
    "get_verifier",
    "list_verifiers",
    "register_verifier",
    "registry",
    "run_verifier_chain",
]
