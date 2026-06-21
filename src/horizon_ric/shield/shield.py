"""The Decision Safety Shield.

    action_ai          = neural_block(state)        # the AI proposes
    safe, certificate  = shield.dispose(action_ai)  # the verified envelope disposes

The Shield wraps every AI-RAN decision — neural-RX, learned constellation, DPoD,
or a RIC policy — in a uniform contract: it checks the proposed action against a
chain of invariants (RAN physics + spectrum regulation + AI-PHY bounds + lawful
intercept), projects it toward the safe set (or to a certified classical
fallback) when it violates, and returns a :class:`SafetyCertificate`. If it
cannot make the action safe it fails closed (``emit_blocked=True``).

This is the AI-RAN-native trust primitive: a poisoned, drifted, or adversarial
model still cannot emit an unsafe or illegal policy, and every disposition is
evidence-by-construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from horizon_ric.shield.certificate import (
    ConstraintViolation,
    InvariantCheck,
    SafetyCertificate,
    ShieldDisposition,
    utc_now_iso,
)
from horizon_ric.shield.invariants import FALLBACK_KEY, Invariant

logger = logging.getLogger(__name__)


@dataclass
class ShieldConfig:
    # Max projection passes before declaring the action unfixable (fail-closed).
    max_passes: int = 8


class Shield:
    """Runtime-assurance envelope over an ordered chain of invariants."""

    def __init__(
        self,
        invariants: Sequence[Invariant],
        config: ShieldConfig | None = None,
    ) -> None:
        self._invariants = list(invariants)
        self._cfg = config or ShieldConfig()

    @property
    def invariant_ids(self) -> list[str]:
        return [inv.id for inv in self._invariants]

    # ── evaluation ──────────────────────────────────────────────────────
    def _evaluate(self, action: dict[str, Any], context: dict[str, Any]) -> list[InvariantCheck]:
        return [inv.evaluate(action, context) for inv in self._invariants]

    # ── main entry point ────────────────────────────────────────────────
    def dispose(
        self,
        action: dict[str, Any],
        context: dict[str, Any] | None = None,
        *,
        decision_id: str = "",
        rng_seed: Optional[int] = None,
        loop_tier: str = "non_rt",
        model_provenance: Optional[dict[str, Any]] = None,
    ) -> ShieldDisposition:
        """Check → project → certify. Returns the safe action and its certificate."""
        context = dict(context or {})
        proposed = dict(action)
        block = str(action.get("block", "unknown"))
        safe = dict(action)
        corrections: list[ConstraintViolation] = []
        projected = False

        for _ in range(self._cfg.max_passes):
            checks = self._evaluate(safe, context)
            unsatisfied = [c for c in checks if not c.satisfied]
            if not unsatisfied:
                break
            projected = True
            for inv in self._invariants:
                safe, corr = inv.project(safe, context)
                corrections.extend(corr)
            if safe.get("emit_blocked"):
                break

        final_checks = self._evaluate(safe, context)
        violated_ids = [c.invariant_id for c in final_checks if not c.satisfied]
        emit_blocked = bool(safe.get("emit_blocked")) or bool(violated_ids)
        safe_ok = not violated_ids
        if emit_blocked and not safe.get("emit_blocked"):
            # Residual hard violation the projections could not resolve.
            safe["emit_blocked"] = True

        fallback_to = safe.get(FALLBACK_KEY)
        block_changed_to_classical = (
            safe.get("block") != block
            and str(safe.get("block", "")).startswith("classical")
        )
        fallback_used = (fallback_to is not None) or block_changed_to_classical

        certificate = SafetyCertificate(
            decision_id=decision_id,
            issued_at=utc_now_iso(),
            loop_tier=loop_tier,
            block=block,
            action_proposed=proposed,
            action_safe=dict(safe),
            invariants=final_checks,
            corrections=corrections,
            violated_ids=violated_ids,
            projected=projected,
            fallback_used=bool(fallback_used),
            fallback_to=fallback_to,
            safe=safe_ok,
            emit_blocked=emit_blocked,
            rng_seed=rng_seed,
            model_provenance=model_provenance,
        )
        if emit_blocked:
            logger.warning(
                "shield.emit_blocked decision_id=%s violated=%s", decision_id, violated_ids
            )
        return ShieldDisposition(safe_action=dict(safe), certificate=certificate)


def default_terrestrial_shield(
    *,
    band_lo_hz: float,
    band_hi_hz: float,
    max_eirp_dBm: float = 33.0,
    li_constraint: Any | None = None,
    max_papr_dB: float = 8.5,
    neural_rx_tolerance_dB: float = 1.0,
    guard_band_hz: float = 0.0,
) -> Shield:
    """Build a Shield with the standard terrestrial AI-RAN invariant chain.

    Order matters: lawful intercept first (fail-closed), then spectrum/power,
    then the AI-PHY envelopes. Pass an :class:`LIConstraint` to enforce TS
    33.127; omit it and the chain runs without LI (lab use only).

    ``guard_band_hz`` keeps the occupied bandwidth that far inside each band edge
    so a clipped carrier does not leak adjacent-channel power (ACLR compliance).
    """
    from horizon_ric.shield.invariants import (
        ConstellationLegalityInvariant,
        LawfulInterceptInvariant,
        MaxEirpInvariant,
        NeuralRxEnvelopeInvariant,
        SpectralMaskInvariant,
    )

    chain: list[Invariant] = []
    if li_constraint is not None:
        chain.append(LawfulInterceptInvariant(li=li_constraint))
    chain.extend(
        [
            SpectralMaskInvariant(
                band_lo_hz=band_lo_hz, band_hi_hz=band_hi_hz, guard_band_hz=guard_band_hz
            ),
            MaxEirpInvariant(max_eirp_dBm=max_eirp_dBm),
            NeuralRxEnvelopeInvariant(tolerance_dB=neural_rx_tolerance_dB),
            ConstellationLegalityInvariant(max_papr_dB=max_papr_dB),
        ]
    )
    return Shield(chain)


def default_ntn_shield(
    *,
    band_lo_hz: float,
    band_hi_hz: float,
    max_eirp_dBm: float = 33.0,
    max_pfd_dBW_m2_MHz: float = -146.0,
    li_constraint: Any | None = None,
    neural_rx_tolerance_dB: float = 1.0,
) -> Shield:
    """Build a Shield for a non-terrestrial (LEO-satellite) AI-RAN link.

    Same chain as the terrestrial shield plus an ITU-R-style downlink
    power-flux-density ceiling (:class:`PfdCeilingInvariant`) for LEO-satellite
    coexistence — the team's privacy-preserving LEO power-control problem class.
    A poisoned LEO power-control agent cannot command a downlink that breaches
    the PFD mask at the Earth's surface.
    """
    from horizon_ric.shield.invariants import (
        LawfulInterceptInvariant,
        MaxEirpInvariant,
        NeuralRxEnvelopeInvariant,
        PfdCeilingInvariant,
        SpectralMaskInvariant,
    )

    chain: list[Invariant] = []
    if li_constraint is not None:
        chain.append(LawfulInterceptInvariant(li=li_constraint))
    chain.extend(
        [
            SpectralMaskInvariant(band_lo_hz=band_lo_hz, band_hi_hz=band_hi_hz),
            MaxEirpInvariant(max_eirp_dBm=max_eirp_dBm),
            PfdCeilingInvariant(max_pfd_dBW_m2_MHz=max_pfd_dBW_m2_MHz),
            NeuralRxEnvelopeInvariant(tolerance_dB=neural_rx_tolerance_dB),
        ]
    )
    return Shield(chain)


__all__ = [
    "Shield",
    "ShieldConfig",
    "default_terrestrial_shield",
    "default_ntn_shield",
]
