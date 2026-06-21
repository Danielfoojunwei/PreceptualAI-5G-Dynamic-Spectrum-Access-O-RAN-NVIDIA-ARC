"""Decision Safety Shield — the AI-RAN-native trust primitive.

The AI proposes; the Shield disposes. Every neural-PHY / RIC decision is
checked against RAN-physics, spectrum-regulatory, AI-PHY, and lawful-intercept
invariants, projected toward the safe set (or a certified classical fallback)
when it violates, and certified. A poisoned, drifted, or adversarial model
cannot emit an unsafe or illegal policy, and every disposition is auditable by
construction.
"""

from horizon_ric.shield.certificate import (
    ConstraintViolation,
    InvariantCheck,
    SafetyCertificate,
    ShieldDisposition,
)
from horizon_ric.shield.invariants import (
    ConstellationLegalityInvariant,
    Invariant,
    LawfulInterceptInvariant,
    MaxEirpInvariant,
    NeuralRxEnvelopeInvariant,
    PfdCeilingInvariant,
    SpectralMaskInvariant,
)
from horizon_ric.shield.shield import (
    Shield,
    ShieldConfig,
    default_ntn_shield,
    default_terrestrial_shield,
)

__all__ = [
    "ConstraintViolation",
    "InvariantCheck",
    "SafetyCertificate",
    "ShieldDisposition",
    "Invariant",
    "SpectralMaskInvariant",
    "MaxEirpInvariant",
    "PfdCeilingInvariant",
    "NeuralRxEnvelopeInvariant",
    "ConstellationLegalityInvariant",
    "LawfulInterceptInvariant",
    "Shield",
    "ShieldConfig",
    "default_terrestrial_shield",
    "default_ntn_shield",
]
