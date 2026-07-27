"""Closed-loop learning against the Decision Safety Shield.

This package hosts the shared machinery for studying how the Shield's
projection operator affects a learner that trains on its own decision history:

* :mod:`horizon_ric.learning.shield_env` — a real-data spectrum environment whose
  reward comes from measured DeepMIMO ray tracing, wrapped by the Shield, and a
  verify-gated replay buffer that reads transitions back out of the hash-chained
  evidence store.
"""

from horizon_ric.learning.shield_env import (
    EvidenceIntegrityError,
    ShieldedSpectrumEnv,
    SpectrumAction,
    StepResult,
    Transition,
    load_transitions,
)

__all__ = [
    "EvidenceIntegrityError",
    "ShieldedSpectrumEnv",
    "SpectrumAction",
    "StepResult",
    "Transition",
    "load_transitions",
]
