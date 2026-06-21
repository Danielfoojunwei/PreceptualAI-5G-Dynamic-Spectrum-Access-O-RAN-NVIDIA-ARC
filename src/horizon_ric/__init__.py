"""Horizon-RIC — a security & trust rApp for AI-RAN.

A torch-free O-RAN Non-RT-RIC rApp that makes AI-RAN neural-PHY / RIC decisions
*safe and auditable by construction*:

  * Decision Safety Shield (`horizon_ric.shield`) — bounds every proposed action
    against RAN-physics, spectrum-regulatory, AI-PHY, and lawful-intercept
    invariants; projects to a safe action or a certified classical fallback and
    emits a SafetyCertificate.
  * At-decision-time evidence chain (`horizon_ric.evidence`) — SHA-256
    hash-chained, RFC-3161-anchored DecisionRecords with replayable counterfactuals.
  * Model provenance (`horizon_ric.provenance`) — signature over
    (weights ‖ training-manifest), verified on promotion.
  * Robust federated aggregation (`horizon_ric.federated`) — Krum / median /
    trimmed-mean against poisoning clients.
  * O-RAN R1/A1/O1 adapters + Zero-Trust platform controls
    (`horizon_ric.rapp`, `horizon_ric.security`).
"""

from horizon_ric.evidence import (
    DecisionRecord,
    JsonlEvidenceStore,
    SqliteEvidenceStore,
    generate_human_explanation,
)
from horizon_ric.rapp import HorizonRAppLifecycle, RAppState
from horizon_ric.shield import SafetyCertificate, Shield, default_terrestrial_shield

__version__ = "0.2.0"

__all__ = [
    "DecisionRecord",
    "HorizonRAppLifecycle",
    "JsonlEvidenceStore",
    "RAppState",
    "SafetyCertificate",
    "Shield",
    "SqliteEvidenceStore",
    "__version__",
    "default_terrestrial_shield",
    "generate_human_explanation",
]
