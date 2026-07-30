"""Assurance interfaces published by WP1.

Two things, both new in WP1 and both deliberately *additive* — nothing in
``src/horizon_ric`` is rewired by importing this package:

* :mod:`horizon_ric.assurance.planner` — the planner interface. Before WP1 there
  was no planner seam anywhere in the codebase:
  :class:`horizon_ric.rapp.pipeline.DecisionPipeline` hard-codes its planner as
  private methods and builds its own Shield. This package *declares* the
  contract (``Planner`` protocol, required/optional action keys, the
  JSON-primitive requirement the certificate signer imposes) and provides
  :func:`~horizon_ric.assurance.planner.shielded` as the adapter. The pipeline
  does not consume it yet.
* :mod:`horizon_ric.assurance.profile` — reflection of a *live* Shield into a
  machine-readable assurance profile plus a stable digest, so the regulatory
  limits actually being enforced are an artefact an auditor can read and pin.

The published JSON Schema documents for both live in ``docs/schemas/`` and are
held to the code by ``tests/test_assurance_schemas.py``.

Stdlib only.
"""

from horizon_ric.assurance.planner import (
    OPTIONAL_ACTION_KEYS,
    OUTPUT_SENTINEL_KEYS,
    REQUIRED_ACTION_KEYS,
    Observation,
    Planner,
    PlannerContractError,
    shielded,
    validate_proposed_action,
)
from horizon_ric.assurance.profile import (
    PROFILE_SCHEMA_VERSION,
    canonical_profile_bytes,
    emit_profile,
    invariant_descriptor,
    profile_digest,
)

__all__ = [
    "OPTIONAL_ACTION_KEYS",
    "OUTPUT_SENTINEL_KEYS",
    "PROFILE_SCHEMA_VERSION",
    "REQUIRED_ACTION_KEYS",
    "Observation",
    "Planner",
    "PlannerContractError",
    "canonical_profile_bytes",
    "emit_profile",
    "invariant_descriptor",
    "profile_digest",
    "shielded",
    "validate_proposed_action",
]
