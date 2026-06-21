"""Evidence record schema (PARADIGMS.md §H1, PRD §2.6, §4.2.7).

Each A1 policy decision persists a DecisionRecord with:
  - the chosen action and its predicted outcome
  - the top-K rejected alternatives, each with structured rejection reason
  - the model versions that produced the decision
  - actual observed outcome (filled in retroactively at +30s/+1min/+5min)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelVersions(BaseModel):
    """Version tags for the models that produced a decision.

    Required by 3GPP TS 28.105 (AI/ML Management) for provenance.
    """

    encoder: str
    risk_heads: str
    dyna: str
    policy: str
    constraint_layer: str
    rapp: str  # rApp version itself


class PredictedOutcome(BaseModel):
    """Per-decision predicted KPI envelope.

    Names align to 3GPP TS 28.554 §6 KPI definitions where possible.
    """

    sla_risk_30s: float = Field(ge=0.0, le=1.0)
    sla_risk_1min: float = Field(ge=0.0, le=1.0)
    sla_risk_5min: float = Field(ge=0.0, le=1.0)
    gateway_loads: dict[str, float] = Field(default_factory=dict)
    ntn_capacity_used: float = Field(ge=0.0, le=1.0, default=0.0)
    energy_kwh: float = Field(ge=0.0, default=0.0)
    constraint_violations: list[str] = Field(default_factory=list)


class RejectionReasonMachine(BaseModel):
    """Structured rejection reason — machine-readable for audit dashboards."""

    primary_cause: Literal[
        "gateway_overload",
        "sla_breach_predicted",
        "compute_overload",
        "energy_cost",
        "ntn_capacity_exhausted",
        "spectrum_unavailable",
        "constraint_violation_hard",
        "constraint_violation_soft",
        "policy_oscillation",
    ]
    primary_metric: str  # e.g. "gateway_load_G3", "compute_cpu_edge_5"
    predicted_value: float
    threshold: float
    horizon: Literal["30s", "60s", "300s"]


class RejectedAlternative(BaseModel):
    """One rejected alternative action with its predicted outcome and rejection reason.

    `random_seed` pins the planner's RNG state at the moment this
    alternative was generated. Closes Devil-A/C Finding 9/26: a regulator
    replaying ``(state, action_space, planner_config, random_seed)`` MUST
    obtain the identical alternative, so the counterfactual is auditable.
    """

    rank: int = Field(ge=1)
    action: dict[str, Any]
    predicted_outcome: PredictedOutcome
    rejection_reason_machine: RejectionReasonMachine
    rejection_reason_human: str  # natural-language explanation
    random_seed: int = Field(
        ...,
        description=(
            "RNG seed pinned by the planner when this alternative was "
            "produced. Required for reproducibility (Devil A/C #9, #26)."
        ),
    )


class ConstraintCorrection(BaseModel):
    """One projection-step correction applied by the constraint layer.

    Persisted to the audit so a regulator can reproduce the projection.
    """

    constraint_id: str
    severity: Literal["hard", "soft"]
    margin_dB: float | None = None
    message: str


class DecisionRecord(BaseModel):
    """Complete record of one rApp policy decision.

    Persisted to the evidence store for regulatory audit (GDPR Art. 30
    records of processing, 3GPP TS 28.105 provenance, NIST SP 800-92 logs).
    """

    decision_id: str
    timestamp: datetime
    rapp_instance_id: str
    # Tenant-scope identity (Casbin domain). Optional for backward
    # compatibility with records produced before multi-tenancy landed;
    # new records produced inside a `TenantScope` are stamped automatically
    # by `EvidenceStore.append`. The hash chain is computed per-tenant.
    tenant_id: str | None = None

    # Input state — z_resource snapshot is too large to inline; we store a
    # hash and reference to a separate object-store blob.
    state_hash: str
    state_blob_uri: str | None = None

    # Predicted vs chosen
    chosen_action: dict[str, Any]
    predicted_outcome_chosen: PredictedOutcome
    rejected_alternatives: list[RejectedAlternative]
    """Counterfactual top-K rejected alternatives (paradigm H1)."""
    constraint_corrections: list[ConstraintCorrection] = Field(default_factory=list)
    """Every projection step the constraint layer applied to make
    `chosen_action` feasible. Empty when no projection was needed."""

    # Provenance
    model_versions: ModelVersions

    # Reproducibility (Devil A/C #9, #26): the RNG seed that produced
    # `chosen_action` AND `rejected_alternatives`. None for the rare
    # decision path that is fully deterministic (no MPPI sampling).
    random_seed: int | None = None

    # Actual outcomes (filled in at +30s, +1min, +5min)
    actual_outcome_30s: PredictedOutcome | None = None
    actual_outcome_1min: PredictedOutcome | None = None
    actual_outcome_5min: PredictedOutcome | None = None

    # Optional operator override
    operator_override: bool = False
    operator_override_reason: str | None = None

    # Active SLA/SLO breaches at decision time, tied to the decision that
    # occurred during them so the audit chain links each breach to its decision.
    # SLO targets are defined in `deploy/SLO.md`; the breach records are supplied
    # by the operator's SLA/SLO pipeline. Held as `list[dict]` (Pydantic v2
    # round-trips each breach record as a dict through `model_validate`);
    # optional, defaults to None when no breach is active.
    sla_breach_context: list[dict[str, Any]] | None = None

    @classmethod
    def new(cls, **kwargs) -> "DecisionRecord":
        """Construct a record stamped with the current UTC timestamp."""
        kwargs.setdefault("timestamp", datetime.now(timezone.utc))
        return cls(**kwargs)
