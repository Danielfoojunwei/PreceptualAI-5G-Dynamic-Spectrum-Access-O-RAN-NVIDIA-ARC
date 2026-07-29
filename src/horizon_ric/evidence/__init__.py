"""Evidence + counterfactual explanation layer (paradigm H1).

Every emitted A1 policy ships with structured rejection reasons for the
top-K rejected alternatives. Schema is queryable + regulator-auditable.
"""

from horizon_ric.evidence.explanation import generate_human_explanation
from horizon_ric.evidence.schema import (
    ConstraintCorrection,
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
    RejectedAlternative,
    RejectionReasonMachine,
)
from horizon_ric.evidence.store import (
    EvidenceStore,
    JsonlEvidenceStore,
    PostgresEvidenceStore,
    SqliteEvidenceStore,
    open_evidence_store,
)

__all__ = [
    "ConstraintCorrection",
    "DecisionRecord",
    "EvidenceStore",
    "JsonlEvidenceStore",
    "ModelVersions",
    "PostgresEvidenceStore",
    "PredictedOutcome",
    "RejectedAlternative",
    "RejectionReasonMachine",
    "SqliteEvidenceStore",
    "generate_human_explanation",
    "open_evidence_store",
]
