"""Counterfactual explanation generator (paradigm H1).

Converts a structured RejectionReasonMachine into a human-readable
sentence that's auditable by an operator or regulator.

Templates are deliberately simple and deterministic — we do NOT use an LLM
to generate explanations because that would itself need certification.
"""

from __future__ import annotations

from horizon_ric.evidence.schema import RejectionReasonMachine

# Per-cause template. Use {primary_metric}, {predicted_value}, {threshold},
# {horizon} as substitution placeholders.
_TEMPLATES: dict[str, str] = {
    "gateway_overload": (
        "Predicted to push {primary_metric} to {predicted_value:.0%} load at +{horizon}, "
        "exceeding the {threshold:.0%} safety threshold and risking SLA breach for "
        "downstream slices."
    ),
    "sla_breach_predicted": (
        "Predicted SLA breach probability {predicted_value:.0%} on {primary_metric} "
        "at +{horizon}, above the {threshold:.0%} threshold."
    ),
    "compute_overload": (
        "Edge compute {primary_metric} predicted at {predicted_value:.0%} utilization "
        "at +{horizon}, exceeding the {threshold:.0%} ceiling and risking AI workload "
        "queueing delays."
    ),
    "energy_cost": (
        "Energy cost on {primary_metric} predicted at {predicted_value:.2f} kWh "
        "at +{horizon}, exceeding the {threshold:.2f} kWh budget without compensating "
        "SLA gain."
    ),
    "ntn_capacity_exhausted": (
        "NTN capacity on {primary_metric} predicted at {predicted_value:.0%} at "
        "+{horizon}, exceeding the {threshold:.0%} reserve target."
    ),
    "spectrum_unavailable": (
        "Spectrum allocation conflict on {primary_metric} at +{horizon}: "
        "predicted utilization {predicted_value:.0%} vs allowed {threshold:.0%}."
    ),
    "constraint_violation_hard": (
        "Action would violate hard regulatory constraint on {primary_metric}: "
        "predicted {predicted_value:.2f}, limit {threshold:.2f}. Hard rejection."
    ),
    "constraint_violation_soft": (
        "Action would violate soft operator-policy constraint on {primary_metric}: "
        "predicted {predicted_value:.2f}, threshold {threshold:.2f}. Soft preference."
    ),
    "policy_oscillation": (
        "Predicted policy oscillation on {primary_metric}: change rate "
        "{predicted_value:.2f} above threshold {threshold:.2f} within {horizon}."
    ),
}


def generate_human_explanation(reason: RejectionReasonMachine) -> str:
    """Convert a machine-readable rejection reason to a human sentence.

    Args:
        reason: structured rejection reason.

    Returns:
        Natural-language explanation suitable for operator dashboards.
    """
    template = _TEMPLATES.get(reason.primary_cause)
    if template is None:
        return (
            f"Rejected due to {reason.primary_cause} on {reason.primary_metric}: "
            f"predicted {reason.predicted_value:.4f} vs threshold {reason.threshold:.4f} "
            f"at +{reason.horizon}."
        )
    return template.format(
        primary_metric=reason.primary_metric,
        predicted_value=reason.predicted_value,
        threshold=reason.threshold,
        horizon=reason.horizon,
    )
