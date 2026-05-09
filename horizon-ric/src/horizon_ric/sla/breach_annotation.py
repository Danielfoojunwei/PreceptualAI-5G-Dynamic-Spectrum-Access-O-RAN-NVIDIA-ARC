"""Annotate DecisionRecord with active SLA breaches.

Called by the rApp main loop right before persisting a DecisionRecord:

    breaches = evaluator.evaluate(observation, dec_record)
    annotate_decision_record(dec_record, breaches)
    evidence_store.append(dec_record)

The annotation lands in `dec_record.sla_breach_context` as a list of
JSON-serialised `SLABreachEvent` dicts. Storing dicts (not the Pydantic
class itself) avoids a circular import between `evidence.schema` and
`sla.policy`, and the values round-trip through the hash chain because
`SLABreachEvent.model_dump(mode="json")` is canonical-JSON-clean.
"""

from __future__ import annotations

from typing import Iterable

from horizon_ric.evidence.schema import DecisionRecord
from horizon_ric.sla.policy import SLABreachEvent


def annotate_decision_record(
    record: DecisionRecord, breaches: Iterable[SLABreachEvent]
) -> DecisionRecord:
    """Mutate `record` to include the breach context. Returns it for chaining."""
    breaches_list = list(breaches)
    if not breaches_list:
        record.sla_breach_context = None
        return record
    record.sla_breach_context = [b.model_dump(mode="json") for b in breaches_list]
    return record


def extract_breach_context(record: DecisionRecord) -> list[SLABreachEvent]:
    """Reverse of `annotate_decision_record` — rebuild typed breach objects."""
    if not record.sla_breach_context:
        return []
    return [SLABreachEvent.model_validate(b) for b in record.sla_breach_context]


__all__ = ["annotate_decision_record", "extract_breach_context"]
