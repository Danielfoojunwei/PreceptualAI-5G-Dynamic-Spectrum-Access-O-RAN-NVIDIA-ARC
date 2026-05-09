"""PreceptualAI SLA management surface.

Operators define SLAs (3GPP TS 28.554 §6 KPIs) — this package evaluates
observations against them, emits breach events, routes them to
Prometheus Alertmanager, and walks an escalation policy on critical
severity.

Submodules:
  policy.py             Pydantic v2 SLO/SLA/SLABreachEvent + SQLAlchemy persistence.
  engine.py             SLAEvaluator with sustained-breach dedup.
  alertmanager.py       AlertManager v2 webhook client.
  escalation.py         Escalation policy engine + channel adapters.
  breach_annotation.py  Annotates DecisionRecord with active breaches.
"""

from horizon_ric.sla.alertmanager import AlertManagerClient, format_breach_for_alertmanager
from horizon_ric.sla.breach_annotation import annotate_decision_record
from horizon_ric.sla.engine import SLAEvaluator
from horizon_ric.sla.escalation import (
    EmailChannel,
    EscalationEngine,
    EscalationPolicy,
    EscalationStep,
    PagerDutyChannel,
    SlackChannel,
    WebhookChannel,
)
from horizon_ric.sla.policy import (
    SLA,
    SLABreachEvent,
    SLOTarget,
    create_schema,
    list_slas,
    load_slas,
    persist_breach,
    persist_sla,
)

__all__ = [
    "SLA",
    "SLOTarget",
    "SLABreachEvent",
    "SLAEvaluator",
    "AlertManagerClient",
    "format_breach_for_alertmanager",
    "EscalationPolicy",
    "EscalationStep",
    "EscalationEngine",
    "SlackChannel",
    "PagerDutyChannel",
    "EmailChannel",
    "WebhookChannel",
    "annotate_decision_record",
    "create_schema",
    "load_slas",
    "list_slas",
    "persist_sla",
    "persist_breach",
]
