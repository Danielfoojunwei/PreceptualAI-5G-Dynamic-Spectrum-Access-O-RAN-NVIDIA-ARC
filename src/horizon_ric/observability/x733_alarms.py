"""ITU-T X.733 alarm-record schema for O-RAN.WG10 OAM compliance.

Devil-A Finding #8 / Solver-2 Fix #20 closure.

ITU-T X.733 (Recommendation, "Information technology — Open Systems
Interconnection — Systems Management: Alarm reporting function") defines
the canonical alarm record consumed by the operator's NMS / SMO Fault
Management plane. Every internal Horizon-RIC failure that an operator's
NMS would need to see MUST be mapped onto an X.733 alarm so the SMO can
ingest it via the ``oran-fm-alarm`` YANG notification channel.

Mandatory X.733 fields (this module covers all of them):

  * ``managedObjectClass`` / ``managedObjectInstance`` — what raised it
  * ``alarmType``        — one of communicationsAlarm / qualityOfServiceAlarm
                           / processingErrorAlarm / equipmentAlarm / environmentalAlarm
  * ``probableCause``    — coded reason from the X.733 / TS 28.532 vocab
  * ``perceivedSeverity``— cleared / indeterminate / critical / major / minor / warning
  * ``specificProblem``  — short human-readable description
  * ``notificationIdentifier`` — globally unique id (uuid4 by default)
  * ``eventTime``        — ISO-8601 UTC timestamp
  * ``additionalText``   — free-form expansion (we put the internal event
                           name + structured payload here for ops debugging)

References
----------
* ITU-T X.733 (1992) — Alarm reporting function
* 3GPP TS 28.532 §11 — Alarm IRP
* 3GPP TS 28.622 — Generic NRM IRP
* O-RAN.WG10.O1-Interface v11.00 §6 — Fault Management
"""

from __future__ import annotations

import datetime as dt
import uuid
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field

# ─── X.733 enumerations ─────────────────────────────────────────────────


class X733AlarmType(str, Enum):
    COMMUNICATIONS = "communicationsAlarm"
    QOS = "qualityOfServiceAlarm"
    PROCESSING_ERROR = "processingErrorAlarm"
    EQUIPMENT = "equipmentAlarm"
    ENVIRONMENTAL = "environmentalAlarm"


class X733Severity(str, Enum):
    CLEARED = "cleared"
    INDETERMINATE = "indeterminate"
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    WARNING = "warning"


# ─── alarm record (Pydantic model) ──────────────────────────────────────


class X733Alarm(BaseModel):
    """X.733 alarm record (ITU-T Rec. X.733, TS 28.532 §11)."""

    alarm_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="notificationIdentifier — globally unique alarm id.",
    )
    managed_object_class: str = Field(default="Horizon-RIC.rApp")
    managed_object_instance: str = Field(default="rApp/horizon_ric")
    alarm_type: X733AlarmType
    probable_cause: str
    perceived_severity: X733Severity
    specific_problem: str
    event_time: dt.datetime = Field(
        default_factory=lambda: dt.datetime.now(tz=dt.timezone.utc)
    )
    additional_text: str = ""
    additional_information: dict[str, Any] = Field(default_factory=dict)

    def is_clearing(self) -> bool:
        return self.perceived_severity == X733Severity.CLEARED

    def to_yang_notification(self) -> dict[str, Any]:
        """Render as ``oran-fm-alarm`` YANG notification dict."""
        return {
            "alarm-id": self.alarm_id,
            "managed-object-class": self.managed_object_class,
            "managed-object-instance": self.managed_object_instance,
            "alarm-type": self.alarm_type.value,
            "probable-cause": self.probable_cause,
            "perceived-severity": self.perceived_severity.value,
            "specific-problem": self.specific_problem,
            "event-time": self.event_time.isoformat(),
            "additional-text": self.additional_text,
            "additional-information": dict(self.additional_information),
        }


# ─── failure-mode → alarm mapping ───────────────────────────────────────
#
# Every internal `horizon.*` event the rApp emits has exactly one X.733
# alarm shape. The map below is the authoritative table; new failure
# modes MUST register here before they can be raised.

_FAILURE_MAP: dict[str, dict[str, Any]] = {
    "horizon.cb.opened": {
        "alarm_type": X733AlarmType.PROCESSING_ERROR,
        "probable_cause": "internalServiceFailure",
        "perceived_severity": X733Severity.MAJOR,
        "specific_problem": "Circuit breaker opened on dependency call",
    },
    "horizon.cb.half_open": {
        "alarm_type": X733AlarmType.PROCESSING_ERROR,
        "probable_cause": "internalServiceFailure",
        "perceived_severity": X733Severity.WARNING,
        "specific_problem": "Circuit breaker entered HALF_OPEN probe state",
    },
    "horizon.cb.closed": {
        "alarm_type": X733AlarmType.PROCESSING_ERROR,
        "probable_cause": "internalServiceFailure",
        "perceived_severity": X733Severity.CLEARED,
        "specific_problem": "Circuit breaker recovered to CLOSED",
    },
    "horizon.degraded.entered": {
        "alarm_type": X733AlarmType.QOS,
        "probable_cause": "qosAlarm",
        "perceived_severity": X733Severity.MAJOR,
        "specific_problem": "rApp entered graceful-degradation mode",
    },
    "horizon.degraded.recovered": {
        "alarm_type": X733AlarmType.QOS,
        "probable_cause": "qosAlarm",
        "perceived_severity": X733Severity.CLEARED,
        "specific_problem": "rApp recovered from graceful-degradation mode",
    },
    "horizon.evidence.tamper_detected": {
        "alarm_type": X733AlarmType.PROCESSING_ERROR,
        "probable_cause": "softwareError",
        "perceived_severity": X733Severity.CRITICAL,
        "specific_problem": "Audit chain SHA-256 verification failed",
    },
    "horizon.evidence.append_failed": {
        "alarm_type": X733AlarmType.PROCESSING_ERROR,
        "probable_cause": "storageError",
        "perceived_severity": X733Severity.MAJOR,
        "specific_problem": "Append to evidence store failed",
    },
    "horizon.watchdog.failed": {
        "alarm_type": X733AlarmType.EQUIPMENT,
        "probable_cause": "equipmentFailure",
        "perceived_severity": X733Severity.CRITICAL,
        "specific_problem": "Liveness watchdog timeout",
    },
    "horizon.sla.breach": {
        "alarm_type": X733AlarmType.QOS,
        "probable_cause": "thresholdCrossed",
        "perceived_severity": X733Severity.MAJOR,
        "specific_problem": "SLA breach predicted",
    },
    "horizon.auth.denied": {
        "alarm_type": X733AlarmType.PROCESSING_ERROR,
        "probable_cause": "authenticationFailure",
        "perceived_severity": X733Severity.MINOR,
        "specific_problem": "Authentication denied at security middleware",
    },
    "horizon.a1.emit_refused": {
        "alarm_type": X733AlarmType.COMMUNICATIONS,
        "probable_cause": "callEstablishmentError",
        "perceived_severity": X733Severity.MAJOR,
        "specific_problem": "A1 policy emit refused by guard chain",
    },
}


def map_event_to_alarm(
    event_name: str,
    *,
    additional_information: dict[str, Any] | None = None,
    severity_override: X733Severity | None = None,
) -> X733Alarm:
    """Look up the X.733 shape for an internal ``horizon.*`` event name.

    ``severity_override`` lets a caller bump severity (e.g. a recurring
    minor warning crossing a counter threshold becomes major).
    """
    if event_name not in _FAILURE_MAP:
        raise KeyError(
            f"event {event_name!r} not registered in X.733 alarm map; "
            f"add it to _FAILURE_MAP before raising."
        )
    spec = _FAILURE_MAP[event_name]
    sev = severity_override or spec["perceived_severity"]
    return X733Alarm(
        alarm_type=spec["alarm_type"],
        probable_cause=spec["probable_cause"],
        perceived_severity=sev,
        specific_problem=spec["specific_problem"],
        additional_text=event_name,
        additional_information=dict(additional_information or {}),
    )


# ─── emitter sink ───────────────────────────────────────────────────────


class X733AlarmBus:
    """Process-wide sink for X.733 alarms.

    The bus accepts subscriber callbacks; the SMO O1 NETCONF notifier
    subscribes here in production. Tests subscribe a list-collector.

    The bus also keeps an in-memory active-alarm registry keyed by
    ``(managed_object_instance, alarm_type, probable_cause)`` so a
    ``cleared`` alarm correctly correlates to the prior active alarm.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[X733Alarm], None]] = []
        self._active: dict[tuple[str, str, str], X733Alarm] = {}
        self._history: list[X733Alarm] = []

    def subscribe(self, fn: Callable[[X733Alarm], None]) -> None:
        self._subscribers.append(fn)

    def emit(self, alarm: X733Alarm) -> X733Alarm:
        key = (
            alarm.managed_object_instance,
            alarm.alarm_type.value,
            alarm.probable_cause,
        )
        if alarm.is_clearing():
            self._active.pop(key, None)
        else:
            self._active[key] = alarm
        self._history.append(alarm)
        for fn in list(self._subscribers):
            try:
                fn(alarm)
            except Exception:
                pass
        return alarm

    def emit_event(
        self,
        event_name: str,
        *,
        additional_information: dict[str, Any] | None = None,
        severity_override: X733Severity | None = None,
    ) -> X733Alarm:
        """Convenience: map ``horizon.*`` event → X.733 alarm and emit."""
        alarm = map_event_to_alarm(
            event_name,
            additional_information=additional_information,
            severity_override=severity_override,
        )
        return self.emit(alarm)

    def active_alarms(self) -> list[X733Alarm]:
        return list(self._active.values())

    def history(self) -> list[X733Alarm]:
        return list(self._history)


# Process-wide default bus — runtime modules import this directly.
_DEFAULT_BUS = X733AlarmBus()


def default_bus() -> X733AlarmBus:
    return _DEFAULT_BUS


__all__ = [
    "X733Alarm",
    "X733AlarmBus",
    "X733AlarmType",
    "X733Severity",
    "default_bus",
    "map_event_to_alarm",
]
