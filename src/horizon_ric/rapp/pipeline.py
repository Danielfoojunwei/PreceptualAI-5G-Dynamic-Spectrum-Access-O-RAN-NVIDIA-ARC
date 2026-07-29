"""Decision pipeline — telemetry → planner → Shield → guards → A1 → evidence.

This is the control-loop core the production daemon (``scripts/run_horizon_rapp``)
drives for every ingested :class:`~horizon_ric.io.schemas.TelemetryEvent`:

    1. deterministic risk-band planner proposes an A1 policy (+ alternatives)
    2. the Decision Safety Shield disposes the proposed action
    3. the pre-emit guard chain makes the final go/no-go call
    4. the policy is emitted over A1 (the adapter persists the DecisionRecord
       and increments the emit metrics itself on success)
    5. enforcement status is polled from the Near-RT RIC
    6. refusals and emit failures are appended to the evidence store too, so
       the audit chain covers every decision — not only the ones that shipped

The planner here is deliberately a *rules-based risk-band planner*, not the
learned TD-MPC2 policy head: it maps the event's ``sla_risk_30s`` onto one of
three A1 policy types with schema-conformant payloads. It exists so the full
Shield → guards → A1 → evidence pipeline is exercised end-to-end; the learned
planner slots in behind the same interface later.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from prometheus_client import Counter, Gauge

from horizon_ric.evidence.schema import (
    ConstraintCorrection,
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import EvidenceStore
from horizon_ric.policy.counterfactual import (
    AlternativeCandidate,
    build_rejected_alternatives,
)
from horizon_ric.policy.emit_guards import GuardFailure, run_guard_chain
from horizon_ric.rapp.a1_adapter import A1Adapter, EvidencePersistError
from horizon_ric.runtime.circuit_breaker import CircuitBreakerError
from horizon_ric.shield import default_terrestrial_shield

logger = structlog.get_logger(__name__)


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no"}

# ---------------------------------------------------------------------------
# Pipeline-scoped Prometheus metrics.
# ---------------------------------------------------------------------------
HORIZON_DECISIONS_BLOCKED = Counter(
    "horizon_decisions_blocked_total",
    "Decisions refused before A1 emit (Shield block or guard-chain failure).",
    labelnames=("reason",),
)
HORIZON_A1_EMIT_FAILURES = Counter(
    "horizon_a1_emit_failures_total",
    "A1 policy emits that failed at the HTTP layer after passing the guards.",
)
HORIZON_DRY_RUN_DECISIONS = Counter(
    "horizon_dry_run_decisions_total",
    "Decisions that passed the Shield and guards but were NOT emitted "
    "because the HORIZON_A1_DRY_RUN kill switch is active.",
)
HORIZON_POLICY_ENFORCED = Gauge(
    "horizon_policy_enforced",
    "1 when the most recent policy of this type reported an enforced status "
    "from the Near-RT RIC, else 0.",
    labelnames=("policy_type",),
)

# Enforcement-status values (across A1 dialects) that mean "in effect".
# A1AP §6.5.2 uses ENFORCED; the OSC PMS mirror reports IN EFFECT; some
# SMO frontends collapse to a bare OK.
_ENFORCED_STATES = frozenset({"ENFORCED", "IN EFFECT", "OK"})

# Minimum implementable transmit power for the radio the planner commands.
# The Shield's EIRP projection is a pure dB subtraction — it will happily
# "fix" an absurd licence ceiling by commanding e.g. -105 dBm, which no real
# PA can emit. A certified-safe action whose Tx power sits below this floor
# is therefore *unfixable in hardware* and the pipeline fails closed on it
# (guard id ``tx_power_below_hw_floor``).
MIN_VIABLE_TX_POWER_DBM = -10.0

# Risk-band centres used to score the three candidate policy types. The
# chosen candidate is the band the risk falls in; scores make the ranking
# of rejected alternatives deterministic and replayable.
_BAND_CENTRES = {
    "horizon.qos.priority": 0.15,
    "horizon.traffic.steering": 0.45,
    "horizon.admission.control": 0.80,
}


@dataclass
class PipelineConfig:
    """Static configuration for one :class:`DecisionPipeline` instance."""

    band_lo_hz: float = 3.40e9
    band_hi_hz: float = 3.50e9
    max_eirp_dbm: float = 33.0
    decision_budget_ms: float = 1000.0
    status_poll_attempts: int = 3
    status_poll_interval_s: float = 1.0
    rapp_id: str = "horizon-ric-rapp"
    rapp_version: str = "0.2.0"

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build from environment variables (Docker/K8s deployment).

        Empty-string values count as unset (Helm ``b64enc`` of "" yields
        an empty string) and fall back to the dataclass defaults.
        """

        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            return float(raw) if raw else default

        def _i(name: str, default: int) -> int:
            raw = os.environ.get(name)
            return int(raw) if raw else default

        return cls(
            band_lo_hz=_f("HORIZON_SHIELD_BAND_LO_HZ", cls.band_lo_hz),
            band_hi_hz=_f("HORIZON_SHIELD_BAND_HI_HZ", cls.band_hi_hz),
            max_eirp_dbm=_f("HORIZON_SHIELD_MAX_EIRP_DBM", cls.max_eirp_dbm),
            decision_budget_ms=_f("HORIZON_DECISION_BUDGET_MS", cls.decision_budget_ms),
            status_poll_attempts=_i("HORIZON_A1_STATUS_POLL_ATTEMPTS", cls.status_poll_attempts),
            status_poll_interval_s=_f(
                "HORIZON_A1_STATUS_POLL_INTERVAL_S", cls.status_poll_interval_s
            ),
            rapp_id=os.environ.get("HORIZON_RAPP_ID") or cls.rapp_id,
            rapp_version=os.environ.get("HORIZON_VERSION") or cls.rapp_version,
        )


@dataclass
class PipelineResult:
    """Outcome of one :meth:`DecisionPipeline.process_event` run.

    ``latency_ms`` is the decision latency up to A1 emit-acceptance (or the
    refusal/failure) — it deliberately EXCLUDES the enforcement-status poll,
    whose wall-clock cost is reported separately in ``poll_duration_ms``.
    """

    decision_id: str
    event_id: str
    policy_type: str | None
    policy_id: str | None
    accepted: bool
    http_status: int | None
    enforcement_status: str | None
    enforced: bool | None
    blocked: bool
    block_reasons: list[str] = field(default_factory=list)
    emit_failed: bool = False
    latency_ms: float = 0.0
    poll_duration_ms: float = 0.0
    # True when the policy is LIVE on the RIC but its DecisionRecord could
    # not be persisted (EvidencePersistError) — "accepted but unaudited".
    evidence_persist_failed: bool = False
    # True when the HORIZON_A1_DRY_RUN kill switch suppressed the A1 PUT.
    dry_run: bool = False


class DecisionPipeline:
    """End-to-end decision pipeline: planner → Shield → guards → A1 → evidence.

    The pipeline never appends to the evidence store on a *successful* emit —
    :meth:`A1Adapter.emit_policy` persists the attached DecisionRecord and
    increments ``A1_POLICIES_EMITTED`` / ``DECISION_RECORDS_PERSISTED`` itself
    once the Near-RT RIC accepts the policy. The pipeline appends directly only
    for refusals (Shield/guard blocks) and emit failures, so those are
    auditable too.
    """

    def __init__(self, a1: A1Adapter, store: EvidenceStore, config: PipelineConfig):
        self.a1 = a1
        self.store = store
        self.cfg = config
        self._shield = default_terrestrial_shield(
            band_lo_hz=config.band_lo_hz,
            band_hi_hz=config.band_hi_hz,
            max_eirp_dBm=config.max_eirp_dbm,
        )
        # Optional Ed25519 certificate signing. Keys are NEVER generated
        # implicitly — a configured-but-unreadable path is a hard startup
        # error (generate via `python -m horizon_ric.shield.signing
        # --generate <path>`).
        self._signing_key = None
        signing_key_path = os.environ.get("HORIZON_CERT_SIGNING_KEY_PATH")
        if signing_key_path:
            from horizon_ric.shield.signing import load_signing_key

            self._signing_key = load_signing_key(signing_key_path)
            logger.info("pipeline.cert_signing.enabled", key_path=signing_key_path)

    # ── planner ─────────────────────────────────────────────────────────
    @staticmethod
    def _risk(event: Any) -> float:
        """Clamp the event's 30-s SLA risk into [0, 1]; default 0.05."""
        try:
            risk = float(event.payload.get("sla_risk_30s", 0.05))
        except (TypeError, ValueError):
            return 0.05
        if not math.isfinite(risk):
            return 0.05
        return min(max(risk, 0.0), 1.0)

    def _candidates(
        self, event: Any, risk: float, decision_id: str
    ) -> dict[str, dict[str, Any]]:
        """Three candidate policy payloads, one per registered risk band.

        Payloads carry ONLY keys declared in the A1 create schemas
        (``A1Adapter._policy_create_schema``; ``additionalProperties: False``)
        plus the optional ``rapp_metadata`` envelope linking back to the
        DecisionRecord.
        """
        meta = {"rapp_metadata": {"decision_id": decision_id}}
        priority = max(1, min(15, 1 + round(risk * 14)))
        if risk < 0.3:
            preferred_path = "terrestrial"
        elif risk < 0.6:
            preferred_path = "hybrid"
        else:
            preferred_path = "ntn"
        admission: dict[str, Any]
        if risk < 0.3:
            admission = {"decision": "accept"}
        elif risk < 0.6:
            admission = {"decision": "defer", "defer_until_seconds": round(30.0 + 270.0 * risk, 1)}
        else:
            admission = {"decision": "reject"}
        return {
            "horizon.qos.priority": {
                "scope": {"slice_id": f"slice-{event.source_id}"},
                "qos_objectives": {"priority": priority},
                **meta,
            },
            "horizon.traffic.steering": {
                "scope": {"ue_group": "default"},
                "steering_objectives": {
                    "preferred_path": preferred_path,
                    "ntn_share_pct": round(min(risk * 100.0, 100.0), 1),
                },
                **meta,
            },
            "horizon.admission.control": {
                "scope": {"workload_class": "ai_inference"},
                "admission": admission,
                **meta,
            },
        }

    @staticmethod
    def _choose(risk: float) -> str:
        """Risk-band dispatch: low → QoS tuning, mid → steering, high → admission."""
        if risk < 0.3:
            return "horizon.qos.priority"
        if risk < 0.6:
            return "horizon.traffic.steering"
        return "horizon.admission.control"

    @staticmethod
    def _predicted_outcome(risk: float) -> PredictedOutcome:
        return PredictedOutcome(
            sla_risk_30s=min(risk, 1.0),
            sla_risk_1min=min(risk * 1.1, 1.0),
            sla_risk_5min=min(risk * 1.5, 1.0),
        )

    def _shield_action(self, policy_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Wrap the policy in an RF-physics action the Shield can validate.

        The Shield's invariant chain (NumericSanityInvariant onward) requires
        finite ``frequency_hz > 0``, ``bandwidth_hz > 0`` and ``tx_power_dBm``
        on every action, so the planner stamps the carrier it is licensed to
        command: centre of the configured band, up to 20 MHz occupied
        bandwidth, and a nominal Tx power sitting safely under the EIRP
        ceiling (24 dBm cap, 9 dB below the ceiling with 5 dBi antenna gain).
        """
        return {
            "block": "policy_emit",
            "policy_type": policy_type,
            "policy_payload": payload,
            "frequency_hz": (self.cfg.band_lo_hz + self.cfg.band_hi_hz) / 2.0,
            "bandwidth_hz": min(20e6, self.cfg.band_hi_hz - self.cfg.band_lo_hz),
            "tx_power_dBm": min(24.0, self.cfg.max_eirp_dbm - 9.0),
            "antenna_gain_dBi": 5.0,
        }

    # ── main entry point ────────────────────────────────────────────────
    async def process_event(self, event: Any) -> PipelineResult:
        """Run one TelemetryEvent through the full decision pipeline."""
        t_start = time.monotonic()
        decision_id = str(uuid.uuid4())
        # Deterministic replay seed pinned to the event identity.
        seed = int.from_bytes(
            hashlib.sha256(event.event_id.encode("utf-8")).digest()[:4], "big"
        )
        state_hash = hashlib.sha256(
            json.dumps(
                json.loads(event.model_dump_json()), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

        # 1. Plan: pick a policy by risk band; the two non-chosen candidates
        #    become the audited counterfactual.
        risk = self._risk(event)
        candidates = self._candidates(event, risk, decision_id)
        policy_type = self._choose(risk)
        payload = candidates[policy_type]
        scores = {name: 1.0 - abs(risk - centre) for name, centre in _BAND_CENTRES.items()}
        rejected = build_rejected_alternatives(
            [
                AlternativeCandidate(
                    action={"policy_type": name, "policy_payload": body},
                    reward_sum=scores[name],
                    constraint_violation_sum=0.0,
                    sla_risk_30s=risk,
                    sla_risk_1min=min(risk * 1.1, 1.0),
                    sla_risk_5min=min(risk * 1.5, 1.0),
                    # build_rejected_alternatives classifies risk > 0.20 as
                    # sla_breach_predicted, so the audited metric/threshold
                    # must be the SLA-risk pair in that band — pairing the
                    # planner score with the risk cause yields contradictory
                    # explanations ("74% above the 91% threshold").
                    primary_metric="sla_risk_30s" if risk > 0.20 else "planner_score",
                    primary_value=risk if risk > 0.20 else scores[name],
                    threshold=0.20 if risk > 0.20 else scores[policy_type],
                )
                for name, body in candidates.items()
                if name != policy_type
            ],
            chosen_score=scores[policy_type],
            random_seed=seed,
        )

        # 2. Shield: the AI proposes, the verified envelope disposes.
        action = self._shield_action(policy_type, payload)
        disposition = self._shield.dispose(action, {}, decision_id=decision_id)
        certificate = disposition.certificate
        if self._signing_key is not None:
            from horizon_ric.shield.signing import signed_certificate

            certificate = signed_certificate(certificate, self._signing_key)
        corrections = [
            ConstraintCorrection(
                constraint_id=c.constraint_id,
                severity=c.severity,
                margin_dB=c.margin_dB,
                message=c.message,
            )
            for c in certificate.corrections
        ]

        # 3. Pre-emit guard chain — the last gate before the A1 wire.
        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        failures = run_guard_chain(
            certificate=certificate,
            elapsed_ms=elapsed_ms,
            audit_corrections_field=[c.model_dump() for c in corrections] or None,
            policy_period_ms=self.cfg.decision_budget_ms,
        )
        # Hardware-floor fail-closed: the EIRP projection subtracts dB without
        # bound, so a nonsensical licence ceiling yields a "safe" Tx power no
        # PA can implement. Refuse rather than emit an un-executable command.
        safe_tx = disposition.safe_action.get("tx_power_dBm")
        try:
            safe_tx_val = float(safe_tx) if safe_tx is not None else None
        except (TypeError, ValueError):
            safe_tx_val = None
        if safe_tx_val is None or safe_tx_val < MIN_VIABLE_TX_POWER_DBM:
            failures.append(
                GuardFailure(
                    guard_id="tx_power_below_hw_floor",
                    message=(
                        f"certified-safe Tx power {safe_tx_val} dBm is below the "
                        f"minimum implementable {MIN_VIABLE_TX_POWER_DBM} dBm — "
                        "the EIRP ceiling cannot be met by any real transmission."
                    ),
                    detail={
                        "safe_tx_power_dBm": safe_tx_val,
                        "min_viable_tx_power_dBm": MIN_VIABLE_TX_POWER_DBM,
                        "max_eirp_dBm": self.cfg.max_eirp_dbm,
                    },
                )
            )

        # 4. Evidence record for this decision (persisted on every path).
        physics_keys = ("frequency_hz", "bandwidth_hz", "tx_power_dBm", "antenna_gain_dBi")
        record = DecisionRecord.new(
            decision_id=decision_id,
            rapp_instance_id=self.cfg.rapp_id,
            state_hash=state_hash,
            chosen_action={
                "policy_type": policy_type,
                "policy_payload": payload,
                "safe_action": {
                    k: disposition.safe_action[k]
                    for k in physics_keys
                    if k in disposition.safe_action
                },
                "source_event": event.event_id,
                "certificate": {
                    "safe": certificate.safe,
                    "projected": certificate.projected,
                    "violated_ids": list(certificate.violated_ids),
                    **(
                        {
                            "signature": certificate.signature,
                            "signing_key_fingerprint": certificate.signing_key_fingerprint,
                        }
                        if certificate.signature is not None
                        else {}
                    ),
                },
            },
            predicted_outcome_chosen=self._predicted_outcome(risk),
            rejected_alternatives=rejected,
            constraint_corrections=corrections,
            # Honest provenance: this pipeline runs the deterministic
            # rules-based risk-band planner, NOT learned models. Stamp
            # identifiers that say exactly that — no phantom JEPA/TD-MPC
            # versions for models that never ran.
            model_versions=ModelVersions(
                encoder="none",
                risk_heads="risk_rules_v1",
                dyna="none",
                policy="risk_band_rules_v1",
                constraint_layer="shield_terrestrial_v1",
                rapp=self.cfg.rapp_version,
            ),
            random_seed=seed,
        )

        # 5. Blocked path — the refusal itself is auditable evidence.
        if failures or certificate.emit_blocked:
            block_reasons = [f.guard_id for f in failures] or ["shield_blocked"]
            record.chosen_action["emit_blocked"] = True
            record.chosen_action["block_reasons"] = block_reasons
            self.store.append(record)
            HORIZON_DECISIONS_BLOCKED.labels(reason=block_reasons[0]).inc()
            try:
                from horizon_ric.rapp.health import DECISION_RECORDS_PERSISTED

                DECISION_RECORDS_PERSISTED.inc()
            except Exception:  # pragma: no cover — metrics are best-effort
                pass
            logger.warning(
                "pipeline.decision.blocked",
                decision_id=decision_id,
                event_id=event.event_id,
                policy_type=policy_type,
                reasons=block_reasons,
            )
            return PipelineResult(
                decision_id=decision_id,
                event_id=event.event_id,
                policy_type=policy_type,
                policy_id=None,
                accepted=False,
                http_status=None,
                enforcement_status=None,
                enforced=None,
                blocked=True,
                block_reasons=block_reasons,
                latency_ms=(time.monotonic() - t_start) * 1000.0,
            )

        # 6. Kill switch — HORIZON_A1_DRY_RUN suppresses the A1 PUT entirely
        #    while keeping the decision auditable (see deploy/RUNBOOK.md).
        if _env_truthy("HORIZON_A1_DRY_RUN"):
            record.chosen_action["dry_run"] = True
            self.store.append(record)
            HORIZON_DRY_RUN_DECISIONS.inc()
            try:
                from horizon_ric.rapp.health import DECISION_RECORDS_PERSISTED

                DECISION_RECORDS_PERSISTED.inc()
            except Exception:  # pragma: no cover — metrics are best-effort
                pass
            logger.warning(
                "pipeline.decision.dry_run",
                decision_id=decision_id,
                event_id=event.event_id,
                policy_type=policy_type,
                note="HORIZON_A1_DRY_RUN active — A1 emit suppressed",
            )
            return PipelineResult(
                decision_id=decision_id,
                event_id=event.event_id,
                policy_type=policy_type,
                policy_id=None,
                accepted=False,
                http_status=None,
                enforcement_status=None,
                enforced=None,
                blocked=False,
                dry_run=True,
                latency_ms=(time.monotonic() - t_start) * 1000.0,
            )

        # 7. Emit over A1. On success the adapter appends the record to the
        #    evidence store and increments the emit/persist metrics itself.
        evidence_persist_failed = False
        try:
            policy_id, http_status = await self.a1.emit_policy(
                policy_type,
                payload,
                decision_record=record,
                safety_certificate=certificate,
            )
        except EvidencePersistError as exc:
            # The policy IS live on the RIC and the emit counters already
            # incremented — only the audit write failed. Best-effort append
            # a distinct audit row and report "accepted but unaudited".
            evidence_persist_failed = True
            policy_id, http_status = exc.policy_id, exc.http_status
            logger.error(
                "pipeline.evidence.persist_failed",
                decision_id=decision_id,
                event_id=event.event_id,
                policy_type=policy_type,
                policy_id=policy_id,
                error=str(exc),
            )
            record.chosen_action["evidence_persist_failed"] = True
            try:
                self.store.append(record)
            except Exception as retry_exc:
                logger.error(
                    "pipeline.evidence.persist_retry_failed",
                    decision_id=decision_id,
                    policy_id=policy_id,
                    error=str(retry_exc),
                )
        except (httpx.HTTPError, CircuitBreakerError) as exc:
            # HTTP failure OR the A1 circuit breaker is OPEN — both mean
            # the policy did not ship; both must land in the audit chain.
            HORIZON_A1_EMIT_FAILURES.inc()
            record.chosen_action["emit_failed"] = True
            self.store.append(record)
            try:
                from horizon_ric.rapp.health import DECISION_RECORDS_PERSISTED

                DECISION_RECORDS_PERSISTED.inc()
            except Exception:  # pragma: no cover — metrics are best-effort
                pass
            response = getattr(exc, "response", None)
            logger.error(
                "pipeline.emit.failed",
                decision_id=decision_id,
                event_id=event.event_id,
                policy_type=policy_type,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return PipelineResult(
                decision_id=decision_id,
                event_id=event.event_id,
                policy_type=policy_type,
                policy_id=None,
                accepted=False,
                http_status=response.status_code if response is not None else None,
                enforcement_status=None,
                enforced=None,
                blocked=False,
                emit_failed=True,
                latency_ms=(time.monotonic() - t_start) * 1000.0,
            )
        except Exception as exc:
            # Unexpected failure — best-effort audit row so the decision
            # doesn't vanish from the chain, then re-raise for the caller.
            try:
                record.chosen_action["emit_failed"] = True
                self.store.append(record)
            except Exception as append_exc:  # pragma: no cover — disk gone
                logger.error(
                    "pipeline.emit.audit_append_failed",
                    decision_id=decision_id,
                    error=str(append_exc),
                )
            logger.error(
                "pipeline.emit.unexpected_error",
                decision_id=decision_id,
                event_id=event.event_id,
                policy_type=policy_type,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

        # Decision latency stops at emit-acceptance; the enforcement poll
        # below is bookkeeping against the Near-RT RIC, not decision time.
        latency_ms = (time.monotonic() - t_start) * 1000.0

        # 8. Poll enforcement status from the Near-RT RIC (A1AP §6.5). The
        #    emit is already accepted — the poll can NEVER fail it.
        poll_t0 = time.monotonic()
        enforcement_status, enforced = await self._poll_enforcement(policy_type, policy_id)
        poll_duration_ms = (time.monotonic() - poll_t0) * 1000.0
        HORIZON_POLICY_ENFORCED.labels(policy_type=policy_type).set(1.0 if enforced else 0.0)

        logger.info(
            "pipeline.decision.accepted",
            decision_id=decision_id,
            event_id=event.event_id,
            policy_type=policy_type,
            policy_id=policy_id,
            http_status=http_status,
            enforcement_status=enforcement_status,
            latency_ms=round(latency_ms, 2),
            poll_duration_ms=round(poll_duration_ms, 2),
            evidence_persist_failed=evidence_persist_failed or None,
        )
        return PipelineResult(
            decision_id=decision_id,
            event_id=event.event_id,
            policy_type=policy_type,
            policy_id=policy_id,
            accepted=True,
            http_status=http_status,
            enforcement_status=enforcement_status,
            enforced=enforced,
            blocked=False,
            latency_ms=latency_ms,
            poll_duration_ms=poll_duration_ms,
            evidence_persist_failed=evidence_persist_failed,
        )

    async def _poll_enforcement(
        self, policy_type: str, policy_id: str
    ) -> tuple[str | None, bool | None]:
        """Poll the policy's enforcement status; stop early once enforced.

        Tolerates transport errors AND malformed bodies — an unreachable
        status endpoint, a non-JSON body (``json.JSONDecodeError`` is a
        ``ValueError``), or JSON that isn't an object all yield
        ``(None, None)`` rather than failing the already-accepted emit.
        """
        enforcement_status: str | None = None
        enforced: bool | None = None
        attempts = max(1, self.cfg.status_poll_attempts)
        for attempt in range(attempts):
            try:
                body = await self.a1.get_policy_status(policy_type, policy_id)
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning(
                    "pipeline.enforcement.poll_failed",
                    policy_id=policy_id,
                    attempt=attempt + 1,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            else:
                if not isinstance(body, dict):
                    logger.warning(
                        "pipeline.enforcement.poll_malformed_body",
                        policy_id=policy_id,
                        attempt=attempt + 1,
                        body_type=type(body).__name__,
                    )
                else:
                    raw = (
                        body.get("enforceStatus")
                        or body.get("instance_status")
                        or body.get("status")
                    )
                    if raw is not None:
                        enforcement_status = str(raw)
                        enforced = enforcement_status in _ENFORCED_STATES
                        if enforced:
                            break
            if attempt + 1 < attempts:
                await asyncio.sleep(self.cfg.status_poll_interval_s)
        return enforcement_status, enforced


__all__ = [
    "MIN_VIABLE_TX_POWER_DBM",
    "DecisionPipeline",
    "PipelineConfig",
    "PipelineResult",
]
