"""A1 adapter — Non-RT RIC → Near-RT RIC ML policy delivery.

References:
    O-RAN.WG2.A1AP-v05.00 §6 (PolicyType, PolicyInstance)
    O-RAN.WG2.A1-GAP-v05.00 §5 (Transport: REST/JSON)
    O-RAN.WG2.A1-EI-v04.00 (Enrichment Information)
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


# Standard A1 policy types (from O-RAN A1AP and 3GPP TS 23.503 alignment).
# Each operator may extend with vendor-specific types; we ship a baseline set.
DEFAULT_POLICY_TYPES = {
    "horizon.qos.priority": {
        "policy_type_id": 20001,
        "name": "QoS priority weights per slice",
        "description": "Adjust per-slice QoS priority based on Horizon-RIC SLA risk",
        "schema_v": "1.0.0",
    },
    "horizon.traffic.steering": {
        "policy_type_id": 20002,
        "name": "Traffic steering across TN/NTN",
        "description": "Direct traffic between terrestrial cell and NTN beam",
        "schema_v": "1.0.0",
    },
    "horizon.admission.control": {
        "policy_type_id": 20003,
        "name": "Slice/UE admission control",
        "description": "Accept, defer, or reject AI workload based on edge load",
        "schema_v": "1.0.0",
    },
    "horizon.spectrum.reservation": {
        "policy_type_id": 20004,
        "name": "Spectrum bin reservation",
        "description": "Reserve spectrum sub-bands for critical traffic classes",
        "schema_v": "1.0.0",
    },
}


@dataclass
class A1AdapterConfig:
    near_rt_ric_base_url: str = "http://nearrtric:10000"
    timeout_seconds: float = 10.0
    rapp_id: str = "horizon-ric-rapp"
    auth: "AuthConfig | None" = None  # noqa: F821 — forward ref
    """When set, the A1 client uses mTLS / OAuth2 / static-bearer per
    O-RAN WG11 §6. None preserves the plain-HTTP path used by the
    OSC reference Near-RT RIC sandbox."""
    policy_status_poll_interval_s: float = 30.0
    policy_types: dict[str, dict[str, Any]] = field(
        default_factory=lambda: dict(DEFAULT_POLICY_TYPES)
    )
    # ------------------------------------------------------------------
    # OSC NONRTRIC dialect switch.
    #
    # The OSC `nonrtric-plt-a1policymanagementservice` reference exposes
    # a *different* URL surface from the historical near-RT-RIC A1AP
    # mirror that was used in the legacy unit tests:
    #   - Base path is "/a1-policy/v2" (lower-case, hyphenated) — see
    #     api/pms-api.json operationId="putPolicy" in
    #     o-ran-sc/nonrtric-plt-a1policymanagementservice.
    #   - Policy create is a flat
    #     `PUT /a1-policy/v2/policies`
    #     with `policy_id`, `policytype_id`, `ric_id`, `policy_data`
    #     in the body (NOT nested under `/policytypes/{id}/policies`).
    #   - Policy delete / status are addressed only by `policy_id`.
    #
    # When `dialect == "osc"` we honour the real OSC paths; the default
    # "legacy" mode retains the historical paths for back-compat with
    # the existing rapp-lifecycle unit tests.
    #
    # EIAP dialect (Ericsson Intelligent Automation Platform / EIAP rApp SDK):
    #   - Base path: /A1-PolicyManagement/v2/
    #   - Policy creation: POST /A1-PolicyManagement/v2/policies with
    #     {policyId, policyTypeId, ricId, policyData}
    #   - Policy status: GET /A1-PolicyManagement/v2/policies/{policyId}/status
    #   - Public reference: Ericsson Developer Studio / Open Source O-RAN
    #     dev portal (full surface requires login, see SMO_INTEGRATION.md).
    #
    # MantaRay dialect (Nokia MantaRay SMO via SDN-R):
    #   - Base path: /sdn-r/api/v1/policies
    #   - rApp catalogue: /sdn-r/api/v1/rapps/
    #   - Auth: keycloak-issued JWT (see auth.py for OAuth2 wiring).
    dialect: str = "legacy"
    osc_ric_id: str = "ric_emulated_horizon"
    osc_service_id: str = "horizon-ric-rapp"
    # EIAP and MantaRay carry similar contextual identifiers under
    # different field names. We keep them as separate keys to make the
    # contract explicit per vendor.
    eiap_ric_id: str = "ric_emulated_horizon"
    mantaray_ric_id: str = "ric_emulated_horizon"
    # Optional TTL-based DNS cache around getaddrinfo for the SMO host.
    # 0 (default) preserves the vanilla httpx behaviour. > 0 wires
    # `runtime.dns_cache.CachingDNSTransport` with this TTL in seconds.
    # See deploy/SLO.md "DNS caching" and src/.../runtime/dns_cache.py.
    dns_cache_ttl_s: float = 0.0


class A1Adapter:
    """A1 policy emitter (rApp side).

    For Phase 1: emit policies as REST/JSON per A1-GAP. Each emission
    returns a (policy_id, status) tuple for the caller to log into the
    evidence store.
    """

    def __init__(self, config: A1AdapterConfig | None = None):
        self.cfg = config or A1AdapterConfig()
        if self.cfg.auth is not None:
            from horizon_ric.rapp.auth import build_secure_async_client

            self._client = build_secure_async_client(
                self.cfg.auth, base_url=self.cfg.near_rt_ric_base_url
            )
        else:
            self._client = httpx.AsyncClient(
                base_url=self.cfg.near_rt_ric_base_url,
                timeout=self.cfg.timeout_seconds,
            )
        self._policies_emitted = 0
        # Explicit asyncio.Lock around the read-prev-then-write critical
        # section that touches `_policies_emitted` AND the evidence
        # store. Keeps the counter and the chain in lock-step under
        # concurrent emit_policy calls. (See test_known_bugs::HIGH-01.)
        self._emit_lock = asyncio.Lock()
        self._evidence_store: "EvidenceStore | None" = None  # noqa: F821
        # Real circuit breaker for the A1 channel.
        from horizon_ric.runtime.circuit_breaker import (
            AsyncCircuitBreaker,
            BreakerConfig,
        )

        self._cb = AsyncCircuitBreaker(
            BreakerConfig(name="horizon.a1", fail_max=5, reset_timeout=30.0)
        )

    @property
    def circuit_breaker(self):
        return self._cb

    async def register_policy_types(self) -> list[int]:
        """Register Horizon-RIC's A1 policy types with the Near-RT RIC.

        Returns:
            List of accepted policy_type_ids.
        """
        accepted: list[int] = []
        for type_name, spec in self.cfg.policy_types.items():
            payload = {
                "policy_type_id": spec["policy_type_id"],
                "name": spec["name"],
                "description": spec["description"],
                "schema_version": spec["schema_v"],
                "create_schema": self._policy_create_schema(type_name),
            }
            url = self._policy_type_url(spec["policy_type_id"])
            try:
                resp = await self._client.put(url, json=payload)
                resp.raise_for_status()
                accepted.append(spec["policy_type_id"])
                logger.info(
                    "a1.policytype.registered",
                    policy_type_id=spec["policy_type_id"],
                    name=type_name,
                    url=url,
                )
            except httpx.HTTPError as e:
                logger.error("a1.policytype.register.failed", type=type_name, error=str(e))
        return accepted

    # -- URL builders that honour the OSC vs legacy dialect ---------------

    def _policy_type_url(self, policy_type_id: int) -> str:
        if self.cfg.dialect == "osc":
            # OSC PMS uses string policytype_id under /a1-policy/v2.
            return f"/a1-policy/v2/policy-types/{policy_type_id}"
        if self.cfg.dialect == "eiap":
            return f"/A1-PolicyManagement/v2/policy-types/{policy_type_id}"
        if self.cfg.dialect == "mantaray":
            return f"/sdn-r/api/v1/policy-types/{policy_type_id}"
        return f"/A1-P/v2/policytypes/{policy_type_id}"

    def _policy_instance_url(
        self, policy_type_id: int, policy_id: str
    ) -> str:
        if self.cfg.dialect == "osc":
            # OSC: flat, addressed only by policy_id.
            return f"/a1-policy/v2/policies/{policy_id}"
        if self.cfg.dialect == "eiap":
            return f"/A1-PolicyManagement/v2/policies/{policy_id}"
        if self.cfg.dialect == "mantaray":
            return f"/sdn-r/api/v1/policies/{policy_id}"
        return f"/A1-P/v2/policytypes/{policy_type_id}/policies/{policy_id}"

    def _policy_create_url(self, policy_type_id: int) -> str:
        # OSC PMS uses a single flat PUT to /a1-policy/v2/policies; the
        # legacy near-RT-RIC mirror addresses the resource directly.
        if self.cfg.dialect == "osc":
            return "/a1-policy/v2/policies"
        if self.cfg.dialect == "eiap":
            # Ericsson EIAP A1 PolicyManagement: flat POST/PUT.
            return "/A1-PolicyManagement/v2/policies"
        if self.cfg.dialect == "mantaray":
            return "/sdn-r/api/v1/policies"
        return f"/A1-P/v2/policytypes/{policy_type_id}/policies"

    def _policy_status_url(self, policy_type_id: int, policy_id: str) -> str:
        if self.cfg.dialect == "osc":
            return f"/a1-policy/v2/policies/{policy_id}/status"
        if self.cfg.dialect == "eiap":
            return f"/A1-PolicyManagement/v2/policies/{policy_id}/status"
        if self.cfg.dialect == "mantaray":
            return f"/sdn-r/api/v1/policies/{policy_id}/status"
        return (
            f"/A1-P/v2/policytypes/{policy_type_id}"
            f"/policies/{policy_id}/status"
        )

    def _policy_list_url(self, policy_type_id: int) -> str:
        if self.cfg.dialect == "osc":
            return f"/a1-policy/v2/policies?policytype_id={policy_type_id}"
        if self.cfg.dialect == "eiap":
            return (
                "/A1-PolicyManagement/v2/policies"
                f"?policyTypeId={policy_type_id}"
            )
        if self.cfg.dialect == "mantaray":
            return f"/sdn-r/api/v1/policies?policyTypeId={policy_type_id}"
        return f"/A1-P/v2/policytypes/{policy_type_id}/policies"

    async def emit_policy(
        self,
        policy_type: str,
        policy_payload: dict[str, Any],
        policy_id: str | None = None,
        decision_record: "DecisionRecord | None" = None,  # noqa: F821
    ) -> tuple[str, int]:
        """Emit an A1 policy instance to the Near-RT RIC.

        If `decision_record` is provided AND an evidence store is attached
        (`A1Adapter.attach_evidence_store()`), the record is appended to the
        store after the PUT succeeds — so audit trails only contain policies
        that the Near-RT RIC actually accepted.

        Args:
            policy_type: registered type name (e.g., "horizon.qos.priority").
            policy_payload: instance-specific data conforming to the policy schema.
            policy_id: optional UUID; one is generated if not provided.
            decision_record: optional DecisionRecord to persist on success.

        Returns:
            (policy_id, http_status_code)
        """
        if policy_type not in self.cfg.policy_types:
            raise ValueError(f"unregistered policy_type: {policy_type}")
        type_spec = self.cfg.policy_types[policy_type]
        policy_id = policy_id or str(uuid.uuid4())
        async def _do_put():
            if self.cfg.dialect == "osc":
                # OSC PMS body is `policy_info` (see pms-api.json schema):
                # required keys = policy_id, policytype_id, ric_id, policy_data.
                osc_body = {
                    "policy_id": policy_id,
                    "policytype_id": str(type_spec["policy_type_id"]),
                    "ric_id": self.cfg.osc_ric_id,
                    "service_id": self.cfg.osc_service_id,
                    "transient": False,
                    "policy_data": policy_payload,
                }
                r = await self._client.put(
                    self._policy_create_url(type_spec["policy_type_id"]),
                    json=osc_body,
                )
            elif self.cfg.dialect == "eiap":
                # Ericsson EIAP rApp SDK contract — camelCase keys.
                eiap_body = {
                    "policyId": policy_id,
                    "policyTypeId": str(type_spec["policy_type_id"]),
                    "ricId": self.cfg.eiap_ric_id,
                    "policyData": policy_payload,
                }
                r = await self._client.put(
                    self._policy_create_url(type_spec["policy_type_id"]),
                    json=eiap_body,
                )
            elif self.cfg.dialect == "mantaray":
                # Nokia MantaRay SDN-R contract — camelCase keys plus
                # rappId for catalogue correlation.
                mantaray_body = {
                    "policyId": policy_id,
                    "policyTypeId": str(type_spec["policy_type_id"]),
                    "ricId": self.cfg.mantaray_ric_id,
                    "rappId": self.cfg.rapp_id,
                    "policyData": policy_payload,
                }
                r = await self._client.put(
                    self._policy_create_url(type_spec["policy_type_id"]),
                    json=mantaray_body,
                )
            else:
                r = await self._client.put(
                    self._policy_instance_url(
                        type_spec["policy_type_id"], policy_id
                    ),
                    json=policy_payload,
                )
            r.raise_for_status()
            return r

        try:
            resp = await self._cb.call(_do_put)
            async with self._emit_lock:
                self._policies_emitted += 1
            logger.info(
                "a1.policy.emitted",
                policy_type=policy_type,
                policy_id=policy_id,
                status=resp.status_code,
            )
            # Best-effort metrics — never break the policy emit on metric failure.
            try:
                from horizon_ric.rapp.health import A1_POLICIES_EMITTED

                A1_POLICIES_EMITTED.labels(policy_type=policy_type).inc()
            except Exception:  # pragma: no cover
                pass
            # Persist evidence if both a record and a store are attached.
            if decision_record is not None and self._evidence_store is not None:
                self._evidence_store.append(decision_record)
                try:
                    from horizon_ric.rapp.health import (
                        COUNTERFACTUAL_ENVELOPE_BYTES,
                        DECISION_RECORDS_PERSISTED,
                        DECISION_REJECTION_REASONS,
                        REJECTED_ALTERNATIVES_COUNT,
                    )

                    DECISION_RECORDS_PERSISTED.inc()
                    alts = getattr(decision_record, "rejected_alternatives", []) or []
                    REJECTED_ALTERNATIVES_COUNT.observe(len(alts))
                    for alt in alts:
                        reason = getattr(
                            alt, "rejection_reason_machine", None
                        ) or "unspecified"
                        DECISION_REJECTION_REASONS.labels(reason=str(reason)).inc()
                    # Envelope size: serialised counterfactual payload only
                    # (rejected_alternatives + predicted_outcome). Cheap to
                    # compute via Pydantic .model_dump_json on the slice.
                    try:
                        import json as _json
                        envelope = {
                            "rejected_alternatives": [
                                a.model_dump() if hasattr(a, "model_dump")
                                else a for a in alts
                            ],
                            "predicted_outcome": (
                                decision_record.predicted_outcome.model_dump()
                                if getattr(decision_record, "predicted_outcome", None)
                                and hasattr(decision_record.predicted_outcome, "model_dump")
                                else None
                            ),
                        }
                        env_bytes = len(_json.dumps(envelope, default=str).encode("utf-8"))
                        COUNTERFACTUAL_ENVELOPE_BYTES.labels(
                            policy_type=policy_type
                        ).set(env_bytes)
                    except Exception:  # pragma: no cover
                        pass
                except Exception:  # pragma: no cover
                    pass
            return policy_id, resp.status_code
        except httpx.HTTPError as e:
            logger.error("a1.policy.emit.failed", policy_type=policy_type, error=str(e))
            raise

    async def get_policy_status(
        self, policy_type: str, policy_id: str
    ) -> dict[str, Any]:
        """A1AP §6.5 PolicyStatus query.

        Returns the current enforcement status from the Near-RT RIC. The
        canonical fields per A1AP §6.5.2 are `enforceStatus` (one of
        ENFORCED, NOT_ENFORCED) and optional `enforceReason` for the latter.
        """
        if policy_type not in self.cfg.policy_types:
            raise ValueError(f"unregistered policy_type: {policy_type}")
        type_spec = self.cfg.policy_types[policy_type]
        try:
            resp = await self._client.get(
                self._policy_status_url(type_spec["policy_type_id"], policy_id)
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error(
                "a1.policy.status.failed",
                policy_type=policy_type,
                policy_id=policy_id,
                error=str(e),
            )
            raise

    async def list_policies(self, policy_type: str) -> list[str]:
        """List policy instance IDs of a given type. A1AP §6.4 GET /policies."""
        if policy_type not in self.cfg.policy_types:
            raise ValueError(f"unregistered policy_type: {policy_type}")
        type_spec = self.cfg.policy_types[policy_type]
        try:
            resp = await self._client.get(
                self._policy_list_url(type_spec["policy_type_id"])
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return [str(x) for x in data]
            return list(data.get("policy_ids", []))
        except httpx.HTTPError as e:
            logger.error(
                "a1.policy.list.failed", policy_type=policy_type, error=str(e)
            )
            raise

    async def create_ei_job(
        self,
        ei_type_id: str,
        ei_job_id: str,
        ei_job_data: dict[str, Any],
        target_uri: str,
    ) -> int:
        """A1-EI Enrichment-Information job creation (O-RAN.WG2.A1-EI §5.4).

        The rApp uses A1-EI to receive enrichment streams (UE mobility,
        slice analytics, etc.) from the Near-RT RIC.
        """
        payload = {
            "ei_type_id": ei_type_id,
            "ei_job_data": ei_job_data,
            "target_uri": target_uri,
            "rapp_id": self.cfg.rapp_id,
        }
        try:
            resp = await self._client.put(
                f"/A1-EI/v1/eijobs/{ei_job_id}", json=payload
            )
            resp.raise_for_status()
            logger.info(
                "a1.ei_job.created", ei_job_id=ei_job_id, ei_type_id=ei_type_id
            )
            return resp.status_code
        except httpx.HTTPError as e:
            logger.error(
                "a1.ei_job.create.failed",
                ei_job_id=ei_job_id,
                error=str(e),
            )
            raise

    async def delete_ei_job(self, ei_job_id: str) -> int:
        """Delete an A1-EI job."""
        try:
            resp = await self._client.delete(f"/A1-EI/v1/eijobs/{ei_job_id}")
            resp.raise_for_status()
            return resp.status_code
        except httpx.HTTPError as e:
            logger.error("a1.ei_job.delete.failed", ei_job_id=ei_job_id, error=str(e))
            raise

    async def rollback_policy(self, policy_type: str, policy_id: str) -> int:
        """Delete (rollback) a previously-emitted A1 policy."""
        type_spec = self.cfg.policy_types[policy_type]
        try:
            resp = await self._client.delete(
                self._policy_instance_url(
                    type_spec["policy_type_id"], policy_id
                )
            )
            resp.raise_for_status()
            logger.info("a1.policy.rollback", policy_type=policy_type, policy_id=policy_id)
            try:
                from horizon_ric.rapp.health import A1_POLICIES_ROLLED_BACK

                A1_POLICIES_ROLLED_BACK.labels(policy_type=policy_type).inc()
            except Exception:  # pragma: no cover
                pass
            return resp.status_code
        except httpx.HTTPError as e:
            logger.error("a1.policy.rollback.failed", policy_id=policy_id, error=str(e))
            raise

    def attach_evidence_store(self, store: "EvidenceStore") -> None:  # noqa: F821
        """Attach an evidence store; subsequent emits with a DecisionRecord
        will be persisted on success."""
        self._evidence_store = store

    def policies_emitted_count(self) -> int:
        return self._policies_emitted

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _policy_create_schema(policy_type: str) -> dict[str, Any]:
        """Return the JSON Schema (Draft-07) for a policy type's instance payload.

        These schemas are tight: each one declares its required scope and
        objective fields and rejects unknown keys. They mirror the policy
        types declared in ``DEFAULT_POLICY_TYPES``.
        """
        common_meta = {
            "rapp_metadata": {
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"},
                    "rapp_version": {"type": "string"},
                    "model_versions": {"type": "object"},
                },
                "required": ["decision_id"],
                "additionalProperties": False,
            },
        }

        if policy_type == "horizon.qos.priority":
            return {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "object",
                        "properties": {
                            "slice_id": {"type": "string"},
                            "ue_id": {"type": "string"},
                        },
                        "required": ["slice_id"],
                        "additionalProperties": False,
                    },
                    "qos_objectives": {
                        "type": "object",
                        "properties": {
                            "priority": {"type": "integer", "minimum": 1, "maximum": 15},
                            "guaranteed_bit_rate_kbps": {"type": "number", "minimum": 0},
                            "max_bit_rate_kbps": {"type": "number", "minimum": 0},
                        },
                        "required": ["priority"],
                        "additionalProperties": False,
                    },
                    **common_meta,
                },
                "required": ["scope", "qos_objectives"],
                "additionalProperties": False,
            }

        if policy_type == "horizon.traffic.steering":
            return {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "object",
                        "properties": {
                            "ue_group": {"type": "string"},
                            "cell_id": {"type": "string"},
                        },
                        "required": ["ue_group"],
                        "additionalProperties": False,
                    },
                    "steering_objectives": {
                        "type": "object",
                        "properties": {
                            "preferred_path": {
                                "type": "string",
                                "enum": ["terrestrial", "ntn", "hybrid"],
                            },
                            "ntn_share_pct": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                            },
                        },
                        "required": ["preferred_path"],
                        "additionalProperties": False,
                    },
                    **common_meta,
                },
                "required": ["scope", "steering_objectives"],
                "additionalProperties": False,
            }

        if policy_type == "horizon.admission.control":
            return {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "object",
                        "properties": {
                            "workload_class": {"type": "string"},
                            "edge_node_id": {"type": "string"},
                        },
                        "required": ["workload_class"],
                        "additionalProperties": False,
                    },
                    "admission": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "enum": ["accept", "defer", "reject"],
                            },
                            "defer_until_seconds": {"type": "number", "minimum": 0},
                        },
                        "required": ["decision"],
                        "additionalProperties": False,
                    },
                    **common_meta,
                },
                "required": ["scope", "admission"],
                "additionalProperties": False,
            }

        if policy_type == "horizon.spectrum.reservation":
            return {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "object",
                        "properties": {
                            "band_id": {"type": "string"},
                            "cell_id": {"type": "string"},
                        },
                        "required": ["band_id"],
                        "additionalProperties": False,
                    },
                    "reservation": {
                        "type": "object",
                        "properties": {
                            "freq_low_hz": {"type": "number", "minimum": 0},
                            "freq_high_hz": {"type": "number", "minimum": 0},
                            "traffic_class": {"type": "string"},
                        },
                        "required": ["freq_low_hz", "freq_high_hz", "traffic_class"],
                        "additionalProperties": False,
                    },
                    **common_meta,
                },
                "required": ["scope", "reservation"],
                "additionalProperties": False,
            }

        # Unknown type: refuse to ship a permissive schema.
        raise ValueError(f"no schema defined for policy_type {policy_type!r}")
