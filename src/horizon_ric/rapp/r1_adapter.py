"""R1 adapter — rApp ↔ SMO communication (O-RAN.WG2.R1AP / R1GAP).

This is a stub for Phase 1. The full implementation calls the SMO's R1AP
service via REST (per the O-RAN R1 reference). For now we provide:
  - rApp registration / deregistration
  - service consumption (DME, AI/ML model pull)
  - service production (publishing the rApp's REST API to SMO catalog)

The transport layer is HTTP/REST per O-RAN.WG2.R1GAP §5.2; we use httpx.

References:
    O-RAN.WG2.R1AP-v06.00 §5 (Application Protocol)
    O-RAN.WG2.R1GAP-v06.00 §5.2 (Transport)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class R1AdapterConfig:
    smo_base_url: str = "http://nonrtric:8080"
    rapp_id: str = "horizon-ric-rapp"
    rapp_version: str = "0.1.0"
    rapp_name: str = "PreceptualAI Resource-Orchestration"
    rapp_description: str = (
        "NTN-Aware AI-RAN Resource-Orchestration rApp Suite (PRD: PreceptualAI)"
    )
    timeout_seconds: float = 30.0
    auth: "AuthConfig | None" = None  # noqa: F821 — forward ref
    """When set, R1 client uses mTLS / OAuth2 / static-bearer per O-RAN
    WG11 §6. None keeps the legacy plain-HTTP behaviour for unsecured
    testbeds."""
    services_produced: list[str] = field(
        default_factory=lambda: [
            "horizon.risk.sla",
            "horizon.risk.beam",
            "horizon.risk.gateway",
            "horizon.risk.compute",
            "horizon.policy.recommend",
            "horizon.policy.rollback",
            "horizon.evidence.query",
        ]
    )
    services_consumed: list[str] = field(
        default_factory=lambda: [
            "smo.dme.subscribe",
            "smo.aiml.model.pull",
            "smo.aiml.model.push",
            "smo.r1.service.discovery",
        ]
    )
    # Optional TTL DNS cache for the SMO host. 0 = OFF (default). See
    # `horizon_ric.runtime.dns_cache` for the wire-up and TTL semantics.
    dns_cache_ttl_s: float = 0.0


class R1Adapter:
    """rApp framework client for the O-RAN R1 interface (Phase 1 stub).

    Production implementation will use the SMO's full R1AP REST catalog;
    this stub validates registration and produces well-formed payloads.
    """

    def __init__(self, config: R1AdapterConfig | None = None):
        self.cfg = config or R1AdapterConfig()
        if self.cfg.auth is not None:
            from horizon_ric.rapp.auth import build_secure_async_client

            self._client = build_secure_async_client(
                self.cfg.auth, base_url=self.cfg.smo_base_url
            )
        else:
            self._client = httpx.AsyncClient(
                base_url=self.cfg.smo_base_url,
                timeout=self.cfg.timeout_seconds,
            )
        self._registered = False
        # Real circuit breaker — fail-fast when the SMO is down so we
        # don't hammer it. See src/horizon_ric/runtime/circuit_breaker.py.
        from horizon_ric.runtime.circuit_breaker import (
            AsyncCircuitBreaker,
            BreakerConfig,
        )

        self._cb = AsyncCircuitBreaker(
            BreakerConfig(name="horizon.r1", fail_max=5, reset_timeout=30.0)
        )

    @property
    def circuit_breaker(self):
        """Expose the breaker for tests / observability."""
        return self._cb

    async def register(self) -> dict[str, Any]:
        """Register the rApp with the SMO/Non-RT RIC.

        Sends an R1AP registration request per O-RAN.WG2.R1AP §5.3.
        """
        payload = self._registration_payload()
        logger.info("r1.register", rapp_id=self.cfg.rapp_id, services=len(payload["services_produced"]))

        async def _do_register():
            resp = await self._client.post(
                "/r1/registration/v1/registration", json=payload
            )
            resp.raise_for_status()
            return resp

        try:
            resp = await self._cb.call(_do_register)
            self._registered = True
            return resp.json()
        except httpx.HTTPError as e:
            logger.error("r1.register.failed", error=str(e))
            raise

    async def deregister(self) -> None:
        """Deregister the rApp."""
        if not self._registered:
            return
        try:
            await self._client.delete(
                f"/r1/registration/v1/registration/{self.cfg.rapp_id}"
            )
            self._registered = False
            logger.info("r1.deregister", rapp_id=self.cfg.rapp_id)
        except httpx.HTTPError as e:
            logger.error("r1.deregister.failed", error=str(e))

    def is_registered(self) -> bool:
        return self._registered

    async def close(self) -> None:
        await self._client.aclose()

    def _registration_payload(self) -> dict[str, Any]:
        """Build the R1 registration payload per spec.

        Schema follows O-RAN.WG2.R1AP §5.3.1 RegistrationRequest.
        """
        return {
            "rapp_id": self.cfg.rapp_id,
            "rapp_version": self.cfg.rapp_version,
            "rapp_name": self.cfg.rapp_name,
            "rapp_description": self.cfg.rapp_description,
            "services_produced": [
                {"service_id": svc, "version": self.cfg.rapp_version}
                for svc in self.cfg.services_produced
            ],
            "services_consumed": [
                {"service_id": svc} for svc in self.cfg.services_consumed
            ],
            "rapp_metadata": {
                "vendor": "Preceptual AI",
                "license": "Apache-2.0",
                "spec_compliance": [
                    "O-RAN.WG2.R1AP-v06.00",
                    "O-RAN.WG2.R1GAP-v06.00",
                    "O-RAN.WG2.AIML-v04.00",
                    "3GPP TS 28.105",
                ],
            },
        }
