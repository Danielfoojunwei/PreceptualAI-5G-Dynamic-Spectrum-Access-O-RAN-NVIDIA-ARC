"""DEPRECATED — legacy entry point for NVIDIA Aerial integrations.

This module historically housed the management-plane ``ARCClient``
(model registry uploads + PM KPI reads) and is the path many existing
imports take. The new real-time I/Q feed consumer lives in
``horizon_ric.integrations.nvidia_arc_ota`` as ``ARCOTAConsumer``;
that is the symbol new code should reach for.

This shim:

  * preserves the legacy ``ARCClient`` / ``ARCClientConfig`` surface
    (the management plane is genuinely separate from the real-time
    consumer, so we do not delete it — we just stop adding to it here);
  * re-exports the new ``ARCOTAConfig`` / ``ARCOTAConsumer`` symbols so
    backwards-compatible imports keep working through 0.2.0;
  * emits a ``DeprecationWarning`` at import time pointing callers at
    ``nvidia_arc_ota``.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

# Re-export the real-time consumer so existing
# `from horizon_ric.integrations.nvidia_arc import ARCOTAConsumer`
# style imports keep working through 0.2.0.
from horizon_ric.integrations.nvidia_arc_ota import (  # noqa: F401
    ARCOTAConfig,
    ARCOTAConsumer,
)

warnings.warn(
    "horizon_ric.integrations.nvidia_arc is deprecated; "
    "use horizon_ric.integrations.nvidia_arc_ota for the real-time "
    "I/Q feed (ARCOTAConsumer). The legacy ARCClient management-plane "
    "API in this module remains available through 0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)

logger = structlog.get_logger(__name__)


@dataclass
class ARCClientConfig:
    """Legacy management-plane client config — kept for compatibility."""

    base_url: str = "https://aerial-arc.local"
    api_key: str | None = None
    timeout_seconds: float = 30.0
    api_root: str = "/api/v1"
    models_path: str = "/models"
    pm_kpis_path: str = "/pm/kpis"


class ARCClient:
    """Legacy async client for NVIDIA Aerial Cloud RAN management API.

    Authentication is bearer-token via ``X-NVIDIA-API-KEY`` header (the
    ARC-published header name). All endpoints accept JSON. New code
    SHOULD migrate to the realtime consumer in ``nvidia_arc_ota`` when
    it needs the I/Q feed; this client remains the right tool for the
    management plane (model registry + PM KPIs).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        config: ARCClientConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        cfg = config or ARCClientConfig()
        if base_url is not None:
            cfg.base_url = base_url
        if api_key is not None:
            cfg.api_key = api_key
        self.cfg = cfg
        headers = {"Accept": "application/json"}
        if cfg.api_key:
            headers["X-NVIDIA-API-KEY"] = cfg.api_key
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            timeout=cfg.timeout_seconds,
            headers=headers,
            transport=transport,
        )

    # -- URL helpers -------------------------------------------------------

    def _models_url(self, model_id: str | None = None) -> str:
        url = f"{self.cfg.api_root}{self.cfg.models_path}"
        if model_id:
            url += f"/{model_id}"
        return url

    def _pm_kpis_url(self, cell_id: str) -> str:
        return f"{self.cfg.api_root}{self.cfg.pm_kpis_path}/{cell_id}"

    # -- Public methods ----------------------------------------------------

    async def submit_inference_model(
        self,
        model_path: str | os.PathLike,
        model_card: dict[str, Any],
    ) -> dict[str, Any]:
        """Upload a checkpoint to ARC's inference model registry."""

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"model checkpoint not found: {path}")

        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                hasher.update(chunk)
        sha256 = hasher.hexdigest()

        card = dict(model_card)
        card.setdefault("artifact_sha256", sha256)
        card.setdefault("artifact_filename", path.name)
        card.setdefault("artifact_bytes", path.stat().st_size)

        files = {
            "model": (path.name, path.read_bytes(), "application/octet-stream"),
            "model_card": (
                "model_card.json",
                json.dumps(card).encode("utf-8"),
                "application/json",
            ),
        }
        try:
            resp = await self._client.post(self._models_url(), files=files)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("arc.model.submit.failed", path=str(path), error=str(e))
            raise
        try:
            body = resp.json()
        except json.JSONDecodeError:
            body = {"raw": resp.text}
        logger.info(
            "arc.model.submitted",
            sha256=sha256,
            registry_id=body.get("model_id"),
            status=resp.status_code,
        )
        return body

    async def get_pm_kpis(
        self,
        cell_id: str,
        time_window: tuple[str, str],
    ) -> dict[str, Any]:
        """Read PM KPIs from ARC's data plane for one cell."""

        start, end = time_window
        params = {"start": start, "end": end}
        try:
            resp = await self._client.get(
                self._pm_kpis_url(cell_id), params=params
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(
                "arc.pm_kpis.fetch.failed",
                cell_id=cell_id,
                window=time_window,
                error=str(e),
            )
            raise
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ARCClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


__all__ = [
    "ARCClient",
    "ARCClientConfig",
    "ARCOTAConfig",
    "ARCOTAConsumer",
]
