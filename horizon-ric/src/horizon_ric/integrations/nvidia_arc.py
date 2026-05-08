"""NVIDIA Aerial Cloud-Native RAN (ARC) management-plane integration.

The cuBB L1/L2 stack itself is reached via FAPI (parquet ingest is in
``horizon_ric.data.aerial`` and the live FAPI socket lives in cuBB's
SDK; PreceptualAI does NOT ship E2 today). This module covers the
*management* plane — Aerial Cloud RAN's REST API for:

  * uploading model checkpoints to its inference model registry
  * reading PM (performance management) KPIs from its data plane

We deliberately keep the surface narrow — the parts an rApp vendor
actually needs to integrate with ARC at deploy time. Anything broader
(E2 service models, FAPI live control) is out of scope and called out
honestly in docs/SMO_INTEGRATION.md.

Reference (public surface):
    NVIDIA Aerial Cloud RAN documentation. The full administrator
    surface lives behind an NVIDIA Developer login; the public reference
    documents the REST shape used here.

The client uses ``httpx.AsyncClient`` so it is fully testable against
``httpx.MockTransport`` — no live network or proprietary stack is
required to validate the payload contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ARCClientConfig:
    base_url: str = "https://aerial-arc.local"
    api_key: str | None = None
    timeout_seconds: float = 30.0
    # Path under which the ARC management API is mounted. The default
    # mirrors the public docs surface; operators override per-deployment.
    api_root: str = "/api/v1"
    # Model registry path.
    models_path: str = "/models"
    # PM KPI path.
    pm_kpis_path: str = "/pm/kpis"


class ARCClient:
    """Async client for NVIDIA Aerial Cloud RAN management API.

    Authentication is bearer-token via ``X-NVIDIA-API-KEY`` header (the
    ARC-published header name). All endpoints accept JSON.
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
        """Upload a checkpoint to ARC's inference model registry.

        The wire shape used by ARC is a multipart/form-data POST with two
        parts:
          * ``model``       — the binary checkpoint file
          * ``model_card``  — the JSON metadata blob (model_id, framework,
                              input/output schema, dataset card, etc.)

        Returns the JSON registry record on success.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"model checkpoint not found: {path}")

        # Hash the checkpoint up-front for the registry record. Real ARC
        # validates the SHA-256 we send matches the bytes uploaded.
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                hasher.update(chunk)
        sha256 = hasher.hexdigest()

        # Inject the hash into the model card so the registry can match.
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
        """Read PM KPIs from ARC's data plane for one cell.

        Args:
            cell_id: ARC cell identifier (matches FAPI cellId).
            time_window: (start_iso8601, end_iso8601) — inclusive.

        Returns a dict mirroring ARC's PM JSON: ``cell_id``, ``kpis``
        (list of {name, unit, samples}), and ``window`` echo-back.
        """
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


__all__ = ["ARCClient", "ARCClientConfig"]
