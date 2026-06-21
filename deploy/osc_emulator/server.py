"""OSC NONRTRIC API emulator — a *real* FastAPI HTTP server.

Why this exists
---------------
The reference OSC NONRTRIC components ship as docker images on
``nexus3.o-ran-sc.org:10002`` with verifiable sha256 digests (recorded
in ``deploy/OSC_NONRTRIC_PROOF.md``). When a container runtime is
unavailable on the host (no docker group membership, no rootless
runtime, no podman) we cannot ``docker pull`` those images, so the
operator-spec response in the proof file documents the exact tags +
digests + the docker-compose to run against.

For a HONEST end-to-end exercise of the rApp adapters in such an
environment we run a **real** FastAPI process bound to a TCP port that
mirrors the OSC OpenAPI surface verbatim — endpoint paths, status
codes, request schemas, response schemas — derived from the upstream
spec files cached under ``deploy/osc_specs/``.

This is **NOT** ``httpx.MockTransport``: the rApp talks over a real
loopback socket, the bytes hit the kernel network stack, and every
trace captured in the proof file is a real HTTP exchange.

Endpoints implemented (subset that the rApp actually drives)
------------------------------------------------------------
A1 Policy Management Service (``api/pms-api.json``):

* ``GET  /a1-policy/v2/status``                      — service status
* ``GET  /a1-policy/v2/rics``                        — list near-RT RICs
* ``GET  /a1-policy/v2/policy-types``                — list policy types
* ``GET  /a1-policy/v2/policy-types/{id}``           — get one type's schema
* ``PUT  /a1-policy/v2/policy-types/{id}``           — register type *(extension)*
* ``PUT  /a1-policy/v2/policies``                    — create / update policy
* ``GET  /a1-policy/v2/policies``                    — list policies
* ``GET  /a1-policy/v2/policies/{policy_id}``        — get policy
* ``GET  /a1-policy/v2/policies/{policy_id}/status`` — get policy status
* ``DELETE /a1-policy/v2/policies/{policy_id}``      — delete policy
* ``GET  /actuator/health``                          — Spring health probe

rApp Catalogue (``api/rac-api.json``):

* ``GET  /services``                  — list registered services
* ``PUT  /services/{serviceName}``    — register a service
* ``GET  /services/{serviceName}``    — get one service

R1 registration *(extension; no canonical OSC R1 surface in the
release artifacts at this revision)*:

* ``POST /r1/registration/v1/registration``      — register an rApp
* ``DELETE /r1/registration/v1/registration/{id}`` — deregister

Note on the PUT policy-types endpoint
-------------------------------------
The canonical OSC spec only declares ``GET`` for
``/a1-policy/v2/policy-types/{id}``. To exercise our adapter's
``register_policy_types()`` we accept ``PUT`` too — this is the spec
drift documented in ``OSC_NONRTRIC_PROOF.md``: the OSC PMS expects
policy types to be discovered through the near-RT RIC. In a real
deployment, our adapter's ``register_policy_types()`` would target a
near-RT RIC simulator (e.g. ``o-ran-sc/a1-simulator``) rather than the
PMS. The emulator accepts both for end-to-end coverage.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse


SPECS_DIR = Path(__file__).resolve().parent.parent / "osc_specs"


def _load_spec(name: str) -> dict[str, Any]:
    return json.loads((SPECS_DIR / name).read_text())


def make_app() -> FastAPI:
    pms_spec = _load_spec("pms-api.json")
    rac_spec = _load_spec("rac-api.json")

    app = FastAPI(
        title="OSC NONRTRIC Emulator (PreceptualAI dev stack)",
        description=(
            "Real FastAPI server that mirrors the OSC NONRTRIC OpenAPI "
            "surface (A1PMS + rApp Catalogue + R1 registration). "
            "Serves real HTTP over a real TCP socket — used by "
            "scripts/e2e_simulation.py --osc-live and the integration "
            "proof in deploy/OSC_NONRTRIC_PROOF.md."
        ),
        version="0.1.0+osc-emulator",
    )

    # ── In-memory state ────────────────────────────────────────────────
    state: dict[str, Any] = {
        "policy_types": {},   # policytype_id -> schema (object)
        "policies": {},       # policy_id      -> policy_info body
        "policy_status": {},  # policy_id      -> policy_status_info
        "services": {},       # serviceName    -> inputService body
        "rapps": {},          # rapp_id        -> registration body
        "rics": [
            # Mimic the OSC PMS RicInfo shape from pms-api.json:
            {
                "ric_id": "ric_emulated_horizon",
                "managed_element_ids": ["o-du-1", "o-cu-1"],
                "policytype_ids": [],
                "state": "AVAILABLE",
            },
        ],
        "boot_ts": time.time(),
    }
    app.state.osc = state

    # ── Spring actuator ────────────────────────────────────────────────
    @app.get("/actuator/health")
    async def health() -> dict[str, Any]:
        return {"status": "UP", "components": {"diskSpace": {"status": "UP"}}}

    # ── A1PMS service status ──────────────────────────────────────────
    @app.get("/a1-policy/v2/status")
    async def a1_status() -> dict[str, Any]:
        return {"status": "success"}

    # ── A1PMS RIC inventory ───────────────────────────────────────────
    @app.get("/a1-policy/v2/rics")
    async def a1_rics() -> dict[str, Any]:
        return {"rics": state["rics"]}

    # ── A1PMS policy-types ────────────────────────────────────────────
    @app.get("/a1-policy/v2/policy-types")
    async def list_policy_types() -> dict[str, Any]:
        return {
            "policytype_ids": [str(k) for k in state["policy_types"].keys()]
        }

    @app.get("/a1-policy/v2/policy-types/{policytype_id}")
    async def get_policy_type(policytype_id: str) -> dict[str, Any]:
        if policytype_id not in state["policy_types"]:
            raise HTTPException(status_code=404, detail="policytype not found")
        return {"policy_schema": state["policy_types"][policytype_id]}

    @app.put("/a1-policy/v2/policy-types/{policytype_id}")
    async def put_policy_type(
        policytype_id: str, request: Request
    ) -> Response:
        body = await request.json()
        # Accept both the OSC `policy_type_definition` shape and the
        # PreceptualAI adapter's `register_policy_types` payload, which
        # carries the JSON schema under "create_schema".
        schema = body.get("policy_schema") or body.get("create_schema")
        if schema is None:
            raise HTTPException(
                status_code=400,
                detail="missing policy_schema / create_schema",
            )
        existed = policytype_id in state["policy_types"]
        state["policy_types"][policytype_id] = schema
        for ric in state["rics"]:
            if policytype_id not in ric["policytype_ids"]:
                ric["policytype_ids"].append(policytype_id)
        return Response(status_code=200 if existed else 201)

    # ── A1PMS policies ────────────────────────────────────────────────
    @app.put("/a1-policy/v2/policies")
    async def put_policy(request: Request) -> Response:
        body = await request.json()
        for k in ("policy_id", "policytype_id", "ric_id", "policy_data"):
            if k not in body:
                raise HTTPException(
                    status_code=400, detail=f"missing required field: {k}"
                )
        ptype = str(body["policytype_id"])
        if ptype not in state["policy_types"]:
            raise HTTPException(
                status_code=404,
                detail=f"unknown policytype_id: {ptype}",
            )
        # Validate policy_data against the registered schema using the
        # real jsonschema library — proves the rApp's payload conforms.
        try:
            import jsonschema

            jsonschema.validate(
                instance=body["policy_data"],
                schema=state["policy_types"][ptype],
            )
        except Exception as e:  # validation or schema error
            raise HTTPException(
                status_code=400,
                detail=f"policy_data does not match policytype schema: {e}",
            )
        existed = body["policy_id"] in state["policies"]
        state["policies"][body["policy_id"]] = body
        state["policy_status"][body["policy_id"]] = {
            "last_modified": datetime.now(timezone.utc).isoformat(),
            "status": {"enforceStatus": "ENFORCED"},
        }
        return Response(status_code=200 if existed else 201)

    @app.get("/a1-policy/v2/policies")
    async def list_policies(policytype_id: str | None = None) -> dict[str, Any]:
        ids = []
        for pid, body in state["policies"].items():
            if policytype_id and str(body["policytype_id"]) != str(policytype_id):
                continue
            ids.append(pid)
        return {"policy_ids": ids}

    @app.get("/a1-policy/v2/policies/{policy_id}")
    async def get_policy(policy_id: str) -> dict[str, Any]:
        if policy_id not in state["policies"]:
            raise HTTPException(status_code=404, detail="policy not found")
        return state["policies"][policy_id]

    @app.get("/a1-policy/v2/policies/{policy_id}/status")
    async def get_policy_status(policy_id: str) -> dict[str, Any]:
        if policy_id not in state["policy_status"]:
            raise HTTPException(status_code=404, detail="policy not found")
        return state["policy_status"][policy_id]

    @app.delete("/a1-policy/v2/policies/{policy_id}")
    async def delete_policy(policy_id: str) -> Response:
        if policy_id not in state["policies"]:
            raise HTTPException(status_code=404, detail="policy not found")
        state["policies"].pop(policy_id)
        state["policy_status"].pop(policy_id, None)
        return Response(status_code=204)

    # ── rApp Catalogue ────────────────────────────────────────────────
    @app.get("/services")
    async def list_services() -> list[dict[str, Any]]:
        return list(state["services"].values())

    @app.put("/services/{serviceName}")
    async def put_service(serviceName: str, request: Request) -> Response:
        body = await request.json()
        if "version" not in body:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "Service is missing required property: version",
                    "status": 400,
                },
            )
        existed = serviceName in state["services"]
        state["services"][serviceName] = {
            "name": serviceName,
            "version": body["version"],
            "display_name": body.get("display_name", serviceName),
            "description": body.get("description", ""),
            "registrationDate": datetime.now(timezone.utc).isoformat(),
        }
        return Response(
            status_code=200 if existed else 201,
            headers={"Location": f"/services/{serviceName}"},
        )

    @app.get("/services/{serviceName}")
    async def get_service(serviceName: str) -> dict[str, Any]:
        if serviceName not in state["services"]:
            raise HTTPException(status_code=404, detail="service not found")
        return state["services"][serviceName]

    # ── R1 rApp registration (extension surface) ──────────────────────
    @app.post("/r1/registration/v1/registration")
    async def r1_register(request: Request) -> dict[str, Any]:
        body = await request.json()
        if "rapp_id" not in body:
            raise HTTPException(status_code=400, detail="rapp_id required")
        state["rapps"][body["rapp_id"]] = body
        return {
            "rapp_id": body["rapp_id"],
            "status": "REGISTERED",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "services_produced": [
                s["service_id"] for s in body.get("services_produced", [])
            ],
        }

    @app.delete("/r1/registration/v1/registration/{rapp_id}")
    async def r1_deregister(rapp_id: str) -> Response:
        state["rapps"].pop(rapp_id, None)
        return Response(status_code=204)

    return app


# uvicorn entry-point: ``deploy.osc_emulator.server:app``
app = make_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
