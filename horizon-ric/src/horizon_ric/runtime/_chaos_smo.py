"""Minimal SMO/Near-RT-RIC HTTP simulator for the chaos test.

A real uvicorn FastAPI server. Accepts every R1 / A1 request the rApp
sends and returns 2xx. We don't simulate failures here — failures
are injected by the chaos harness via `kill -9`.
"""

from __future__ import annotations

import argparse
import sys

from fastapi import FastAPI


def build_app() -> FastAPI:
    app = FastAPI()

    @app.post("/r1/registration/v1/registration")
    async def register(payload: dict) -> dict:
        return {"status": "registered", "rapp_id": payload.get("rapp_id")}

    @app.delete("/r1/registration/v1/registration/{rapp_id}")
    async def deregister(rapp_id: str) -> dict:
        return {"status": "deregistered", "rapp_id": rapp_id}

    @app.put("/A1-P/v2/policytypes/{type_id}")
    async def policy_type(type_id: int, payload: dict) -> dict:
        return {"policy_type_id": type_id, "status": "ok"}

    @app.put("/A1-P/v2/policytypes/{type_id}/policies/{policy_id}")
    async def policy_instance(type_id: int, policy_id: str, payload: dict) -> dict:
        return {"policy_id": policy_id, "type_id": type_id}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()

    import uvicorn

    uvicorn.run(
        build_app(),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        lifespan="off",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
