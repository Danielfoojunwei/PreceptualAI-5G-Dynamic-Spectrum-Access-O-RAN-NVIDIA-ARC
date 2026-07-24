#!/usr/bin/env python3
"""Exercise Horizon-RIC against a live official O-RAN-SC A1 simulator.

This is intentionally a narrow interoperability check: policy-type
registration, policy create, list, status and rollback over the OSC 2.1.0
REST surface. It is not a radio, operator-traffic, vendor-stack, conformance
or production-readiness test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig


async def run(base_url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        health = await client.get("/a1-p/healthcheck")
        health.raise_for_status()

    adapter = A1Adapter(
        A1AdapterConfig(
            near_rt_ric_base_url=base_url,
            dialect="osc_a1",
            timeout_seconds=10.0,
        )
    )
    policy_id = "horizon-osc-a1-live-smoke"
    try:
        accepted = await adapter.register_policy_types()
        expected = [20001, 20002, 20003, 20004]
        if accepted != expected:
            raise RuntimeError(
                f"policy type registration mismatch: {accepted!r} != {expected!r}"
            )

        emitted_id, create_status = await adapter.emit_policy(
            "horizon.qos.priority",
            {
                "scope": {"slice_id": "live-smoke-slice"},
                "qos_objectives": {"priority": 5},
            },
            policy_id=policy_id,
        )
        listed_after_create = await adapter.list_policies(
            "horizon.qos.priority"
        )
        if emitted_id not in listed_after_create:
            raise RuntimeError(f"created policy missing from list: {listed_after_create}")

        status_body = await adapter.get_policy_status(
            "horizon.qos.priority", emitted_id
        )
        if "enforceStatus" not in status_body:
            raise RuntimeError(f"unexpected OSC status body: {status_body!r}")

        delete_status = await adapter.rollback_policy(
            "horizon.qos.priority", emitted_id
        )
        listed_after_delete = await adapter.list_policies(
            "horizon.qos.priority"
        )
        if emitted_id in listed_after_delete:
            raise RuntimeError("rolled-back policy remains in simulator")
    finally:
        await adapter.close()

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "system_under_test": "Horizon-RIC A1Adapter osc_a1 dialect",
        "external_system": {
            "name": "O-RAN-SC sim-a1-interface",
            "repository": "https://github.com/o-ran-sc/sim-a1-interface",
            "commit": os.environ.get("OSC_A1_SIM_COMMIT", "unknown"),
            "interface": "OSC_2.1.0",
        },
        "operations": {
            "healthcheck": health.status_code,
            "registered_policy_type_ids": accepted,
            "policy_create": create_status,
            "policy_list_after_create_contains_id": True,
            "policy_status_has_enforce_status": True,
            "policy_delete": delete_status,
            "policy_list_after_delete_contains_id": False,
        },
        "result": "pass",
        "scope": (
            "Open-source OSC simulator REST interoperability only; no live radio, "
            "operator traffic, OSC platform deployment, commercial vendor stack, "
            "conformance certification, or carrier-scale load is exercised."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8085")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args.base_url))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
