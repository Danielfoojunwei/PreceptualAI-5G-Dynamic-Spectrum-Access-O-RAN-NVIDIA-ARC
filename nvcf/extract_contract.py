#!/usr/bin/env python3
"""Extract NVCF's invocation contract from its own OpenAPI specification.

Same discipline as ``ocudu/extract_catalogue.py``: the constants our client
uses are *derived from the vendor's published specification*, not transcribed
from documentation into a Python file where they rot silently. If NVIDIA
changes a header name or adds a terminal status code, re-running this against
a newer spec changes ``contract/nvcf-invocation-contract.json``, and gate G10
fails until someone looks.

The spec is vendored at ``nvcf/spec/nvcf-openapi-<version>.json`` — fetched
live from ``https://api.nvcf.nvidia.com/v3/openapi`` and re-serialised
canonically (``indent=2, sort_keys=True``) so its SHA-256 is stable
independent of transport framing.

Run ``python nvcf/extract_contract.py --out contract/nvcf-invocation-contract.json``
to regenerate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC_DIR = HERE / "spec"
CONTRACT_PATH = HERE / "contract" / "nvcf-invocation-contract.json"

# The two operations the whole integration turns on. Named by OpenAPI
# operationId so a path change upstream is visible as a path change here,
# rather than as a lookup that silently finds nothing.
INVOKE_OPS = ("invokeFunction", "invokeFunction_1")
POLL_OP = "getFunctionInvocationResult"


class ExtractError(RuntimeError):
    """The vendored spec does not contain what the contract needs."""


def spec_path() -> Path:
    """The single vendored spec. More than one is an error, not a choice."""
    found = sorted(SPEC_DIR.glob("nvcf-openapi-*.json"))
    if len(found) != 1:
        raise ExtractError(
            f"expected exactly one vendored spec in {SPEC_DIR}, found "
            f"{[p.name for p in found]}"
        )
    return found[0]


def load_spec() -> tuple[dict[str, Any], str, str]:
    """Return ``(spec, sha256, version)`` for the vendored specification."""
    p = spec_path()
    raw = p.read_bytes()
    spec = json.loads(raw.decode("utf-8"))
    return spec, hashlib.sha256(raw).hexdigest(), str(spec["info"]["version"])


def _operations(spec: dict[str, Any]) -> dict[str, tuple[str, str, dict[str, Any]]]:
    """``operationId -> (method, path, operation)`` across the whole spec."""
    out: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue
            op_id = op.get("operationId")
            if op_id:
                out[op_id] = (method.upper(), path, op)
    return out


def _describe(method: str, path: str, op: dict[str, Any]) -> dict[str, Any]:
    responses = op.get("responses") or {}
    return {
        "method": method,
        "path": path,
        "summary": op.get("summary"),
        "request_headers": sorted(
            p["name"] for p in op.get("parameters", []) if p.get("in") == "header"
        ),
        "path_params": [
            p["name"] for p in op.get("parameters", []) if p.get("in") == "path"
        ],
        "status_codes": {
            code: {
                "description": " ".join((r.get("description") or "").split()),
                "headers": sorted((r.get("headers") or {}).keys()),
            }
            for code, r in sorted(responses.items())
        },
    }


def build() -> dict[str, Any]:
    """Rebuild the whole contract from the vendored spec."""
    spec, digest, version = load_spec()
    ops = _operations(spec)

    missing = [o for o in (*INVOKE_OPS, POLL_OP) if o not in ops]
    if missing:
        raise ExtractError(
            f"vendored spec {version} does not define {missing}; the "
            "invocation contract cannot be extracted from it"
        )

    invoke = {op_id: _describe(*ops[op_id]) for op_id in INVOKE_OPS}
    poll = _describe(*ops[POLL_OP])

    # Every NVCF-* header the spec mentions anywhere, request or response.
    nvcf_headers: set[str] = set()
    for _method, _path, op in ops.values():
        for prm in op.get("parameters", []):
            if prm.get("in") == "header" and prm["name"].upper().startswith("NVCF-"):
                nvcf_headers.add(prm["name"])
        for r in (op.get("responses") or {}).values():
            for h in (r.get("headers") or {}):
                if h.upper().startswith("NVCF-"):
                    nvcf_headers.add(h)

    schemas = spec.get("components", {}).get("schemas", {})

    def _enum(schema_name: str, prop: str) -> list[str]:
        s = schemas.get(schema_name, {}).get("properties", {}).get(prop, {})
        return list(s.get("enum") or [])

    def _required(schema_name: str) -> list[str]:
        return list(schemas.get(schema_name, {}).get("required") or [])

    # Terminal vs retryable, decided by the spec's own descriptions rather
    # than by our reading of them: 202 is the only "keep polling" code the
    # invocation endpoints declare, and 302 is the only redirect.
    invoke_codes = set()
    for d in invoke.values():
        invoke_codes |= set(d["status_codes"])
    poll_codes = set(poll["status_codes"])

    asset_ops = {
        op_id: _describe(*ops[op_id])
        for op_id in ("createAsset", "getAsset", "deleteAsset", "getAssets")
        if op_id in ops
    }

    return {
        "schema": "horizon-ric.nvcf.contract/1",
        "source": {
            "url": "https://api.nvcf.nvidia.com/v3/openapi",
            "vendored_as": str(spec_path().relative_to(HERE)),
            "openapi_version": spec["openapi"],
            "api_version": version,
            "sha256": digest,
            "server": (spec.get("servers") or [{}])[0].get("url"),
        },
        "invoke": invoke,
        "poll": poll,
        "assets": asset_ops,
        "nvcf_headers": sorted(nvcf_headers),
        "invoke_status_codes": sorted(invoke_codes),
        "poll_status_codes": sorted(poll_codes),
        "function_status_enum": _enum("FunctionDto", "status"),
        "function_type_enum": _enum("FunctionDto", "functionType"),
        "api_body_format_enum": _enum("FunctionDto", "apiBodyFormat"),
        "health_protocol_enum": _enum("HealthDto", "protocol"),
        "health_required": _required("HealthDto"),
        "function_required": _required("FunctionDto"),
        "gpu_specification_required": _required("GpuSpecificationDto"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(CONTRACT_PATH),
        help=f"write the contract JSON here (default {CONTRACT_PATH})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed contract differs from a fresh "
        "extraction, instead of writing",
    )
    args = parser.parse_args(argv)

    try:
        contract = build()
    except ExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    out = Path(args.out)

    if args.check:
        if not out.exists():
            print(f"error: {out} does not exist", file=sys.stderr)
            return 1
        if out.read_text(encoding="utf-8") != text:
            print(
                f"error: {out} differs from a fresh extraction of "
                f"{spec_path().name}; re-run without --check",
                file=sys.stderr,
            )
            return 1
        print(f"{out.name} is up to date (API {contract['source']['api_version']})")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(
        f"wrote {out} from {spec_path().name} "
        f"(API {contract['source']['api_version']}, "
        f"{len(contract['nvcf_headers'])} NVCF-* headers)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
