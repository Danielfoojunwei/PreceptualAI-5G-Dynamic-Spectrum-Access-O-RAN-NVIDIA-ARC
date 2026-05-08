#!/usr/bin/env python
"""Validate the PreceptualAI CycloneDX SBOM.

Used in lieu of `cyclonedx-cli` (Go binary) when only Python is available.
Loads the JSON SBOM, runs it through the official `cyclonedx-python-lib`
JSON validator for the schema version declared in the file, and asserts
that every component has a name, version and PURL.

Exits 0 on success, non-zero on failure.

Usage:
    python deploy/sbom/validate_sbom.py [path/to/sbom.json]

Default path: deploy/sbom/horizon-ric-sbom.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: Path) -> int:
    if not path.is_file():
        print(f"FAIL: {path} not found", file=sys.stderr)
        return 2
    with path.open() as f:
        data = json.load(f)

    # Required CycloneDX top-level keys
    required = {"bomFormat", "specVersion", "components"}
    missing = required - data.keys()
    if missing:
        print(f"FAIL: missing top-level keys: {sorted(missing)}", file=sys.stderr)
        return 3
    if data["bomFormat"] != "CycloneDX":
        print(f"FAIL: bomFormat is {data['bomFormat']!r}, expected 'CycloneDX'")
        return 4

    spec_version = data["specVersion"]
    print(f"specVersion: {spec_version}")

    # Use the official validator from cyclonedx-python-lib (mirrors what
    # cyclonedx-cli does internally).
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator

    sv_attr = "V" + spec_version.replace(".", "_")
    if not hasattr(SchemaVersion, sv_attr):
        print(f"FAIL: unsupported specVersion {spec_version}", file=sys.stderr)
        return 5
    schema_version = getattr(SchemaVersion, sv_attr)
    validator = JsonStrictValidator(schema_version)
    err = validator.validate_str(path.read_text())
    if err is not None:
        print(f"FAIL: schema validation: {err}", file=sys.stderr)
        return 6

    components = data.get("components", [])
    # Local source components (this project + its in-tree SDK) legitimately
    # have no PURL; everything else must.
    bad = []
    for i, c in enumerate(components):
        if not c.get("name") or not c.get("version"):
            bad.append(i)
            continue
        is_local = any(
            (ref.get("url") or "").startswith("file://")
            for ref in c.get("externalReferences", [])
        )
        if not is_local and not c.get("purl"):
            bad.append(i)
    if bad:
        print(f"FAIL: {len(bad)} components missing required fields: {bad[:5]}")
        return 7

    print(f"OK: CycloneDX {spec_version} SBOM, {len(components)} components.")
    return 0


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "deploy/sbom/horizon-ric-sbom.json"
    )
    raise SystemExit(main(p))
