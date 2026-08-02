#!/usr/bin/env python3
"""Build the committed, self-verifying evidence fixture for the audit gate.

``audit/verify_evidence.py`` (the offline evidence auditor) needs a committed
evidence export to run against on a clean checkout. The live E2E export
(``data/e2e/audit.jsonl``) is produced by a stack run and is not version
controlled, so this script generates a small, deterministic export whose
per-tenant hash chain is computed by the SAME ``horizon_audit.chain.chain_hash``
the verifier recomputes. The cryptographic linkage is therefore genuine — the
verifier independently re-derives every hash — while the record payloads are
clearly-labelled illustrative Shield decisions.

Run from the repo root:

    python audit/fixtures/build_evidence_sample.py

It is deterministic: re-running reproduces byte-identical output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "audit" / "src"))

from horizon_audit.chain import (  # noqa: E402
    ZERO_HASH_HEX,
    canonical_json,
    chain_hash,
)

OUT = HERE / "evidence-sample.jsonl"

TENANT = "audit-demo"

# Illustrative Shield decision records. The numbers are fixture values, not a
# measured result; the point of the fixture is the hash chain, which is real.
RECORDS = [
    {
        "decision_id": "repro-fixture-0001",
        "tenant_id": TENANT,
        "chosen_action": {"eirp_dbm": 33.0, "band": "n78"},
        "shield": "PASS",
        "illegal_requested": True,
        "projected": True,
        "note": "audit fixture; hashes are real, payload is illustrative",
    },
    {
        "decision_id": "repro-fixture-0002",
        "tenant_id": TENANT,
        "chosen_action": {"eirp_dbm": 30.0, "band": "n78"},
        "shield": "PASS",
        "illegal_requested": False,
        "projected": False,
        "note": "audit fixture; hashes are real, payload is illustrative",
    },
    {
        "decision_id": "repro-fixture-0003",
        "tenant_id": TENANT,
        "chosen_action": {"eirp_dbm": 33.0, "band": "n77"},
        "shield": "PASS",
        "illegal_requested": True,
        "projected": True,
        "note": "audit fixture; hashes are real, payload is illustrative",
    },
]


def build() -> list[dict]:
    lines: list[dict] = []
    prev = ZERO_HASH_HEX
    for rec in RECORDS:
        # Match the producer contract: the stored record is already the
        # canonicalised dict, so the verifier's re-canonicalisation reproduces
        # the same bytes and therefore the same hash.
        canon = json.loads(canonical_json(rec))
        digest = chain_hash(prev, canon)
        lines.append({"hash": digest, "tenant_id": TENANT, "record": canon})
        prev = digest
    return lines


def main() -> int:
    lines = build()
    OUT.write_text(
        "\n".join(json.dumps(obj, sort_keys=True) for obj in lines) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
