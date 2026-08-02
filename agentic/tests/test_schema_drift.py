"""The published schemas must track the dataclasses, or they are documentation.

Same discipline as ``tests/test_assurance_schemas.py`` in the main package, and
for the same reason: a schema that drifts from the object it describes is worse
than no schema, because an integrator writes against it and their parser
diverges silently from what the wire actually carries.

No ``jsonschema`` dependency is added — the main package deliberately avoided
one, and introspecting ``dataclasses.fields`` catches the failure that actually
happens (a field added to the code and not to the schema) without pulling in a
validator to do it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from horizon_agentic.envelope import SCOPE_ACTION_KEYS, AgentActionEnvelope
from horizon_agentic.evidence import TransactionCertificate

from horizon_ric.assurance.planner import REQUIRED_ACTION_KEYS

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_envelope_schema_covers_every_field() -> None:
    schema = load("agent-action-envelope.schema.json")
    declared = set(schema["properties"])
    actual = {f.name for f in dataclasses.fields(AgentActionEnvelope)}
    assert actual - declared == set(), (
        f"envelope fields with no schema property: {sorted(actual - declared)}"
    )
    assert declared - actual == set(), (
        f"schema describes properties the envelope does not have: "
        f"{sorted(declared - actual)}"
    )


def test_certificate_schema_covers_every_field() -> None:
    schema = load("transaction-certificate.schema.json")
    declared = set(schema["properties"])
    actual = {f.name for f in dataclasses.fields(TransactionCertificate)}
    assert actual - declared == set(), (
        f"certificate fields with no schema property: {sorted(actual - declared)}"
    )
    assert declared - actual == set(), (
        f"schema describes properties the certificate does not have: "
        f"{sorted(declared - actual)}"
    )


def test_envelope_schema_requires_what_authentication_needs() -> None:
    """A schema that made the signature optional would invite unsigned traffic."""
    schema = load("agent-action-envelope.schema.json")
    required = set(schema["required"])
    for name in ("agent_id", "signature", "nonce", "issued_at", "mutates"):
        assert name in required, f"{name} must be required"


def test_schema_scope_enum_matches_the_code() -> None:
    schema = load("agent-action-envelope.schema.json")
    enum = set(schema["properties"]["granted_scopes"]["items"]["enum"])
    assert enum == set(SCOPE_ACTION_KEYS), (
        f"scope enum drifted: schema {sorted(enum)}, code {sorted(SCOPE_ACTION_KEYS)}"
    )


def test_schema_action_requirements_match_the_planner_contract() -> None:
    schema = load("agent-action-envelope.schema.json")
    required = set(schema["properties"]["requested_action"]["required"])
    assert required == set(REQUIRED_ACTION_KEYS), (
        f"requested_action required-keys drifted from REQUIRED_ACTION_KEYS: "
        f"schema {sorted(required)}, code {sorted(REQUIRED_ACTION_KEYS)}"
    )


@pytest.mark.parametrize(
    "name",
    ["agent-action-envelope.schema.json", "transaction-certificate.schema.json"],
)
def test_schemas_are_well_formed(name: str) -> None:
    schema = load(name)
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert schema["$id"]
    assert schema["title"]
    assert schema.get("additionalProperties") is False, (
        "an open schema cannot detect an unexpected field, which is the drift "
        "this file exists to catch"
    )


def test_certificate_serialises_against_its_own_schema_shape() -> None:
    """Round-trip: a real certificate's keys must be exactly the declared ones."""
    from horizon_agentic.evidence import canonical_transaction_bytes

    cert = TransactionCertificate(
        transaction_id="t1", issued_at="2026-01-01T00:00:00Z", committed=True
    )
    payload = json.loads(canonical_transaction_bytes(cert))
    declared = set(load("transaction-certificate.schema.json")["properties"])
    # canonical bytes omit the signature fields by construction
    assert set(payload) | {"signature", "signing_key_fingerprint"} == declared
