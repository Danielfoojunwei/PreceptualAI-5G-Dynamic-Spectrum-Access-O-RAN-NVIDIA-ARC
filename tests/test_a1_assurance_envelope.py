"""Contract tests for the A1 ``assurance`` envelope — the certificate on the wire.

PROPERTY PINNED: a Horizon A1 policy body may carry a *verifiable reference* to
the SafetyCertificate for the decision that produced it, and that reference is
verifiable by someone who holds only the wire body and the public key.

Four things have to be simultaneously true for that to mean anything, and each
one is a separate failure mode:

1. **The digest must be the signed bytes.** ``assurance.certificate_digest`` is
   ``sha256(canonical_certificate_bytes(cert))``. If the envelope builder ever
   grows its own canonicalisation, the digest stops being the thing
   ``assurance.signature`` signs and the whole envelope becomes decoration. The
   test recomputes it from :mod:`horizon_ric.shield.signing` directly rather
   than from any helper in the adapter.
2. **The envelope must be optional in the byte-exact sense.** Every registered
   create schema is ``additionalProperties: false``, so this change is only
   additive if a caller that passes no certificate produces the *same bytes* as
   before. Asserted as ``json.dumps`` equality, not key-set equality.
3. **The schema must declare it, for all four policy types, without requiring
   it.** An undeclared field is a 400 from any schema-validating near-RT RIC
   (the official ``o-ran-sc/sim-a1-interface`` simulator validates every PUT
   against the registered ``create_schema``); a *required* field would break
   every existing caller.
4. **A refusal must be legible as a refusal.** A blocked certificate has to
   arrive as ``safe: false`` with a non-empty ``violated_ids``, so a receiver
   reading only the wire can tell an endorsement from a rejection.

The repo has no ``jsonschema`` dependency and this file does not add one, so the
Draft-07 validator below is hand-written and covers exactly the keywords the
create schemas use: ``type`` (including type unions), ``properties``,
``required``, ``additionalProperties``, ``items``, ``enum``, ``pattern``,
``minimum``, ``maximum``. Its own non-vacuity is asserted first — a permissive
validator would make every schema test here green and meaningless.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from horizon_ric.rapp.a1_adapter import (
    ASSURANCE_ENVELOPE_KEYS,
    DEFAULT_POLICY_TYPES,
    A1Adapter,
    assurance_envelope,
)
from horizon_ric.shield import default_terrestrial_shield
from horizon_ric.shield.certificate import SafetyCertificate
from horizon_ric.shield.signing import (
    canonical_certificate_bytes,
    generate_signing_key,
    key_fingerprint,
    signed_certificate,
    verify_certificate,
)

POLICY_TYPES = tuple(DEFAULT_POLICY_TYPES)


# ---------------------------------------------------------------------------
# Draft-07 subset validator (no jsonschema dependency — see module docstring)
# ---------------------------------------------------------------------------
_TYPE_CHECKS: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    # bool is a subclass of int in Python; JSON says true is not a number.
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def _validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of Draft-07 violations of ``instance`` against ``schema``."""
    errors: list[str] = []

    declared = schema.get("type")
    if declared is not None:
        allowed = [declared] if isinstance(declared, str) else list(declared)
        if not any(_TYPE_CHECKS[name](instance) for name in allowed):
            errors.append(f"{path}: {type(instance).__name__} is not {allowed}")
            return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum {schema['enum']}")
    if "pattern" in schema and isinstance(instance, str):
        if re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: {instance!r} does not match {schema['pattern']!r}")
    if "minimum" in schema and isinstance(instance, (int, float)):
        if instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
    if "maximum" in schema and isinstance(instance, (int, float)):
        if instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance:
                errors.append(f"{path}: missing required property {name!r}")
        if schema.get("additionalProperties") is False:
            for name in instance:
                if name not in properties:
                    errors.append(f"{path}: additional property {name!r} is not allowed")
        for name, value in instance.items():
            if name in properties:
                errors.extend(_validate(value, properties[name], f"{path}.{name}"))

    if isinstance(instance, list) and "items" in schema:
        for index, value in enumerate(instance):
            errors.extend(_validate(value, schema["items"], f"{path}[{index}]"))

    return errors


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _shield():
    return default_terrestrial_shield(
        band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0
    )


def _action(**overrides: Any) -> dict[str, Any]:
    action = {
        "block": "policy_emit",
        "frequency_hz": 3.45e9,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 24.0,
        "antenna_gain_dBi": 5.0,
    }
    action.update(overrides)
    return action


def _certificate(**overrides: Any) -> SafetyCertificate:
    """A real certificate from a real ``default_terrestrial_shield`` disposition."""
    return _shield().dispose(_action(**overrides), {}, decision_id="d-envelope").certificate


def _qos_payload(assurance: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scope": {"slice_id": "slice-1"},
        "qos_objectives": {"priority": 5},
        "rapp_metadata": {"decision_id": "d-envelope"},
    }
    if assurance is not None:
        payload["assurance"] = assurance
    return payload


# ---------------------------------------------------------------------------
# (0) the validator itself is not vacuous
# ---------------------------------------------------------------------------
def test_subset_validator_rejects_what_the_create_schemas_forbid():
    schema = A1Adapter._policy_create_schema("horizon.qos.priority")
    assert _validate(_qos_payload(), schema) == []
    # additionalProperties: false, at the top level and inside a sub-object.
    assert _validate({**_qos_payload(), "bogus": 1}, schema)
    assert _validate(
        {**_qos_payload(), "scope": {"slice_id": "s", "bogus": 1}}, schema
    )
    # required, type, and the numeric bounds on priority.
    assert _validate({"scope": {"slice_id": "s"}}, schema)
    assert _validate(
        {"scope": {"slice_id": 5}, "qos_objectives": {"priority": 5}}, schema
    )
    assert _validate(
        {"scope": {"slice_id": "s"}, "qos_objectives": {"priority": 99}}, schema
    )
    # A boolean is not an integer priority, however cheerfully Python coerces.
    assert _validate(
        {"scope": {"slice_id": "s"}, "qos_objectives": {"priority": True}}, schema
    )


# ---------------------------------------------------------------------------
# (1) the digest is the signed bytes
# ---------------------------------------------------------------------------
def test_certificate_digest_is_sha256_of_the_canonical_signing_bytes():
    cert = _certificate()
    envelope = assurance_envelope(cert)
    expected = hashlib.sha256(canonical_certificate_bytes(cert)).hexdigest()
    assert envelope["certificate_digest"] == expected
    assert re.fullmatch(r"[0-9a-f]{64}", envelope["certificate_digest"])


def test_digest_is_unchanged_by_signing_so_the_signature_covers_it():
    """Signing must not move the digest, or the reference cannot be verified.

    ``canonical_certificate_bytes`` strips the signature fields, so the digest
    of the unsigned certificate and of the signed one are the same 32 bytes —
    which is exactly what makes ``sha256(canonical)`` a usable handle for
    ``verify_certificate``: a holder of the certificate recomputes the digest,
    confirms it is the object the wire named, and verifies the signature over
    the same bytes.
    """
    key = Ed25519PrivateKey.generate()
    cert = _certificate()
    signed = signed_certificate(cert, key)

    unsigned_envelope = assurance_envelope(cert)
    signed_envelope = assurance_envelope(signed)
    assert signed_envelope["certificate_digest"] == unsigned_envelope["certificate_digest"]

    # The wire carries the signature and the key that made it; both are real.
    assert signed_envelope["signature"] == signed.signature
    assert signed_envelope["signing_key_fingerprint"] == key_fingerprint(key.public_key())
    assert verify_certificate(signed, key.public_key()) is True

    # And the digest a receiver would recompute from the certificate it later
    # obtains equals the one it was handed on the wire.
    assert (
        hashlib.sha256(canonical_certificate_bytes(signed)).hexdigest()
        == signed_envelope["certificate_digest"]
    )


def test_unsigned_certificate_omits_the_signature_fields_rather_than_nulling_them():
    envelope = assurance_envelope(_certificate())
    assert "signature" not in envelope
    assert "signing_key_fingerprint" not in envelope
    assert envelope["certificate_digest"]


def test_envelope_keys_are_exactly_the_declared_schema_properties():
    declared = A1Adapter._policy_create_schema("horizon.qos.priority")
    properties = declared["properties"]["assurance"]["properties"]
    assert set(ASSURANCE_ENVELOPE_KEYS) == set(properties)

    key = Ed25519PrivateKey.generate()
    fullest = assurance_envelope(
        signed_certificate(_certificate(), key), profile_digest="ab" * 32
    )
    # Every key the builder can emit is declared, and the tuple is the order.
    assert list(fullest) == [k for k in ASSURANCE_ENVELOPE_KEYS if k in fullest]
    assert set(fullest) == set(ASSURANCE_ENVELOPE_KEYS)


# ---------------------------------------------------------------------------
# (2) absent certificate → byte-identical payload
# ---------------------------------------------------------------------------
def test_absent_certificate_yields_an_empty_envelope():
    assert assurance_envelope(None) == {}
    assert assurance_envelope(None, profile_digest="ab" * 32) == {}


def test_absent_certificate_leaves_the_policy_body_byte_identical():
    """Backward compatibility, asserted on bytes rather than on key sets.

    This is the whole basis on which the change is additive: an uncertified
    caller (``tests/test_a1_osc_a1_dialect.py``, ``tests/test_multi_vendor_
    integration.py``, ``scripts/osc_a1_live_smoke.py``) must put the same bytes
    on the wire as before the envelope existed.
    """
    payload = _qos_payload()
    before = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    envelope = assurance_envelope(None)
    wire = {**payload, "assurance": envelope} if envelope else payload

    assert json.dumps(wire, sort_keys=True, separators=(",", ":")) == before
    assert wire is payload  # not even a copy was made
    assert "assurance" not in wire


# ---------------------------------------------------------------------------
# (3) the schema declares it, everywhere, without requiring it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_every_policy_type_declares_the_assurance_envelope(policy_type: str):
    schema = A1Adapter._policy_create_schema(policy_type)
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"

    assurance = schema["properties"]["assurance"]
    assert assurance["type"] == "object"
    assert assurance["additionalProperties"] is False
    # The verdict is mandatory *within* the envelope...
    assert set(assurance["required"]) == {
        "certificate_digest",
        "safe",
        "projected",
        "violated_ids",
    }
    # ...but the envelope itself is not, alongside the unchanged rapp_metadata.
    assert "assurance" not in schema["required"]
    assert "rapp_metadata" not in schema["required"]
    assert set(schema["properties"]["rapp_metadata"]["properties"]) == {
        "decision_id",
        "rapp_version",
        "model_versions",
    }


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_schema_version_was_bumped_for_the_additive_field(policy_type: str):
    """A-4 of ASSURANCE_PROFILE.md §6.2: the change must be versioned.

    ``additionalProperties: false`` means a receiver holding the 1.0.0 schema
    rejects a body carrying the envelope, so the schema version cannot stay
    at 1.0.0. Minor, because a 1.0.0 body is still a valid 1.1.0 body.
    """
    assert DEFAULT_POLICY_TYPES[policy_type]["schema_v"] == "1.1.0"


def test_qos_payload_with_the_envelope_validates_against_the_create_schema():
    key = Ed25519PrivateKey.generate()
    cert = signed_certificate(_certificate(), key)
    payload = _qos_payload(assurance_envelope(cert, profile_digest="cd" * 32))
    schema = A1Adapter._policy_create_schema("horizon.qos.priority")
    assert _validate(payload, schema) == []


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_the_envelope_validates_inside_every_policy_type(policy_type: str):
    """The envelope is cross-cutting, so check it against all four schemas.

    Each type gets a minimal, schema-valid scope/objective block plus the same
    real envelope; only the envelope is under test.
    """
    bodies: dict[str, dict[str, Any]] = {
        "horizon.qos.priority": {
            "scope": {"slice_id": "slice-1"},
            "qos_objectives": {"priority": 5},
        },
        "horizon.traffic.steering": {
            "scope": {"ue_group": "default"},
            "steering_objectives": {"preferred_path": "hybrid", "ntn_share_pct": 40.0},
        },
        "horizon.admission.control": {
            "scope": {"workload_class": "ai_inference"},
            "admission": {"decision": "defer", "defer_until_seconds": 120.0},
        },
        "horizon.spectrum.reservation": {
            "scope": {"band_id": "n78"},
            "reservation": {
                "freq_low_hz": 3.40e9,
                "freq_high_hz": 3.42e9,
                "traffic_class": "critical",
            },
        },
    }
    key = Ed25519PrivateKey.generate()
    payload = {
        **bodies[policy_type],
        "rapp_metadata": {"decision_id": "d-envelope"},
        "assurance": assurance_envelope(
            signed_certificate(_certificate(), key), profile_digest="ef" * 32
        ),
    }
    schema = A1Adapter._policy_create_schema(policy_type)
    assert _validate(payload, schema) == []


def test_a_1_0_0_receiver_would_reject_the_envelope():
    """Why the version bump is load-bearing, not bookkeeping.

    Reconstructs the pre-1.1.0 schema by removing the declaration, and shows
    the envelope-bearing body is then a validation failure — the same 400 the
    real ``o-ran-sc/sim-a1-interface`` simulator returns
    (``a1_controller_create_or_replace_policy_instance`` validates every PUT
    body against the registered ``create_schema``).
    """
    schema = A1Adapter._policy_create_schema("horizon.qos.priority")
    legacy = {**schema, "properties": dict(schema["properties"])}
    del legacy["properties"]["assurance"]

    payload = _qos_payload(assurance_envelope(_certificate()))
    errors = _validate(payload, legacy)
    assert any("additional property 'assurance'" in e for e in errors), errors
    # ...and the same body is fine under 1.1.0.
    assert _validate(payload, schema) == []


def test_envelope_missing_the_verdict_fails_the_schema():
    schema = A1Adapter._policy_create_schema("horizon.qos.priority")
    envelope = assurance_envelope(_certificate())
    del envelope["safe"]
    assert _validate(_qos_payload(envelope), schema)


def test_undeclared_key_inside_the_envelope_fails_the_schema():
    schema = A1Adapter._policy_create_schema("horizon.qos.priority")
    envelope = assurance_envelope(_certificate())
    envelope["invariants"] = [{"invariant_id": "max_eirp_etsi", "satisfied": True}]
    errors = _validate(_qos_payload(envelope), schema)
    assert any("additional property 'invariants'" in e for e in errors), errors


def test_a_truncated_digest_fails_the_schema_pattern():
    schema = A1Adapter._policy_create_schema("horizon.qos.priority")
    envelope = assurance_envelope(_certificate())
    envelope["certificate_digest"] = envelope["certificate_digest"][:32]
    assert _validate(_qos_payload(envelope), schema)


# ---------------------------------------------------------------------------
# (4) a refusal is legible as a refusal
# ---------------------------------------------------------------------------
def test_blocked_certificate_reports_safe_false_with_violated_invariants():
    """A Shield refusal must be readable off the wire, not inferred from silence.

    A negative bandwidth is unfixable by projection, so the Shield fails closed:
    ``emit_blocked`` with a non-empty ``violated_ids``. The envelope carries
    both the verdict and the invariant ids, so a near-RT RIC receiving such a
    policy (Horizon's own pipeline refuses to emit it at all — see
    ``HORIZON_A1_REQUIRE_CERT`` and ``policy/emit_guards.run_guard_chain``) can
    still see that it was refused rather than endorsed.
    """
    cert = _certificate(bandwidth_hz=-1.0)
    assert cert.emit_blocked is True

    envelope = assurance_envelope(cert)
    assert envelope["safe"] is False
    assert envelope["violated_ids"]
    assert "numeric_domain_sanity" in envelope["violated_ids"]

    schema = A1Adapter._policy_create_schema("horizon.qos.priority")
    assert _validate(_qos_payload(envelope), schema) == []


def test_a_safe_certificate_is_distinguishable_from_a_blocked_one_on_the_wire():
    safe = assurance_envelope(_certificate())
    blocked = assurance_envelope(_certificate(bandwidth_hz=-1.0))
    assert (safe["safe"], safe["violated_ids"]) == (True, [])
    assert blocked["safe"] is False and blocked["violated_ids"] != []
    assert safe["certificate_digest"] != blocked["certificate_digest"]


# ---------------------------------------------------------------------------
# (5) projection and margin are reported honestly
# ---------------------------------------------------------------------------
def test_projected_and_min_margin_track_the_certificate():
    projected = _certificate(tx_power_dBm=90.0)
    assert projected.projected is True
    envelope = assurance_envelope(projected)
    assert envelope["projected"] is True
    assert envelope["safe"] is True  # projection reached the safe set
    assert envelope["min_margin_dB"] == projected.min_margin_dB

    untouched = assurance_envelope(_certificate())
    assert untouched["projected"] is False


def test_min_margin_may_be_null_and_the_schema_allows_it():
    """``min_margin_dB`` is ``None`` when no invariant reports a dB margin."""
    cert = SafetyCertificate(
        decision_id="d-no-db",
        issued_at="2026-07-30T00:00:00+00:00",
        loop_tier="non_rt",
        block="policy_emit",
        action_proposed={},
        action_safe={},
        invariants=[],
        corrections=[],
        violated_ids=[],
        projected=False,
        fallback_used=False,
        fallback_to=None,
        safe=True,
        emit_blocked=False,
    )
    envelope = assurance_envelope(cert)
    assert envelope["min_margin_dB"] is None

    schema = A1Adapter._policy_create_schema("horizon.qos.priority")
    assert _validate(_qos_payload(envelope), schema) == []


def test_profile_digest_is_carried_only_when_supplied():
    cert = _certificate()
    assert "profile_digest" not in assurance_envelope(cert)
    assert (
        assurance_envelope(cert, profile_digest="ab" * 32)["profile_digest"] == "ab" * 32
    )


async def test_pipeline_emit_does_not_invalidate_the_signature_it_shipped():
    """Regression test for the sharpest hazard in this change.

    ``Shield.dispose`` shallow-copies the action, so
    ``certificate.action_proposed["policy_payload"]`` is the *same dict object*
    as the payload the pipeline is about to emit. Merging the envelope into that
    dict in place would retroactively change the bytes
    ``canonical_certificate_bytes`` produces, and the signature — computed
    before the merge — would stop verifying. The pipeline therefore merges into
    a copy. This test verifies the signature *after* the emit has happened, so
    an in-place merge fails it.

    No HTTP here: the property under test is in-process aliasing, and the A1
    collaborator only has to record what it was handed. The real-socket
    coverage is ``deploy/xapp-e2e/a1_assurance_proof.py``, which does the same
    verification against the returned body of the official O-RAN-SC simulator.
    """
    import tempfile
    from datetime import datetime, timezone

    from horizon_ric.io.schemas import TelemetryEvent
    from horizon_ric.rapp.pipeline import DecisionPipeline, PipelineConfig

    class _CapturingA1:
        def __init__(self) -> None:
            self.bodies: list[dict[str, Any]] = []
            self.certificates: list[SafetyCertificate] = []
            self.records: list[Any] = []

        async def emit_policy(
            self,
            policy_type: str,
            policy_payload: dict[str, Any],
            policy_id: str | None = None,
            decision_record: Any = None,
            safety_certificate: SafetyCertificate | None = None,
        ) -> tuple[str, int]:
            # Snapshot through JSON: the wire only ever sees serialised bytes.
            self.bodies.append(json.loads(json.dumps(policy_payload)))
            assert safety_certificate is not None
            self.certificates.append(safety_certificate)
            self.records.append(decision_record)
            return "p-signature-survival", 202

        async def get_policy_status(
            self, policy_type: str, policy_id: str
        ) -> dict[str, str]:
            return {"enforceStatus": "ENFORCED"}

    class _Store:
        def append(self, record: Any) -> None:  # pragma: no cover — happy path
            pass

    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path as _Path

        key = generate_signing_key(_Path(tmp) / "signing.pem")
        previous = os.environ.get("HORIZON_CERT_SIGNING_KEY_PATH")
        os.environ["HORIZON_CERT_SIGNING_KEY_PATH"] = str(_Path(tmp) / "signing.pem")
        try:
            a1 = _CapturingA1()
            pipeline = DecisionPipeline(
                a1, _Store(), PipelineConfig(status_poll_interval_s=0.01)
            )
            event = TelemetryEvent(
                event_id="evt-assurance-0001",
                modality="kpm_5g",
                source_id="test-src",
                ts_utc=datetime.now(timezone.utc),
                sequence=1,
                payload={"sla_risk_30s": 0.05},
            )
            result = await pipeline.process_event(event)
        finally:
            if previous is None:
                os.environ.pop("HORIZON_CERT_SIGNING_KEY_PATH", None)
            else:
                os.environ["HORIZON_CERT_SIGNING_KEY_PATH"] = previous

    assert result.accepted is True
    body = a1.bodies[0]
    certificate = a1.certificates[0]
    envelope = body["assurance"]

    # The signature is on the wire and the certificate still verifies — checked
    # after the emit, which is the whole point.
    assert envelope["signature"] == certificate.signature
    assert verify_certificate(certificate, key.public_key()) is True
    assert (
        envelope["certificate_digest"]
        == hashlib.sha256(canonical_certificate_bytes(certificate)).hexdigest()
    )
    key.public_key().verify(
        bytes.fromhex(envelope["signature"]), canonical_certificate_bytes(certificate)
    )
    assert envelope["signing_key_fingerprint"] == key_fingerprint(key.public_key())
    assert envelope["profile_digest"] == pipeline._profile_digest
    assert body["rapp_metadata"] == {"decision_id": result.decision_id}

    # The certificate's own view of the proposed payload is the pre-envelope
    # one, and so is the DecisionRecord's: both must keep matching what the
    # Shield actually graded.
    assert "assurance" not in certificate.action_proposed["policy_payload"]
    record = a1.records[0]
    assert "assurance" not in record.chosen_action["policy_payload"]
    # The evidence side-channel keeps its three-field subset plus signature.
    assert set(record.chosen_action["certificate"]) == {
        "safe",
        "projected",
        "violated_ids",
        "signature",
        "signing_key_fingerprint",
    }


def test_pipeline_stamps_the_live_shield_profile_digest():
    """The digest on the wire is reflected off the pipeline's own Shield.

    Not a constant and not a config string: two pipelines whose Shields enforce
    different EIRP ceilings must stamp different profile digests, or a receiver
    cannot tell which invariant set graded the action.
    """
    from horizon_ric.assurance.profile import emit_profile, profile_digest
    from horizon_ric.rapp.pipeline import DecisionPipeline, PipelineConfig

    # Neither collaborator is touched by __init__; the digest is computed from
    # the Shield the pipeline builds for itself.
    loose = DecisionPipeline(None, None, PipelineConfig(max_eirp_dbm=33.0))
    tight = DecisionPipeline(None, None, PipelineConfig(max_eirp_dbm=20.0))

    assert re.fullmatch(r"[0-9a-f]{64}", loose._profile_digest)
    assert loose._profile_digest != tight._profile_digest
    assert loose._profile_digest == profile_digest(
        emit_profile(
            loose._shield,
            profile_id="horizon-ric-rapp/terrestrial",
            description="band 3400000000-3500000000 Hz, max EIRP 33.0 dBm",
        )
    )
