"""Anti-drift gate for the WP1 assurance interface and its published schemas.

PROPERTY PINNED: the JSON Schema documents in ``docs/schemas/`` are the published
contract for three objects that live in code — the planner's proposed action, the
Shield's ``SafetyCertificate``, and the assurance profile reflected off a live
Shield. A hand-written schema document rots the moment someone adds a field, and
a rotted *published* schema is worse than none: an integrator validates against
it, passes, and is still rejected at runtime.

So none of these tests trust the documents. They derive the truth from the code
and fail if the documents disagree:

* Certificate / ``InvariantCheck`` / ``ConstraintViolation`` property sets are
  compared against ``dataclasses.fields(...)``, plus the one derived
  ``min_margin_dB`` property that ``to_dict()`` also emits. Add a certificate
  field without touching the schema and this file fails.
* The proposed-action property set is compared against
  ``REQUIRED_ACTION_KEYS`` + ``OPTIONAL_ACTION_KEYS``.
* The known-invariant-id enum is compared against ``.id`` read off
  default-constructed instances of all eight concrete invariant classes.
* A *real* certificate from a *real* ``default_terrestrial_shield`` disposition
  is validated against the certificate schema, so the schema is checked against
  observed output and not only against declarations.

The repo has no ``jsonschema`` dependency and WP1 does not add one, so the
validator below is hand-written and covers exactly the keywords these three
documents use: ``$ref`` (local), ``type``, ``enum``, ``const``, ``pattern``,
``required``, ``properties``, ``additionalProperties``, ``items``, ``minItems``,
``minLength``, and the four numeric bounds. Unknown keywords are ignored by
design — ``allOf`` / ``not`` / ``if`` / ``then`` in the action schema are
asserted structurally instead, since a partial validator that silently skipped
them would be the most dangerous kind of green test.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from horizon_ric.assurance.planner import (
    OPTIONAL_ACTION_KEYS,
    OUTPUT_SENTINEL_KEYS,
    REQUIRED_ACTION_KEYS,
    Planner,
    PlannerContractError,
    shielded,
    validate_proposed_action,
)
from horizon_ric.assurance.profile import (
    PROFILE_SCHEMA_VERSION,
    emit_profile,
    invariant_descriptor,
    profile_digest,
)
from horizon_ric.shield.certificate import (
    ConstraintViolation,
    InvariantCheck,
    SafetyCertificate,
    ShieldDisposition,
)
from horizon_ric.shield.invariants import (
    ConstellationLegalityInvariant,
    LawfulInterceptInvariant,
    MaxEirpInvariant,
    NeuralRxEnvelopeInvariant,
    NumericSanityInvariant,
    PfdCeilingInvariant,
    ProtectedSliceFloorInvariant,
    SpectralMaskInvariant,
)
from horizon_ric.shield.shield import Shield, default_terrestrial_shield

# n78-style 100 MHz block; the numbers only have to be a real licensed channel
# the clean action fits inside with margin.
BAND_LO_HZ = 3.40e9
BAND_HI_HZ = 3.50e9
MAX_EIRP_DBM = 33.0
CENTRE_HZ = 3.45e9
BANDWIDTH_HZ = 20.0e6
TX_POWER_DBM = 24.0
ANTENNA_GAIN_DBI = 6.0
# 24 + 6 = 30 dBm EIRP against a 33 dBm ceiling.
CLEAN_EIRP_MARGIN_DB = 3.0
# An EIRP this high is 13 dB over the ceiling, so the projection must cut it.
HOT_TX_POWER_DBM = 40.0
HOT_EIRP_OVERAGE_DB = 13.0

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "docs" / "schemas"
ACTION_SCHEMA_PATH = SCHEMA_DIR / "proposed-action-v1.schema.json"
CERTIFICATE_SCHEMA_PATH = SCHEMA_DIR / "safety-certificate-v1.schema.json"
PROFILE_SCHEMA_PATH = SCHEMA_DIR / "invariant-profile-v1.schema.json"

DERIVED_CERTIFICATE_KEYS = ("min_margin_dB",)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


def _shield() -> Shield:
    return default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ,
        band_hi_hz=BAND_HI_HZ,
        max_eirp_dBm=MAX_EIRP_DBM,
    )


def _clean_action() -> dict[str, Any]:
    """An action inside the safe set of :func:`_shield` — no projection needed."""
    return {
        "block": "policy_emit",
        "frequency_hz": CENTRE_HZ,
        "bandwidth_hz": BANDWIDTH_HZ,
        "tx_power_dBm": TX_POWER_DBM,
        "antenna_gain_dBi": ANTENNA_GAIN_DBI,
    }


class _LiStub:
    """Stand-in for an ``LIConstraint``. Never called — the profile emitter must
    report the field as delegated without touching the value, and ``.id`` is a
    plain dataclass default that does not read ``li`` either."""


def _all_invariants() -> list[Any]:
    """One default-constructed instance of every concrete invariant class."""
    return [
        NumericSanityInvariant(),
        SpectralMaskInvariant(band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ),
        MaxEirpInvariant(),
        ProtectedSliceFloorInvariant(),
        PfdCeilingInvariant(),
        NeuralRxEnvelopeInvariant(),
        ConstellationLegalityInvariant(),
        LawfulInterceptInvariant(li=_LiStub()),
    ]


def _planner(action: dict[str, Any], planner_id: str = "test-planner") -> Any:
    """A minimal structural :class:`Planner` returning a fixed action."""

    class _Fixed:
        def __init__(self) -> None:
            self.planner_id = planner_id

        def propose(self, observation: Any) -> dict[str, Any]:
            return dict(action)

    return _Fixed()


# ---------------------------------------------------------------------------
# a very small JSON Schema validator (see module docstring for scope)
# ---------------------------------------------------------------------------
def _type_ok(instance: Any, name: str) -> bool:
    if name == "null":
        return instance is None
    if name == "boolean":
        return isinstance(instance, bool)
    if name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if name == "string":
        return isinstance(instance, str)
    if name == "array":
        return isinstance(instance, list)
    if name == "object":
        return isinstance(instance, dict)
    raise AssertionError(f"validator does not implement JSON type {name!r}")


def _resolve(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    assert ref.startswith("#/"), f"only local $refs are supported, got {ref!r}"
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    assert isinstance(node, dict), f"$ref {ref!r} does not point at a schema object"
    return node


def _errors(instance: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> list[str]:
    """Every way ``instance`` violates ``schema``. Empty list means valid."""
    if "$ref" in schema:
        return _errors(instance, _resolve(schema["$ref"], root), root, path)

    problems: list[str] = []
    declared = schema.get("type")
    if declared is not None:
        names = [declared] if isinstance(declared, str) else list(declared)
        if not any(_type_ok(instance, name) for name in names):
            # A type mismatch makes every other keyword meaningless.
            return [f"{path}: expected type {names}, got {type(instance).__name__}"]

    if "enum" in schema and instance not in schema["enum"]:
        problems.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if "const" in schema and instance != schema["const"]:
        problems.append(f"{path}: {instance!r} != const {schema['const']!r}")

    if isinstance(instance, str):
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            problems.append(f"{path}: {instance!r} does not match {schema['pattern']!r}")
        if "minLength" in schema and len(instance) < schema["minLength"]:
            problems.append(f"{path}: shorter than minLength {schema['minLength']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            problems.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            problems.append(f"{path}: {instance} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            problems.append(
                f"{path}: {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}"
            )
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            problems.append(
                f"{path}: {instance} >= exclusiveMaximum {schema['exclusiveMaximum']}"
            )

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            problems.append(f"{path}: fewer than minItems {schema['minItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                problems.extend(_errors(item, item_schema, root, f"{path}[{index}]"))

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                problems.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                problems.extend(_errors(value, properties[key], root, f"{path}.{key}"))
                continue
            extra = schema.get("additionalProperties", True)
            if extra is False:
                problems.append(f"{path}: additional property {key!r} is not allowed")
            elif isinstance(extra, dict):
                problems.extend(_errors(value, extra, root, f"{path}.{key}"))

    return problems


def _assert_valid(instance: Any, schema: dict[str, Any]) -> None:
    problems = _errors(instance, schema, schema, "$")
    assert problems == [], "instance failed schema validation:\n  " + "\n  ".join(problems)


# ---------------------------------------------------------------------------
# the validator itself must be able to fail
# ---------------------------------------------------------------------------
def test_the_hand_written_validator_actually_rejects() -> None:
    """A validator that never says no is a green light, not a gate."""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["a", "b"],
        "properties": {
            "a": {"type": ["number", "null"], "minimum": 0},
            "b": {"type": "string", "enum": ["hard", "soft"]},
        },
    }
    _assert_valid({"a": 1.0, "b": "hard"}, schema)
    _assert_valid({"a": None, "b": "soft"}, schema)
    assert _errors({"a": 1.0}, schema, schema, "$") != []            # missing required
    assert _errors({"a": -1.0, "b": "hard"}, schema, schema, "$") != []  # below minimum
    assert _errors({"a": 1.0, "b": "medium"}, schema, schema, "$") != []  # off-enum
    assert _errors({"a": "1.0", "b": "hard"}, schema, schema, "$") != []  # wrong type
    assert _errors({"a": 1.0, "b": "hard", "c": 1}, schema, schema, "$") != []  # extra key


# ---------------------------------------------------------------------------
# the documents themselves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [ACTION_SCHEMA_PATH, CERTIFICATE_SCHEMA_PATH, PROFILE_SCHEMA_PATH],
    ids=lambda p: p.name,
)
def test_every_schema_is_a_self_describing_2020_12_document(path: Path) -> None:
    schema = _schema(path)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(path.name), "the $id must name the document it identifies"
    assert schema["title"]
    assert len(schema["description"]) > 200, "a published schema must explain itself"


# ---------------------------------------------------------------------------
# certificate schema <-> dataclass
# ---------------------------------------------------------------------------
def test_certificate_schema_properties_match_the_dataclass_exactly() -> None:
    """18 declared fields + the derived ``min_margin_dB`` property = 19 keys.

    ``to_dict()`` emits the derived property alongside the fields, so the schema
    has to carry it too. Introspecting the dataclass means a new certificate
    field cannot be shipped without updating the published schema.
    """
    schema = _schema(CERTIFICATE_SCHEMA_PATH)
    declared = tuple(f.name for f in fields(SafetyCertificate))
    assert len(declared) == 18

    expected = set(declared) | set(DERIVED_CERTIFICATE_KEYS)
    assert set(schema["properties"]) == expected
    assert len(schema["properties"]) == 19
    # Every key is always emitted, so optionality is a nullable type, never an
    # absent key.
    assert set(schema["required"]) == expected


def test_certificate_schema_property_order_and_nullability_are_honest() -> None:
    schema = _schema(CERTIFICATE_SCHEMA_PATH)
    nullable = {
        name
        for name, sub in schema["properties"].items()
        if isinstance(sub.get("type"), list) and "null" in sub["type"]
    }
    assert nullable == {
        "fallback_to",
        "min_margin_dB",
        "rng_seed",
        "model_provenance",
        "signature",
        "signing_key_fingerprint",
    }
    assert "DERIVED" in schema["properties"]["min_margin_dB"]["description"]


def test_invariant_check_schema_matches_the_dataclass_exactly() -> None:
    schema = _schema(CERTIFICATE_SCHEMA_PATH)
    sub = schema["$defs"]["invariantCheck"]
    declared = {f.name for f in fields(InvariantCheck)}
    assert declared == {"invariant_id", "satisfied", "margin", "unit", "detail"}
    assert set(sub["properties"]) == declared
    assert set(sub["required"]) == declared
    assert sub["additionalProperties"] is False


def test_constraint_violation_schema_matches_the_dataclass_exactly() -> None:
    schema = _schema(CERTIFICATE_SCHEMA_PATH)
    sub = schema["$defs"]["constraintViolation"]
    declared = {f.name for f in fields(ConstraintViolation)}
    assert declared == {"constraint_id", "severity", "margin_dB", "message"}
    assert set(sub["properties"]) == declared
    assert set(sub["required"]) == declared
    assert sub["additionalProperties"] is False


def test_unenforced_conventions_are_declared_as_unenforced() -> None:
    """``severity`` and ``loop_tier`` are pinned by the schema but NOT by the
    code: ``LOOP_TIERS`` is a module constant nothing checks, and ``severity``
    is a bare ``str``. The schema may be stricter than the code, but it must say
    so rather than implying a guarantee that does not exist."""
    schema = _schema(CERTIFICATE_SCHEMA_PATH)
    loop_tier = schema["properties"]["loop_tier"]
    severity = schema["$defs"]["constraintViolation"]["properties"]["severity"]
    assert loop_tier["enum"] == ["rt", "near_rt", "non_rt"]
    assert severity["enum"] == ["hard", "soft"]
    assert "UNVALIDATED" in loop_tier["description"]
    assert "UNVALIDATED" in severity["description"]
    assert "UNVALIDATED IN THE CURRENT CODE" in schema["description"]


# ---------------------------------------------------------------------------
# a real disposition must validate against the published certificate schema
# ---------------------------------------------------------------------------
def test_a_real_clean_disposition_validates_against_the_certificate_schema() -> None:
    disposition = _shield().dispose(_clean_action(), {}, decision_id="dec-clean")
    cert = disposition.certificate

    assert cert.projected is False
    assert cert.safe is True
    assert cert.emit_blocked is False
    assert cert.violated_ids == []
    assert cert.min_margin_dB == pytest.approx(CLEAN_EIRP_MARGIN_DB)

    payload = cert.to_dict()
    assert len(payload) == 19
    _assert_valid(payload, _schema(CERTIFICATE_SCHEMA_PATH))


def test_a_real_projected_disposition_validates_including_its_corrections() -> None:
    """The clean path never populates ``corrections``, so the
    ``ConstraintViolation`` subschema would go unexercised. Force a projection."""
    hot = {**_clean_action(), "tx_power_dBm": HOT_TX_POWER_DBM}
    cert = _shield().dispose(hot, {}, decision_id="dec-hot").certificate

    assert cert.projected is True
    assert cert.safe is True
    assert cert.emit_blocked is False
    assert [c.constraint_id for c in cert.corrections] == ["max_eirp"]
    assert cert.corrections[0].severity == "hard"
    assert cert.corrections[0].margin_dB == pytest.approx(-HOT_EIRP_OVERAGE_DB)
    assert cert.action_safe["tx_power_dBm"] == pytest.approx(
        HOT_TX_POWER_DBM - HOT_EIRP_OVERAGE_DB
    )

    _assert_valid(cert.to_dict(), _schema(CERTIFICATE_SCHEMA_PATH))


def test_a_blocked_disposition_still_validates() -> None:
    """Failing closed is a normal outcome, not an error path — its certificate
    must be as well-formed as a clean one, because that is the certificate an
    auditor will actually read."""
    unplaceable = {**_clean_action(), "bandwidth_hz": 400.0e6}  # wider than the block
    cert = _shield().dispose(unplaceable, {}, decision_id="dec-blocked").certificate

    assert cert.emit_blocked is True
    assert cert.safe is False
    assert "spectral_mask_ts38104" in cert.violated_ids
    _assert_valid(cert.to_dict(), _schema(CERTIFICATE_SCHEMA_PATH))


# ---------------------------------------------------------------------------
# invariant id registry <-> the implementation
# ---------------------------------------------------------------------------
def test_known_invariant_id_enum_matches_the_class_defaults() -> None:
    """Read every id off a default-constructed instance. Adding an invariant
    without listing it in the published registry fails here."""
    enum = _schema(PROFILE_SCHEMA_PATH)["$defs"]["knownInvariantId"]["enum"]
    live = {inv.id for inv in _all_invariants()}

    assert live == {
        "numeric_domain_sanity",
        "spectral_mask_ts38104",
        "max_eirp",
        "protected_slice_floor",
        "pfd_ceiling_ntn",
        "neural_rx_envelope",
        "constellation_legality",
        "lawful_intercept",
    }
    assert set(enum) == live
    assert enum == sorted(enum), "keep the registry sorted so diffs stay readable"
    assert len(enum) == len(set(enum))


def test_the_id_field_is_last_on_every_invariant_dataclass() -> None:
    """``invariant_descriptor`` reports ``id`` separately and drops it from
    ``limits``; that only reads cleanly because ``id`` is the trailing field with
    a default on every concrete invariant."""
    for inv in _all_invariants():
        names = [f.name for f in fields(inv)]
        assert names[-1] == "id", f"{type(inv).__name__} field order changed: {names}"


# ---------------------------------------------------------------------------
# proposed-action schema <-> planner.py
# ---------------------------------------------------------------------------
def test_action_schema_properties_match_the_declared_key_tuples() -> None:
    schema = _schema(ACTION_SCHEMA_PATH)
    assert set(schema["required"]) == set(REQUIRED_ACTION_KEYS)
    assert set(schema["properties"]) == set(REQUIRED_ACTION_KEYS) | set(OPTIONAL_ACTION_KEYS)


def test_action_schema_forbids_the_shield_output_sentinels() -> None:
    """The action shape is open (``additionalProperties: true``), so these
    ``not``/``required`` clauses are the only thing keeping a planner from
    forging the Shield's own verdict."""
    schema = _schema(ACTION_SCHEMA_PATH)
    assert schema["additionalProperties"] is True
    forbidden = {
        tuple(clause["not"]["required"])[0] for clause in schema["allOf"] if "not" in clause
    }
    assert forbidden == set(OUTPUT_SENTINEL_KEYS)
    assert OUTPUT_SENTINEL_KEYS == ("emit_blocked", "shield_fallback_to")
    # None of the sentinels may be described as a legitimate property either.
    assert not set(schema["properties"]) & set(OUTPUT_SENTINEL_KEYS)


def test_action_schema_requires_a_slant_range_for_ntn() -> None:
    schema = _schema(ACTION_SCHEMA_PATH)
    conditionals = [clause for clause in schema["allOf"] if "if" in clause]
    assert len(conditionals) == 1
    assert conditionals[0]["if"]["properties"]["ntn"]["const"] is True
    assert conditionals[0]["then"]["required"] == ["slant_range_m"]


def test_a_conformant_action_validates_against_the_action_schema() -> None:
    _assert_valid(_clean_action(), _schema(ACTION_SCHEMA_PATH))


# ---------------------------------------------------------------------------
# validate_proposed_action
# ---------------------------------------------------------------------------
def test_a_conformant_action_reports_no_problems() -> None:
    assert validate_proposed_action(_clean_action()) == []


def test_a_missing_required_key_is_reported() -> None:
    action = _clean_action()
    del action["bandwidth_hz"]
    problems = validate_proposed_action(action)
    assert any("missing required key 'bandwidth_hz'" in p for p in problems)


@pytest.mark.parametrize(
    "value",
    [
        Decimal("1"),
        datetime(2026, 7, 30, tzinfo=timezone.utc),
    ],
    ids=["decimal", "datetime"],
)
def test_non_json_primitive_values_are_rejected(value: Any) -> None:
    """The signing constraint. ``canonical_certificate_bytes`` has no
    ``default=`` handler and ``action_proposed`` embeds this dict verbatim, so
    either of these would raise ``TypeError`` on the evidence path instead of at
    the planner boundary."""
    action = {**_clean_action(), "vendor_hint": value}
    problems = validate_proposed_action(action)
    assert any("not JSON-primitive" in p for p in problems)
    assert any("action.vendor_hint" in p for p in problems)

    # And confirm the premise rather than asserting it: the same value really
    # does break the canonical serialisation the signer performs.
    with pytest.raises(TypeError):
        json.dumps(action, sort_keys=True, separators=(",", ":"))


def test_non_primitives_are_found_recursively() -> None:
    action = {**_clean_action(), "beams": [{"gain": Decimal("3")}]}
    problems = validate_proposed_action(action)
    assert any("action.beams[0].gain" in p for p in problems)


def test_presetting_an_output_sentinel_is_rejected() -> None:
    """A planner that stamps ``emit_blocked`` is claiming the Shield's verdict
    for itself, which must never be accepted regardless of the value."""
    for sentinel in OUTPUT_SENTINEL_KEYS:
        problems = validate_proposed_action({**_clean_action(), sentinel: True})
        assert any(sentinel in p and "OUTPUT sentinel" in p for p in problems)


def test_out_of_domain_optional_values_are_reported() -> None:
    problems = validate_proposed_action(
        {
            **_clean_action(),
            "predicted_tbler": 1.5,
            "papr_dB": -1.0,
            "constellation_order": 6.5,
        }
    )
    assert any("predicted_tbler" in p for p in problems)
    assert any("papr_dB" in p for p in problems)
    assert any("constellation_order" in p for p in problems)


def test_ntn_without_a_slant_range_is_reported() -> None:
    problems = validate_proposed_action({**_clean_action(), "ntn": True})
    assert any("slant_range_m" in p for p in problems)
    assert validate_proposed_action({**_clean_action(), "ntn": True, "slant_range_m": 5.5e5}) == []


def test_a_non_mapping_proposal_is_reported_rather_than_crashing() -> None:
    assert validate_proposed_action(None) != []
    assert validate_proposed_action([1, 2, 3]) != []


def test_nan_is_rejected_because_json_cannot_represent_it() -> None:
    problems = validate_proposed_action({**_clean_action(), "frequency_hz": float("nan")})
    assert any("strict JSON" in p for p in problems)


# ---------------------------------------------------------------------------
# the shielded() adapter
# ---------------------------------------------------------------------------
def test_a_fixed_planner_satisfies_the_structural_protocol() -> None:
    assert isinstance(_planner(_clean_action()), Planner) is True


def test_shielded_returns_a_disposition_for_a_conformant_planner() -> None:
    disposition = shielded(
        _planner(_clean_action()), _shield(), {"cell_id": "c-1"}, decision_id="dec-1"
    )
    assert isinstance(disposition, ShieldDisposition) is True
    assert disposition.certificate.decision_id == "dec-1"
    assert disposition.certificate.safe is True
    assert disposition.safe_action["tx_power_dBm"] == pytest.approx(TX_POWER_DBM)


def test_shielded_fails_at_the_boundary_not_at_signing_time() -> None:
    bad = {**_clean_action(), "issued": datetime(2026, 7, 30, tzinfo=timezone.utc)}
    with pytest.raises(PlannerContractError) as excinfo:
        shielded(_planner(bad, "rogue-planner"), _shield(), {}, decision_id="dec-2")
    assert excinfo.value.planner_id == "rogue-planner"
    assert excinfo.value.problems != []
    assert "rogue-planner" in str(excinfo.value)


def test_shielded_still_disposes_of_an_unsafe_but_wellformed_action() -> None:
    """The contract check must not become a second safety gate. An action that is
    merely *unsafe* is the Shield's business, and the Shield fails it closed with
    a certificate rather than an exception."""
    hot = {**_clean_action(), "tx_power_dBm": HOT_TX_POWER_DBM}
    disposition = shielded(_planner(hot), _shield(), {}, decision_id="dec-3")
    assert disposition.certificate.projected is True
    assert disposition.certificate.emit_blocked is False


# ---------------------------------------------------------------------------
# emit_profile / profile_digest
# ---------------------------------------------------------------------------
def test_emit_profile_round_trips_through_json() -> None:
    profile = emit_profile(_shield(), profile_id="cell-42/band-n78", description="lab bring-up")
    reloaded = json.loads(json.dumps(profile, sort_keys=True, separators=(",", ":")))
    assert reloaded == profile
    assert profile["schema_version"] == PROFILE_SCHEMA_VERSION
    assert profile["max_passes"] == 8
    assert [d["id"] for d in profile["invariant_chain"]] == _shield().invariant_ids


def test_emit_profile_validates_against_the_published_profile_schema() -> None:
    profile = emit_profile(_shield(), profile_id="cell-42/band-n78")
    _assert_valid(profile, _schema(PROFILE_SCHEMA_PATH))


def test_profile_records_the_live_regulatory_limits() -> None:
    """The point of the profile: the numbers actually being enforced, read off
    the instance rather than restated by hand."""
    profile = emit_profile(_shield(), profile_id="p-1")
    limits = {d["id"]: d["limits"] for d in profile["invariant_chain"]}
    assert limits["spectral_mask_ts38104"]["band_lo_hz"] == pytest.approx(BAND_LO_HZ)
    assert limits["spectral_mask_ts38104"]["band_hi_hz"] == pytest.approx(BAND_HI_HZ)
    assert limits["max_eirp"]["max_eirp_dBm"] == pytest.approx(MAX_EIRP_DBM)
    assert limits["numeric_domain_sanity"] == {}
    # Tuples become arrays so the profile is JSON, not repr.
    assert limits["constellation_legality"]["allowed_orders"] == [4, 16, 64, 256]
    assert limits["neural_rx_envelope"]["require_measurement"] is True


def test_profile_digest_is_stable_across_calls() -> None:
    first = emit_profile(_shield(), profile_id="p-1", description="d")
    second = emit_profile(_shield(), profile_id="p-1", description="d")
    assert profile_digest(first) == profile_digest(second)
    assert len(profile_digest(first)) == 64


def test_profile_digest_moves_when_a_limit_moves() -> None:
    tight = emit_profile(
        default_terrestrial_shield(
            band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ, max_eirp_dBm=MAX_EIRP_DBM - 3.0
        ),
        profile_id="p-1",
    )
    assert profile_digest(tight) != profile_digest(emit_profile(_shield(), profile_id="p-1"))


def test_profile_digest_moves_when_the_chain_is_reordered() -> None:
    """Chain order is part of the safety claim, so it must be part of the digest.
    JSON arrays are ordered and ``sort_keys`` sorts keys, never elements."""
    forward = emit_profile(
        Shield([NumericSanityInvariant(), MaxEirpInvariant()]), profile_id="p-1"
    )
    reversed_ = emit_profile(
        Shield([MaxEirpInvariant(), NumericSanityInvariant()]), profile_id="p-1"
    )
    assert [d["id"] for d in forward["invariant_chain"]] != [
        d["id"] for d in reversed_["invariant_chain"]
    ]
    assert profile_digest(forward) != profile_digest(reversed_)


def test_the_lawful_intercept_catalogue_is_never_serialised() -> None:
    """``LawfulInterceptInvariant.li`` carries the operator's LIMF rule
    catalogue. The rApp sees warrant tags, never warrant content, so the profile
    records delegation and nothing else."""
    descriptor = invariant_descriptor(LawfulInterceptInvariant(li=_LiStub()))
    assert descriptor["id"] == "lawful_intercept"
    assert descriptor["class_name"] == "LawfulInterceptInvariant"
    assert descriptor["limits"] == {"li": {"delegated_to": "LIConstraint"}}
    assert "_LiStub" not in json.dumps(descriptor)


def test_a_profile_with_lawful_intercept_still_validates_and_digests() -> None:
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ,
        band_hi_hz=BAND_HI_HZ,
        max_eirp_dBm=MAX_EIRP_DBM,
        li_constraint=_LiStub(),
    )
    profile = emit_profile(shield, profile_id="p-li")
    assert profile["invariant_chain"][1]["id"] == "lawful_intercept"
    _assert_valid(profile, _schema(PROFILE_SCHEMA_PATH))
    assert len(profile_digest(profile)) == 64


def test_the_profile_action_contract_mirrors_planner_module() -> None:
    contract = emit_profile(_shield(), profile_id="p-1")["action_contract"]
    assert contract["required_keys"] == list(REQUIRED_ACTION_KEYS)
    assert contract["optional_keys"] == list(OPTIONAL_ACTION_KEYS)
    assert contract["forbidden_keys"] == list(OUTPUT_SENTINEL_KEYS)
    assert contract["json_primitive_values_only"] is True
