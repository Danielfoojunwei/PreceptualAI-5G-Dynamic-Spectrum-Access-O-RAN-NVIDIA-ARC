"""E2SM-RC control-payload tests — the property is *only Shield-approved
actions can reach E2*.

Two things are pinned here.

1. **Wire correctness.** The vendored O-RAN E2SM-RC v1.03 *standard*
   ASN.1 text (``src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn``,
   sha256-pinned against the FlexRIC tree it was copied from) compiles
   and exposes the control types, and every payload this repo encodes
   decodes back to the same fields through the same spec — so a bad
   vendor copy, a renamed IE, or a wrong CHOICE arm fails here rather
   than on someone's E2 wire. Aligned PER throughout, matching
   ``horizon_ric.e2.kpm_bridge``.

2. **Fail-closed enforcement.** ``control_from_disposition`` is the only
   route from a Horizon decision to an E2SM-RC control payload, and a
   real ``default_terrestrial_shield`` refusal must make encoding
   *impossible*, not merely discouraged. The paired
   blocked/approved test is the whole point of the module: it is what
   stops E2 from being a parallel unguarded route to the RAN while A1 is
   guarded. The approved half also checks the encoded values are the
   Shield's *projected* ones, not the planner's proposal.
"""

from __future__ import annotations

import hashlib

import pytest

from horizon_ric.e2.rc_control import (
    DEFAULT_PARAMETER_IDS,
    RC_ASN1_FILE,
    RC_ASN1_SHA256,
    RcControlError,
    control_from_disposition,
    control_message_parameters,
    decode_control_header,
    decode_control_message,
    encode_control_header,
    encode_control_message,
    rc_spec,
)
from horizon_ric.shield import default_terrestrial_shield

# The FlexRIC submodule pin the spec text was vendored from.
FLEXRIC_COMMIT = "ef6d722f22191eea74089966983da1f5ec1fedd4"

# Control types the encoder and its round-trip decode depend on.
REQUIRED_RC_CONTROL_TYPES = (
    "E2SM-RC-ControlHeader",
    "E2SM-RC-ControlHeader-Format1",
    "E2SM-RC-ControlHeader-Format2",
    "E2SM-RC-ControlMessage",
    "E2SM-RC-ControlMessage-Format1",
    "E2SM-RC-ControlMessage-Format1-Item",
    "E2SM-RC-ControlMessage-Format2",
    "E2SM-RC-ControlMessage-Format2-Style-Item",
    "E2SM-RC-ControlMessage-Format2-ControlAction-Item",
)

# The licensed channel and EIRP ceiling used for every Shield in this module:
# 3.40–3.50 GHz (n78 sub-block), 33 dBm EIRP.
BAND_LO_HZ = 3.40e9
BAND_HI_HZ = 3.50e9
MAX_EIRP_DBM = 33.0

# A UE identity in AMF-UE-NGAP-ID range — the same value FlexRIC's gNB
# emulator reports in its KPM UE reports, so the two E2 paths line up.
UE_ID = 112358132134

# In-band, under-ceiling: the Shield must pass this through untouched.
CLEAN_ACTION = {
    "block": "risk_band_planner",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 20e6,
    "tx_power_dBm": 20.0,
    "antenna_gain_dBi": 10.0,
}

# In-band but 17 dB over the EIRP ceiling: fixable by projection
# (33 dBm ceiling − 10 dBi gain ⇒ 23 dBm conducted).
OVER_EIRP_ACTION = {
    "block": "risk_band_planner",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 20e6,
    "tx_power_dBm": 40.0,
    "antenna_gain_dBi": 10.0,
}
PROJECTED_TX_POWER_DBM = 23.0

# Out of band AND over-ceiling, and 400 MHz wide — four times the 100 MHz
# licensed channel, so the spectral-mask projection cannot place the
# carrier at all and the Shield fails closed (emit_blocked).
UNFIXABLE_ACTION = {
    "block": "poisoned_planner",
    "frequency_hz": 3.90e9,
    "bandwidth_hz": 400e6,
    "tx_power_dBm": 46.0,
    "antenna_gain_dBi": 12.0,
}
SPECTRAL_MASK_ID = "spectral_mask_ts38104"
MAX_EIRP_ID = "max_eirp"


def _shield():
    return default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ, max_eirp_dBm=MAX_EIRP_DBM
    )


# ---------------------------------------------------------------------------
# Vendored spec provenance and compilation
# ---------------------------------------------------------------------------
def test_vendored_rc_spec_matches_flexric_provenance():
    digest = hashlib.sha256(RC_ASN1_FILE.read_bytes()).hexdigest()
    assert digest == RC_ASN1_SHA256
    provenance = (RC_ASN1_FILE.parent / "PROVENANCE_RC.md").read_text()
    assert digest in provenance
    assert FLEXRIC_COMMIT in provenance


def test_vendored_rc_spec_compiles_and_exposes_control_types():
    spec = rc_spec()
    for name in REQUIRED_RC_CONTROL_TYPES:
        assert name in spec.types, f"vendored E2SM-RC spec is missing {name}"
    # Cached: the same compiled object comes back, not a second compile.
    assert rc_spec() is spec


# ---------------------------------------------------------------------------
# Header round-trip (E2SM-RC-ControlHeader / controlHeader-Format1)
# ---------------------------------------------------------------------------
def test_control_header_round_trip_preserves_fields():
    encoded = encode_control_header(ue_id=UE_ID, ric_style_type=2, control_action_id=3)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0

    decoded = decode_control_header(encoded)
    kind, fmt1 = decoded["ric-controlHeader-formats"]
    assert kind == "controlHeader-Format1"
    assert fmt1["ric-Style-Type"] == 2
    assert fmt1["ric-ControlAction-ID"] == 3
    ue_kind, ue_body = fmt1["ueID"]
    assert ue_kind == "gNB-UEID"
    assert ue_body["amf-UE-NGAP-ID"] == UE_ID
    # Minimal spec-valid UEID-GNB: mandatory GUAMI, no optional id lists.
    guami = ue_body["guami"]
    assert set(guami) == {"pLMNIdentity", "aMFRegionID", "aMFSetID", "aMFPointer"}
    assert len(guami["pLMNIdentity"]) == 3
    assert guami["aMFRegionID"][1] == 8
    assert guami["aMFSetID"][1] == 10
    assert guami["aMFPointer"][1] == 6
    # The OPTIONAL ric-ControlDecision is deliberately not asserted.
    assert "ric-ControlDecision" not in fmt1


def test_control_header_rejects_out_of_range_ids():
    with pytest.raises(RcControlError, match="AMF-UE-NGAP-ID range"):
        encode_control_header(ue_id=-1, ric_style_type=1, control_action_id=1)
    with pytest.raises(RcControlError, match="RIC-ControlAction-ID range"):
        encode_control_header(ue_id=1, ric_style_type=1, control_action_id=0)


# ---------------------------------------------------------------------------
# Message round-trip (E2SM-RC-ControlMessage / controlMessage-Format1)
# ---------------------------------------------------------------------------
def test_control_message_round_trip_integer_and_real_values():
    encoded = encode_control_message(parameters=[(1, 3450000000.0), (2, 42), (3, -7)])
    decoded = decode_control_message(encoded)
    kind, fmt1 = decoded["ric-controlMessage-formats"]
    assert kind == "controlMessage-Format1"
    items = fmt1["ranP-List"]
    assert len(items) == 3

    # ranParameter-ID order is preserved; each value takes the CHOICE arm
    # its Python type dictates.
    assert [item["ranParameter-ID"] for item in items] == [1, 2, 3]
    arms = []
    for item in items:
        vt_kind, vt_body = item["ranParameter-valueType"]
        assert vt_kind == "ranP-Choice-ElementTrue"
        arms.append(vt_body["ranParameter-value"][0])
    assert arms == ["valueReal", "valueInt", "valueInt"]

    values = control_message_parameters(encoded)
    assert values[1] == pytest.approx(3.45e9)
    assert values[2] == 42
    assert values[3] == -7
    # The integer arm must not have been floated on the way through.
    assert isinstance(values[2], int) is True
    assert isinstance(values[1], float) is True


def test_control_message_rejects_unsupported_value_type_rather_than_coercing():
    with pytest.raises(RcControlError, match="unsupported RAN parameter value type"):
        encode_control_message(parameters=[(1, "3450000000")])
    with pytest.raises(RcControlError, match="unsupported RAN parameter value type"):
        encode_control_message(parameters=[(1, None)])
    # bool is an int subclass in Python; encoding it as valueInt would be
    # exactly the silent coercion the encoder refuses.
    with pytest.raises(RcControlError, match="bool RAN parameter values"):
        encode_control_message(parameters=[(1, True)])


def test_control_message_rejects_out_of_range_parameter_id():
    with pytest.raises(RcControlError, match="RANParameter-ID range"):
        encode_control_message(parameters=[(0, 1)])


# ---------------------------------------------------------------------------
# Determinism — identical input must give byte-identical payloads
# ---------------------------------------------------------------------------
def test_encoded_payloads_are_deterministic():
    header_digests = {
        hashlib.sha256(
            encode_control_header(ue_id=UE_ID, ric_style_type=1, control_action_id=1)
        ).hexdigest()
        for _ in range(2)
    }
    assert len(header_digests) == 1

    params = [(1, 3.45e9), (2, 20000000.0), (3, 20.0), (4, 10)]
    message_digests = {
        hashlib.sha256(encode_control_message(parameters=params)).hexdigest()
        for _ in range(2)
    }
    assert len(message_digests) == 1


def test_control_request_from_identical_disposition_is_deterministic():
    shield = _shield()
    first = control_from_disposition(
        shield.dispose(CLEAN_ACTION, decision_id="dec-determinism"), ue_id=UE_ID
    )
    second = control_from_disposition(
        shield.dispose(CLEAN_ACTION, decision_id="dec-determinism"), ue_id=UE_ID
    )
    assert first.header_bytes == second.header_bytes
    assert first.message_bytes == second.message_bytes
    assert first.header_sha256 == second.header_sha256
    assert first.message_sha256 == second.message_sha256


# ---------------------------------------------------------------------------
# THE PROPERTY: only Shield-approved actions can reach E2
# ---------------------------------------------------------------------------
def test_shield_refused_action_cannot_become_an_e2_control_request():
    shield = _shield()
    disposition = shield.dispose(UNFIXABLE_ACTION, decision_id="dec-blocked")

    # The Shield genuinely refused: out-of-band beyond repair and over-EIRP.
    assert disposition.certificate.emit_blocked is True
    assert disposition.certificate.safe is False
    assert SPECTRAL_MASK_ID in disposition.certificate.violated_ids
    assert MAX_EIRP_ID in disposition.certificate.violated_ids

    with pytest.raises(RcControlError) as excinfo:
        control_from_disposition(disposition, ue_id=UE_ID)
    message = str(excinfo.value)
    assert SPECTRAL_MASK_ID in message
    assert MAX_EIRP_ID in message
    assert "emit_blocked" in message
    assert "dec-blocked" in message


def test_shield_approved_action_reaches_e2_with_the_projected_values():
    shield = _shield()
    disposition = shield.dispose(CLEAN_ACTION, decision_id="dec-clean")
    assert disposition.certificate.emit_blocked is False
    assert disposition.certificate.safe is True
    assert disposition.certificate.violated_ids == []
    assert disposition.certificate.projected is False

    request = control_from_disposition(disposition, ue_id=UE_ID)
    assert request.decision_id == "dec-clean"
    assert len(request.certificate_digest) == 64
    assert request.parameter_names == (
        "frequency_hz",
        "bandwidth_hz",
        "tx_power_dBm",
        "antenna_gain_dBi",
    )

    # The decoded echo carried on the request equals a fresh decode.
    assert request.decoded_header == decode_control_header(request.header_bytes)
    assert request.decoded_message == decode_control_message(request.message_bytes)

    values = control_message_parameters(request.message_bytes)
    assert values[DEFAULT_PARAMETER_IDS["frequency_hz"]] == pytest.approx(3.45e9)
    assert values[DEFAULT_PARAMETER_IDS["bandwidth_hz"]] == pytest.approx(20e6)
    assert values[DEFAULT_PARAMETER_IDS["tx_power_dBm"]] == pytest.approx(20.0)
    assert values[DEFAULT_PARAMETER_IDS["antenna_gain_dBi"]] == pytest.approx(10.0)

    _, fmt1 = request.decoded_header["ric-controlHeader-formats"]
    assert fmt1["ueID"][1]["amf-UE-NGAP-ID"] == UE_ID


def test_control_request_encodes_the_projection_not_the_proposal():
    """A fixable violation must reach E2 as the *corrected* value."""
    shield = _shield()
    disposition = shield.dispose(OVER_EIRP_ACTION, decision_id="dec-projected")
    assert disposition.certificate.emit_blocked is False
    assert disposition.certificate.safe is True
    assert disposition.certificate.projected is True
    assert disposition.certificate.action_proposed["tx_power_dBm"] == pytest.approx(40.0)
    assert disposition.safe_action["tx_power_dBm"] == pytest.approx(
        PROJECTED_TX_POWER_DBM
    )

    request = control_from_disposition(disposition, ue_id=UE_ID)
    values = control_message_parameters(request.message_bytes)
    assert values[DEFAULT_PARAMETER_IDS["tx_power_dBm"]] == pytest.approx(
        PROJECTED_TX_POWER_DBM
    )
    # The 40 dBm the planner asked for is nowhere on the wire.
    assert values[DEFAULT_PARAMETER_IDS["tx_power_dBm"]] != pytest.approx(40.0)


def test_control_from_disposition_rejects_non_disposition_and_empty_mapping():
    shield = _shield()
    with pytest.raises(RcControlError, match="must carry .safe_action"):
        control_from_disposition(object(), ue_id=UE_ID)

    disposition = shield.dispose(CLEAN_ACTION, decision_id="dec-empty-map")
    with pytest.raises(RcControlError, match="refusing to encode an empty"):
        control_from_disposition(
            disposition, ue_id=UE_ID, parameter_ids={"no_such_key": 9}
        )


def test_control_request_to_dict_is_json_safe():
    shield = _shield()
    request = control_from_disposition(
        shield.dispose(CLEAN_ACTION, decision_id="dec-json"), ue_id=UE_ID
    )
    summary = request.to_dict()
    assert summary["decision_id"] == "dec-json"
    assert summary["asn1_sha256"] == RC_ASN1_SHA256
    assert summary["e2sm_rc_version"] == "v1.03"
    assert bytes.fromhex(summary["header_hex"]) == request.header_bytes
    assert bytes.fromhex(summary["message_hex"]) == request.message_bytes
    assert summary["header_sha256"] == hashlib.sha256(request.header_bytes).hexdigest()
    assert summary["message_sha256"] == hashlib.sha256(request.message_bytes).hexdigest()
