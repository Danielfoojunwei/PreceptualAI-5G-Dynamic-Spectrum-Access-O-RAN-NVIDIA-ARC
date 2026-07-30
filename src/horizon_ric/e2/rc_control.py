"""E2SM-RC RIC Control payload **construction** — Shield-gated.

Horizon's enforcement used to stop at A1. A1 is the non-real-time
policy interface; it states intent ("admission control on, risk band
high") and leaves the RAN to interpret it. The near-real-time knob —
"set this UE's carrier to 3.45 GHz at 20 dBm, now" — is E2SM-RC RIC
Control, and this package had no encoder for it, so the near-RT plane
was reachable only by code that had never seen a Shield certificate.

This module closes that. It builds the two OCTET STRING payloads a RIC
Control Request carries:

* ``E2SM-RC-ControlHeader`` — who to control and which control style /
  action (``controlHeader-Format1``);
* ``E2SM-RC-ControlMessage`` — the RAN parameter values to apply
  (``controlMessage-Format1``).

Both are encoded with ``asn1tools`` in **aligned PER** against the
vendored O-RAN E2SM-RC v1.03 *standard* ASN.1 text
(``horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn``, provenance and
sha256 in ``PROVENANCE_RC.md`` beside it) — the same file FlexRIC's own
asn1c wire codec for RC was generated from, and the same codec choice
``horizon_ric.e2.kpm_bridge`` already uses to decode KPM indications.

Scope — construction, not transport
-----------------------------------
This package **deliberately contains no E2AP/SCTP transport**, and this
module does not add any. The near-RT RIC owns the E2 association; what
Horizon contributes is the *E2SM payload*, byte-exact and
certificate-bound. A caller with an E2 termination places
``header_bytes`` into the RIC Control Request's *RIC Control Header* IE
and ``message_bytes`` into its *RIC Control Message* IE. Nothing here
opens a socket, and nothing here can be shown to have been accepted by
a live RAN — see ``deploy/e2-companion/E2_RC_PROOF.md``.

The load-bearing property — fail closed
---------------------------------------
:func:`control_from_disposition` is the only intended way to produce a
control request from a Horizon decision, and it **cannot** encode a
Shield-refused action: if the certificate says ``emit_blocked``, or
``not safe``, or lists any ``violated_ids``, it raises
:class:`RcControlError` naming the violated invariant ids. When it does
encode, it encodes strictly from ``disposition.safe_action`` — the
*projected* action — never from the action the planner proposed. That
makes the E2 path structurally subject to the same enforcement as A1
instead of being a parallel, unguarded route to the RAN.

RAN Parameter IDs are node-specific
-----------------------------------
E2SM-RC does not fix universal ids for RAN parameters: an E2 node
advertises them in its RAN Function Definition
(``RANFunctionDefinition-Control-Action-Item.ran-ControlActionParameters-List``,
a list of ``ControlAction-RANParameter-Item`` pairing a
``ranParameter-ID`` with its ``ranParameter-name``). The ids in
:data:`DEFAULT_PARAMETER_IDS` are therefore **local placeholders for
the four Shield-governed action keys**, not spec constants. Against a
real node, pass ``parameter_ids=`` with the ids that node advertises.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import asn1tools
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# ASN.1 spec (E2SM-RC v1.03, aligned PER) — committed with provenance.
# ---------------------------------------------------------------------------
ASN1_DIR = Path(__file__).resolve().parent / "asn1"
RC_ASN1_FILE = ASN1_DIR / "e2sm_rc_v1_03_standard.asn"

# sha256 of the vendored spec text; equals the FlexRIC original at commit
# ef6d722f22191eea74089966983da1f5ec1fedd4. See asn1/PROVENANCE_RC.md.
RC_ASN1_SHA256 = "09f295cb59d4145efd64fe510f602c994d6df65e9beba33aab2e98cdf5bc642b"

# The E2SM-RC version string this module encodes against.
E2SM_RC_VERSION = "v1.03"

_spec_cache: Any | None = None


class RcControlError(ValueError):
    """Raised when an E2SM-RC control payload cannot be constructed.

    Also the fail-closed signal: a Shield disposition that was refused,
    unsafe, or carries violated invariants raises this rather than
    yielding encodable bytes.
    """


def rc_spec() -> Any:
    """Compile (once) and return the asn1tools E2SM-RC v1.03 spec.

    ``codec="per"`` is asn1tools' ALIGNED PER — the transfer syntax
    E2SM mandates and the one ``horizon_ric.e2.kpm_bridge`` already
    uses. Compilation takes roughly half a second and is cached for the
    process lifetime.

    Raises :class:`RcControlError`, naming the path, if the vendored
    ASN.1 text is missing or does not compile — a bad vendor copy must
    not surface later as a confusing encode failure.
    """
    global _spec_cache
    if _spec_cache is None:
        if not RC_ASN1_FILE.is_file():
            raise RcControlError(
                f"vendored E2SM-RC {E2SM_RC_VERSION} ASN.1 spec not found: {RC_ASN1_FILE}"
            )
        try:
            _spec_cache = asn1tools.compile_files([str(RC_ASN1_FILE)], codec="per")
        except Exception as exc:  # asn1tools raises its own error hierarchy
            raise RcControlError(
                f"vendored E2SM-RC ASN.1 spec {RC_ASN1_FILE} failed to compile: {exc}"
            ) from exc
        logger.info(
            "rc_control.asn1.compiled",
            file=str(RC_ASN1_FILE),
            e2sm=f"rc-{E2SM_RC_VERSION}",
        )
    return _spec_cache


# ---------------------------------------------------------------------------
# Minimal valid GUAMI for the gNB-UEID arm.
# ---------------------------------------------------------------------------
# E2SM-RC-ControlHeader-Format1 requires a mandatory ``ueID UEID``, and
# every UEID arm is a SEQUENCE with mandatory identity fields — there is
# no bare-integer arm. The ``gNB-UEID`` arm (UEID-GNB) is the narrowest
# one for a 5G SA gNB: mandatory ``amf-UE-NGAP-ID`` (INTEGER
# 0..1099511627775) plus a mandatory ``guami``; every other field is
# OPTIONAL and the spec's own comments say the CU-CP/DU F1AP/E1AP id
# lists "may not be included" in NearRT-RIC → E2 Node messages. So the
# minimal spec-valid instance is amf-UE-NGAP-ID + GUAMI only, which is
# what this module encodes.
#
# GUAMI (3GPP TS 38.413) = PLMN identity + AMF Region/Set/Pointer. Those
# identify the serving AMF, not the UE, so they are not something a
# radio action carries; they are pinned here to documented test-network
# values so the encoding is deterministic and reproducible. Against a
# real core they must be replaced with the serving AMF's real GUAMI.
#
# PLMN 001/01 — the 3GPP TS 23.003 §12.1 test PLMN — TBCD-encoded as
# MCC digits 0,0,1 then MNC filler + digits: 0x00 0xf1 0x10.
DEFAULT_PLMN_IDENTITY = b"\x00\xf1\x10"
# AMFRegionID  BIT STRING (SIZE(8))
DEFAULT_AMF_REGION_ID = (b"\x80", 8)
# AMFSetID     BIT STRING (SIZE(10))
DEFAULT_AMF_SET_ID = (b"\x00\x40", 10)
# AMFPointer   BIT STRING (SIZE(6))
DEFAULT_AMF_POINTER = (b"\x00", 6)

# AMF-UE-NGAP-ID ::= INTEGER (0..1099511627775)
AMF_UE_NGAP_ID_MAX = 1_099_511_627_775
# RANParameter-ID ::= INTEGER (1..4294967295, ...)
RAN_PARAMETER_ID_MIN = 1
RAN_PARAMETER_ID_MAX = 4_294_967_295
# RIC-ControlAction-ID ::= INTEGER (1..65535, ...)
RIC_CONTROL_ACTION_ID_MIN = 1
RIC_CONTROL_ACTION_ID_MAX = 65_535

# Local placeholder RANParameter-IDs for the Shield-governed action keys.
# NOT spec constants — see the module docstring. Insertion order is the
# encoding order, which keeps the encoded bytes deterministic.
DEFAULT_PARAMETER_IDS: dict[str, int] = {
    "frequency_hz": 1,
    "bandwidth_hz": 2,
    "tx_power_dBm": 3,
    "antenna_gain_dBi": 4,
}


def _guami() -> dict[str, Any]:
    return {
        "pLMNIdentity": DEFAULT_PLMN_IDENTITY,
        "aMFRegionID": DEFAULT_AMF_REGION_ID,
        "aMFSetID": DEFAULT_AMF_SET_ID,
        "aMFPointer": DEFAULT_AMF_POINTER,
    }


# ---------------------------------------------------------------------------
# Header (E2SM-RC-ControlHeader, controlHeader-Format1)
# ---------------------------------------------------------------------------
def encode_control_header(
    *,
    ue_id: int,
    ric_style_type: int,
    control_action_id: int,
) -> bytes:
    """Encode an ``E2SM-RC-ControlHeader`` via ``controlHeader-Format1``.

    The field names below are the spec's own, read from the vendored
    text (``E2SM-RC-IEs`` module)::

        E2SM-RC-ControlHeader ::= SEQUENCE {
            ric-controlHeader-formats  CHOICE {
                controlHeader-Format1  E2SM-RC-ControlHeader-Format1, ...
            }, ... }

        E2SM-RC-ControlHeader-Format1 ::= SEQUENCE {
            ueID                  UEID,
            ric-Style-Type        RIC-Style-Type,
            ric-ControlAction-ID  RIC-ControlAction-ID,
            ric-ControlDecision   ENUMERATED {accept, reject, ...} OPTIONAL,
            ... }

    **UEID CHOICE arm used: ``gNB-UEID`` (``UEID-GNB``).** ``ueID`` is
    mandatory in Format1 and ``UEID`` has no scalar arm — every arm is a
    SEQUENCE. ``gNB-UEID`` is chosen because it is the 5G-SA gNB arm and
    the narrowest: only ``amf-UE-NGAP-ID`` and ``guami`` are mandatory,
    and the spec's inline comments state the CU-CP/DU F1AP and E1AP id
    lists "may not be included" in NearRT-RIC → E2 Node messages. So
    ``ue_id`` is carried as ``amf-UE-NGAP-ID`` (the NGAP-level UE
    identity a CU-CP knows) alongside a documented minimal GUAMI (see
    :data:`DEFAULT_PLMN_IDENTITY` and friends — test-network values,
    replace for a real core).

    The OPTIONAL ``ric-ControlDecision`` is omitted: it is not required
    for a RIC-initiated control request, and inventing an ``accept`` /
    ``reject`` value here would assert a semantic the caller has not
    asked for.

    Returns the aligned-PER bytes to place in the RIC Control Request's
    *RIC Control Header* IE.
    """
    if not isinstance(ue_id, int) or isinstance(ue_id, bool):
        raise RcControlError(f"ue_id must be an int, got {type(ue_id).__name__}")
    if not 0 <= ue_id <= AMF_UE_NGAP_ID_MAX:
        raise RcControlError(
            f"ue_id {ue_id} outside AMF-UE-NGAP-ID range 0..{AMF_UE_NGAP_ID_MAX}"
        )
    if not isinstance(ric_style_type, int) or isinstance(ric_style_type, bool):
        raise RcControlError(
            f"ric_style_type must be an int, got {type(ric_style_type).__name__}"
        )
    if not isinstance(control_action_id, int) or isinstance(control_action_id, bool):
        raise RcControlError(
            f"control_action_id must be an int, got {type(control_action_id).__name__}"
        )
    if not RIC_CONTROL_ACTION_ID_MIN <= control_action_id <= RIC_CONTROL_ACTION_ID_MAX:
        raise RcControlError(
            f"control_action_id {control_action_id} outside RIC-ControlAction-ID range "
            f"{RIC_CONTROL_ACTION_ID_MIN}..{RIC_CONTROL_ACTION_ID_MAX}"
        )

    pdu = {
        "ric-controlHeader-formats": (
            "controlHeader-Format1",
            {
                "ueID": (
                    "gNB-UEID",
                    {"amf-UE-NGAP-ID": ue_id, "guami": _guami()},
                ),
                "ric-Style-Type": ric_style_type,
                "ric-ControlAction-ID": control_action_id,
            },
        )
    }
    try:
        return rc_spec().encode("E2SM-RC-ControlHeader", pdu)
    except Exception as exc:
        raise RcControlError(f"E2SM-RC ControlHeader encode failed: {exc}") from exc


def decode_control_header(header_bytes: bytes) -> dict[str, Any]:
    """Decode ``E2SM-RC-ControlHeader`` aligned-PER bytes to a dict.

    Present so encoded bytes are independently checkable — the CHOICEs
    come back as asn1tools ``(arm_name, value)`` tuples, so a third
    party can confirm which arms the encoder actually used.
    """
    try:
        return dict(rc_spec().decode("E2SM-RC-ControlHeader", header_bytes))
    except Exception as exc:
        raise RcControlError(f"E2SM-RC ControlHeader decode failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Message (E2SM-RC-ControlMessage, controlMessage-Format1)
# ---------------------------------------------------------------------------
def _ran_parameter_value(value: float | int) -> tuple[str, Any]:
    """Pick the ``RANParameter-Value`` CHOICE arm for a Python value.

    From the vendored spec::

        RANParameter-Value ::= CHOICE {
            valueBoolean          BOOLEAN,
            valueInt              INTEGER,
            valueReal             REAL,
            valueBitS             BIT STRING,
            valueOctS             OCTET STRING,
            valuePrintableString  PrintableString,
            ... }

    ``int`` → ``valueInt``; ``float`` → ``valueReal``. Nothing else is
    accepted. ``bool`` is rejected on purpose even though it is an
    ``int`` subclass in Python: silently encoding ``True`` as integer 1
    would be exactly the coercion this function exists to refuse, and a
    genuinely boolean RAN parameter belongs in ``valueBoolean``, which
    this typed API does not reach.
    """
    if isinstance(value, bool):
        raise RcControlError(
            "bool RAN parameter values are not supported: encoding True/False as "
            "valueInt would silently coerce a boolean into an integer parameter "
            "(the spec's valueBoolean arm is not reachable through this API)"
        )
    if isinstance(value, int):
        return ("valueInt", value)
    if isinstance(value, float):
        return ("valueReal", value)
    raise RcControlError(
        f"unsupported RAN parameter value type {type(value).__name__!r} "
        f"({value!r}): expected int (valueInt) or float (valueReal)"
    )


def encode_control_message(*, parameters: Sequence[tuple[int, float | int]]) -> bytes:
    """Encode an ``E2SM-RC-ControlMessage`` via ``controlMessage-Format1``.

    ``parameters`` is an ordered sequence of ``(ranParameter-ID, value)``
    pairs; the order is preserved into ``ranP-List``, so identical input
    yields identical bytes. The spec's own field names::

        E2SM-RC-ControlMessage ::= SEQUENCE {
            ric-controlMessage-formats  CHOICE {
                controlMessage-Format1  E2SM-RC-ControlMessage-Format1, ...
            }, ... }

        E2SM-RC-ControlMessage-Format1 ::= SEQUENCE {
            ranP-List  SEQUENCE (SIZE(0..maxnoofAssociatedRANParameters))
                       OF E2SM-RC-ControlMessage-Format1-Item, ... }

        E2SM-RC-ControlMessage-Format1-Item ::= SEQUENCE {
            ranParameter-ID         RANParameter-ID,
            ranParameter-valueType  RANParameter-ValueType, ... }

    ``RANParameter-ValueType`` is a CHOICE; the arm used is
    ``ranP-Choice-ElementTrue`` (``RANParameter-ValueType-Choice-ElementTrue``),
    whose single mandatory field is ``ranParameter-value``. That is the
    right arm for a control request: the sibling
    ``ranP-Choice-ElementFalse`` makes ``ranParameter-value`` OPTIONAL and
    the spec annotates it *"C-ifControl: this IE shall be present if it is
    part of a RIC Control Request message"* — i.e. a control request must
    carry the value, so the arm that mandates it is the correct choice.
    ``ranP-Choice-Structure`` / ``ranP-Choice-List`` are for nested
    structures and are not used here.

    Value arm selection is by Python type; see :func:`_ran_parameter_value`.
    """
    items: list[dict[str, Any]] = []
    for param_id, value in parameters:
        if not isinstance(param_id, int) or isinstance(param_id, bool):
            raise RcControlError(
                f"ranParameter-ID must be an int, got {type(param_id).__name__}"
            )
        if not RAN_PARAMETER_ID_MIN <= param_id <= RAN_PARAMETER_ID_MAX:
            raise RcControlError(
                f"ranParameter-ID {param_id} outside RANParameter-ID range "
                f"{RAN_PARAMETER_ID_MIN}..{RAN_PARAMETER_ID_MAX}"
            )
        items.append(
            {
                "ranParameter-ID": param_id,
                "ranParameter-valueType": (
                    "ranP-Choice-ElementTrue",
                    {"ranParameter-value": _ran_parameter_value(value)},
                ),
            }
        )

    pdu = {"ric-controlMessage-formats": ("controlMessage-Format1", {"ranP-List": items})}
    try:
        return rc_spec().encode("E2SM-RC-ControlMessage", pdu)
    except Exception as exc:
        raise RcControlError(f"E2SM-RC ControlMessage encode failed: {exc}") from exc


def decode_control_message(message_bytes: bytes) -> dict[str, Any]:
    """Decode ``E2SM-RC-ControlMessage`` aligned-PER bytes to a dict."""
    try:
        return dict(rc_spec().decode("E2SM-RC-ControlMessage", message_bytes))
    except Exception as exc:
        raise RcControlError(f"E2SM-RC ControlMessage decode failed: {exc}") from exc


def control_message_parameters(message_bytes: bytes) -> dict[int, float | int]:
    """``{ranParameter-ID: value}`` from encoded ControlMessage bytes.

    A convenience over :func:`decode_control_message` for asserting what
    a control request actually carries, without walking the CHOICE
    tuples by hand.
    """
    decoded = decode_control_message(message_bytes)
    kind, body = decoded["ric-controlMessage-formats"]
    if kind != "controlMessage-Format1":
        raise RcControlError(f"expected controlMessage-Format1, got {kind!r}")
    out: dict[int, float | int] = {}
    for item in body.get("ranP-List") or []:
        vt_kind, vt_body = item["ranParameter-valueType"]
        if vt_kind != "ranP-Choice-ElementTrue":
            raise RcControlError(f"unexpected RANParameter-ValueType arm {vt_kind!r}")
        out[item["ranParameter-ID"]] = vt_body["ranParameter-value"][1]
    return out


# ---------------------------------------------------------------------------
# Shield-gated control request — the enforcement crossing E2.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RcControlRequest:
    """The two E2SM-RC payloads of one Shield-certified control action.

    ``header_bytes`` goes into the RIC Control Request's *RIC Control
    Header* IE and ``message_bytes`` into its *RIC Control Message* IE.
    Everything else is the audit binding: ``decision_id`` and
    ``certificate_digest`` tie these exact bytes to the Shield
    certificate that authorised them, and ``decoded_header`` /
    ``decoded_message`` are the round-trip echo of what was encoded, so
    the bytes can be checked without re-running the encoder.
    """

    header_bytes: bytes
    message_bytes: bytes
    decision_id: str
    certificate_digest: str
    decoded_header: dict[str, Any]
    decoded_message: dict[str, Any]
    parameters: tuple[tuple[int, float | int], ...] = ()
    parameter_names: tuple[str, ...] = ()
    e2sm_rc_version: str = E2SM_RC_VERSION
    asn1_sha256: str = RC_ASN1_SHA256

    @property
    def header_sha256(self) -> str:
        return hashlib.sha256(self.header_bytes).hexdigest()

    @property
    def message_sha256(self) -> str:
        return hashlib.sha256(self.message_bytes).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary (bytes as hex) for the evidence chain."""
        return {
            "e2sm_rc_version": self.e2sm_rc_version,
            "asn1_sha256": self.asn1_sha256,
            "decision_id": self.decision_id,
            "certificate_digest": self.certificate_digest,
            "header_hex": self.header_bytes.hex(),
            "header_sha256": self.header_sha256,
            "message_hex": self.message_bytes.hex(),
            "message_sha256": self.message_sha256,
            "parameter_names": list(self.parameter_names),
            "parameters": [[pid, value] for pid, value in self.parameters],
        }


def certificate_digest(certificate: Any) -> str:
    """sha256 over the certificate's canonical JSON.

    Uses ``horizon_ric.shield.signing.canonical_certificate_bytes`` when
    available (the same canonicalisation the Ed25519 signature covers),
    so the digest recorded on a control request is comparable with the
    signed evidence. Falls back to a sorted-key JSON dump of
    ``certificate.to_dict()`` if the optional signing dependency
    (``cryptography``) is not installed.
    """
    try:
        from horizon_ric.shield.signing import canonical_certificate_bytes

        return hashlib.sha256(canonical_certificate_bytes(certificate)).hexdigest()
    except Exception:  # pragma: no cover — optional signing dependency
        payload = json.dumps(
            certificate.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def control_from_disposition(
    disposition: Any,
    *,
    ue_id: int,
    ric_style_type: int = 1,
    control_action_id: int = 1,
    parameter_ids: Mapping[str, int] | None = None,
) -> RcControlRequest:
    """Turn a Shield disposition into E2SM-RC control payloads — fail closed.

    This is the only intended route from a Horizon decision to E2, and
    it refuses to build one for an action the Shield did not approve.
    :class:`RcControlError` is raised when the certificate reports

    * ``emit_blocked`` — the Shield could not make the action safe, or
    * ``not safe`` — the final action still breaches a hard invariant, or
    * a non-empty ``violated_ids`` — any residual invariant breach,

    and the message names the violated invariant ids. There is no
    ``force`` flag and no second path: a Shield-refused action is
    *structurally unable* to become an E2 control message, which is what
    puts the near-real-time plane under the same enforcement as A1.

    On approval the payloads are encoded **only** from
    ``disposition.safe_action`` — the projected action — never from
    ``certificate.action_proposed``. So if the Shield clipped a carrier
    back inside the licensed band or cut Tx power to meet the EIRP
    ceiling, the bytes that would reach the RAN carry the clipped
    values.

    ``parameter_ids`` maps ``safe_action`` keys to ``RANParameter-ID``s;
    it defaults to :data:`DEFAULT_PARAMETER_IDS` (local placeholders —
    see the module docstring on node-advertised ids). Keys absent from
    the safe action are skipped; a mapping that matches nothing in the
    action is an error, because an empty control message would silently
    command nothing. Encoding order follows the mapping's iteration
    order, so identical input gives identical bytes.
    """
    certificate = getattr(disposition, "certificate", None)
    safe_action = getattr(disposition, "safe_action", None)
    if certificate is None or safe_action is None:
        raise RcControlError(
            "disposition must carry .safe_action and .certificate "
            "(a horizon_ric.shield.certificate.ShieldDisposition)"
        )

    violated = list(getattr(certificate, "violated_ids", []) or [])
    decision_id = str(getattr(certificate, "decision_id", "") or "")
    if certificate.emit_blocked or not certificate.safe or violated:
        reason = []
        if certificate.emit_blocked:
            reason.append("emit_blocked")
        if not certificate.safe:
            reason.append("not safe")
        if violated:
            reason.append("violated invariants " + ", ".join(violated))
        logger.warning(
            "rc_control.fail_closed",
            decision_id=decision_id,
            violated_ids=violated,
            emit_blocked=bool(certificate.emit_blocked),
        )
        raise RcControlError(
            "Shield refused this action; it cannot be encoded as an E2SM-RC "
            f"control request (decision_id={decision_id or '<unset>'}): "
            + "; ".join(reason)
        )

    ids = dict(parameter_ids) if parameter_ids is not None else dict(DEFAULT_PARAMETER_IDS)
    parameters: list[tuple[int, float | int]] = []
    names: list[str] = []
    for key, param_id in ids.items():
        if key not in safe_action:
            continue
        parameters.append((param_id, safe_action[key]))
        names.append(key)
    if not parameters:
        raise RcControlError(
            "safe_action carries none of the mapped RAN parameter keys "
            f"{sorted(ids)}; refusing to encode an empty control message"
        )

    header_bytes = encode_control_header(
        ue_id=ue_id,
        ric_style_type=ric_style_type,
        control_action_id=control_action_id,
    )
    message_bytes = encode_control_message(parameters=parameters)

    request = RcControlRequest(
        header_bytes=header_bytes,
        message_bytes=message_bytes,
        decision_id=decision_id,
        certificate_digest=certificate_digest(certificate),
        decoded_header=decode_control_header(header_bytes),
        decoded_message=decode_control_message(message_bytes),
        parameters=tuple(parameters),
        parameter_names=tuple(names),
    )
    logger.info(
        "rc_control.request",
        decision_id=decision_id,
        ue_id=ue_id,
        ric_style_type=ric_style_type,
        control_action_id=control_action_id,
        parameter_names=names,
        header_sha256=request.header_sha256,
        message_sha256=request.message_sha256,
    )
    return request


__all__ = [
    "AMF_UE_NGAP_ID_MAX",
    "ASN1_DIR",
    "DEFAULT_PARAMETER_IDS",
    "E2SM_RC_VERSION",
    "RC_ASN1_FILE",
    "RC_ASN1_SHA256",
    "RcControlError",
    "RcControlRequest",
    "certificate_digest",
    "control_from_disposition",
    "control_message_parameters",
    "decode_control_header",
    "decode_control_message",
    "encode_control_header",
    "encode_control_message",
    "rc_spec",
]
