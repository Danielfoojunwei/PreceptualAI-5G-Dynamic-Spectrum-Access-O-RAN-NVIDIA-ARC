"""Offline RFC-3161 timestamp-anchor verification.

A Horizon evidence chain can be *anchored* to a public RFC-3161 Time-Stamp
Authority (TSA, e.g. freetsa.org): the producer asks the TSA to sign the
current chain-head hash and stores the returned token as a
``rfc3161_anchor`` record / anchor JSON
(``horizon_ric.evidence.rfc3161.TimestampAnchor.to_record``)::

    {"type": "rfc3161_anchor", "tsa": "<url>", "token_b64": "<base64>",
     "chain_head_hex": "<sha256-hex>", "gen_time_utc": "<iso>",
     "anchored_at_unix": <float>}

**Offline verification** — done here without ever contacting the TSA — checks
the property that actually binds the token to the evidence: the token's
``messageImprint`` must equal ``H(chain_head)`` under the hash algorithm named
inside the token. This is the RFC-3161 invariant that ties a signed timestamp
to a specific message. When the token embeds the TSA's signing certificate we
additionally verify the CMS ``SignerInfo`` signature (real ``cryptography``
public-key verification) and that the signed ``messageDigest`` attribute
matches the ``eContent`` — i.e. the token is internally, cryptographically
self-consistent.

**Documented offline limit.** Establishing that the signing certificate
chains to a *trusted* TSA root is a trust-anchor decision that requires the
TSA's published certificate / CA bundle, which is out of band. Without it,
this verifier reports ``trust_anchored = None`` ("not established offline")
rather than pretending a full X.509 path validation happened. Supply the
TSA's certificate via ``--tsa-cert`` to have the signer certificate checked
against it.

Parsing uses ``asn1tools`` (a pure-Python ASN.1 DER codec) with a vendored
RFC-3161 / RFC-5652 (CMS) grammar defined in this module. It never imports
``horizon_ric``.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── OIDs ────────────────────────────────────────────────────────────────
ID_SIGNED_DATA = "1.2.840.113549.1.7.2"
ID_CT_TSTINFO = "1.2.840.113549.1.9.16.1.4"
ID_CONTENTTYPE_ATTR = "1.2.840.113549.1.9.3"
ID_MESSAGEDIGEST_ATTR = "1.2.840.113549.1.9.4"

# hash-algorithm OID → (hashlib factory name, cryptography-hashes attribute)
_HASH_OIDS: dict[str, str] = {
    "1.3.14.3.2.26": "sha1",
    "2.16.840.1.101.3.4.2.1": "sha256",
    "2.16.840.1.101.3.4.2.2": "sha384",
    "2.16.840.1.101.3.4.2.3": "sha512",
}

# signatureAlgorithm OID → (public-key family, digest name) for CMS verify.
_SIG_ALGS: dict[str, tuple[str, str]] = {
    "1.2.840.113549.1.1.1": ("rsa", ""),           # rsaEncryption (digest from SI)
    "1.2.840.113549.1.1.11": ("rsa", "sha256"),
    "1.2.840.113549.1.1.12": ("rsa", "sha384"),
    "1.2.840.113549.1.1.13": ("rsa", "sha512"),
    "1.2.840.113549.1.1.5": ("rsa", "sha1"),
    "1.2.840.10045.4.3.2": ("ecdsa", "sha256"),
    "1.2.840.10045.4.3.3": ("ecdsa", "sha384"),
    "1.2.840.10045.4.3.4": ("ecdsa", "sha512"),
    "1.3.101.112": ("ed25519", ""),               # Ed25519
}

# ── Vendored ASN.1 grammar (RFC 5652 CMS + RFC 3161 TSTInfo) ────────────
_CMS_ASN1 = """
HorizonCms DEFINITIONS IMPLICIT TAGS ::= BEGIN
  AlgorithmIdentifier ::= SEQUENCE { algorithm OBJECT IDENTIFIER, parameters ANY OPTIONAL }
  MessageImprint ::= SEQUENCE { hashAlgorithm AlgorithmIdentifier, hashedMessage OCTET STRING }
  Accuracy ::= SEQUENCE {
    seconds INTEGER OPTIONAL,
    millis [0] IMPLICIT INTEGER OPTIONAL,
    micros [1] IMPLICIT INTEGER OPTIONAL }
  TSTInfo ::= SEQUENCE {
    version INTEGER,
    policy OBJECT IDENTIFIER,
    messageImprint MessageImprint,
    serialNumber INTEGER,
    genTime GeneralizedTime,
    accuracy Accuracy OPTIONAL,
    ordering BOOLEAN DEFAULT FALSE,
    nonce INTEGER OPTIONAL,
    tsa [0] EXPLICIT ANY OPTIONAL,
    extensions [1] IMPLICIT ANY OPTIONAL }
  EncapsulatedContentInfo ::= SEQUENCE {
    eContentType OBJECT IDENTIFIER,
    eContent [0] EXPLICIT OCTET STRING OPTIONAL }
  Attribute ::= SEQUENCE { attrType OBJECT IDENTIFIER, attrValues SET OF ANY }
  IssuerAndSerialNumber ::= SEQUENCE { issuer ANY, serialNumber INTEGER }
  SignerInfo ::= SEQUENCE {
    version INTEGER,
    sid IssuerAndSerialNumber,
    digestAlgorithm AlgorithmIdentifier,
    signedAttrs [0] IMPLICIT SET OF Attribute OPTIONAL,
    signatureAlgorithm AlgorithmIdentifier,
    signature OCTET STRING,
    unsignedAttrs [1] IMPLICIT SET OF Attribute OPTIONAL }
  SignedData ::= SEQUENCE {
    version INTEGER,
    digestAlgorithms SET OF AlgorithmIdentifier,
    encapContentInfo EncapsulatedContentInfo,
    certificates [0] IMPLICIT SET OF ANY OPTIONAL,
    crls [1] IMPLICIT SET OF ANY OPTIONAL,
    signerInfos SET OF SignerInfo }
  ContentInfo ::= SEQUENCE {
    contentType OBJECT IDENTIFIER,
    content [0] EXPLICIT SignedData }
  SignedAttrs ::= SET OF Attribute
  OidValue ::= OBJECT IDENTIFIER
  OctetValue ::= OCTET STRING
END
"""

_spec = None


def _compiled():
    """Compile the vendored ASN.1 grammar once (lazy, keeps import cheap)."""
    global _spec
    if _spec is None:
        import asn1tools

        _spec = asn1tools.compile_string(_CMS_ASN1, "der")
    return _spec


class TimestampParseError(RuntimeError):
    """Raised when an RFC-3161 token cannot be decoded."""


@dataclass(frozen=True)
class ParsedToken:
    imprint_alg_oid: str
    imprint_alg_name: str | None
    hashed_message: bytes
    serial_number: int | None
    gen_time: str | None
    policy_oid: str | None
    nonce: int | None
    tstinfo_der: bytes
    signer_infos: list[dict[str, Any]]
    certificates_der: list[bytes]


def parse_timestamp_token(der: bytes) -> ParsedToken:
    """Decode an RFC-3161 ``TimeStampToken`` (CMS SignedData) DER blob."""
    spec = _compiled()
    try:
        ci = spec.decode("ContentInfo", der)
    except Exception as exc:  # asn1tools raises various types
        raise TimestampParseError(f"could not decode ContentInfo: {exc}") from exc
    if ci.get("contentType") != ID_SIGNED_DATA:
        raise TimestampParseError(
            f"not a CMS SignedData token (contentType={ci.get('contentType')})"
        )
    sd = ci["content"]
    eci = sd["encapContentInfo"]
    if eci.get("eContentType") != ID_CT_TSTINFO:
        raise TimestampParseError(
            f"eContentType is not id-ct-TSTInfo ({eci.get('eContentType')})"
        )
    econtent = eci.get("eContent")
    if econtent is None:
        raise TimestampParseError("token carries no eContent (detached TSTInfo)")
    tst_der = bytes(econtent)
    try:
        tst = spec.decode("TSTInfo", tst_der)
    except Exception as exc:
        raise TimestampParseError(f"could not decode TSTInfo: {exc}") from exc

    imp = tst["messageImprint"]
    alg_oid = imp["hashAlgorithm"]["algorithm"]
    certs = [bytes(c) for c in (sd.get("certificates") or [])]
    return ParsedToken(
        imprint_alg_oid=alg_oid,
        imprint_alg_name=_HASH_OIDS.get(alg_oid),
        hashed_message=bytes(imp["hashedMessage"]),
        serial_number=tst.get("serialNumber"),
        gen_time=_fmt_gentime(tst.get("genTime")),
        policy_oid=tst.get("policy"),
        nonce=tst.get("nonce"),
        tstinfo_der=tst_der,
        signer_infos=list(sd.get("signerInfos") or []),
        certificates_der=certs,
    )


def _fmt_gentime(v: Any) -> str | None:
    if v is None:
        return None
    try:
        formatted: str = v.isoformat()
    except AttributeError:
        return str(v)
    return formatted


@dataclass(frozen=True)
class AnchorResult:
    """Outcome of verifying one RFC-3161 anchor, offline."""

    chain_head_hex: str
    imprint_match: bool
    imprint_alg: str
    # None => could not be evaluated (e.g. token unparsable / no signer cert).
    signature_verified: bool | None = None
    trust_anchored: bool | None = None
    message_digest_ok: bool | None = None
    tsa: str | None = None
    gen_time: str | None = None
    serial_number: int | None = None
    ok: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain_head_hex": self.chain_head_hex,
            "imprint_match": self.imprint_match,
            "imprint_alg": self.imprint_alg,
            "signature_verified": self.signature_verified,
            "message_digest_ok": self.message_digest_ok,
            "trust_anchored": self.trust_anchored,
            "tsa": self.tsa,
            "gen_time": self.gen_time,
            "serial_number": self.serial_number,
            "ok": self.ok,
            "notes": list(self.notes),
        }


def _anchor_fields(anchor: Any) -> tuple[str | None, str | None, str | None]:
    """Return (token_b64, chain_head_hex, tsa) from a dict or TimestampAnchor."""
    if isinstance(anchor, dict):
        token_b64 = anchor.get("token_b64")
        head = anchor.get("chain_head_hex")
        tsa = anchor.get("tsa") or anchor.get("tsa_url")
    else:
        token_b64 = getattr(anchor, "token_b64", None)
        head = getattr(anchor, "chain_head_hex", None)
        tsa = getattr(anchor, "tsa_url", None) or getattr(anchor, "tsa", None)
    return token_b64, head, tsa


def verify_anchor(
    anchor: Any,
    *,
    expected_chain_head_hex: str | None = None,
    tsa_cert_pem: bytes | None = None,
) -> AnchorResult:
    """Verify one RFC-3161 anchor offline.

    ``ok`` is True when the imprint binds the chain head (and, if the token
    carries a signer certificate, the CMS signature verifies). Trust-anchor
    validation is only asserted when ``tsa_cert_pem`` is supplied.
    """
    notes: list[str] = []
    token_b64, head, tsa = _anchor_fields(anchor)
    if not token_b64 or not head:
        return AnchorResult(
            chain_head_hex=head or "",
            imprint_match=False,
            imprint_alg="",
            ok=False,
            notes=["anchor missing token_b64 / chain_head_hex"],
        )
    if expected_chain_head_hex is not None and head != expected_chain_head_hex:
        return AnchorResult(
            chain_head_hex=head,
            imprint_match=False,
            imprint_alg="",
            tsa=tsa,
            ok=False,
            notes=[
                f"anchor chain_head {head} != expected {expected_chain_head_hex}"
            ],
        )

    try:
        token = base64.b64decode(token_b64)
    except Exception as exc:
        return AnchorResult(
            chain_head_hex=head, imprint_match=False, imprint_alg="", tsa=tsa,
            ok=False, notes=[f"token_b64 is not valid base64: {exc}"],
        )

    try:
        parsed = parse_timestamp_token(token)
    except TimestampParseError as exc:
        return AnchorResult(
            chain_head_hex=head, imprint_match=False, imprint_alg="", tsa=tsa,
            ok=False,
            notes=[
                f"token could not be parsed offline: {exc}",
                "imprint binding could not be evaluated",
            ],
        )

    # ── Core check: imprint binds the chain head ────────────────────────
    alg_name = parsed.imprint_alg_name
    if alg_name is None:
        notes.append(
            f"unknown imprint hash OID {parsed.imprint_alg_oid}; cannot recompute"
        )
        imprint_match = False
    else:
        chain_head_bytes = bytes.fromhex(head)
        recomputed = hashlib.new(alg_name, chain_head_bytes).digest()
        imprint_match = recomputed == parsed.hashed_message
        if not imprint_match:
            notes.append(
                f"messageImprint != {alg_name}(chain_head) — token does NOT "
                "cover this chain head"
            )

    # ── Optional: CMS SignerInfo signature self-consistency ─────────────
    sig_verified: bool | None = None
    md_ok: bool | None = None
    if parsed.signer_infos:
        sig_verified, md_ok, sig_notes = _verify_signer_info(parsed)
        notes.extend(sig_notes)
    else:
        notes.append("token carries no SignerInfo; signature not evaluated")

    # ── Optional: trust anchor against a supplied TSA certificate ───────
    trust_anchored: bool | None = None
    if tsa_cert_pem is not None:
        trust_anchored, trust_notes = _check_trust_anchor(parsed, tsa_cert_pem)
        notes.extend(trust_notes)
    else:
        notes.append(
            "trust_anchored=None: TSA root not established offline (supply "
            "--tsa-cert with the TSA's published certificate to establish it)"
        )

    ok = imprint_match and (sig_verified is not False) and (trust_anchored is not False)
    return AnchorResult(
        chain_head_hex=head,
        imprint_match=imprint_match,
        imprint_alg=alg_name or parsed.imprint_alg_oid,
        signature_verified=sig_verified,
        message_digest_ok=md_ok,
        trust_anchored=trust_anchored,
        tsa=tsa,
        gen_time=parsed.gen_time,
        serial_number=parsed.serial_number,
        ok=ok,
        notes=notes,
    )


def _verify_signer_info(parsed: ParsedToken) -> tuple[bool | None, bool | None, list[str]]:
    """Verify the first SignerInfo's signature + messageDigest attribute."""
    from cryptography import x509
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

    notes: list[str] = []
    if not parsed.certificates_der:
        return None, None, ["no signer certificate embedded; CMS signature not verified"]

    si = parsed.signer_infos[0]
    signed_attrs = si.get("signedAttrs")
    if not signed_attrs:
        return None, None, ["SignerInfo has no signedAttrs; CMS signature not verified"]

    spec = _compiled()

    # messageDigest signed-attr must equal digest(eContent).
    md_ok: bool | None = None
    si_digest_oid = si["digestAlgorithm"]["algorithm"]
    di_name = _HASH_OIDS.get(si_digest_oid)
    for attr in signed_attrs:
        if attr.get("attrType") == ID_MESSAGEDIGEST_ATTR and di_name:
            try:
                stored_md = bytes(
                    spec.decode("OctetValue", bytes(attr["attrValues"][0]))
                )
                md_ok = stored_md == hashlib.new(di_name, parsed.tstinfo_der).digest()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"could not evaluate messageDigest attr: {exc}")
    if md_ok is False:
        notes.append("signed messageDigest attribute != digest(eContent)")

    # Reconstruct the signed bytes: DER of the signedAttrs as a universal SET OF.
    try:
        signed_bytes = spec.encode("SignedAttrs", signed_attrs)
    except Exception as exc:  # noqa: BLE001
        return None, md_ok, notes + [f"could not re-encode signedAttrs: {exc}"]

    # Load the signer certificate (first embedded cert; adequate for a
    # single-signer TSA token). A more thorough matcher would select by
    # IssuerAndSerialNumber; single-cert tokens are the common case.
    try:
        cert = x509.load_der_x509_certificate(parsed.certificates_der[0])
    except Exception as exc:  # noqa: BLE001
        return None, md_ok, notes + [f"could not load signer certificate: {exc}"]
    pub = cert.public_key()

    sig_alg_oid = si["signatureAlgorithm"]["algorithm"]
    family, alg_digest = _SIG_ALGS.get(sig_alg_oid, ("", ""))
    digest_name = alg_digest or di_name
    signature = bytes(si["signature"])

    # Dispatch on the ACTUAL key type in the certificate, never on the
    # algorithm the token declares. A token that declares one family and
    # carries a key of another is exactly the confusion an attacker would
    # want us to resolve in the token's favour, so the declared family is
    # recorded as a disagreement note and otherwise ignored.
    if isinstance(pub, ed25519.Ed25519PublicKey):
        actual_family = "ed25519"
    elif isinstance(pub, ec.EllipticCurvePublicKey):
        actual_family = "ecdsa"
    elif isinstance(pub, rsa.RSAPublicKey):
        actual_family = "rsa"
    else:
        return None, md_ok, notes + [
            f"unsupported signer key type {type(pub).__name__}"
        ]
    if family and family != actual_family:
        notes.append(
            f"signatureAlgorithm declares {family!r} but the signer key is "
            f"{actual_family!r}; verified against the key, not the declaration"
        )

    try:
        if isinstance(pub, ed25519.Ed25519PublicKey):
            pub.verify(signature, signed_bytes)
        elif isinstance(pub, ec.EllipticCurvePublicKey):
            pub.verify(signature, signed_bytes, ec.ECDSA(_hashes(digest_name, hashes)))
        else:
            pub.verify(
                signature, signed_bytes, padding.PKCS1v15(),
                _hashes(digest_name, hashes),
            )
    except InvalidSignature:
        notes.append("CMS SignerInfo signature is INVALID")
        return False, md_ok, notes
    except Exception as exc:  # noqa: BLE001
        return None, md_ok, notes + [f"could not verify CMS signature: {exc}"]

    notes.append("CMS SignerInfo signature verified against embedded certificate")
    return True, md_ok, notes


def _hashes(name: str | None, hashes_mod):
    mapping = {
        "sha1": hashes_mod.SHA1,
        "sha256": hashes_mod.SHA256,
        "sha384": hashes_mod.SHA384,
        "sha512": hashes_mod.SHA512,
    }
    factory = mapping.get(name or "sha256", hashes_mod.SHA256)
    return factory()


def _check_trust_anchor(parsed: ParsedToken, tsa_cert_pem: bytes) -> tuple[bool | None, list[str]]:
    """Check the embedded signer cert equals / is issued by the supplied cert."""
    from cryptography import x509
    from cryptography.exceptions import InvalidSignature

    notes: list[str] = []
    if not parsed.certificates_der:
        return None, ["no embedded signer certificate to match against --tsa-cert"]
    try:
        trusted = x509.load_pem_x509_certificate(tsa_cert_pem)
        signer = x509.load_der_x509_certificate(parsed.certificates_der[0])
    except Exception as exc:  # noqa: BLE001
        return None, [f"could not load certificates for trust check: {exc}"]

    if signer.fingerprint(_sha256_hash()) == trusted.fingerprint(_sha256_hash()):
        notes.append("signer certificate equals supplied --tsa-cert")
        return True, notes
    # Otherwise check the signer cert is issued (signed) by the supplied cert.
    try:
        _verify_issuance(signer, trusted)
    except InvalidSignature:
        notes.append("signer certificate is NOT issued by supplied --tsa-cert")
        return False, notes
    except Exception as exc:  # noqa: BLE001
        return None, [f"could not verify issuance against --tsa-cert: {exc}"]
    notes.append("signer certificate is issued by supplied --tsa-cert")
    return True, notes


def _sha256_hash():
    from cryptography.hazmat.primitives import hashes

    return hashes.SHA256()


def _verify_issuance(signer: Any, issuer: Any) -> None:
    """Verify ``signer`` was signed by ``issuer``; raise otherwise.

    Dispatches on the issuer's actual key type. An issuer key of a family we
    cannot use to verify an X.509 signature (a key-agreement key, say) is a
    hard error, not a silent pass.
    """
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

    pub = issuer.public_key()
    if isinstance(pub, ed25519.Ed25519PublicKey):
        pub.verify(signer.signature, signer.tbs_certificate_bytes)
    elif isinstance(pub, ec.EllipticCurvePublicKey):
        pub.verify(
            signer.signature,
            signer.tbs_certificate_bytes,
            ec.ECDSA(signer.signature_hash_algorithm),
        )
    elif isinstance(pub, rsa.RSAPublicKey):
        pub.verify(
            signer.signature,
            signer.tbs_certificate_bytes,
            padding.PKCS1v15(),
            signer.signature_hash_algorithm,
        )
    else:
        raise TypeError(
            f"--tsa-cert public key type {type(pub).__name__} cannot verify an "
            "X.509 signature"
        )


def load_anchor(path: str | Path) -> dict[str, Any]:
    """Load an anchor JSON file (a single anchor object)."""
    import json

    loaded: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return loaded


__all__ = [
    "AnchorResult",
    "ParsedToken",
    "TimestampParseError",
    "load_anchor",
    "parse_timestamp_token",
    "verify_anchor",
]
