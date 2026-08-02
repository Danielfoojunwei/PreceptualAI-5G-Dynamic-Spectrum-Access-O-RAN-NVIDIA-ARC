"""Tests for the STANDALONE offline evidence verifier (WP4 deliverable 1).

Run::

    PYTHONPATH=audit/src:src /home/user/venv/bin/python -m pytest \
        audit/tests/test_offline_verifier.py -q

``src`` is on the path ONLY so these tests can drive the real
``horizon_ric`` product to build genuine fixtures (a real JSONL evidence
store, a real Ed25519-signed certificate). The verifier under test
(``horizon_audit`` / ``audit/verify_evidence.py``) must NEVER import
``horizon_ric`` — the purity test below proves it does not.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

# ── horizon_audit (the code under test) ─────────────────────────────────
from horizon_audit import certificate as cert_mod
from horizon_audit import timestamp as ts_mod
from horizon_audit.chain import ZERO_HASH_HEX, canonical_json, chain_hash, verify_chain

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_CLI = REPO_ROOT / "audit" / "verify_evidence.py"
AUDIT_SRC = REPO_ROOT / "audit" / "src"
RIC_SRC = REPO_ROOT / "src"
SAMPLE_RECORD = REPO_ROOT / "deploy" / "xapp-e2e" / "results" / "sample-decision-record.json"


# ── fixture builders that DO drive horizon_ric ──────────────────────────
def _make_record(decision_id: str, tenant_id: str, seed: int):
    from horizon_ric.evidence.schema import (
        DecisionRecord,
        ModelVersions,
        PredictedOutcome,
    )

    return DecisionRecord.new(
        decision_id=decision_id,
        tenant_id=tenant_id,
        rapp_instance_id="horizon-ric-rapp",
        state_hash=hashlib.sha256(decision_id.encode()).hexdigest(),
        chosen_action={"policy_type": "horizon.admission.control", "seed": seed},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.1, sla_risk_1min=0.2, sla_risk_5min=0.3
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="none",
            risk_heads="risk_rules_v1",
            dyna="none",
            policy="risk_band_rules_v1",
            constraint_layer="shield_terrestrial_v1",
            rapp="0.2.0",
        ),
        random_seed=seed,
    )


def _build_store(path: Path, specs):
    """Append (decision_id, tenant_id, seed) records to a real JSONL store."""
    from horizon_ric.evidence.store import JsonlEvidenceStore

    store = JsonlEvidenceStore(path)
    for did, tid, seed in specs:
        store.append(_make_record(did, tid, seed))
    return store


# ════════════════════════════════════════════════════════════════════════
# 1. THE EQUIVALENCE GATE — standalone verifier agrees with EvidenceStore
# ════════════════════════════════════════════════════════════════════════
def test_intact_chain_equivalence(tmp_path):
    """Intact multi-tenant chain: standalone == EvidenceStore.verify()."""
    from horizon_ric.evidence.store import JsonlEvidenceStore

    path = tmp_path / "evidence.jsonl"
    _build_store(
        path,
        [
            ("d0", "tenant-a", 1),
            ("d1", "tenant-b", 2),
            ("d2", "tenant-a", 3),
            ("d3", "tenant-b", 4),
            ("d4", "tenant-a", 5),
        ],
    )

    reference = JsonlEvidenceStore(path).verify()  # -1 when intact
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    standalone = verify_chain(lines)

    assert reference == -1
    assert standalone.intact is True
    assert standalone.first_broken_index == reference  # both -1
    assert standalone.tenant_count == 2


@pytest.mark.parametrize("tamper_line", [0, 1, 2, 3, 4])
def test_tampered_chain_equivalence(tmp_path, tamper_line):
    """Tampering any line: standalone finds the SAME first-broken index."""
    from horizon_ric.evidence.store import JsonlEvidenceStore

    path = tmp_path / "evidence.jsonl"
    _build_store(
        path,
        [
            ("d0", "tenant-a", 1),
            ("d1", "tenant-b", 2),
            ("d2", "tenant-a", 3),
            ("d3", "tenant-b", 4),
            ("d4", "tenant-a", 5),
        ],
    )

    # Tamper: mutate the stored record body WITHOUT recomputing its hash.
    raw_lines = [x for x in path.read_text().splitlines() if x.strip()]
    obj = json.loads(raw_lines[tamper_line])
    obj["record"]["chosen_action"]["seed"] = 999999
    raw_lines[tamper_line] = json.dumps(obj, sort_keys=True)
    path.write_text("\n".join(raw_lines) + "\n")

    reference = JsonlEvidenceStore(path).verify()
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    standalone = verify_chain(lines)

    assert reference != -1, "reference should detect the tamper"
    assert standalone.intact is False
    assert standalone.first_broken_index == reference, (
        f"standalone first-broken {standalone.first_broken_index} != "
        f"reference {reference}"
    )


def test_single_tenant_unscoped_equivalence(tmp_path):
    """A store with no explicit tenant lands in the _unscoped_ chain and the
    standalone verifier agrees on both intact and tampered."""
    from horizon_ric.evidence.schema import (
        DecisionRecord,
        ModelVersions,
        PredictedOutcome,
    )
    from horizon_ric.evidence.store import JsonlEvidenceStore

    path = tmp_path / "evidence.jsonl"
    store = JsonlEvidenceStore(path)
    for i in range(4):
        rec = DecisionRecord.new(
            decision_id=f"u{i}",
            rapp_instance_id="r",
            state_hash="00",
            chosen_action={"n": i},
            predicted_outcome_chosen=PredictedOutcome(
                sla_risk_30s=0.0, sla_risk_1min=0.0, sla_risk_5min=0.0
            ),
            rejected_alternatives=[],
            model_versions=ModelVersions(
                encoder="none", risk_heads="r", dyna="none", policy="p",
                constraint_layer="c", rapp="0.2.0",
            ),
        )
        store.append(rec)

    assert JsonlEvidenceStore(path).verify() == -1
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    res = verify_chain(lines)
    assert res.intact
    assert res.tenant_count == 1
    assert res.tenants[0].tenant_id == "_unscoped_"

    # Tamper line 2.
    lines[2]["record"]["chosen_action"]["n"] = -1
    path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in lines) + "\n")
    ref = JsonlEvidenceStore(path).verify()
    lines2 = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert verify_chain(lines2).first_broken_index == ref == 2


# ════════════════════════════════════════════════════════════════════════
# 2. FALSIFICATION — a deliberately broken recompute FAILS the gate
# ════════════════════════════════════════════════════════════════════════
def test_falsification_broken_recompute_disagrees(tmp_path):
    """Prove the equivalence gate has teeth: a verifier that canonicalises
    the record WRONG (space separators instead of compact) disagrees with
    the reference EvidenceStore.verify() on an intact chain — i.e. the gate
    would catch a broken standalone implementation."""
    from horizon_ric.evidence.store import JsonlEvidenceStore

    path = tmp_path / "evidence.jsonl"
    _build_store(path, [("d0", "t", 1), ("d1", "t", 2), ("d2", "t", 3)])
    reference = JsonlEvidenceStore(path).verify()
    assert reference == -1

    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def _broken_first_index(lines):
        prev = ZERO_HASH_HEX
        for i, obj in enumerate(lines):
            # WRONG canonicalisation: default separators (", ", ": ").
            payload = json.dumps(obj["record"], sort_keys=True)  # note: spaces
            h = hashlib.sha256(bytes.fromhex(prev) + payload.encode()).hexdigest()
            if h != obj["hash"]:
                return i
            prev = obj["hash"]
        return -1

    broken = _broken_first_index(lines)
    # The correct verifier agrees with reference (intact); the broken one does
    # NOT — it reports a false break at line 0. The gate distinguishes them.
    assert verify_chain(lines).first_broken_index == reference  # correct: -1
    assert broken != reference, "broken recompute must disagree with reference"
    assert broken == 0


# ════════════════════════════════════════════════════════════════════════
# 3. PURITY — the standalone verifier never pulls in horizon_ric
# ════════════════════════════════════════════════════════════════════════
def test_verifier_does_not_import_horizon_ric():
    """Import every standalone module (with horizon_ric AVAILABLE on the
    path) and assert none pulled a horizon_ric module into sys.modules."""
    probe = (
        "import sys\n"
        "import horizon_audit\n"
        "import horizon_audit.chain, horizon_audit.certificate, horizon_audit.timestamp\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('verify_evidence', r'{VERIFY_CLI}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        # Exercise real code paths so lazy imports (asn1tools/cryptography) fire.
        "horizon_audit.chain.verify_chain([])\n"
        "leaked = sorted(k for k in sys.modules if k == 'horizon_ric' or k.startswith('horizon_ric.'))\n"
        "print('LEAKED:' + ','.join(leaked))\n"
        "sys.exit(1 if leaked else 0)\n"
    )
    env = {
        "PYTHONPATH": f"{AUDIT_SRC}:{RIC_SRC}",  # horizon_ric IS importable here
        "PATH": "/usr/bin:/bin",
    }
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        f"standalone verifier pulled in horizon_ric: {proc.stdout}{proc.stderr}"
    )
    assert "LEAKED:" in proc.stdout and proc.stdout.strip().endswith("LEAKED:")


# ════════════════════════════════════════════════════════════════════════
# 4. Ed25519 CERTIFICATE — standalone agrees with horizon_ric.shield.signing
# ════════════════════════════════════════════════════════════════════════
def test_certificate_equivalence_valid_and_tampered():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from horizon_ric.shield.certificate import SafetyCertificate
    from horizon_ric.shield.signing import (
        signed_certificate,
    )
    from horizon_ric.shield.signing import (
        verify_certificate as ric_verify,
    )

    key = Ed25519PrivateKey.generate()
    cert = SafetyCertificate(
        decision_id="dcert",
        issued_at="2026-08-02T00:00:00+00:00",
        loop_tier="non_rt",
        block="risk_band",
        action_proposed={"tx_power_dBm": 30.0},
        action_safe={"tx_power_dBm": 24.0},
        invariants=[],
        corrections=[],
        violated_ids=[],
        projected=True,
        fallback_used=False,
        fallback_to=None,
        safe=True,
        emit_blocked=False,
    )
    signed = signed_certificate(cert, key)
    cert_dict = signed.to_dict()
    pub = key.public_key()

    # Valid: both verifiers agree True.
    assert ric_verify(signed, pub) is True
    assert cert_mod.verify_certificate(cert_dict, pub) is True
    assert cert_mod.fingerprint_matches(cert_dict, pub) is True

    # Standalone canonical bytes match the reference signer's bytes exactly.
    from horizon_ric.shield.signing import canonical_certificate_bytes as ric_bytes

    assert cert_mod.canonical_certificate_bytes(cert_dict) == ric_bytes(signed)

    # Tampered: flip a signed field; both verifiers agree False.
    tampered = dict(cert_dict)
    tampered["action_safe"] = {"tx_power_dBm": 99.0}
    assert cert_mod.verify_certificate(tampered, pub) is False


def test_sample_record_has_no_full_embedded_certificate():
    """The committed real DecisionRecord carries only the certificate DIGEST
    subset (safe/projected/violated_ids), not a full signed certificate — the
    verifier reports that honestly rather than failing."""
    record = json.loads(SAMPLE_RECORD.read_text())
    embedded = cert_mod.find_embedded_certificates(record)
    # The chosen_action.certificate subset has no 'signature' -> not found as
    # certificate-like at all here (signing was not configured for this run).
    full = [e for e in embedded if e.full]
    assert full == [], "sample record should not contain a full signed certificate"


# ════════════════════════════════════════════════════════════════════════
# 5. RFC-3161 ANCHOR — real DER, real signature, verified fully offline
# ════════════════════════════════════════════════════════════════════════
def _mint_token(chain_head_hex: str, *, tamper_sig: bool = False):
    """Mint a REAL RFC-3161 TimeStampToken signed by a self-issued RSA TSA.

    Uses the verifier's own vendored ASN.1 grammar for encode/decode symmetry
    and the ``cryptography`` package for a genuine RSA signature. This is real
    crypto over real DER — only the TSA is self-issued (there is no network
    to reach freetsa in this environment)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.x509.oid import NameOID

    spec = ts_mod._compiled()
    OID_SHA512 = "2.16.840.1.101.3.4.2.3"
    OID_SHA512_RSA = "1.2.840.113549.1.1.13"

    chain_head = bytes.fromhex(chain_head_hex)
    imprint = hashlib.sha512(chain_head).digest()  # rfc3161-client default alg
    tstinfo = {
        "version": 1,
        "policy": "1.3.6.1.4.1.4146.2.3",
        "messageImprint": {
            "hashAlgorithm": {"algorithm": OID_SHA512, "parameters": b"\x05\x00"},
            "hashedMessage": imprint,
        },
        "serialNumber": 4242,
        "genTime": datetime.datetime(2026, 8, 2, 12, 0, 0, tzinfo=datetime.timezone.utc),
        "nonce": 987654321,
    }
    tstinfo_der = spec.encode("TSTInfo", tstinfo)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Horizon Test TSA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1234)
        .not_valid_before(datetime.datetime(2026, 1, 1))
        .not_valid_after(datetime.datetime(2035, 1, 1))
        .sign(key, hashes.SHA256())
    )
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)

    md = hashlib.sha512(tstinfo_der).digest()
    attrs = [
        {"attrType": ts_mod.ID_CONTENTTYPE_ATTR,
         "attrValues": [spec.encode("OidValue", ts_mod.ID_CT_TSTINFO)]},
        {"attrType": ts_mod.ID_MESSAGEDIGEST_ATTR,
         "attrValues": [spec.encode("OctetValue", md)]},
    ]
    signed_attrs_der = spec.encode("SignedAttrs", attrs)
    signature = key.sign(signed_attrs_der, padding.PKCS1v15(), hashes.SHA512())
    if tamper_sig:
        signature = bytes([signature[0] ^ 0xFF]) + signature[1:]

    signer = {
        "version": 1,
        "sid": {"issuer": cert.issuer.public_bytes(), "serialNumber": 1234},
        "digestAlgorithm": {"algorithm": OID_SHA512, "parameters": b"\x05\x00"},
        "signedAttrs": attrs,
        "signatureAlgorithm": {"algorithm": OID_SHA512_RSA, "parameters": b"\x05\x00"},
        "signature": signature,
    }
    signed_data = {
        "version": 3,
        "digestAlgorithms": [{"algorithm": OID_SHA512, "parameters": b"\x05\x00"}],
        "encapContentInfo": {"eContentType": ts_mod.ID_CT_TSTINFO, "eContent": tstinfo_der},
        "certificates": [cert_der],
        "signerInfos": [signer],
    }
    token = spec.encode("ContentInfo", {"contentType": ts_mod.ID_SIGNED_DATA, "content": signed_data})
    anchor = {
        "type": "rfc3161_anchor",
        "tsa": "https://freetsa.org/tsr",
        "token_b64": base64.b64encode(token).decode("ascii"),
        "chain_head_hex": chain_head_hex,
        "gen_time_utc": "2026-08-02T12:00:00+00:00",
        "anchored_at_unix": 0.0,
    }
    return anchor, cert_pem


def test_anchor_offline_imprint_and_signature():
    head = hashlib.sha256(b"the-chain-head").hexdigest()
    anchor, cert_pem = _mint_token(head)

    res = ts_mod.verify_anchor(anchor)
    assert res.imprint_match is True
    assert res.imprint_alg == "sha512"
    assert res.signature_verified is True
    assert res.message_digest_ok is True
    assert res.trust_anchored is None  # not established without --tsa-cert
    assert res.ok is True

    # With the TSA cert supplied, trust is established.
    res2 = ts_mod.verify_anchor(anchor, tsa_cert_pem=cert_pem)
    assert res2.trust_anchored is True
    assert res2.ok is True


def test_anchor_rejects_wrong_chain_head():
    head = hashlib.sha256(b"the-chain-head").hexdigest()
    anchor, _ = _mint_token(head)
    # Claim the token covers a DIFFERENT head than it really does.
    anchor["chain_head_hex"] = hashlib.sha256(b"a-different-head").hexdigest()
    res = ts_mod.verify_anchor(anchor)
    assert res.imprint_match is False
    assert res.ok is False


def test_anchor_rejects_tampered_signature():
    head = hashlib.sha256(b"the-chain-head").hexdigest()
    anchor, _ = _mint_token(head, tamper_sig=True)
    res = ts_mod.verify_anchor(anchor)
    assert res.imprint_match is True  # imprint still binds
    assert res.signature_verified is False  # but the signature is broken
    assert res.ok is False


# ════════════════════════════════════════════════════════════════════════
# 6. CLI — exit codes and JSON contract
# ════════════════════════════════════════════════════════════════════════
def _run_cli(*args):
    env = {"PYTHONPATH": str(AUDIT_SRC), "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [sys.executable, str(VERIFY_CLI), *args],
        capture_output=True, text=True, env=env,
    )


def test_cli_intact_exit_zero(tmp_path):
    path = tmp_path / "evidence.jsonl"
    _build_store(path, [("d0", "t", 1), ("d1", "t", 2)])
    proc = _run_cli(str(path), "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["result"] == "PASS"
    assert report["chain"]["intact"] is True


def test_cli_tampered_exit_nonzero(tmp_path):
    path = tmp_path / "evidence.jsonl"
    _build_store(path, [("d0", "t", 1), ("d1", "t", 2), ("d2", "t", 3)])
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    lines[1]["record"]["chosen_action"]["seed"] = 42
    path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in lines) + "\n")
    proc = _run_cli(str(path), "--json")
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["result"] == "FAIL"
    assert report["chain"]["first_broken_index"] == 1


def test_cli_anchor_end_to_end(tmp_path):
    path = tmp_path / "evidence.jsonl"
    _build_store(path, [("d0", "t", 1)])
    # Anchor over the actual final stored hash of the export.
    last_hash = json.loads(path.read_text().splitlines()[-1])["hash"]
    anchor, _ = _mint_token(last_hash)
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text(json.dumps(anchor))
    proc = _run_cli(str(path), "--anchor", str(anchor_path), "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["anchor"]["imprint_match"] is True
    assert report["anchor"]["chain_head_present_in_export"] is True


def test_canonical_json_matches_reference():
    """Sanity: our canonical_json equals the store's separators/sort rule."""
    obj = {"b": 1, "a": [3, 2, 1], "c": {"z": 0, "y": 9}}
    assert canonical_json(obj) == json.dumps(obj, sort_keys=True, separators=(",", ":"))
    assert chain_hash(ZERO_HASH_HEX, obj) == hashlib.sha256(
        bytes.fromhex(ZERO_HASH_HEX) + canonical_json(obj).encode()
    ).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
