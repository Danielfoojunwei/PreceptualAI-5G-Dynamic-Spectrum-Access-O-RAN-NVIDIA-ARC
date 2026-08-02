"""horizon_audit — a STANDALONE, offline evidence-chain verifier.

This package lets a spectrum authority / regulator verify a Horizon RIC
evidence export *without* installing the ``horizon_ric`` product. It imports
only the Python standard library, the ``cryptography`` package (Ed25519 /
RFC-3161 signature checks) and — for RFC-3161 token parsing only —
``asn1tools`` (a pure-Python ASN.1 codec). It NEVER imports ``horizon_ric``.

Three independent checks are provided:

* :mod:`horizon_audit.chain`        — per-tenant SHA-256 hash-chain integrity.
* :mod:`horizon_audit.certificate`  — Ed25519 SafetyCertificate signatures.
* :mod:`horizon_audit.timestamp`    — offline RFC-3161 anchor imprint binding.

The reference implementation these agree with byte-for-byte lives in
``horizon_ric.evidence.store`` / ``horizon_ric.shield.signing`` /
``horizon_ric.evidence.rfc3161``, but a regulator does not need it: the
formats are reproduced here from first principles and pinned by an
equivalence test (``audit/tests/test_offline_verifier.py``).
"""

from __future__ import annotations

from horizon_audit.chain import (
    ZERO_HASH_HEX,
    ChainResult,
    TenantChainResult,
    canonical_json,
    chain_hash,
    verify_chain,
    verify_jsonl_file,
)

__all__ = [
    "ChainResult",
    "TenantChainResult",
    "ZERO_HASH_HEX",
    "canonical_json",
    "chain_hash",
    "verify_chain",
    "verify_jsonl_file",
]

__version__ = "1.0.0"
