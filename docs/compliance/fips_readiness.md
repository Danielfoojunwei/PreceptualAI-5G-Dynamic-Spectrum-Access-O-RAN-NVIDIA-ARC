# FIPS 140-3 Readiness Disclosure

> See [`../../README.md`](../../README.md) for current state, performance, tests, and roadmap.


**Version:** 0.1.0
**Date:** 2026-05-07
**Audience:** regulated-industry compliance teams and security architects evaluating Horizon-RIC against FIPS-140-required workloads.
**Companion docs:** [`docs/compliance/nist_csf.md`](nist_csf.md), [`docs/THREAT_MODEL.md`](../THREAT_MODEL.md).

This document is the honest answer to the question *"Is Horizon-RIC FIPS 140-3 validated?"* It is intentionally precise about what is true today, what is true when the operator deploys onto a FIPS-mode host, and what would be required to claim a module-level CMVP certificate.

We will not claim a validation we do not have.

---

## 1. Executive classification

Horizon-RIC has **no FIPS 140-3 module-level certificate** today. We are not on the [CMVP active-validation list](https://csrc.nist.gov/projects/cryptographic-module-validation-program/validated-modules) and have not begun a CMVP submission.

What Horizon-RIC *does* offer:

1. **Transitive FIPS inheritance on a FIPS-mode host.** Horizon-RIC's cryptographic primitives are sourced from CPython's `ssl` module (against the host OpenSSL) and the `cryptography` package (which uses OpenSSL via `cffi`). On a FIPS-mode RHEL 9 / Ubuntu 22.04 LTS host where the OpenSSL provider is itself a CMVP-validated module (e.g. Red Hat's certificate #4746 against OpenSSL 3.0.7), Horizon-RIC's TLS, hashing, and JWT signing operations execute against the validated provider. The application boundary is not validated, but the cryptography boundary is.
2. **A documented cryptographic-primitive inventory.** §2 below lists every primitive Horizon-RIC uses, with `file:line` citations to the source. An auditor can verify the claim by `grep`. There are no hidden primitives.
3. **Explicit out-of-boundary callouts.** §3 lists the libraries that touch crypto-adjacent surfaces but are *not* covered by the OpenSSL inheritance. These are the boundaries an FedRAMP-Moderate auditor will flag. We flag them ourselves.

A 6-week path to module-level validation is documented in §4.

---

## 2. Cryptographic primitive inventory (auditable)

Every entry below cites the exact file and line where the primitive is invoked. The "FIPS-Approved mapping" column references NIST SP 800-140C (approved security functions) and SP 800-140D (approved sensitive security parameter generation).

| # | Primitive | Used for | Source citation | FIPS-Approved (SP 800-140C/D) |
|---|---|---|---|---|
| 1 | TLS 1.3 — `TLS_AES_256_GCM_SHA384` | R1 / A1 transport | `src/horizon_ric/rapp/auth.py:35` | AES-256-GCM (SP 800-38D); SHA-384 (FIPS 180-4) |
| 2 | TLS 1.3 — `TLS_CHACHA20_POLY1305_SHA256` | R1 / A1 transport (fallback) | `src/horizon_ric/rapp/auth.py:36` | **NOT FIPS-Approved** (ChaCha20-Poly1305 is not in SP 800-140C). Operator MUST set `cipher_allowlist=["TLS_AES_256_GCM_SHA384", "TLS_AES_128_GCM_SHA256"]` to stay in-policy. |
| 3 | TLS 1.3 — `TLS_AES_128_GCM_SHA256` | R1 / A1 transport (fallback) | `src/horizon_ric/rapp/auth.py:37` | AES-128-GCM (SP 800-38D); SHA-256 (FIPS 180-4) |
| 4 | mTLS X.509 client cert + RSA key | R1 / A1 client authentication | `src/horizon_ric/rapp/auth.py:146-149` (`SSLContext.load_cert_chain`) | RSA (FIPS 186-5 §5); SHA-256 sig (FIPS 180-4) |
| 5 | JWT **RS256** (RSASSA-PKCS1-v1_5 + SHA-256) | rApp dashboard / SDK auth | `src/horizon_ric/security/jwt.py:69` (`_DEFAULT_ALG = "RS256"`); minted at `src/horizon_ric/security/jwt.py:172-176`; verified at `src/horizon_ric/security/jwt.py:208-214` | RSA-PKCS1-v1_5 with SHA-256: FIPS 186-5 §5.4 (signature generation), FIPS 186-5 §5.5 (verification). **Approved.** |
| 6 | RSA private-key load (PEM, PKCS#8) | JWT signing key bootstrap | `src/horizon_ric/security/jwt.py:86` (`serialization.load_pem_private_key`) | Key import — covered by SP 800-140D §6.1 (CSP entry). |
| 7 | RSA public-key derivation | JWT verification key publication | `src/horizon_ric/security/jwt.py:91-99` | FIPS 186-5 §A.1 (key generation parameters); we do not generate, we derive from the private key. |
| 8 | SHA-256 hash chain | DecisionRecord tamper-evidence | `src/horizon_ric/evidence/store.py:23,76` (`hashlib.sha256`); chain construction `src/horizon_ric/evidence/store.py:13-18` | SHA-256 — FIPS 180-4 §6.2. **Approved.** |
| 9 | SHA-256 over training-corpus manifest | TS 28.105 model-card evidence | `src/horizon_ric/observability/model_card.py:25,129` | SHA-256 — FIPS 180-4 §6.2. **Approved.** |
| 10 | HMAC-SHA-256 | RAaS rApp-archive signature | `src/horizon_ric/integrations/raas.py:342,347` (`hashlib.sha256` keyed) | HMAC — FIPS 198-1; SHA-256 — FIPS 180-4. **Approved.** |
| 11 | RFC 3161 timestamp signatures | Audit chain anchoring | `src/horizon_ric/evidence/rfc3161.py:47-52` (uses `rfc3161-client`) | RSA-PKCS1-v1_5 + SHA-256 inside the TSA's response. **Approved** (TSA-side); Horizon-RIC only verifies, it does not generate. |

**What's *not* in the inventory** (intentional negative claim): no MD5, no SHA-1, no DES/3DES, no RC4, no static-key TLS, no client-side random number generation outside `os.urandom` (which on a FIPS-mode kernel is `getrandom(2)` against the validated DRBG).

---

## 3. Boundaries that are explicitly NOT FIPS today

The transitive-inheritance argument in §1 covers OpenSSL and `cryptography`. It does **not** cover the following surfaces. An FedRAMP / DoD auditor will (correctly) flag each.

### 3.1 `python-jose` for JWT codec

`src/horizon_ric/security/jwt.py:63` imports `from jose import jwt as jose_jwt`. While the underlying RSA + SHA-256 primitives are routed through `cryptography` (which inherits OpenSSL FIPS), the *codec* — base64url, JSON canonicalisation, JWS header construction — is python-jose's own pure-Python code path. python-jose is **not** CMVP-validated. Risk: the codec layer is non-cryptographic (no key material flows through it that doesn't also flow through `cryptography`), so the FIPS exposure is "module-level boundary expansion" rather than "weak crypto", but a strict auditor will still call it out.

**Mitigation (Phase 2):** swap `python-jose` for `pyjwt` configured to use `cryptography`'s backend exclusively, OR switch to direct `cryptography.hazmat.primitives.asymmetric.padding.PKCS1v15` calls and serialize the JWS ourselves (~120 LOC, tested by `tests/test_security_jwt_rotation.py`).

### 3.2 `pybreaker.CircuitBreakerListener` callbacks

`src/horizon_ric/runtime/circuit_breaker.py:61-133` registers a `_BreakerLogger` listener that fires on `state_change`, `failure`, and `success`. The listener itself is non-cryptographic, but it **logs structured events containing tenant identifiers and request URLs**. If those identifiers are themselves keys/tokens (they are not, by design — see `tests/test_circuit_breaker.py`), the listener path would constitute a CSP-disclosure boundary. We assert here for the auditor's record: no key material, no JWT, no client secret ever transits the breaker listener path.

### 3.3 `httpx` request-ID generation

`httpx.AsyncClient` (used at `src/horizon_ric/rapp/auth.py:285`) generates per-request UUIDv4 IDs internally for retries and trace headers. CPython's `uuid.uuid4` uses `os.urandom`, which on FIPS-mode kernels is the validated DRBG; on a non-FIPS host it is a non-validated PRNG. Since the request IDs are not used as security parameters (they are observability identifiers only), this is informational rather than blocking — but listed for completeness.

### 3.4 RFC 3161 client (`rfc3161-client`)

`src/horizon_ric/evidence/rfc3161.py:47-52`. The `rfc3161-client` package is a thin CMS/ASN.1 wrapper that delegates the actual signature verification to `cryptography`. The wrapper code itself has not been reviewed against FIPS boundaries. Treat the same way as python-jose: codec layer outside the validated module.

---

## 4. Path to module-level FIPS 140-3 validation (6-week effort)

The cheapest path that produces a defensible "FIPS-Inside" claim, without seeking our own CMVP certificate:

| Week | Activity | Owner | Artifact |
|---|---|---|---|
| 1 | Vendor `cryptography` 42.x at a pinned version known to consume `OPENSSL_FIPS=1` | platform | `pyproject.toml` constraint, `deploy/sbom/horizon-ric-sbom.json` regenerated |
| 2 | Replace `python-jose` with `pyjwt` configured for `cryptography` backend; update `tests/test_security_jwt_rotation.py` | security | unified diff on `src/horizon_ric/security/jwt.py` |
| 3 | Audit `rfc3161-client` against FIPS provider — fall back to `cryptography.x509` direct verification if any non-Approved primitive is invoked | security | `tests/test_audit_rfc3161_anchor.py` extended with FIPS-mode CI matrix |
| 4 | CI lane: build container against `registry.access.redhat.com/ubi9/ubi-minimal:latest` with `crypto-policies-scripts` set to `FIPS`, run full test suite | infra | `.github/workflows/fips-mode.yml` |
| 5 | Document operator-side prerequisites: kernel boot with `fips=1`, OpenSSL 3.0.7 FIPS provider loaded, `crypto-policies-scripts` = `FIPS` | docs | `docs/compliance/fips_operator_runbook.md` |
| 6 | External crypto review (1 reviewer-week, specialist) — sign-off letter on the inventory in §2 and the boundary list in §3 | external | signed PDF, retained with the conformance dossier |

This path produces **"FIPS-Inside" deployability**, not a CMVP certificate. A full Horizon-RIC-as-a-cryptographic-module CMVP submission is a separate, substantial engagement (industry estimates are on the order of 18 months) that is out of scope for the current programme. It remains a later-phase roadmap item.

---

## 5. Auditable artefacts (what to grep before signing the SOC)

For a compliance reviewer who wants to verify this document's claims in five minutes:

```bash
# Cipher allow-list (entry 1-3 of inventory)
grep -n "_DEFAULT_WG11_CIPHERS" src/horizon_ric/rapp/auth.py

# JWT algorithm pin (entry 5)
grep -n "_DEFAULT_ALG" src/horizon_ric/security/jwt.py

# Hash chain primitive (entry 8)
grep -n "hashlib.sha256" src/horizon_ric/evidence/store.py

# Negative-claim verification: no MD5, SHA-1, DES, RC4 anywhere
grep -RnE "md5|sha1|\bdes\b|rc4|hashlib\.md5|hashlib\.sha1" src/horizon_ric/ \
  --include="*.py" | grep -v "test_\|sha1_signature_disabled"
```

The last command should return zero matches for `md5|hashlib.md5|hashlib.sha1|rc4`. SHA-1 strings appearing in *test* files (asserting we *don't* use SHA-1) are expected and listed by name in the exclusion above.

---

## 6. What this document is not

- It is **not** a CMVP certificate.
- It is **not** a substitute for the operator's own ATO process.
- It is **not** a guarantee that running on a FIPS-mode RHEL host is sufficient for FedRAMP-Moderate (FedRAMP additionally requires ATO boundary documentation, continuous monitoring, and the SAR/SAP cycle — out of scope here).

It **is** the honest, line-cited inventory an evaluator can put in front of their security engineer to decide whether Horizon-RIC clears the bar for *their* deployment posture today, and what closing the residual gap would require.
