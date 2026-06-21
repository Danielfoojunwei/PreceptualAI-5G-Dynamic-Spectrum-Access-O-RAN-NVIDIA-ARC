# HSM Key Custody for Federated Aggregation

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


**Scope.** Horizon-RIC's secure aggregation path (`ShamirSecretSharing` /
`secure_mean` in `src/horizon_ric/federated/secure.py`, Shamir secret-sharing
over GF(2¹²⁷ − 1)) and its model-provenance signing path
(`src/horizon_ric/provenance/signing.py`, `sign_model(...)`) require that the
signing key never live in process memory in production. This document covers
the HSM-abstraction layer (`src/horizon_ric/security/hsm.py`), the supported
backends, key-rotation procedure, and FIPS 140-3 inheritance.

Establishes HSM-backed key custody for the provenance-signing path.

---

## 1. Architecture

```
                                    ┌──────────────────────────┐
                                    │  HSM Backend (PKCS#11)   │
                                    │ ──────────────────────── │
                                    │  CKM_RSA_PKCS_PSS  sign  │
   operator (R1) ──┐                │  CKM_RSA_PKCS_OAEP enc   │
                   │                │  CKM_AES_GCM      wrap   │
                   ▼                │                          │
   sign_model() ──> HSMBackend ───> ├─── SoftHSM2Backend ──────│  test labs
   (provenance/     (security/      ├─── AWS CloudHSMBackend ──│  prod (FIPS 140-2 L3)
    signing.py)      hsm.py)        ├─── ThalesLunaBackend ────│  prod (FIPS 140-3 L3)
                                    └─── InMemoryHSMBackend ───│  unit tests ONLY
```

The Protocol-style abstract `HSMBackend` exposes six operations (six
PKCS#11 verbs, one factory):

| Op                  | Mechanism                       | Notes                         |
| ------------------- | ------------------------------- | ----------------------------- |
| `generate_keypair`  | `CKM_RSA_PKCS_KEY_PAIR_GEN`     | RSA-2048; label-scoped        |
| `sign`              | `CKM_RSA_PKCS_PSS` (SHA-256)    | salt = digest length          |
| `encrypt`/`decrypt` | `CKM_RSA_PKCS_OAEP` (SHA-256)   | MGF1-SHA-256                  |
| `export_public`     | (read-only attribute)           | SubjectPublicKeyInfo DER      |
| `list_keys`         | C_FindObjects                   | label enumeration             |
| `from_config`       | factory                         | dispatches by `cfg["backend"]` |

DEKs (data-encryption keys for payloads-at-rest) wrap under
`CKM_AES_GCM`; the wiring is documented but not bundled — see §6.

---

## 2. Backend matrix (honest)

| Backend                | File support          | FIPS status                    | Test status                                                                                                                  |
| ---------------------- | --------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `InMemoryHSMBackend`   | Bundled, real RSA-2048 via `cryptography` | NONE — process memory          | TEST-ONLY. Exercised by `tests/test_provenance.py` as the HSM backing `sign_model(...)` in the provenance-signing path.       |
| `SoftHSM2Backend`      | Bundled, PKCS#11      | NOT FIPS — SoftHSM2 is a test HSM | Functional when `python-pkcs11` is installed and `libsofthsm2.so` is on the host. Validated against SoftHSM2 v2.6.x.          |
| AWS CloudHSM           | Documented, NOT bundled | **FIPS 140-2 Level 3**         | Production deployment requires the CloudHSM Client + JCE provider. `from_config({"backend":"aws_cloudhsm"})` raises `NotImplementedError("contact ops")`. |
| Thales Luna Network HSM | Documented, NOT bundled | **FIPS 140-3 Level 3**         | Production deployment requires the Luna client + Universal Client SDK. Same factory path, same error surface.                |

The PKCS#11 mechanism set above is identical across SoftHSM2 / CloudHSM /
Luna, so the wire-format of signed announcements does not change when
upgrading the backend — only the key-custody guarantee strengthens.

---

## 3. SoftHSM2 reference setup (test labs)

```bash
sudo apt-get install -y softhsm2 libsofthsm2
softhsm2-util --init-token --slot 0 \
    --label horizon-ric \
    --pin 1234 --so-pin 1234
pip install python-pkcs11
```

```python
from horizon_ric.security.hsm import HSMBackend
from horizon_ric.provenance import sign_model

hsm = HSMBackend.from_config({"backend": "softhsm2",
                              "token_label": "horizon-ric",
                              "user_pin": "1234"})
prov = sign_model(weights, trainer_id="trainer-a",
                  training_manifest=manifest, hsm=hsm)
```

The SoftHSM2 token is initialised under `~/.config/softhsm2/` per user.
For multi-tenant labs, set `SOFTHSM2_CONF` to a tenant-scoped config file.

---

## 4. Production deployment (AWS CloudHSM / Thales Luna)

Both production backends are **documented but not bundled** because:

* CloudHSM client is licensed per-instance and ships out-of-band via
  `aws cloudhsm-cli`; bundling the client would conflict with AWS
  redistribution terms.
* Thales Luna client (`lunaclient`) ships under a Thales master license
  and is distributed only to direct customers.

The factory call `HSMBackend.from_config({"backend": "aws_cloudhsm"})`
intentionally raises `NotImplementedError("contact ops")`. To enable in
production:

1. Provision the HSM (CloudHSM cluster or Luna network appliance).
2. Install the vendor PKCS#11 module on the aggregator host.
3. Set `$PKCS11_MODULE` to the vendor `.so` path.
4. Subclass `HSMBackend`, copy the SoftHSM2Backend body, swap the token
   discovery / PIN handling for the vendor's session API.
5. Wire into `from_config()` by editing `src/horizon_ric/security/hsm.py`.

Each step is reviewed by the security team (PKCS#11 wiring is small but
HSM-vendor-specific quirks — Luna's CKA_DERIVE-must-be-true, CloudHSM's
multi-region clustering — bite quickly).

---

## 5. Key rotation

Cite **`docs/runbooks/cert_rotation.md`** for the canonical rotation
runbook. The short version, specific to share-dealer keys:

| Step | Action                                                             | Operator |
| ---- | ------------------------------------------------------------------ | -------- |
| 1    | `softhsm2-util --import …` or vendor equivalent — generate new key under label `share-dealer-v{n+1}` | SRE      |
| 2    | Re-deploy aggregator with `hsm_key_label="share-dealer-v{n+1}"` | SRE      |
| 3    | Drain in-flight signing operations (≤ 30 s; `sign_model` is stateless across calls) | SRE      |
| 4    | `softhsm2-util --delete-object … --label share-dealer-v{n}` after 30-day audit window | Sec      |
| 5    | Rotate aggregator client trust-store (NIS-2 incident-response trail) | Sec      |

Rotation cadence: **90 days** for share-dealer keys (NIST SP 800-57 Part 1
Rev 5, §5.3 — RSA-2048, Class 1 protection, originator-usage period
≤ 2 years; we are well inside).

---

## 6. Compliance crosswalk

| Standard                            | Clause                       | Coverage                                                                  |
| ----------------------------------- | ---------------------------- | ------------------------------------------------------------------------- |
| **NIST SP 800-57 Part 1 Rev 5**     | §5.6.3 Cryptoperiod          | 90-day rotation under §5 above; well inside the §5.3 originator window.   |
| **NIST SP 800-57 Part 2 Rev 1**     | §3.2 Key-management roles    | Mapped to RBAC roles `security.rotate-keys`, `ops.deploy-aggregator`.     |
| **FIPS 140-3 Level 3**              | (Thales Luna)                | Inherited when `backend = thales_luna`. SoftHSM2 / InMemory: NOT covered. |
| **FIPS 140-2 Level 3**              | (AWS CloudHSM)               | Inherited when `backend = aws_cloudhsm`.                                  |
| **eIDAS QSCD profile**              | EN 419 211-2 / -3            | Inherited from FIPS 140-3 L3 (Thales Luna QSCD-listed).                   |
| **NIS-2 Article 21(2)(g)**          | Cryptography & key mgmt      | This document + `docs/compliance/nist_csf.md` map to NIS-2 controls.      |
| **EU AI Act Annex IV §1(g)**        | Cybersecurity                | Share-dealer keys in HSM = "appropriate level of cybersecurity" baseline. |

---

## 7. Testing

* `tests/test_provenance.py` — exercises the `InMemoryHSMBackend` as the
  HSM backing `sign_model(...)`: sign/verify roundtrip (RSA-PSS-SHA256),
  tampered-weights rejection, tampered-manifest rejection, untrusted-signer
  rejection under a pinned public key, `verify_or_raise` promotion gate, and
  provenance dict roundtrip. The SoftHSM2 / CloudHSM / Thales backends and the
  `from_config` factory are present in `src/horizon_ric/security/hsm.py` but
  are not covered by the unit suite (they require a live PKCS#11 token).
* Shamir secure aggregation (`ShamirSecretSharing` / `secure_mean`) is covered
  by `tests/test_robust_aggregation.py`
  (`test_secure_mean_matches_plain_mean_privately`) and
  `tests/test_spectrum_dsa.py` (`test_secure_aggregation_path_runs`).

```bash
.venv/bin/python -m pytest tests/test_provenance.py tests/test_robust_aggregation.py -v
```
