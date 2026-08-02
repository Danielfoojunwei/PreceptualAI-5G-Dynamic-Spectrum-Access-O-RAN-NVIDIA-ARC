# Horizon RIC — Offline Evidence-Chain Verifier

**Audience:** a spectrum authority, regulator, or independent auditor who has
received a Horizon RIC *evidence export* and needs to verify its integrity
**without installing the Horizon RIC product**.

This tool (`horizon_audit`) is deliberately standalone. It imports only:

* the **Python standard library**,
* the **`cryptography`** package (Ed25519 and RFC-3161 signature checks),
* **`asn1tools`** (a pure-Python ASN.1 DER codec) — used *only* to parse
  RFC-3161 timestamp tokens.

It **never** imports `horizon_ric`. This is enforced by a test
(`test_verifier_does_not_import_horizon_ric`) that imports every module of
the verifier with `horizon_ric` on the path and asserts nothing from it is
pulled into `sys.modules`.

---

## What it verifies

| Check | Module | What it proves |
|-------|--------|----------------|
| **Hash chain** | `horizon_audit.chain` | Every record's SHA-256 chain link is intact, per tenant. Any edit to any earlier record breaks the chain at that record and every later one. |
| **Ed25519 certificates** | `horizon_audit.certificate` | Any embedded, full `SafetyCertificate` was signed by the operator's published Ed25519 key and has not been altered. |
| **RFC-3161 anchor** | `horizon_audit.timestamp` | A saved TSA timestamp token genuinely covers the claimed chain-head hash (message-imprint binding), verified **offline** — the TSA is never contacted. |

---

## Install

```bash
python -m pip install cryptography asn1tools
```

Nothing else is required. Python 3.9+ is sufficient.

---

## Run

```bash
python audit/verify_evidence.py <evidence.jsonl> \
    [--pubkey <ed25519_pub.pem | 64-char-hex>] \
    [--anchor <anchor.json>] \
    [--tsa-cert <tsa_cert.pem>] \
    [--json]
```

The script adds its own `audit/src` to `sys.path`, so it runs from a plain
checkout with no `PYTHONPATH` or install step.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Every enabled check passed. |
| `1` | A check failed — a broken chain line, an invalid certificate, or a bad anchor. The report names the first broken `(tenant, index)`. |
| `2` | Usage / input error (file missing, unparsable, bad key). |

### Examples

Verify just the chain:

```bash
python audit/verify_evidence.py evidence.jsonl
```

```
evidence : evidence.jsonl
chain    : 512 line(s) across 3 tenant(s)
           INTACT — every per-tenant hash chain verifies
result   : PASS
```

A tampered export:

```
chain    : 512 line(s) across 3 tenant(s)
           BROKEN — first broken line index 129 (tenant 'tenant-b')
             tenant 'tenant-b': broken at global index 129 (per-tenant #42) —
             recomputed hash a1b2… != stored 9f8e…
result   : FAIL
```

Machine-readable output for a pipeline:

```bash
python audit/verify_evidence.py evidence.jsonl --json
```

---

## The formats it checks (reproduced, not imported)

### Evidence export (`.jsonl`)

One JSON object per line, in append order:

```json
{"hash": "<sha256-hex>", "tenant_id": "<str>", "record": { ... }}
```

### Chain rule (per line)

```
curr_hash = sha256( bytes.fromhex(prev_hash_hex)
                    + canonical_json(record).encode("utf-8") ).hexdigest()

canonical_json(obj) = json.dumps(obj, sort_keys=True, separators=(",", ":"))
```

* Genesis / previous hash of the first record in a chain: `"0" * 64`.
* **Chains are per-tenant.** Lines are grouped by `tenant_id` (default
  `"_unscoped_"`); each tenant's chain starts from the genesis hash
  independently. The reported `first_broken_index` is the **global line
  index** — identical to the value the producer's own
  `EvidenceStore.verify()` returns.

### Ed25519 certificate

A signed certificate is a JSON object carrying `signature` (hex) and
`signing_key_fingerprint` (SHA-256 of the raw 32-byte public key). The signed
bytes are the certificate's canonical JSON **with those two fields removed**:

```
signed_bytes = json.dumps(
    {k: v for k, v in cert.items()
     if k not in ("signature", "signing_key_fingerprint")},
    sort_keys=True, separators=(",", ":")).encode("utf-8")
```

Verification uses `Ed25519PublicKey.verify`. A missing, malformed, or invalid
signature verifies as **false** — a tampered certificate never passes.

> **Note on records.** A `DecisionRecord` stores only a *digest reference* to
> its certificate at `chosen_action.certificate`
> (`{safe, projected, violated_ids, signature, …}`), **not** the full signed
> document. That subset cannot be verified on its own — the signature was
> computed over the complete certificate. The tool detects this and reports
> "signature-digest reference only" rather than failing. To verify a
> certificate end-to-end, provide the full certificate document. The committed
> sample decision record carries no signature at all (signing was not
> configured for that run), and the tool reports "no embedded certificate".

### RFC-3161 anchor (`anchor.json`)

The anchor object produced by
`horizon_ric.evidence.rfc3161.TimestampAnchor.to_record`:

```json
{"type": "rfc3161_anchor", "tsa": "<url>", "token_b64": "<base64>",
 "chain_head_hex": "<sha256-hex>", "gen_time_utc": "<iso>",
 "anchored_at_unix": <float>}
```

**Offline verification checks the property that binds the token to the
evidence:** the token's `messageImprint.hashedMessage` must equal
`H(chain_head)` under the hash algorithm named inside the token (the Horizon
producer uses SHA-512 for the imprint). When the token embeds the TSA's
signing certificate, the tool additionally:

* verifies the CMS `SignerInfo` signature with `cryptography` (RSA / ECDSA /
  Ed25519 signer keys are supported), and
* checks the signed `messageDigest` attribute equals `digest(eContent)`,

i.e. the token is internally, cryptographically self-consistent.

#### Documented offline limit

Establishing that the signing certificate chains to a **trusted TSA root** is
a trust-anchor decision that needs the TSA's *published* certificate / CA
bundle, obtained out of band. Without it the tool reports
`trust_anchored = null` ("not established offline") rather than pretending a
full X.509 path validation was performed. Supply the TSA's certificate with
`--tsa-cert <tsa_cert.pem>` to have the embedded signer certificate checked
against it (equality or issuance).

This is a deliberate fail-closed / honest posture: the tool never claims more
assurance than the offline inputs support.

---

## Reproduce the tests

```bash
PYTHONPATH=audit/src:src python -m pytest audit/tests/test_offline_verifier.py -q
```

`src` is on the path **only** so the tests can drive the real `horizon_ric`
product to build genuine fixtures. The suite includes:

* **Equivalence gate** — builds a real multi-tenant JSONL evidence store with
  `horizon_ric`, tampers a line, and asserts this standalone verifier reports
  the **same** first-broken index as `EvidenceStore.verify()`, on both intact
  and tampered chains.
* **Falsification** — proves the gate has teeth: a deliberately wrong
  canonicalisation (spaces instead of compact separators) is shown to
  *disagree* with the reference, so a broken standalone implementation would
  be caught.
* **Purity** — imports every verifier module with `horizon_ric` available and
  asserts none of it leaks into `sys.modules`.
* **Certificate equivalence** — signs a real `SafetyCertificate` with
  `horizon_ric.shield.signing` and confirms this verifier's canonical bytes
  and verdicts match byte-for-byte, valid and tampered.
* **RFC-3161** — mints a real DER TimeStampToken signed by a self-issued RSA
  TSA (there is no network to reach a public TSA in the build environment),
  and verifies imprint binding + CMS signature offline, plus rejection of a
  wrong chain-head and a tampered signature.
