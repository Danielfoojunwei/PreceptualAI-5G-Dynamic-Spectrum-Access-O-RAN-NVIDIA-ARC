# Cosign image signing for PreceptualAI

PreceptualAI publishes container images signed with [Sigstore Cosign](https://docs.sigstore.dev/cosign/overview/),
in line with O-RAN.WG11 (Security Specification) supply-chain integrity
requirements and SLSA L3 attestation.

## SECURITY NOTICE — KEY ROTATION 2026-05-06

The previous `cosign.key` file (sha256
`b4fa4c49c9149959fddbbb0ddfb23e1283ce402f6a7bf781e1c20940355216bb`,
653 bytes) was **committed to this repository in plain text**. As of the
DEVIL_B remediation it is **REVOKED**. Anyone holding a copy of that key
can no longer publish signed PreceptualAI images: the public key in
`cosign.pub` has been replaced with a new keypair and any signature made
with the old private key fails `cosign verify` against this repo.

The revoked private key has been moved to `/tmp/cosign_old_REVOKED.key`
on the build host (mode `0600`) and **MUST NOT** be re-imported into any
trust store. Operators that pinned the previous public-key fingerprint
must update their pin to the value in `cosign.pub`.

## Files

| File         | Purpose                                                           |
| ------------ | ----------------------------------------------------------------- |
| `cosign.pub` | Public key (the only file in this directory). Operators verify with this. |
| `cosign.key` | **NEVER COMMITTED.** Lives in KMS (production) or a Sealed Secret (test/dev). |

The `.gitignore` at the repo root explicitly excludes `*.key`, `*.pem`,
and `deploy/cosign/cosign.key` to make recommitting impossible without
`git add -f` (which the `tests/test_no_secrets_in_repo.py` CI gate
catches anyway).

## Production: KMS-backed signing (REQUIRED)

Production signing **MUST** use a Cloud KMS key. The private key never
leaves the HSM. The exact command (substitute your project + key ring):

```bash
# One-time: generate a new keypair backed by GCP KMS
cosign generate-key-pair \
  --kms gcpkms://projects/horizon-ric-prod/locations/us-central1/keyRings/cosign/cryptoKeys/horizon-rapp-signing/versions/1

# Each release: sign by digest using the same KMS reference
cosign sign \
  --key gcpkms://projects/horizon-ric-prod/locations/us-central1/keyRings/cosign/cryptoKeys/horizon-rapp-signing/versions/1 \
  --yes \
  horizonric/rapp@${DIGEST}
```

Equivalent KMS URIs are supported for the other clouds:

| Cloud | URI                                                                 |
| ----- | ------------------------------------------------------------------- |
| AWS   | `awskms:///${ARN}` or `awskms://${ENDPOINT}/${KEY_ID}`              |
| Azure | `azurekms://${VAULT_NAME}.vault.azure.net/${KEY_NAME}`              |
| HV    | `hashivault://${KEY_NAME}` (with `VAULT_ADDR` + `VAULT_TOKEN` env)  |

Cosign also supports keyless OIDC (`--identity-token`) for CI workflows
via Sigstore Fulcio + Rekor — see
[the cosign docs](https://docs.sigstore.dev/cosign/sign/) for the
GitHub Actions identity-token flow.

## Test / dev: Sealed Kubernetes Secret

For `kind` and local edge-cluster testing the private key still lives in
a Kubernetes Secret, but the Secret is encrypted at rest using
[`bitnami-labs/sealed-secrets`](https://github.com/bitnami-labs/sealed-secrets)
or the equivalent `external-secrets-operator` integration. The plain
`Secret` resource is **never** committed; only the encrypted
`SealedSecret` is.

```bash
# Encrypt a fresh local key into a SealedSecret
kubectl create secret generic horizon-cosign \
  --from-file=cosign.key=/tmp/cosign_new.key \
  --dry-run=client -o yaml \
  | kubeseal --format=yaml > deploy/kubernetes/cosign-sealed.yaml
# kubeseal output is safe to commit; only the cluster controller can decrypt it.
```

## Operator verification

```bash
# Install cosign 2.x (Linux arm64 example; pick the right asset for your host)
curl -sSL -o /usr/local/bin/cosign \
  https://github.com/sigstore/cosign/releases/download/v2.4.1/cosign-linux-arm64
chmod +x /usr/local/bin/cosign

# Verify the rApp image
cosign verify \
  --key deploy/cosign/cosign.pub \
  horizonric/rapp:0.2.0
```

Expected output ends with a JSON payload that includes the digest, the
signature, and the timestamping authority. A non-zero exit code indicates
the image is unsigned, signed by a different (revoked) key, or tampered
with — **do not deploy** in that case.

## Key rotation cadence

Rotate `cosign.key` quarterly or on suspected exposure. The 2026-05-06
rotation was forced: see SECURITY NOTICE above.

1. Provision a new KMS key version: `cosign generate-key-pair --kms ...`
2. Update `cosign.pub` (the `--output-key-prefix` flag pulls the new public key down).
3. Commit the new `cosign.pub`.
4. Re-sign all currently-supported image tags with the new KMS key version.
5. Announce the new public-key fingerprint in the release notes.
6. After the deprecation window, delete the old KMS key version.

## Public key fingerprint (this repo, post 2026-05-06 rotation)

The current public key is in `cosign.pub` (ECDSA P-256,
SubjectPublicKeyInfo PEM). Operators should pin the fingerprint and treat
any change as a rotation event:

```bash
sha256sum deploy/cosign/cosign.pub
```

## Standards mapping

| Requirement                              | Where met                                              |
| ---------------------------------------- | ------------------------------------------------------ |
| O-RAN.WG11 §6 supply-chain integrity     | Image signature + SBOM attached as cosign attestation. |
| SLSA L3 (build provenance)               | `cosign attest --predicate slsa-provenance.json …`     |
| 3GPP TS 28.105 §5 trustworthy AI/ML lifecycle | Each model card includes `sha256` matched to artifact (see `tests/test_ts28105_model_card_emit.py`). |
| NIST SP 800-204D supply-chain hygiene    | KMS-backed signing key (no private material at rest).   |
