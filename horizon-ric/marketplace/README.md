# Marketplace listings

This directory contains the per-marketplace submission artefacts for
PreceptualAI. Each subdirectory ships a manifest in the format the target
rApp catalogue / marketplace expects, plus any auxiliary metadata that
is published alongside the listing.

| Marketplace          | Artefact                          | Format                        |
|---------------------|-----------------------------------|-------------------------------|
| O-RAN SC rApp       | `oran-sc/manifest.yaml`           | `rapp.o-ran-sc.org/v1` CRD    |
| Ericsson Marketplace| `ericsson/listing.yaml`           | Ericsson Developer Studio     |
| Nokia DAC           | `nokia/listing.yaml`              | Nokia partner-portal template |

The dialect each marketplace expects PreceptualAI to speak is wired in
`A1AdapterConfig.dialect`; see `docs/SMO_INTEGRATION.md` for the
URL/body contract per dialect.

## Submission steps

### O-RAN SC rApp catalogue

1. Fork `o-ran-sc/nonrtric` and add `marketplace/oran-sc/manifest.yaml`
   under `nonrtric/rapp-manager-catalogue/catalog/horizon-ric.yaml`.
2. Open a PR against `o-ran-sc/nonrtric` referencing this repo as the
   build source and the linked Helm chart OCI URL.
3. Run the upstream conformance suite locally (`./conformance/run.sh`)
   and attach the JUnit XML to the PR.
4. After merge, the catalogue Pod will pick up the new entry and the
   rApp manager can install it.

Reference: <https://wiki.o-ran-sc.org/display/RICNR/rApp+catalogue>

### Ericsson Marketplace (EIAP rApp Studio)

1. Sign in to the Ericsson Developer Hub
   (<https://developer.ericsson.com>) with a registered partner account.
2. Open the rApp Studio submission flow and select "Import listing".
3. Upload `marketplace/ericsson/listing.yaml`.
4. Attach the Helm chart OCI URL referenced in the manifest, the
   conformance statement (`docs/conformance/SoC.yaml`), and the
   integration guide (`docs/SMO_INTEGRATION.md`).
5. Ericsson's security team runs the platform-mandated SCA/SAST review;
   the listing flips to `pending → published` on pass.

Honest blocker: the full submission template, schema, and review
checklist live behind the partner portal login. The committed YAML
mirrors only the publicly-documented fields; partner-only fields are
filled in via the portal at submission time.

### Nokia DAC

1. Sign in to the Nokia Network Solutions developer portal
   (<https://developer.nokia.com>).
2. Open Digital Automation Cloud → Marketplace → Submit listing.
3. Upload `marketplace/nokia/listing.yaml`.
4. Provide the OCI Helm chart URL, OpenAPI bundle
   (`docs/openapi/horizon-ric.yaml`), and conformance statement.
5. Nokia's security team runs MantaRay platform compatibility tests
   and signs the listing with their internal key.

Honest blocker: the precise schema for the Nokia DAC listing template
also requires a Nokia partner login. The committed YAML is filled in
according to the public marketplace docs; portal-only fields will be
added at submission time.

## Validating a listing locally

A simple YAML schema validation runs as part of the `make conformance`
target. To validate by hand:

```bash
.venv/bin/python -c "import yaml, sys; yaml.safe_load(open(sys.argv[1])); print('ok')" \
    marketplace/oran-sc/manifest.yaml
```

If the listing is valid YAML the script prints `ok` and exits 0.
