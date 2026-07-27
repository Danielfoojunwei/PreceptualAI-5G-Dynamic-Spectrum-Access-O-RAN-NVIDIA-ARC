# ASN.1 module provenance

## `e2sm_kpm_v03.00_standard.asn1`

- **What it is**: the O-RAN WG3 E2SM-KPM v3.00 ASN.1 definition
  (`O-RAN.WG3.E2SM-KPM-v03.00`), containing the two modules
  `E2SM-KPM-IEs` and `E2SM-COMMON-IEs` in one file.
- **Where it came from**: byte-for-byte copy of
  `src/sm/kpm_sm/kpm_sm_v03.00/ie/asn/e2sm_kpm_v03.00_standard.asn1`
  from the FlexRIC source tree
  (<https://gitlab.eurecom.fr/mosaic5g/flexric.git>, branch `dev`,
  commit `ef6d722f22191eea74089966983da1f5ec1fedd4`).
  FlexRIC labels this file "standard" — it is the unmodified spec text
  as shipped with the O-RAN specification. FlexRIC's own wire
  encoder/decoder (the asn1c-generated C in
  `src/sm/kpm_sm/kpm_sm_v03.00/ie/asn/`) was generated from THIS file
  — every generated header carries the banner
  `From ASN.1 module "E2SM-KPM-IEs" found in
  "e2sm_kpm_v03.00_standard.asn1"` — so the bytes FlexRIC puts on the
  E2 wire and the spec this repo decodes them with share one source
  text.
- **Verified**: `sha256sum` of this copy equals the FlexRIC original:
  `ab473a9cc60bc3e3b1e32d032ce3b7158175c210282a006d662b1c48bc8a573a`.
- **How it is used**: compiled at runtime by `asn1tools`
  (`codec="per"`, i.e. aligned PER — the encoding E2SM-KPM mandates)
  in `horizon_ric.e2.kpm_bridge` to decode real
  `E2SM-KPM-IndicationMessage` / `E2SM-KPM-IndicationHeader` PDUs.
- **License note**: the ASN.1 text is part of the O-RAN Alliance
  specification reproduced in the FlexRIC repository; it is carried
  here unmodified solely for protocol interoperability.
