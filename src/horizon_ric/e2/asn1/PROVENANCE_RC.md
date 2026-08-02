# ASN.1 module provenance — E2SM-RC

## `e2sm_rc_v1_03_standard.asn`

- **What it is**: the O-RAN WG3 E2SM-RC v1.03 ASN.1 definition
  (`O-RAN.WG3.E2SM-RC-v01.03`). The one file carries two ASN.1 modules:
  - `E2SM-COMMON-IEs`
    (`{iso(1) identified-organization(3) dod(6) internet(1) private(4)
    enterprise(1) 53148 e2(1) version1(1) e2sm(2) e2sm-COMMON-IEs(0)}`)
  - `E2SM-RC-IEs`
    (`{iso(1) identified-organization(3) dod(6) internet(1) private(4)
    enterprise(1) oran(53148) e2(1) version1(1) e2sm(2) e2sm-RC-IEs(3)}`)
- **Where it came from**: byte-for-byte copy of
  `src/sm/rc_sm/ie/asn/e2sm_rc_v1_03_standard.asn`
  from the FlexRIC source tree
  (<https://gitlab.eurecom.fr/mosaic5g/flexric.git>, branch `dev`,
  commit `ef6d722f22191eea74089966983da1f5ec1fedd4`, described as
  `v2.0.0-224-gef6d722f`) — the same submodule pin this repo already
  uses for E2SM-KPM.
  FlexRIC labels this file "standard" — it is the unmodified spec text
  as shipped with the O-RAN specification. The sibling
  `e2sm_rc_v1_03_modified.asn` in that tree is FlexRIC's own edited
  variant and is deliberately **not** the file vendored here.
  FlexRIC's wire encoder/decoder for RC (the asn1c-generated C beside
  it in `src/sm/rc_sm/ie/asn/`) was generated from THIS file — all 484
  generated sources carry the banner
  `From ASN.1 module "E2SM-RC-IEs" found in "e2sm_rc_v1_03_standard.asn"`
  — so the bytes a FlexRIC E2 node parses off the E2 wire and the
  bytes this repo encodes share one source text.
- **Verified**: `sha256sum` of this copy equals the FlexRIC original:
  `09f295cb59d4145efd64fe510f602c994d6df65e9beba33aab2e98cdf5bc642b`
  (50 985 bytes, 1 575 lines). `diff` against the upstream path above
  is empty.
- **ASN.1 types this repo exercises** (all defined in `E2SM-RC-IEs`):
  `E2SM-RC-ControlHeader`, `E2SM-RC-ControlHeader-Format1`,
  `E2SM-RC-ControlMessage`, `E2SM-RC-ControlMessage-Format1`,
  `E2SM-RC-ControlMessage-Format1-Item`, and the IEs they reference —
  `UEID` (CHOICE, arm `gNB-UEID` → `UEID-GNB`), `AMF-UE-NGAP-ID`,
  `GUAMI`, `RIC-Style-Type`, `RIC-ControlAction-ID`,
  `RANParameter-ID`, `RANParameter-ValueType` (CHOICE, arm
  `ranP-Choice-ElementTrue`) and `RANParameter-Value` (CHOICE, arms
  `valueInt` / `valueReal`).
  `E2SM-RC-ControlHeader-Format2` and `E2SM-RC-ControlMessage-Format2`
  (plus its style/action items) compile and are available but are not
  encoded by `horizon_ric.e2.rc_control`.
- **How it is used**: compiled at runtime by `asn1tools`
  (`codec="per"`, i.e. aligned PER — the encoding E2SM mandates, and
  the same codec `horizon_ric.e2.kpm_bridge` already uses) in
  `horizon_ric.e2.rc_control` to **construct** and round-trip-decode
  `E2SM-RC-ControlHeader` / `E2SM-RC-ControlMessage` OCTET STRING
  payloads. Construction only: this package carries no E2AP/SCTP
  transport, by design (see `horizon_ric/e2/__init__.py`).
- **License note**: the ASN.1 text is part of the O-RAN Alliance
  specification reproduced in the FlexRIC repository; it is carried
  here unmodified solely for protocol interoperability.
