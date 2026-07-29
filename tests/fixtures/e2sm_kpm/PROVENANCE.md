# E2SM-KPM v3.00 indication PDU fixtures — provenance

These fixtures were **captured live off the wire** from a running
FlexRIC near-RT RIC ↔ E2-agent association on 2026-07-27 (epoch
1785142431.227, loopback):

- `e2ap_ricindication.bin` (1332 B) — the complete E2AP **RICindication**
  PDU (E2AP v2.03, procedureCode 5, initiatingMessage) exactly as the
  emulated gNB E2 agent sent it to the nearRT-RIC on E2 port 36421,
  under an accepted REPORT Style 4 KPM subscription created by the
  stock `xapp_kpm_moni` xApp.
- `indication_hdr.aper.bin` (32 B) — the `E2SM-KPM-IndicationHeader`
  OCTET STRING carved from that PDU (offset 38), aligned-PER,
  Format 1, sender "My OAI-MONO"/"MONO"/"OAI".
- `indication_msg.aper.bin` (1255 B) — the `E2SM-KPM-IndicationMessage`
  OCTET STRING carved from that PDU (offset 77), aligned-PER,
  Format 3 ("Common condition-based, UE-level"): 6 UE reports × 7 real
  3GPP TS 28.552 measurements each.

The carve offsets were located by scanning the E2AP bytes for APER
length-determinant-prefixed slices that decode as the E2SM-KPM types
AND re-encode byte-identically (exact round-trip equality), and the
decoded measurement values were cross-checked against the KPM monitor
xApp's independently logged output for the same indication
(`DRB.PdcpSduVolumeDL = 13`, `DRB.PdcpSduVolumeUL = 951`,
`DRB.RlcSduDelayDl = 5.50`, `DRB.UEThpDl = 5.47`, `DRB.UEThpUl = 5.84`,
`RRU.PrbTotDl = 261`, `RRU.PrbTotUl = 791` — three independent
witnesses: sniffer bytes, asn1tools decode, FlexRIC's asn1c decode).

Stack that produced them (see `deploy/e2-companion/E2_KPM_PROOF.md`):

- FlexRIC @ `ef6d722f22191eea74089966983da1f5ec1fedd4`
  (<https://gitlab.eurecom.fr/mosaic5g/flexric.git>, branch `dev`),
  built `-DKPM_VERSION=KPM_V3_00 -DCMAKE_BUILD_TYPE=Release`;
  `nearRT-RIC` + `emu_agent_gnb` + `xapp_kpm_moni`, all stock binaries.
- **Transport disclosure**: this sandbox kernel has no SCTP
  (`EPROTONOSUPPORT`, no modules). The association ran over an
  `LD_PRELOAD` shim (`deploy/e2-companion/sctp_udp_shim.c`) that maps
  FlexRIC's one-to-many SCTP calls onto loopback **UDP datagrams**
   1:1 (send/recv with explicit peer addresses, message boundaries
  preserved). Every E2AP/E2SM byte is produced and parsed by
  unmodified FlexRIC code; only the L4 transport differs from a
  standards-conformant deployment.

Checksums:

```
d6b7725e5b759972b436f1c4bd49c0b8d0449349f90694cfe5d02441ec3c4596  indication_hdr.aper.bin
61304fc25175b0d863498b31b73127ab2c92f914759986f32ce7fe74eec51c60  indication_msg.aper.bin
d307360a176bc41d4da9f8d560a4f7ef807146cc6261546b51a1b60488b5ae0e  e2ap_ricindication.bin
```
