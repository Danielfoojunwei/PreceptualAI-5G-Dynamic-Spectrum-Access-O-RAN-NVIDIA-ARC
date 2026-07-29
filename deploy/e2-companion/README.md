# E2 companion — real near-RT RIC E2 termination feeding Horizon

This directory stands up a **real O-RAN E2 termination** — FlexRIC's
nearRT-RIC with a real E2AP + E2SM-KPM v3.00 implementation — and
bridges the **real E2SM-KPM RIC Indications** it receives into Horizon
`TelemetryEvent`s that drive the existing Shield-gated
`DecisionPipeline` and a real A1 policy consumer. No mocks; every PDU
is produced and parsed by unmodified FlexRIC code.

```
emu_agent_gnb ──E2AP/E2SM-KPM (SCTP*)──► nearRT-RIC ◄──E42── xapp_kpm_moni
      │                                       │
      │   RIC Indication (KPM v3.00, APER)    │ sniffed on the wire
      ▼                                       ▼
  sniff_e2ap.py ──► e2ap_capture.jsonl ──► bridge_capture_to_a1.py
                                               │  horizon_ric.e2.KpmMeasurementBridge
                                               │  (asn1tools decode, kpm_risk_v1)
                                               ▼
              DecisionPipeline (planner → Shield → guards)
                                               │  A1AP policy PUT
                                               ▼
              real A1 endpoint (e.g. the deploy/xapp-e2e a1mediator :10000)
                                               │  RMR A1_POLICY_REQ/RESP
                                               ▼
              hw-python xApp → enforceStatus = ENFORCED
```

| Component | Source | Pin |
| --- | --- | --- |
| FlexRIC (nearRT-RIC, E2 agent emulator, KPM xApp) | `gitlab.eurecom.fr/mosaic5g/flexric` (dev) | `ef6d722f22191eea74089966983da1f5ec1fedd4` |
| asn1c (NR-RRC codegen for the xApp SDK) | `github.com/mouse07410/asn1c` (FlexRIC's own CI pin) | `940dd5fa9f3917913fd487b13dfddfacd0ded06e` |
| E2SM-KPM ASN.1 spec | copied from the FlexRIC tree, sha256-verified | see `src/horizon_ric/e2/asn1/PROVENANCE.md` |

## Usage

```bash
sudo deploy/e2-companion/run_e2_stack.sh        # build + self-test + live run + bridge
# or drive the committed live-captured fixture PDU through the pipeline only:
python deploy/e2-companion/bridge_capture_to_a1.py \
    --a1-url http://127.0.0.1:10000 --dialect legacy
```

The proof of the live run this harness produced (E2 Setup, accepted KPM
subscription, real indications, and the resulting Horizon
DecisionRecord ENFORCED by a real xApp) is in `E2_KPM_PROOF.md`.

## Files

- `run_e2_stack.sh` — pinned build + FlexRIC KPM self-tests (`ctest -R KPM`)
  + live stack + capture + bridge run.
- `bridge_capture_to_a1.py` — carves E2SM-KPM header/message OCTET
  STRINGs out of captured E2AP RICindication datagrams (APER
  length-determinant scan, exact re-encode verification) and runs them
  through `horizon_ric.e2` → `DecisionPipeline` → A1.
- `sniff_e2ap.py` — AF_PACKET loopback sniffer for ports 36421/36422.
- `sctp_udp_shim.c` — LD_PRELOAD transport substitute for kernels
  without SCTP (see below).
- `kpm_capture.c` — offline harness that produces indication PDUs with
  FlexRIC's emulator callback + production APER encoder without any
  transport; used to validate the decode path before the live run.

## Local deviations from stock upstream (full disclosure)

1. **Transport shim** — this sandbox kernel has no SCTP
   (`socket(AF_INET, SOCK_SEQPACKET, IPPROTO_SCTP)` →
   `EPROTONOSUPPORT`, no loadable modules), which FlexRIC requires.
   `sctp_udp_shim.c` (LD_PRELOAD) maps FlexRIC's one-to-many SCTP
   calls — `sctp_sendmsg`/`sctp_recvmsg` with explicit peer addresses,
   no `accept()`/`connect()` — 1:1 onto loopback UDP datagrams, which
   have identical message-boundary semantics. E2AP/E2SM bytes are
   untouched. On kernels with SCTP the shim is skipped automatically
   and the stack runs standards-conformant transport.
2. **No FlexRIC source modifications.** The stock
   `xapp_kpm_moni` crashes after ~4 indications when paired with the
   emulator (a display-path issue in its handler for the emulator's
   randomly-formatted second Style-1 subscription); the measurement
   stream up to that point is unaffected and the wire capture is
   independent of the xApp process.
