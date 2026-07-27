# E2 → A1 proof — real FlexRIC E2 termination driving a Horizon decision

Date: 2026-07-27. Host: Ubuntu 24.04 sandbox (4 cores), loopback only.

## What was proven

A **real near-RT RIC E2 termination** (FlexRIC) was built and run: a
real E2AP association was established, a real E2SM-KPM v3.00 REPORT
Style 4 subscription was accepted, real RIC Indications carrying 3GPP
TS 28.552 measurement records flowed on the wire, one of those
indications was captured **off the wire**, decoded by
`horizon_ric.e2.KpmMeasurementBridge` (asn1tools, aligned PER, against
the sha256-verified spec text FlexRIC's own encoder was generated
from), bridged into a `TelemetryEvent`, and driven through the full
Horizon `DecisionPipeline` — planner → Decision Safety Shield → guard
chain → A1 policy PUT — against the **real O-RAN-SC a1mediator**,
which delivered it over RMR to the **real hw-python xApp** and
reported `enforceStatus = ENFORCED`.

## Stack and pins

| Component | Version / pin |
| --- | --- |
| FlexRIC | `gitlab.eurecom.fr/mosaic5g/flexric.git` branch `dev`, commit `ef6d722f22191eea74089966983da1f5ec1fedd4` ("Merge branch 'bug-fixes-better-ci' into 'dev'") |
| FlexRIC build | `cmake -DKPM_VERSION=KPM_V3_00 -DCMAKE_BUILD_TYPE=Release -DASN1C_EXEC=/opt/asn1c/bin/asn1c`, `make -j2`, `sudo make install` — log at `~/oran-deps/flexric-build.log` (final `MAKE3_EXIT=0`) |
| asn1c | `github.com/mouse07410/asn1c` @ `940dd5fa9f3917913fd487b13dfddfacd0ded06e` (the exact pin in FlexRIC's `docker/Dockerfile.flexric.ubuntu`) |
| E2SM-KPM spec | `src/horizon_ric/e2/asn1/e2sm_kpm_v03.00_standard.asn1`, sha256 `ab473a9c…8a573a`, byte-identical to the FlexRIC tree copy that its asn1c wire codec was generated from |
| A1 side | official O-RAN-SC a1mediator (Go) on `:10000` + hw-python xApp over RMR — the pinned `deploy/xapp-e2e` stack |

FlexRIC's own KPM self-tests pass on this build:

```
1/2 Test  #4: Unit_test_enc_dec_KPM_v3_00 ......   Passed    0.01 sec
2/2 Test #13: Unit_test_KPM_v3_00 ..............   Passed    0.21 sec
100% tests passed, 0 tests failed out of 2
```

## Honest scope — the SCTP substitution

This sandbox kernel **cannot run SCTP**: `socket(AF_INET,
SOCK_STREAM/SEQPACKET, IPPROTO_SCTP)` fails with `EPROTONOSUPPORT`
(errno 93), `/lib/modules/$(uname -r)` does not exist, and containers
share the kernel, so no container/modprobe workaround exists. A first
run without any workaround got exactly this far (captured logs):

```
nearRT-RIC:    errno = 9                      # bind on the failed SCTP fd
emu_agent_gnb: [E2-AGENT]: E2 SETUP-REQUEST tx
               Error sending sctp message
xapp_kpm_moni: [xApp]: E42 SETUP-REQUEST tx
               Error sending sctp message
```

The live association was then run under `sctp_udp_shim.c`
(LD_PRELOAD): FlexRIC's entire message IO is one-to-many SCTP —
`sctp_sendmsg`/`sctp_recvmsg` with explicit peer `sockaddr_in`, no
`accept()`, no `connect()` — which maps 1:1 onto loopback **UDP
datagrams** (identical message-boundary semantics). Every E2AP/E42AP/
E2SM byte is produced and parsed by unmodified FlexRIC binaries; only
the L4 transport differs from a conformant deployment. This is a
disclosed transport substitution, not a mock: the sniffer captures
below are FlexRIC's own wire bytes.

## The real E2 evidence (verbatim log lines)

nearRT-RIC (`~/oran-deps/flexric-nearrt-ric.log`):

```
[E2AP]: E2 SETUP-REQUEST rx from PLMN 505. 1 Node ID 1 RAN type ngran_gNB
[NEAR-RIC]: Accepting RAN function ID 2 with def = ORAN-E2SM-KPM
[iApp]: E42 SETUP-REQUEST rx
[iApp]: E42 SETUP-RESPONSE tx
[iApp]: SUBSCRIPTION-REQUEST RAN_FUNC_ID 2 RIC_REQ_ID 1 tx
[iApp]: SUBSCRIPTION-REQUEST RAN_FUNC_ID 2 RIC_REQ_ID 2 tx
```

E2 agent (`~/oran-deps/flexric-emu-agent.log`):

```
[E2-AGENT]: E2 SETUP-REQUEST tx
[E2-AGENT]: E2 SETUP RESPONSE rx
[E2-AGENT]: Transaction ID E2 SETUP-REQUEST 0 E2 SETUP-RESPONSE 0
[E2 AGENT]: RIC_SUBSCRIPTION_REQUEST rx RAN_FUNC_ID 2 RIC_REQ_ID 1021
[E2 AGENT]: RIC_SUBSCRIPTION_REQUEST rx RAN_FUNC_ID 2 RIC_REQ_ID 1022
```

KPM monitor xApp (`~/oran-deps/flexric-kpm-xapp.log`; 4 indications
received before a stock display-path crash unrelated to the KPM
stream — see README deviation #2):

```
[xApp]: E42 SETUP-RESPONSE rx
[xApp]: xApp ID = 7
Connected E2 nodes = 1
[xApp]: E42 RIC SUBSCRIPTION REQUEST tx RAN_FUNC_ID 2 RIC_REQ_ID 1
[xApp]: SUBSCRIPTION RESPONSE rx
[xApp]: Successfully subscribed to RAN_FUNC_ID 2
      1 KPM ind_msg latency = 1443 [μs]
UE ID type = gNB, amf_ue_ngap_id = 112358132134
DRB.PdcpSduVolumeDL = 13 [Mb]
DRB.PdcpSduVolumeUL = 951 [Mb]
DRB.RlcSduDelayDl = 5.50 [μs]
DRB.UEThpDl = 5.47 [kbps]
DRB.UEThpUl = 5.84 [kbps]
RRU.PrbTotDl = 261 [%]
RRU.PrbTotUl = 791 [%]
```

Wire capture (AF_PACKET sniffer, `sniff_e2ap.py`): the E2 port 36421
carried the full E2AP procedure sequence — procedureCode 1
(E2setup request 1153 B / response 128 B), procedureCode 8
(RICsubscription request 209 B / response 38 B), then a stream of
procedureCode 5 (RICindication) datagrams at the 1000 ms granularity
period.

## The captured KPM record

Fixture `tests/fixtures/e2sm_kpm/e2ap_ricindication.bin` (sha256
`d307360a…b5ae0e`, 1332 B) is the first RICindication datagram of the
capture, taken at epoch 1785142431.227 (2026-07-27T08:53:51Z). The
E2SM-KPM OCTET STRINGs carved from it (exact re-encode verified):

- `indication_hdr.aper.bin` — Format 1 header, `colletStartTime`
  2026-07-27T08:53:5xZ, sender `My OAI-MONO` / `MONO` / `OAI`;
- `indication_msg.aper.bin` — Format 3 message, 6 UE reports × 7
  TS 28.552 measurements. UE 1 decodes to exactly what the xApp
  independently logged (three witnesses: sniffer bytes, asn1tools
  decode, FlexRIC's asn1c decode):
  `DRB.PdcpSduVolumeDL=13, DRB.PdcpSduVolumeUL=951,
  DRB.RlcSduDelayDl=5.504, DRB.UEThpDl=5.470, DRB.UEThpUl=5.837,
  RRU.PrbTotDl=261, RRU.PrbTotUl=791`.

Values are the emulator's synthetic per-UE data ("all RIC INDICATION
messages contain random data, as there is no UE connected" — FlexRIC
README §4.1); the PDUs, procedures and encodings are real.

## The resulting Horizon decision

`bridge_capture_to_a1.py --e2ap-capture ~/oran-deps/e2ap_capture.jsonl
--a1-url http://127.0.0.1:10000 --dialect legacy` bridged the newest
carvable live indication (UE report: `RRU.PrbTotDl=878` →
`kpm_risk_v1` risk 1.0, dominant metric `RRU.PrbTotDl`) and produced:

```
event_id            922f05ae-eed2-450e-bc22-8bae61192d51  (kpm_5g, source flexric-emu-gnb)
decision_id         efd92110-37a2-4674-96ee-6d3dcfc44edf
policy_type         horizon.admission.control   (risk band ≥ 0.6)
policy_id           d098010e-bb05-4cf6-b646-422ade7a064e
A1 PUT              202 accepted (a1mediator :10000, legacy dialect)
enforcement_status  ENFORCED
latency_ms          7.31 (decision → A1 accept)
```

Corroboration from the a1mediator's own receive log — the real
hw-python xApp acknowledged the policy instance:

```
"message recieved : {... \"policy_instance_id\": \"d098010e-bb05-4cf6-b646-422ade7a064e\",
 \"policy_type_id\": \"20003\", \"handler_id\": \"hw-python\", \"status\": \"OK\"}"
```

The DecisionRecord (with Shield certificate, rejected alternatives
and the `rapp_metadata.decision_id` link the mediator stored) was
persisted to the evidence store; the run report is
`~/oran-deps/../..//tmp/e2-companion-report.json` (regenerated on each
run of the harness).

Note: the OSC A1 simulator on `:8085` (osc_a1 dialect) was not
running in this environment; the loop was closed against the running
**real** O-RAN-SC a1mediator + hw-python xApp stack from
`deploy/xapp-e2e` instead, which is a strictly stronger consumer.

## Reproduce

```bash
sudo deploy/e2-companion/run_e2_stack.sh
# offline (no FlexRIC needed) — committed live-captured PDU through the same path:
python deploy/e2-companion/bridge_capture_to_a1.py --a1-url http://127.0.0.1:10000 --dialect legacy
python -m pytest tests/test_e2_kpm_bridge.py -q
```
