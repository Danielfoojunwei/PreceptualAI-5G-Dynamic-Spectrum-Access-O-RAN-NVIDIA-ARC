# OCUDU gNB joined to FlexRIC over E2 — proof

_Recorded 2026-07-28 on the same 4 vCPU / 15 GiB Linux host as
[`E2_KPM_PROOF.md`](E2_KPM_PROOF.md). This closes the gap that document and
[`../../docs/ECOSYSTEM.md`](../../docs/ECOSYSTEM.md) §3 listed as **P1**: the
OCUDU CU/DU was built and its E2 agent surface confirmed, but it had never
been joined to the near-RT RIC — the earlier E2SM-KPM proof used FlexRIC's
**emulated** agent (`emu_agent_gnb`), not a real gNB._

## What is new

The **real OCUDU gNB binary** (`apps/gnb/gnb`, commit `f46f580`, the same
binary whose sha256 is recorded in
[`../ocudu-build/binaries-sha256.txt`](../ocudu-build/binaries-sha256.txt))
now performs a real **E2 Setup** against the **real FlexRIC near-RT RIC**
(`ef6d722f`), which accepts it as an `ngran_gNB` E2 node and registers its
E2SM-KPM and E2SM-RC RAN functions. A stock FlexRIC xApp then completes its
E42 setup against that RIC and issues subscription requests.

Reproduce with:

```sh
gcc -shared -fPIC -O2 -o shim.so deploy/e2-companion/sctp_udp_shim.c -ldl
LD_PRELOAD=$PWD/shim.so  <flexric>/build/examples/ric/nearRT-RIC &
LD_PRELOAD=$PWD/shim.so  <ocudu>/build/apps/gnb/gnb -c deploy/e2-companion/ocudu-gnb-e2.yml
```

(omit `LD_PRELOAD` on a kernel that has SCTP — see *Transport* below).

## Witness 1 — the gNB

Reproduced on four consecutive runs; timestamps from the final clean run:

```
[E2-CU-CP] [I] "RIC Connection Setup Routine" started.
[E2-CU-CP] [D] Trying to establish E2 connection to Near-RT RIC (configured addrs 127.0.0.1, port 36421)...
[E2-CU-CP] [I] E2: Connection to Near-RT-RIC on 127.0.0.1:36421 established
[E2-CU-CP] [I] Generate RAN function definition for OID: 1.3.6.1.4.1.53148.1.2.2.2
[E2-CU-CP] [I] Generate RAN function definition for OID: 1.3.6.1.4.1.53148.1.1.2.3
[E2-CU-CP] [I] E2AP msg, "successfulOutcome.E2setupResponse", transaction id=0
[E2-CU-CP] [I] E2 Setup procedure successful.
[E2-CU-CP] [I] Added supported RAN function with id 2 and OID 1.3.6.1.4.1.53148.1.2.2.2
[E2-CU-CP] [I] Added supported RAN function with id 3 and OID 1.3.6.1.4.1.53148.1.1.2.3
[E2-CU-CP] [I] "RIC Connection Setup Routine" finished successfully.
```

An `E2setupResponse` cannot be self-generated: it is produced by the RIC.

## Witness 2 — the RIC

Independently, from the near-RT RIC's own stdout:

```
[E2AP]: E2 SETUP-REQUEST rx from PLMN   1. 1 Node ID 411 RAN type ngran_gNB
[NEAR-RIC]: Accepting RAN function ID 2 with def = ORAN-E2SM-KPM
[NEAR-RIC]: Accepting RAN function ID 3 with def = ORAN-E2SM-RC
[NEAR-RIC]: Registered E2 nodes = 5.
```

The RIC classifies the peer as `ngran_gNB` and accepts both service models.
(`Registered E2 nodes = 5` reflects five gNB runs against one long-lived RIC
process; stale nodes age out via `Pending event timeout`.)

## Witness 3 — a third-party xApp

The stock FlexRIC `xapp_kpm_moni` attaches to the same RIC and drives
subscriptions, so the E2 node is visible through the RIC's northbound side and
not only in its log:

```
[iApp]: nearRT-RIC IP Address = 127.0.0.1, PORT = 36422
[iApp]: E42 SETUP-REQUEST rx
[iApp]: E42 SETUP-RESPONSE tx
[iApp]: SUBSCRIPTION-REQUEST RAN_FUNC_ID 2 RIC_REQ_ID 2 tx
[iApp]: SUBSCRIPTION-REQUEST RAN_FUNC_ID 2 RIC_REQ_ID 3 tx
[iApp]: SUBSCRIPTION-REQUEST RAN_FUNC_ID 2 RIC_REQ_ID 4 tx
[iApp]: SUBSCRIPTION-REQUEST RAN_FUNC_ID 2 RIC_REQ_ID 5 tx
```

## Transport — still the disclosed substitution

This kernel still has no SCTP. Re-verified on the day of this run:
`socket(AF_INET, SOCK_STREAM, IPPROTO_SCTP)` → `OSError [Errno 93] Protocol not
supported`, `/proc/net/sctp` absent, no `modprobe` in an unprivileged
container. The stack therefore runs under
[`sctp_udp_shim.c`](sctp_udp_shim.c), which maps SCTP socket calls onto
loopback UDP datagrams. Every E2AP/E2SM byte is produced and parsed by
unmodified FlexRIC and OCUDU code; only L4 differs.

Joining OCUDU required extending that shim beyond what FlexRIC needed, because
OCUDU uses the one-to-one SCTP **client** style: `sctp_bindx`, `sctp_connectx`,
`sctp_getpaddrs`, `sctp_freepaddrs`, and — the call that actually blocked the
join — `getsockopt(IPPROTO_SCTP, …)`. OCUDU reads `SCTP_RTOINFO`,
`SCTP_INITMSG`, `SCTP_PEER_ADDR_PARAMS` and `SCTP_ASSOCINFO` back before
modifying them, and the real `EOPNOTSUPP` aborted the connection with
`E2: Error getting RTO_INFO sockopts`.

**This proves E2AP/E2SM interoperability between the two real
implementations. It does not prove SCTP transport conformance.**

## Scope — what this still does NOT prove

- **Only the CU-CP E2 agent attaches.** `enable_du_e2: true` is present in the
  gNB's effective configuration, but the DU-side agent never starts and never
  opens a socket. That is the agent carrying cell and scheduler KPMs, so the
  DU-level measurement path remains unjoined. Unresolved.
- **No UE and no traffic.** The gNB runs on the ZMQ RF driver with no UE
  attached, so any KPM indication would report zeros. E2 Setup and RAN-function
  registration are proven; measurement content is not.
- **No E2AP pcap.** `pcap.e2ap_enable` is set, but the gNB cannot shut down
  cleanly in this sandbox (`Could not stop application after 5 seconds`), so the
  capture never flushes and the files are zero-length. The three log witnesses
  above are the evidence.
- **No Horizon policy traverses this path.** Horizon-RIC's own control loop
  still emits over A1 (see [`../XAPP_E2E_PROOF.md`](../XAPP_E2E_PROOF.md)).
  Near-real-time enforcement over E2SM-RC remains roadmap work.
