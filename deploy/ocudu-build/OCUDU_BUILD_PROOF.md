# OCUDU CU/DU build proof

_Built 2026-07-27 from source on a clean Linux host (Ubuntu 24.04, 4 vCPU,
15 GiB). Real compiler output, real binaries — the raw evidence files in
this directory were produced by the binaries themselves._

[OCUDU](https://ocudu.org/) is the Linux Foundation open-source 5G-and-beyond
CU/DU stack (the srsRAN Project successor, BSD-3-Clause-Open-MPI). Horizon-RIC
is a **Non-RT RIC rApp** and does not contain a RAN; this build demonstrates
that the CU/DU substrate the [6G integration path](../../docs/SIXG_READINESS.md)
targets compiles and runs here, and — critically — that its **E2 agent** is
present and configurable, which is the interface a companion near-RT RIC xApp
uses to feed Horizon's telemetry bus.

## Source

Pinned as a git submodule at [`third_party/ocudu`](../../third_party/ocudu)
— `gitlab.com/ocudu/ocudu` commit `f46f5804e53fede1e5c3353420ce1bcd3f4e57c6`.

## Build

```bash
sudo apt-get install -y cmake make gcc g++ pkg-config \
    libfftw3-dev libmbedtls-dev libsctp-dev libyaml-cpp-dev libgtest-dev libzmq3-dev
cd third_party/ocudu && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_ZMQ=ON -DENABLE_UHD=OFF \
         -DENABLE_DPDK=OFF -DAUTO_DETECT_ISA=OFF
make -j4 gnb ocu odu
```

`-DENABLE_ZMQ=ON` compiles the ZeroMQ RF front-end so the gNB runs without
SDR hardware; UHD/DPDK are off (no radio, no dataplane offload on this host).

## Result — three binaries, all run

| Binary | Target | Size | sha256 |
| --- | --- | --- | --- |
| `apps/gnb/gnb` | monolithic gNB (CU+DU) | 54,153,600 | `9aad49c4…23ef17` |
| `apps/cu/ocu` | split CU | 40,728,656 | `580a7d30…f77a78` |
| `apps/du/odu` | split DU | 34,280,768 | `9ac80da5…1f1e06` |

```
$ apps/gnb/gnb --version
--== OCUDU gNB (commit f46f580) ==--
OCUDU 5G gNB version 26.04.0 (f46f580)
```

Full checksums in [`binaries-sha256.txt`](binaries-sha256.txt); version banner
in [`gnb-version.txt`](gnb-version.txt).

## E2 termination is real and configured

The gNB links `lib/asn1/libe2ap_asn1.a` (E2AP ASN.1) and `lib/e2/libocudu_e2.a`
(the E2 agent + E2SM-KPM/RC/CCC service models), and exposes an E2 config
section — the actual `--help` output ([`gnb-e2-config.txt`](gnb-e2-config.txt)):

```
E2 parameters
  --enable_cu_cp_e2 BOOLEAN [false]   Enable CU E2 agent
  --enable_cu_up_e2 BOOLEAN [false]   Enable CU-UP E2 agent
  --enable_du_e2    BOOLEAN [false]   Enable DU E2 agent
  --addr [[127.0.0.1]]                RIC addresses for the E2 interface (SCTP multi-homing)
  --port :INT in [20000-40000] [36421]  RIC port
```

This is the concrete integration point: OCUDU's DU/CU E2 agent associates over
SCTP (default port 36421) with a near-RT RIC's E2 termination, subscribes to
E2SM-KPM, and streams RIC Indication measurement records. The companion path
that turns those records into Horizon telemetry lives under
[`../e2-companion/`](../e2-companion/) (Horizon E2SM-KPM bridge). Horizon
itself terminates A1/O1/R1, not E2 — the E2 leg is the near-RT RIC's, per the
[6G readiness doc](../../docs/SIXG_READINESS.md).

## Scope

A build-and-run proof plus an E2-surface demonstration. No RF, no UE
attach, no live E2 association is claimed *here* — the live E2 KPM flow is
documented separately under `deploy/e2-companion/`. The binaries are not
committed (35–54 MB each); they are reproducible from the pinned submodule
with the commands above.
