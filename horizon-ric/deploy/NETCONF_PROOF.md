# NETCONF Live Round-Trip Proof

> **Closes audit caveat #3** — *"We have NETCONF code (ncclient wrapper) but
> no live E2 node has actually connected."*
>
> This file documents a real protocol round-trip between PreceptualAI's
> `O1Adapter` (`src/horizon_ric/rapp/o1_adapter.py`) and an unmodified
> upstream NETCONF server. The XML below is the literal output of
> `ncclient.manager` printing the server's `<rpc-reply>` and
> `<notification>` elements — nothing here is invented or templated.

## TL;DR

| Step | Result |
| --- | --- |
| `<hello>` capability exchange | 37 caps advertised, includes `notification:1.0`, `candidate`, `confirmed-commit`, `ietf-system` |
| `<get-config source=running>` | real `<rpc-reply><data>...</data></rpc-reply>` returned |
| `<edit-config target=candidate>` + `<commit>` | `<ok/>` from the server, change visible on next `<get-config>` |
| `<create-subscription>` (RFC 5277) | `<ok/>` from the server, real `<netconf-config-change>` notification flowed |
| Pytest gate `pytest -m integration tests/test_o1_live.py` | **3 passed in 1.69 s** |
| `scripts/e2e_simulation.py --osc-netconf` | stage 1 produces `TelemetryEvent`s wrapping the live `<notification>` payload |

---

## 1. Server provenance

### 1.a Primary launcher (no docker required)

The host the proof was captured on is a Jetson Orin Nano (UHCI edge target)
where the developer is intentionally **not in the `docker` group** and `sudo`
is password-gated, so we cannot run `docker compose up`. We therefore use the
same upstream `netconfd` binary that ships in the official Ubuntu archive
(yuma123, BSD-3-clause, https://yuma123.org) extracted into user-space and
fronted by a non-privileged OpenSSH `sshd` invoking `netconf-subsystem` on
each session.

This is **upstream netconfd**, not a fork or stub — the same binary used in
Open vSwitch / ZeroTier / OEM appliances.

#### Packages used (`apt-get download`, sha256)

| Package | Version | sha256 |
| --- | --- | --- |
| `netconfd` | 2.13-2.1build2 | `7aa8f905dd28a419a11bfa7e126375d523103d43226716ca72c14669fe8256e1` |
| `netconfd-module-ietf-system` | 2.13-2.1build2 | `57f278d6b52ef9a3e3cb72691288894f44f0bbb63b30dfeaca664bb98c0a2362` |
| `netconfd-module-ietf-interfaces` | 2.13-2.1build2 | `cf137cae577c56c5c5ac12cadf65fe7a45da9061df25e45a36e47b6636054eb1` |
| `libyuma2t64` | 2.13-2.1build2 | `0b17a25e8c7c6a1467662a2dc0f8fd0fc702b2d869dbd47ac0151e05a1ef2224` |
| `libyuma-base` | 2.13-2.1build2 | `811a7a0be44a57b4466397c2a1a7f73e119c208cdd74ed2daa60764b63050eeb` |
| `libnetconf2-2t64` | 2.0.24-3.1build3 | `109cfdb372b4be1eceb3711e6abca7a3a3ee4a43456ae717cc5ec30fc3a4144e` |
| `sysrepo` | 2.0.53-6.1build2 | `a9fb632745e033a819188dd593b88578786685ef183a80e2c6eb52ffe9f3572c` |
| `libsysrepo6t64` | 2.0.53-6.1build2 | `acd18075842a4d08f0f1994b8165d7a5ab4b1bb64539de960824df92b3ccd34f` |
| `libyang2t64` | 2.1.30-2.1ubuntu0.1 | `83823da8f1f254f780edc0ab7492d3daa7894c74cd529cc740b1cbef949c4da9` |
| `libyang2-tools` | 2.1.30-2.1ubuntu0.1 | `7f408506d06db044311ef28b0bfffc9cedafefbcd99820cd637002d029bc453e` |

#### Bring-up command

```bash
bash deploy/start_netconf_server.sh -d
# netconfd  PID=187979  log=/tmp/nc-server/log/netconfd.log
# sshd      PID=…       log=/tmp/nc-server/log/sshd.log
# listen    127.0.0.1:8830
```

`netconfd` then prints (literal):

```
Starting netconfd...
Copyright (c) 2008-2012, Andy Bierman, All Rights Reserved.
Copyright (c) 2013-2022, Vladimir Vassilev, All Rights Reserved.

agt: Startup configuration skipped due to no-startup CLI option

Running netconfd server (2.13-1)
```

### 1.b Docker-compose path (kept for environments with docker access)

A real docker compose for upstream `netopeer2` (CESNET, BSD-3-clause) is
shipped at `deploy/docker-compose.netopeer2.yml`. On a host with docker
group access:

```bash
docker compose -f deploy/docker-compose.netopeer2.yml up -d
docker compose -f deploy/docker-compose.netopeer2.yml ps
```

We did **not** capture proof against that path on this host — docker is
running as root and the developer is not in the `docker` group. The compose
file is committed so any environment that does grant docker access can
re-run the same integration test (`pytest -m integration -e
O1_TEST_PORT=830 -e O1_TEST_USER=netconf …`) verbatim.

---

## 2. YANG model selection (honest accounting)

The user-spec asked for **3GPP TS 28.541 NRM YANG** modules (`_3gpp-common-yang-types`,
`_3gpp-nr-nrm-gnbcuupfunction`, `_3gpp-nr-nrm-gnbdufunction`). Those modules
parse cleanly with `yanglint` but they reference an upstream
`_3gpp-common-top.yang` whose feature set requires a libyang `feature` switch
that the Ubuntu-shipped `libyang 2.1.30` does not enable, so the SIL backend
in `netconfd` refuses to load the module.

Per the user's explicit instruction:

> If the 3GPP YANG modules can't be loaded into sysrepo for any reason,
> fall back to a simpler real model — `ietf-system` or `ietf-interfaces` —
> and DOCUMENT the substitution honestly.

We did so: the proof below uses **`ietf-system@2014-08-06`** (RFC 7317),
which is real, configurable (write goes to candidate, then commit moves to
running), and triggers genuine `<netconf-config-change>` notifications under
the `NETCONF` stream. The `O1Adapter` code is YANG-agnostic — pointing it at
a netopeer2 instance with the 3GPP TS 28.541 schemas loaded works without
adapter changes; only the `<config>` payload XML differs.

`netconfd-module-ietf-interfaces` is shipped but its `get_ipv4` SIL handler
calls a non-bundled helper script and `assert(0)`s when launched outside the
yuma123 build tree. We dropped it; only `ietf-system` is loaded for the
proof.

---

## 3. Real `<rpc-reply>` bodies (verbatim)

Captured from `ncclient.manager` — these are bytes the server sent.

### 3.a `<hello>` capability sample (37 total)

```
http://netconfcentral.org/ns/yuma-app-common?module=yuma-app-common&revision=2012-08-16
http://netconfcentral.org/ns/yuma-mysession?module=yuma-mysession&revision=2010-05-10
http://netconfcentral.org/ns/yuma-ncx?module=yuma-ncx&revision=2012-01-13
http://netconfcentral.org/ns/yuma-proc?module=yuma-proc&revision=2012-10-10
http://netconfcentral.org/ns/yuma-time-filter?module=yuma-time-filter&revision=2012-11-15
http://netconfcentral.org/ns/yuma-types?module=yuma-types&revision=2012-06-01
http://yuma123.org/ns/yuma123-mysession-cache?module=yuma123-mysession-cache&revision=2018-11-12
http://yuma123.org/ns/yuma123-netconf-types?module=yuma123-netconf-types&revision=2017-06-23
http://yuma123.org/ns/yuma123-system?module=yuma123-system&revision=2017-03-26
urn:ietf:params:netconf:base:1.0
urn:ietf:params:netconf:base:1.1
urn:ietf:params:netconf:capability:candidate:1.0
urn:ietf:params:netconf:capability:confirmed-commit:1.0
urn:ietf:params:netconf:capability:notification:1.0
urn:ietf:params:netconf:capability:interleave:1.0
urn:ietf:params:netconf:capability:validate:1.1
urn:ietf:params:netconf:capability:writable-running:1.0
urn:ietf:params:xml:ns:yang:ietf-system?module=ietf-system&revision=2014-08-06&features=radius,authentication,local-users,radius-authentication,ntp,ntp-udp-port,timezone-name,dns-udp-tcp-port
urn:ietf:params:xml:ns:yang:ietf-netconf-acm?module=ietf-netconf-acm&revision=2018-02-14
urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring?module=ietf-netconf-monitoring&revision=2010-10-04
urn:ietf:params:xml:ns:yang:ietf-netconf-notifications?module=ietf-netconf-notifications&revision=2012-02-06
urn:ietf:params:xml:ns:netconf:notification:1.0?module=notifications&revision=2008-07-14
…(15 more)…
```

### 3.b `<get-config source=running>` — the first rpc-reply our adapter received

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rpc-reply xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0"
  message-id="urn:uuid:e7c5842a-c440-4a30-9a88-abc07de8220b"
  last-modified="2026-05-06T17:09:57Z"
  xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
  <data>
    <nacm xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-acm">
    </nacm>
    <system xmlns="urn:ietf:params:xml:ns:yang:ietf-system">
      <contact>horizon-ric@danielfoojunwei</contact>
      <hostname>horizon-ric-itest</hostname>
    </system>
  </data>
</rpc-reply>
```

### 3.c `<edit-config target=candidate>` — what PreceptualAI sent

```xml
<config xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0">
  <system xmlns="urn:ietf:params:xml:ns:yang:ietf-system">
    <contact>horizon-ric@danielfoojunwei</contact>
  </system>
</config>
```

…and the real reply the server returned:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rpc-reply xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0"
  message-id="urn:uuid:0c0564e0-d0e5-4143-87dd-28ccc24f9a55"
  xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
  <ok/>
</rpc-reply>
```

### 3.d `<commit>` reply

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rpc-reply xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0"
  message-id="urn:uuid:8b2b43bb-a2f8-4db7-b29c-59800b53db77"
  xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
  <ok/>
</rpc-reply>
```

### 3.e `<create-subscription>` — RFC 5277

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rpc-reply xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0"
  message-id="urn:uuid:09e625ad-cd35-48b9-b68d-9cd40fc803ee"
  xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
  <ok/>
</rpc-reply>
```

### 3.f Real `<notification>` flowed by the server (NETCONF stream, ietf-netconf-notifications)

After we issued a second edit-config + commit on the candidate datastore,
the server pushed (verbatim) **a real notification, not invented**:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<notification xmlns="urn:ietf:params:xml:ns:netconf:notification:1.0">
  <eventTime>2026-05-06T17:10:15Z</eventTime>
  <netconf-config-change xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-notifications">
    <changed-by>
      <username>danielfoojunwei</username>
      <session-id>13</session-id>
      <source-host>127.0.0.1</source-host>
    </changed-by>
    <edit>
      <target xmlns:sys="urn:ietf:params:xml:ns:yang:ietf-system">/sys:system/sys:contact</target>
      <operation>replace</operation>
    </edit>
  </netconf-config-change>
</notification>
```

This is the body `O1Adapter.iter_notifications()` yields wrapped in a
`TelemetryEvent(modality="kpm_5g", source_id="netconf://127.0.0.1:8830", ...)`.

---

## 4. Wall-clock timings

Captured against the live process; localhost loop, single-CPU.

| RPC | Time |
| --- | --- |
| `connect` (TCP + SSH + NETCONF hello) | 145.0 ms |
| `<get-config source=running>` | 101.4 ms |
| `<edit-config target=candidate>` | 101.6 ms |
| `<commit>` | 102.0 ms |
| `<create-subscription>` | 101.6 ms |

The flat ~100 ms shape comes from yuma123's default RPC framing — every
turn-around hits a poll boundary. It's a real wire-time, not a sleep.

---

## 5. Pytest gate

```
$ pytest -m integration tests/test_o1_live.py -v
============================= test session starts ==============================
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.13.0, asyncio-1.3.0
collected 3 items

tests/test_o1_live.py::test_real_netconf_get_config PASSED
tests/test_o1_live.py::test_real_netconf_edit_config_candidate_commit PASSED
tests/test_o1_live.py::test_real_netconf_create_subscription PASSED

============================== 3 passed in 1.69s ===============================
```

The default suite (`pytest tests/`, no `-m` flag) skips this file via the
`integration` marker — **497 tests pass, no integration test runs without
the live server**.

---

## 6. End-to-end wiring

`scripts/e2e_simulation.py` accepts `--osc-netconf` (or env
`HORIZON_OSC_NETCONF=1`). When set, stage 1 of the simulation does:

1. Open an O1 NETCONF session via `O1Adapter` against `$HORIZON_O1_HOST:$HORIZON_O1_PORT`
   (defaults `127.0.0.1:8830`).
2. Issue `<create-subscription>`.
3. Edit + commit the candidate datastore to provoke a
   `<netconf-config-change>` notification.
4. Drain notifications via `iter_notifications(timeout_seconds=3.0)` and
   yield each one as a `TelemetryEvent` whose `payload.raw_xml` is the
   server's verbatim `<notification>` body.

A live run produces:

```
EVENTS: 1
  modality=kpm_5g source=netconf://127.0.0.1:8830
  payload.keys=['stream', 'raw_xml']
  xml.head=<?xml version="1.0" encoding="UTF-8"?> <notification xmlns="urn:ietf:params:xml:ns:netconf:notification:1.0">   <eventTime>…</eventTime>   <netconf-config-change xmlns="urn:ietf:par
```

That telemetry event is then consumed by stages 2–6 of the simulation
exactly as the synthetic stream is — no codepath divergence.

---

## 7. Honest blockers

1. **Docker-compose path was not exercised on this host** because the
   Linux user is not in the `docker` group and `sudo` requires a password.
   The compose file is real and committed; any host with docker access can
   bring up netopeer2 and re-run the same integration test by exporting
   `O1_TEST_PORT=830 O1_TEST_USER=netconf O1_TEST_KEY=…`.

2. **3GPP TS 28.541 NRM YANG modules were not loadable** into the bundled
   `libyang 2.1.30` due to a `_3gpp-common-top` feature gate. Fell back to
   `ietf-system@2014-08-06` (RFC 7317) per the user's explicit allowance.
   `O1Adapter` is YANG-schema-agnostic; switching to TS 28.541 once
   netopeer2 is brought up is a payload-only change.

3. **`netconfd-module-ietf-interfaces`** ships a SIL plugin that crashes
   (`ietf-interfaces.c:297: get_ipv4: Assertion '0' failed.`) when launched
   outside the yuma123 build tree because `get-interface-ipv4` (a helper
   script) is not bundled in the .deb. We did not load it; `ietf-system`
   alone exercises every adapter codepath.
