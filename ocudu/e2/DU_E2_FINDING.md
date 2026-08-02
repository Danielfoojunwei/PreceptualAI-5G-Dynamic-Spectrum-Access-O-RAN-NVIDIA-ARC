# The DU-side E2 agent DOES join — and this document previously said otherwise

> **CORRECTION.** An earlier version of this file concluded that the OCUDU
> DU-side E2 agent "never attempts a connection" and that closing G4's
> delivery half "needs a change to OCUDU, not to Horizon". **Both statements
> were wrong.** The agent is created, its Style 2 Action 6 executor is
> registered, and it joins the RIC successfully — once the gNB is given an RF
> driver that does not block forever waiting for a counterparty that is not
> there. The fix was ours, and it is two lines of configuration.
>
> The superseded claim, and how it was refuted, are kept below rather than
> deleted.

_Verified 2026-08-02 on OCUDU `f46f580` and FlexRIC `ef6d722f` under the
SCTP→UDP shim. Working configuration:
[`ocudu-gnb-e2-du-joins.yml`](ocudu-gnb-e2-du-joins.yml)._

## The result

Both ends witness three E2 nodes, not one.

**RIC:**

```
[E2AP]: E2 SETUP-REQUEST rx from PLMN   1. 1 Node ID 411 RAN type ngran_gNB
[NEAR-RIC]: Accepting RAN function ID 2 with def = ORAN-E2SM-KPM
[NEAR-RIC]: Accepting RAN function ID 3 with def = ORAN-E2SM-RC
[E2AP]: E2 SETUP-REQUEST rx from PLMN   1. 1 Node ID 411 RAN type ngran_gNB_CUUP ID 0
[NEAR-RIC]: Accepting RAN function ID 2 with def = ORAN-E2SM-KPM
[NEAR-RIC]: Accepting RAN function ID 3 with def = ORAN-E2SM-RC
[E2AP]: E2 SETUP-REQUEST rx from PLMN   1. 1 Node ID 411 RAN type ngran_gNB_DU ID 0
[NEAR-RIC]: Accepting RAN function ID 2 with def = ORAN-E2SM-KPM
[NEAR-RIC]: Accepting RAN function ID 3 with def = ORAN-E2SM-RC
```

**gNB:**

```
[E2-DU   ] [I] "RIC Connection Setup Routine" started.
[E2-DU   ] [I] E2: Connection to Near-RT-RIC on 127.0.0.1:36421 established
[E2-DU   ] [D] E2 Setup: node component config [0] interface_type=3 req_bytes=178 resp_bytes=52
[E2-DU   ] [I] E2AP msg, "successfulOutcome.E2setupResponse", transaction id=0
[E2-DU   ] [I] E2 Setup procedure successful.
[E2-DU   ] [I] Added supported RAN function with id 3 and OID 1.3.6.1.4.1.53148.1.1.2.3
```

`E2-CU-CP`, `E2-CU-UP` and `E2-DU` all reach *E2 Setup procedure successful*.

## The cause

`lib/du/du_high/o_du_high_impl.cpp`:

```cpp
du_hi->start();          // with the ZMQ driver and no peer: NEVER RETURNS
if (e2agent) {
  e2agent->start();      // therefore never reached
}
```

With `device_driver: zmq` and nothing listening on `tcp://127.0.0.1:2001`,
`du_hi->start()` blocks permanently inside `mac_cell_processor::start()`. A
gdb backtrace of the hung process shows the main thread parked in
`futex_util::wait` under `du_manager_controller_impl::start()`, while the sole
`main_pool#0` worker is blocked in a `sync_task_executor` and `phy_worker`
sits idle — a self-deadlock on a single-threaded main pool.

So the DU E2 agent was never disabled, never misconfigured, and never lost its
flag. It was **constructed, given its Style 2 Action 6 executor, and then
stranded behind a call that never returned.**

The single-variable experiment — the reference config with *only* the RF
driver changed:

| arm | E2 units reaching setup | DU bring-up lines |
|---|---|---|
| `device_driver: zmq` | CU-CP | **0** |
| `device_driver: realtime_loopback` | CU-CP, CU-UP, **DU** | **734** |

The second required change, `expert_execution.threads`, is not tuning: without
it `realtime_loopback` aborts with *"Maximum PUSCH and SRS concurrency (i.e.,
2) exceeds the number of main pool threads (i.e., 1)"*.

## What the earlier version claimed, and why it was wrong

It asserted:

> "The DU comes up. `Cell pci=1, bw=10 MHz, 1T1R, dl_arfcn=368500 (n3)` and
> the ZMQ front-end binds. **This is not a DU that failed to start.**"

That was the load-bearing error. Those lines are printed while the cell is
being *configured*; they say nothing about `start()` completing. The DU never
came up. The giveaway sat in the same logs the whole time and went unread:
**zero** `[FAPI]`/`[RU]`/`[SCHED]`/`[MAC]` lines in the ZMQ run, against 734
in the working one. A configuration banner was mistaken for evidence of
bring-up.

It also declined to name a cause — correctly, given two prior wrong answers —
but then framed the blocker as upstream: *"closing it needs a host where
OCUDU's DU E2 agent starts, a change to OCUDU, not to Horizon."* Wrong in both
halves. No host change was needed and no OCUDU change was needed.

The full list of superseded claims about this one subsystem:

1. *Each unit creates its own `e2:` section.* Refuted: `add_subcommand` is
   get-or-create and `add_option` fans out.
2. *The DU E2 schema is never registered.* Refuted: it is, via the
   split-specific unit schemas.
3. *The SCTP shim / a second concurrent association is the candidate.*
   Refuted: the DU never opened a socket, and CU-UP joins over that same shim.
4. *The DU E2 agent never attempts a connection, and closing G4 needs an
   OCUDU change.* Refuted here.

Four wrong answers, each produced by reasoning over source or partial logs
rather than by an experiment that varied one thing. What settled it was
instrumentation, a gdb backtrace of the live hung process, and a
single-variable config diff — not more reading.

## Consequence for G4

G4 is *"a RIC Control Request derived from a signed certificate is accepted by
an E2 node, and one derived from a refused decision cannot be constructed at
all."*

- **Readability half — closed.** FlexRIC's independent asn1c C codec decodes
  Horizon's payload in full; see
  [`../verify_g4_codec_interop.py`](../verify_g4_codec_interop.py).
- **Refusal half — closed.** `control_from_disposition` raises on a refused
  disposition; tested in `tests/test_e2_rc_control.py`.
- **Delivery half — NOW ATTEMPTABLE, and not yet done.** The DU E2 agent is
  joined and the RIC has accepted RAN function 3 (`ORAN-E2SM-RC`) for it, so
  the Style 2 Action 6 executor finally has a live association. What remains
  is an xApp that issues the control and an observation of the executor
  accepting it. That has **not** been run. Nothing here should be read as
  claiming G4 is closed.

## Reproduce

```sh
gcc -shared -fPIC -O2 -o shim.so deploy/e2-companion/sctp_udp_shim.c -ldl
LD_PRELOAD=$PWD/shim.so <flexric>/build/examples/ric/nearRT-RIC &
LD_PRELOAD=$PWD/shim.so <ocudu>/build/apps/gnb/gnb \
    -c ocudu/e2/ocudu-gnb-e2-du-joins.yml
```

Then check for three `E2 SETUP-REQUEST rx` lines at the RIC, including
`RAN type ngran_gNB_DU`, and for `[E2-DU   ] … E2 Setup procedure successful`
at the gNB. `ocudu/e2/du_e2_probe.py` automates the check;
`ocudu/tests/test_du_e2_probe.py` is its positive control.

## Still not determined

Why the main pool reports a single thread.
`expert_execution.threads.main_pool.nof_threads: 4` was supplied both as YAML
and on the command line and the pool was still reported as 1, on a host with
`nproc` = 4. Whether the ZMQ hang would also occur with a genuinely larger
main pool was not tested. This does not affect the result above — the working
configuration is verified end to end — but it means the ZMQ hang is
*characterised*, not fully explained.
