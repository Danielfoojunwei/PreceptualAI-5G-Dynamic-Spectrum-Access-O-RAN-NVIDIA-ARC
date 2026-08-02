# The DU-side E2 agent does not join — and that is what blocks G4

_Recorded 2026-08-02, on binaries built from scratch in this container:
OCUDU `f46f580` (`apps/gnb/gnb`) and FlexRIC `ef6d722f`
(`examples/ric/nearRT-RIC`), both rebuilt because the previous container was
recycled. SHA-256 of both in
[`../results/du-e2-probe.json`](../results/du-e2-probe.json)._

## Why this decides G4

Gate G4 is *"a RIC Control Request derived from a signed certificate is
accepted by an E2 node, and one derived from a refused decision cannot be
constructed at all."* Horizon's control message is **E2SM-RC Style 2,
Action 6 — slice-level PRB quota**, because that is the quantity
`ProtectedSliceFloorInvariant` guards.

Reading OCUDU's E2 factories:

| Unit | Control style registered | Executor |
|---|---|---|
| CU-CP | Style **3**, Action **1** (handover control) | `e2sm_rc_control_action_3_1_cu_executor` — `e2_cu_cp_factory.cpp:57` |
| DU | Style **2**, Action **6** (slice PRB quota) | `e2sm_rc_control_action_2_6_du_executor` — `e2_du_factory.cpp:59` |

**The executor for the action Horizon encodes exists only in the DU.** So
whether the DU E2 agent joins is not a detail — it decides whether G4's
delivery half is attemptable at all on this stack.

## What was run

[`du_e2_probe.py`](du_e2_probe.py), twice, against the real binaries under the
[SCTP→UDP shim](../../deploy/e2-companion/sctp_udp_shim.c) (this kernel still
has no SCTP):

| Run | gNB command | RIC saw | gNB E2 units that *tried* | DU joined |
|---|---|---|---|---|
| A | `-c ocudu-gnb-e2.yml` | 2 setups: `ngran_gNB`, `ngran_gNB_CUUP` | `E2-CU-CP`, `E2-CU-UP` | **No** |
| B | `… e2 --enable_du_e2 true` | 1 setup: `ngran_gNB` | `E2-CU-CP` | **No** |

## What is established

- **The option exists.** `gnb e2 --help` lists `--enable_du_e2 BOOLEAN [false]`
  alongside `--enable_cu_cp_e2` and `--enable_cu_up_e2`, in the one shared
  `e2` subcommand.
- **The gNB app wires a DU E2 gateway.** `gnb.cpp:476` builds
  `create_e2_gateway_client(… E2_DU_PPID)` and `gnb.cpp:543` assigns
  `odu_dependencies.e2_client_handler = e2_gw_du.get()`.
- **The DU comes up.** `Cell pci=1, bw=10 MHz, 1T1R, dl_arfcn=368500 (n3),
  dl_freq=1842.5 MHz` and the ZMQ front-end binds `tcp://127.0.0.1:2000`. This
  is not a DU that failed to start.
- **The other two agents join.** CU-CP *and* CU-UP both attach; the RIC
  accepts RAN functions 2 (`ORAN-E2SM-KPM`) and 3 (`ORAN-E2SM-RC`).
- **No DU E2 logger appears under any name.** `ocudulog` pads logger names to
  a fixed width (`[GNB     ]`), and the probe's regexes were widened to
  tolerate trailing spaces *before* this was concluded — the un-padded version
  was also hiding `E2-CU-UP`, so the padding mattered.
- **Forcing the flag changes nothing.** `gnb -c cfg.yml e2 --enable_du_e2 true`
  parses without error and produces the same result.

## What is NOT established

**Where** the DU's `enable_unit_e2` is lost, between the parsed CLI option and
`o_du_high_unit_factory.cpp:225` where it gates the agent. Finding that needs
an instrumented build or a debugger.

No cause is asserted here on purpose. Static reading of this exact subsystem
has produced two wrong answers already, both withdrawn:

1. That CU-CP and DU-high each create their own `e2:` section — refuted:
   `add_subcommand` returns the existing subcommand, and `add_option` *fans
   out*, writing one option's value into every registered parameter.
2. That the DU-high E2 schema is never registered at all — refuted: it is
   registered via `configure_cli11_with_o_du_high_config_schema`, which the
   split-specific unit schemas call.

A third guess is not worth more than the two that were wrong.

## Two defects this probe found in itself

Both were caught by the design rather than by luck, and both are worth stating
because either would have produced a confident wrong answer:

1. **Reading logs before teardown.** FlexRIC's stdout is C stdio
   block-buffered when redirected to a file, so its `E2 SETUP-REQUEST rx` line
   sat unflushed until exit. The first run reported *the RIC saw 0 setups*
   while the gNB reported a successful one. Having two independent witnesses
   is what exposed it; a single-witness probe would have published "no E2
   setup happened".
2. **Assuming logger names are not padded.** The first regexes matched
   `[E2-CU-CP]` but would have missed `[E2-DU   ]`. They were widened before
   the conclusion was drawn — and doing so immediately surfaced `E2-CU-UP`,
   which the narrow version had been silently dropping.

## Consequence

G4 splits into readability and delivery.

- **Readability: closed.** FlexRIC's independent asn1c-generated C codec
  decodes Horizon's payload in full, with correct values — see
  [`../verify_g4_codec_interop.py`](../verify_g4_codec_interop.py).
- **Delivery: blocked here.** Horizon's Style 2 Action 6 control has no
  registered executor on either association that does join. Closing it needs a
  host on which OCUDU's DU E2 agent starts — which is a change to OCUDU's
  configuration or code, not to Horizon's.
