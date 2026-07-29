# xApp end-to-end proof — full pipeline against the real near-RT RIC A1 stack

_Recorded 2026-07-27 on a clean Linux host (4 vCPU, 15 GiB). Everything
below was produced by real processes on real sockets: no
``httpx.MockTransport``, no in-repo emulators, no stubbed components.
Raw artifacts live in [`xapp-e2e/results/`](xapp-e2e/results/); the
harness that reproduces the stack is [`xapp-e2e/`](xapp-e2e/) and runs
in CI via `.github/workflows/xapp-e2e.yml`._

## What was proven

The complete Horizon-RIC control loop —

```
telemetry (12-event JSONL replay)
  → risk-band planner (rules-based; counterfactual alternatives recorded)
  → Decision Safety Shield (default_terrestrial_shield, 3.40–3.50 GHz, 33 dBm EIRP)
  → emit-guard chain (certificate / budget / audit-completeness gates)
  → A1Adapter.emit_policy()  [dialect "legacy", /A1-P/v2]
  → official O-RAN-SC Go A1 mediator (:10000, RMR 4.9.4, SDL on redis)
  → RMR A1_POLICY_REQ (mtype 20010) → official hw-python xApp (:4560)
  → xApp A1_POLICY_RESP (mtype 20011, handler_id "hw-python")
  → mediator marks enforceStatus=ENFORCED
  → Horizon polls /status and stamps enforcement into its report
  → hash-chained DecisionRecord persisted per decision
```

— executed live, twice, against two independent real targets.

## Run A — official OSC A1 simulator (dialect `osc_a1`)

Target: `o-ran-sc/sim-a1-interface` @ `be2943f5…` (OSC_2.1.0 interface),
built from official source and served on `:8085`.

| Witness | Result |
| --- | --- |
| Live smoke (`scripts/osc_a1_live_smoke.py`) | `"result": "pass"` — healthcheck 200, types 20001–20004 registered, create 202, status has `enforceStatus`, delete 202 ([artifact](xapp-e2e/results/osc-a1-live-result.json)) |
| Full pipeline `--once`, 12 events | `accepted=12 blocked=0 emit_failed=0`, exit 0 ([report](xapp-e2e/results/pipeline-report-osc-sim.json)) |
| Simulator state after run | `GET /a1-p/policytypes` → `[20001, 20002, 20003, 20004]`; the 12 instance ids split 6 (qos.priority) / 3 (traffic.steering) / 3 (admission.control) across the risk bands |
| Enforcement | `NOT_ENFORCED` — correct: no xApp sits behind the simulator, and Horizon reports what the RIC actually said |
| Evidence chain | 12 records, `verify()` first-broken-index `-1` (intact) |

## Run B — real A1 mediator + real hw-python xApp (dialect `legacy`)

Target: `ric-plt/a1` @ `09a757b4…` (Go, one disclosed receive-side
patch — see below) with RMR `8b9a2149…` (4.9.4) built from source,
SDL on stock `redis-server`, and the **official `hw-python` reference
xApp** @ `a6d00525…` on `ricxappframe==2.2.0`.

Three independent witnesses, cross-checked by
`scripts/xapp_e2e_proof.py` → **pass**
([proof artifact](xapp-e2e/results/xapp-e2e-proof.json)):

1. **Pipeline report** ([artifact](xapp-e2e/results/pipeline-report-xapp.json)):
   `events=12 accepted=12 blocked=0 emit_failed=0 enforced=12`,
   audit chain length 12, verify intact, exit 0 under
   `HORIZON_ONCE_REQUIRE_ACCEPTED=1`.
2. **Mediator northbound**: all 12 policy instances returned
   `{"enforceStatus": "ENFORCED"}` — a state the mediator only enters
   after receiving the xApp's `A1_POLICY_RESP` over RMR.
3. **Receiver-side log + RMR stats**: the mediator's log records each
   response with the xApp's self-declared identity, e.g.

   ```json
   {"payload": "{\"qos_objectives\":{\"priority\":2},
                 \"rapp_metadata\":{\"decision_id\":\"31502a17-77d9-461b-850d-4c21f057c5f1\"},
                 \"scope\":{\"slice_id\":\"slice-e2e-replay\"}}",
    "policy_instance_id": "5f4565b4-e203-4a44-bd47-d0ed97748a7a",
    "policy_type_id": "20001", "handler_id": "hw-python", "status": "OK"}
   ```

   — note the xApp echoes the **full Horizon policy payload including
   the `decision_id`**, tying the wire message back to one specific
   hash-chained DecisionRecord. The xApp's RMR stats line confirms the
   count: `target=127.0.0.1:4562 … succ=12 fail=0`.

Every persisted DecisionRecord carries the Shield certificate summary,
the physics-validated safe action (`frequency_hz=3.45e9`,
`bandwidth_hz=2e7`, `tx_power_dBm=24.0` inside the 33 dBm EIRP
envelope), two counterfactual rejected alternatives with
machine-readable causes (e.g. `sla_breach_predicted`), and the
deterministic replay seed.

## Upstream interop bug found (and disclosed)

Pairing the two stock components crashed the mediator on the very first
ACK: `pkg/rmr/messages.go` serialises `policy_type_id` as a JSON
*string* in `A1_POLICY_REQ`, while `Consume()` type-asserts `float64`
on the echoed `A1_POLICY_RESP` —
`panic: interface conversion: interface {} is string, not float64`.
The one-hunk receive-side patch
([`xapp-e2e/patches/a1mediator-policy-resp-type.patch`](xapp-e2e/patches/a1mediator-policy-resp-type.patch))
accepts both encodings. Candidate for an upstream report to
`ric-plt/a1`.

## Scope (what this does NOT prove)

Standalone RMR deployment with a static route table — no platform Route
Manager, no E2 nodes, no RAN traffic, no radio. `hw-python` verifies
the A1 envelope and ACKs; it does not act on Horizon's policy
semantics. Ericsson EIAP / Nokia MantaRay dialects remain wire-contract
validated only (see [`../docs/VENDOR_ONBOARDING.md`](../docs/VENDOR_ONBOARDING.md)).
