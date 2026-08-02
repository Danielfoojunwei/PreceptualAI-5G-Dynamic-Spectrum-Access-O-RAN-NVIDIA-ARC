# The "runs unattended" demonstration — runbook and honesty ledger

## What the proposal promises

§VII of the proposal (`docs/proposal/content.py:410-425`) tells the Alliance:

> Prototype: the enforcement layer running as a non-real-time rApp with a live
> demonstration — **policy emission through the production A1 mediator into a
> real xApp, including a Shield-corrected over-power proposal refused rather
> than emitted — which runs unattended** and is available to the Alliance on
> request.

This document says, without embellishment, exactly what a reader can run, what
it needs, what the committed evidence shows, and **where the proposal's single
sentence is broader than the artifact** so the author can fix either the text
or the artifact before anyone shows this to the Alliance.

Gate this document's claims against the committed evidence at any time, with no
Docker and no build:

```bash
/home/user/venv/bin/python audit/check_unattended.py
```

It exits non-zero if any claim below has drifted from the committed files.

---

## The honest headline

The proposal's one sentence describes **one** unattended demonstration. In the
repository it is actually **two separate proofs, run by two scripts, in two
different CI jobs**:

| # | What §VII says | Artifact that proves it | CI job | Committed evidence |
|---|----------------|--------------------------|--------|--------------------|
| A | "policy emission through the production A1 mediator into a real xApp" | `scripts/xapp_e2e_proof.py` driven after `deploy/xapp-e2e/run_stack.sh` | `.github/workflows/xapp-e2e.yml` (`xapp-e2e`) | `deploy/xapp-e2e/results/xapp-e2e-proof.json` |
| B | "a Shield-corrected over-power proposal refused rather than emitted" | `deploy/xapp-e2e/a1_assurance_proof.py` | `.github/workflows/osc-a1-integration.yml` (`a1-assurance-wire`) | `deploy/xapp-e2e/results/a1-assurance-wire-proof.json` |

They do **not** run in a single process, and B does **not** go through the
mediator+xApp path that A exercises — B runs against the O-RAN-SC A1
*simulator* in-process. See the two gaps at the bottom of this file.

"Runs unattended" is true in the sense that **both jobs run without a human in
CI** (they are `pull_request`/`workflow_dispatch` GitHub Actions jobs). There
was, until this runbook, **no single local one-command launcher** for path A.
This runbook ships one: `audit/run_unattended.sh`.

---

## Path A — emission into a real xApp (the main event)

### The one command (live)

```bash
XAPP_PYTHON="$(command -v python3.11)" \
PYTHON=/home/user/venv/bin/python \
audit/run_unattended.sh
```

That wrapper chains the three upstream steps that `deploy/xapp-e2e/README.md`
documents separately:

1. `deploy/xapp-e2e/run_stack.sh` — clone and **build from pinned O-RAN-SC
   source** the RMR C library, the Go A1 mediator, and the `hw-python`
   reference xApp; start redis, mediator, xApp; wait for the mediator
   northbound on `:10000`.
2. `scripts/run_horizon_rapp.py --once --once-max-events 12` — drive the full
   Horizon pipeline (telemetry → planner → Shield → A1 emit) against the live
   mediator, writing a pipeline report.
3. `scripts/xapp_e2e_proof.py` — cross-check three independent witnesses and
   emit the proof JSON.

### Real prerequisites (this fails closed if they are missing)

- **Network egress** to `https://gerrit.o-ran-sc.org` — the stack is cloned,
  not vendored.
- **Build toolchain**: `gcc`, `cmake`, `make`, **Go ≥ 1.21**.
- **`redis-server`** on `PATH` (the SDL backend).
- **A Python 3.11 interpreter for the xApp venv only** — the xApp's
  `ricxappframe → redis → hiredis` chain has a `setup.py` that imports the
  stdlib `imp`, removed in 3.12. Point `XAPP_PYTHON` at 3.11; the Horizon
  pipeline itself runs on `PYTHON` (3.12+ is fine).
- **Free TCP ports**: `10000` (mediator northbound), `4560`/`4562` (RMR),
  `6379` (redis).
- **No Docker** is needed for path A — `run_stack.sh` builds from source.

### What a viewer will see

- `run_stack.sh` prints `A1 mediator northbound live on :10000; hw-python xApp
  on RMR :4560.`
- The pipeline report shows `accepted: 12`, `enforced: 12`, `blocked: 0`,
  `emit_failed: 0`.
- `xapp_e2e_proof.py` prints a JSON proof ending in `"result": "pass"` in which:
  - the mediator reports `enforceStatus: ENFORCED` for all 12 policy ids —
    which it only does after the xApp ACKs each one over RMR;
  - the mediator receive-log carries a `hw-python` `A1_POLICY_RESP` per policy;
  - the xApp's RMR send-stats show ≥ 12 successful sends to the mediator.

### If you cannot run Docker / cannot build the stack

Read the committed proof directly:

```
deploy/xapp-e2e/results/xapp-e2e-proof.json      # the three-witness proof
deploy/xapp-e2e/results/pipeline-report-xapp.json # the rApp's own report
```

and verify it is internally coherent (no build, no network) with:

```bash
/home/user/venv/bin/python audit/check_unattended.py
```

---

## Path B — the Shield-blocked proposal refused rather than emitted

### The one command (live)

```bash
git submodule update --init --depth 1 third_party/sim-a1-interface
/home/user/venv/bin/python -m pip install -r deps/locks/a1sim.txt
/home/user/venv/bin/python deploy/xapp-e2e/a1_assurance_proof.py \
    --out /tmp/a1-assurance-wire.json
```

### Real prerequisites

- The **`third_party/sim-a1-interface` git submodule** (the real, pinned
  O-RAN-SC A1 simulator — commit `be2943f…`).
- The simulator's pinned Flask/Connexion runtime (`deps/locks/a1sim.txt`).
- **No Docker** — `a1_assurance_proof.py` starts the simulator in-process on a
  free port and tears it down; on a host with Docker it would use it, but it
  falls back to native launch and records `docker_available: false`.

### What a viewer will see (committed at `deploy/xapp-e2e/results/a1-assurance-wire-proof.json`)

The `negative_case` block:

```json
"negative_case": {
  "refused": true,
  "certificate": { "emit_blocked": true, "safe": false,
                   "violated_ids": ["numeric_domain_sanity"] },
  "blocked_policy_absent_from_simulator": true,
  "policy_get_status_after": 404,
  "refusal_message": "HORIZON_A1_REQUIRE_CERT is set and the safety
    certificate for the horizon.qos.priority emit is not clean
    (safe=False, emit_blocked=True) — refusing"
}
```

i.e. a Shield-blocked proposal offered to `emit_policy` under
`HORIZON_A1_REQUIRE_CERT=1` is **refused before any HTTP happens**, and a
follow-up `GET` on that policy id returns `404` — proof that nothing was
emitted. The positive case in the same file shows a clean certificate crossing
the A1 socket with an Ed25519 signature that verifies.

---

## The two gaps between §VII and the artifact (read before quoting §VII)

**Gap 1 — one sentence, two proofs.** §VII reads as a single unattended
demonstration. In the repo, emission (path A) and refusal (path B) are separate
scripts, separate CI jobs, and path B does not traverse the mediator+xApp that
path A proves. The emission path itself refuses **nothing** — its committed
proof has `blocked: 0`. Either soften §VII to "two complementary
demonstrations", or wire a combined driver that emits *and* refuses through the
same mediator run.

**Gap 2 — "over-power" is not the invariant actually exercised, and it cannot
be.** §VII says the refused proposal is an **over-power** proposal. The
committed refused proposal (`a1_assurance_proof.py:463`) sets
`bandwidth_hz = -1.0`, which violates `numeric_domain_sanity` (negative
bandwidth is *unfixable by projection*) — **not** `max_eirp`.

This was first recorded here as "the proof is missing", with the suggested fix
"change the negative case to a real over-power violation". **That suggestion
was wrong**, and [`verify_refusal_semantics.py`](verify_refusal_semantics.py)
is the gate that establishes why:

&gt; `MaxEirpInvariant.project` subtracts the overage from `tx_power_dBm`
&gt; unconditionally. There is no over-power action it cannot repair, so there
&gt; is no over-power action it refuses. **2688 over-power actions** — 0.1 dB to
&gt; 967 dB above the ceiling, crossed with antenna gain, constellation order,
&gt; PAPR and TBLER — produce **zero** refusals.

Refusal in this Shield is reserved for actions no projection can repair. That
is the design, not an omission: the Shield is a *projection operator*. So
§VII's sentence describes behaviour the chain deliberately does not have, and
no test will produce it. **The fix is a wording change, not a missing proof.**

The gate proposes this replacement, which is what the demonstration actually
shows and is the stronger claim anyway — the action reaching the RAN is not
the action the planner proposed:

&gt; ~~"a Shield-corrected over-power proposal refused rather than emitted"~~
&gt; **"an over-power proposal corrected to the EIRP ceiling before emission, and
&gt; a non-physical proposal refused outright"**

Both halves are proven: a 40.0 dBm proposal is clamped to exactly 33.0 dBm and
emitted, and the negative-bandwidth proposal is refused with a 404 on the
policy id.

Changing §VII is the author's call — nothing under `audit/` edits the proposal.

`audit/check_unattended.py` asserts the refusal that is *actually committed*
(negative bandwidth) and emits a `[NOTE]` flagging this wording gap; it
deliberately does **not** assert "over-power", because that would be asserting
something the evidence does not contain.

### A claim this subtree got wrong, and the correction

Chasing Gap 2 turned up a defect in `audit/constants/swap_harness.py`, which
had published:

&gt; "The default chain converges in &lt;= 1 projection pass across 4 infeasible
&gt; scenarios, so the disposition is INSENSITIVE to max_passes for any value
&gt; &gt;= 1."

**False.** `tx_power_dBm = 33.1` at `antenna_gain_dBi = 60.0` clamps to
`-26.999999999999993`, whose EIRP is `33.00000000000001` — margin `-7.1e-15`,
still violated. It needs a **second** pass. At `max_passes = 1` that action is
*refused*; at 2 it is corrected. So the 1 → 2 boundary is genuinely
behavioural and only values ≥ 2 are headroom.

Two independent defects produced the wrong answer, and both are now fixed:

1. **The battery held only large, obvious overages.** A 7 dB overage converges
   in one pass and a 0.1 dB one does not — what matters is whether the float
   subtraction rounds back exactly, not how big the violation is. The
   counterexample is now scenario 5.
2. **The measurement stopped at the first pair of agreeing dispositions.** The
   residue case is blocked at *both* `max_passes=0` and `max_passes=1`, so
   "first repeat" concluded zero passes were needed. Convergence is now read
   from the settled end: the smallest `max_passes` whose disposition equals
   the one at 8 and never changes again.

Per-scenario convergence is now reported as `[1, 1, 1, 1, 2]`.

This also reclassifies `max_eirp` itself. It is **DEFERRING**, not
CORRECTING: on 96 of 2688 probes its own projection neither repairs the
violation nor refuses, leaving it to the Shield's fixed-point loop. That loop
is doing real work here rather than being unused headroom.

---

## Files this runbook references (all exist; `audit/tests/test_unattended.py` enforces it)

- `audit/run_unattended.sh` — the one-command path-A launcher (new here).
- `audit/check_unattended.py` — the offline documentation-integrity gate.
- `deploy/xapp-e2e/run_stack.sh`, `deploy/xapp-e2e/README.md`,
  `deploy/xapp-e2e/source-replay.yaml`.
- `scripts/run_horizon_rapp.py`, `scripts/xapp_e2e_proof.py`,
  `deploy/xapp-e2e/a1_assurance_proof.py`.
- `deploy/xapp-e2e/results/xapp-e2e-proof.json`,
  `deploy/xapp-e2e/results/pipeline-report-xapp.json`,
  `deploy/xapp-e2e/results/a1-assurance-wire-proof.json`.
- `.github/workflows/xapp-e2e.yml`, `.github/workflows/osc-a1-integration.yml`.
