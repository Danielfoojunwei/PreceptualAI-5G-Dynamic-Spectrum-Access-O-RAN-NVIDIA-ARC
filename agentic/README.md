# Horizon-Agentic — the deterministic action-control plane for many agents

**This is a separate, additive part of the repository.** It does not modify
`horizon_ric`, it is not wired into the shipped runtime, and none of it is
claimed in the AI-RAN Alliance proposal under `docs/proposal/`. It imports the
existing Shield read-only and composes on top of it.

That separation is the point. The proposal's evidence base is about one planner
behind one Shield, and it should stay that way until this has evidence of its
own. What is here is real code with real gates — but it is a second track, and
mixing the two would let an unproven architecture borrow the credibility of a
measured result.

## Why it exists

Horizon-RIC's argument is about **outputs**: whatever a planner proposes, only
the projected action reaches the radio. That argument does not weaken with more
agents — but it stops being *sufficient*, in two specific ways this package
addresses.

**Enforcement assumed its inputs were true.** Projection guarantees the emitted
action satisfies the invariants *as evaluated against the state it was given*.
Feed it false state and it computes a different, still-"safe", still-wrong
action, and signs a certificate saying so. One planner consuming telemetry from
its own operator made that tolerable. A shared cross-domain data layer feeding
many agents does not: one stale or misattributed stream steers several closed
loops at once, and every one of them emits a valid signature over a decision
made from fiction. `telemetry_trust.py` replaces the unconditional assumption
with a stated, checkable one.

**Some limits are properties of the combination.** Every invariant in
`horizon_ric.shield.invariants` takes one action and decides it alone. That is
correct for one planner and structurally insufficient for two, and the gap is
not hypothetical — it is already latent in the shipped code:

`ProtectedSliceFloorInvariant` guards a **fraction** of the cell's PRBs. It has
no view of what that fraction is a fraction *of*. So an energy agent that
narrows `bandwidth_hz`, and a slice agent that sets a share at or above the
floor, can each be individually impeccable and jointly cut the protected
slice's real capacity by a factor of four:

| state | bandwidth | `safety_critical` share | its capacity |
|---|---|---|---|
| baseline | 100 MHz | 0.50 | 50.0 MHz |
| energy agent alone | 50 MHz | 0.50 (declared) | 25.0 MHz — ok |
| slice agent alone | 100 MHz (declared) | 0.25 | 25.0 MHz — ok |
| **both applied** | 50 MHz | 0.25 | **12.5 MHz — below a 20 MHz floor** |

Neither action is projected by the Shield. Neither agent exceeds its authority.
The fractional floor is satisfied throughout — by more than double. The
denominator moved, and no per-action invariant can see a denominator another
agent changed.

This is not a defect in `ProtectedSliceFloorInvariant`; a per-action check
cannot be responsible for state it never receives. It is the boundary of what
per-action checking means.

## What is in here

| Module | What it does |
|---|---|
| `telemetry_trust.py` | Authenticated source, freshness, replay, range and physical bounds, cross-source corroboration. Missing evidence is a refusal, never a default. |
| `envelope.py` | `AgentActionEnvelope` — identity, delegated authority, target domain, and which keys that authority may **mutate**. Enforced at admission. |
| `aggregate.py` | Invariants over a *set* of actions, reusing `InvariantCheck` so aggregate evidence has the same shape as per-action evidence. |
| `bundle.py` | `SafetyTransaction` — trust, authority, per-action projection, aggregates, resolution, then commit-all-or-refuse-all. |
| `conflict.py` | Deterministic resolution ordered by operator-programmed priority. Drops requests; never rewrites values. |

Three design decisions worth stating, because each had a tempting alternative:

**Declaring state is not mutating it.** `REQUIRED_ACTION_KEYS` forces every
action to carry `frequency_hz`, `bandwidth_hz` and `tx_power_dBm`, so an
authority model keyed on *key presence* would make a legal slice action
impossible to express. The envelope separates the keys an agent **mutates**
from the state it merely declares, and the transaction checks declared state
against a baseline — otherwise `mutates` is a self-report and an agent can
smuggle a change by omitting it.

**Resolution drops requests, it does not compute replacement values.**
Computing the value that satisfies a violated limit means inventing a number no
agent asked for and no operator authorised, and the number that satisfies one
aggregate limit routinely violates another. Dropping a member means "this
request is not applied this cycle", which returns that dimension to what the
network already has: unambiguously a narrowing, no baseline arithmetic, and
explainable in one sentence to whoever has to answer for it.

**The per-action guarantee is never relaxed.** Every action still goes through
the unmodified `horizon_ric` Shield. The aggregate layer can only subtract from
what is emitted.

## Running it

```sh
PYTHONPATH=src:agentic/src python -m pytest agentic/tests -q
python agentic/verify_g6_multi_agent.py --out agentic/results/g6-multi-agent.json
```

No new dependencies. `telemetry_trust` uses `hmac`/`hashlib` from the standard
library, matching the discipline that kept `jsonschema` out of the main package.

## Gate G6, and that it can fail

`verify_g6_multi_agent.py` pins seven checks and the SHA-256 of every source
file its numbers depend on — including `shield.py` and `invariants.py`, so
editing an invariant to make the result come out right changes the digest.

The first check is the load-bearing one. A "multi-agent conflict" whose members
were not individually legal proves nothing about multi-agent interaction; it is
a single-agent violation with a second actor standing nearby. G6 asserts
non-staging *before* it asserts detection, at both layers, and treats a Shield
projection of either action as disqualifying.

Each check was falsified by hand:

| Falsification | Result |
|---|---|
| Energy agent's bandwidth cut so it alone breaches the floor | `not_staged` fails |
| Absolute floor lowered to 1 MHz so the joint case passes | `joint_violation_detected` fails |
| Resolution ordered by arrival position instead of policy | `order_independent` fails |
| Resolution ignores operator priority | `priority_decides` fails |
| Aggregate loses its baseline fallback (resolution by amnesia) | `atomic_refusal` fails |
| `authorize` forced to admit everything | `authority_enforced` fails |
| Replay detection disabled | `telemetry_gate_refuses` fails |
| `TransactionResult.members` readable when refused | `atomic_refusal` fails |

**One falsification found a real bug rather than confirming a gate.** Forcing
`authorize` to return `authorized=True` left G6 *green*: `SafetyTransaction`
was refusing on the verdict's `problems` list being non-empty, and never read
the `authorized` flag at all. The two agree today and nothing made them agree —
an `authorize` that ever reported an advisory alongside an admission would have
refused it, and a verdict that refused without explaining itself would have
been committed. The transaction now keys off the verdict;
`test_refusal_keys_off_the_verdict_not_off_the_problem_text` pins both
directions.

That is the second time in this repository that falsifying a gate found a
defect in the thing being gated rather than in the gate.

## What is deliberately not claimed

- **No cross-domain enforcement.** Nothing here touches a core network. The
  `target_domain` field admits the *possibility* of a non-RAN domain; no core
  action, no cApp, and no RAN–core transaction has been built or run. Calling
  this cross-domain enforcement would exceed the evidence by a wide margin.
- **Not a conflict-mitigation specification.** O-RAN already carries work on
  A1 policy conflict between rApps in the Non-RT RIC, and there is a literature
  on Near-RT RIC conflict mitigation. This is not an attempt to reinvent or
  replace it — see [`STANDARDS_MAP.md`](STANDARDS_MAP.md).
- **Telemetry is not proven truthful.** Nothing can prove that from inside the
  receiver. The unconditional trust assumption is removed and replaced with a
  stated one; a source that is authentic, punctual, in-range and corroborated
  can still be wrong.
- **No live deployment, no vendor platform, no operator pilot.** Everything
  here runs offline against the repository's own Shield.
- **Not in the proposal.** See the first paragraph.
