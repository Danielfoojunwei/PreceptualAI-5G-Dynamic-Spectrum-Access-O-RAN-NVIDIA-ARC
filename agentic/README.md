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
| `telemetry_trust.py` | Authenticated source (HMAC, domain-separated), freshness both ways, replay by sequence and in-batch, declared physical bounds, per-source field authorization, median corroboration. Missing evidence is a refusal, never a default. |
| `identity.py` | Ed25519 signature over the envelope plus a nonce, verified against an operator-held registry. Authority decides what a principal may do; this decides *which principal it is*. |
| `envelope.py` | `AgentActionEnvelope` — delegated authority, target domain, resource, and which keys that authority may **mutate**. Enforced at admission. |
| `aggregate.py` | Invariants over a *set* of actions, grouped by resource, reusing `InvariantCheck` so aggregate evidence has the same shape as per-action evidence. |
| `bundle.py` | `SafetyTransaction` — trust, identity, authority, per-action projection, aggregates, resolution, then commit-all-or-refuse-all, certified either way. |
| `conflict.py` | Deterministic resolution: culpability first, then operator-programmed priority. Drops requests; never rewrites values. |
| `evidence.py` | `TransactionCertificate` for **every** transaction including refusals, signed and hash-chained with tamper localisation. |

Aggregate limits implemented: absolute slice capacity floor, PRB conservation,
site EIRP budget, spectral separation between carriers, aggregate PFD.

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
| Envelope nonce replay check disabled | `identity_enforced` fails |
| Refusals no longer carry their reasons into the certificate | `evidence_signed_and_chained` fails |
| Predecessor hash no longer hashed into each chain entry | `evidence_signed_and_chained` fails |
| Guard removed that turns "dropped everyone" into a refusal | `no_silent_partial_commit` fails |

Two properties are pinned by tests rather than by G6, and it is worth saying
which: culpability-before-priority in conflict resolution, and telemetry being
unable to overwrite the decision baseline. Mutating either fails
`test_adversarial_findings.py` and leaves the gate green.

**Two falsifications were wrong before they were right.** Removing the
empty-drop guard left G6 green because the scenario refused earlier, at
authority — the envelopes declared a bandwidth the starved baseline did not
have, so the branch under test was never reached. And the chain falsification
only rewrote a record without re-linking its successor, which the stored
`prev_hash` comparison catches on its own; the property that actually needs
hashing the predecessor is resistance to a *fully re-linked* chain. Both checks
were rebuilt to exercise the real branch.

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

## The adversarial review, and what it found

Four independent reviewers were pointed at this subtree and told to break it.
Everything below was reproduced against running code, not theorised, and each
is now pinned by a test in `test_adversarial_findings.py`.

| Finding | Why it mattered |
|---|---|
| A slice-scoped agent raised transmit power 7 dB | `mutates` defaulted to empty and the baseline check was optional, so undeclared keys were never scope-checked. The Shield clamped the value to something *legal* and emitted it — clamping restores physics, not authority. |
| The operator root could be impersonated | A one-element chain `(ROOT,)` passes every structural check, and the narrowing loop `zip(chain, chain[1:])` is empty, so nothing is ever narrowed. The root now may not present envelopes at all. |
| Telemetry could overwrite the decision baseline | Trusted telemetry and caller context shared one flat namespace; a field named `baseline_action` replaced the baseline with a float and switched the anti-smuggling check off. Telemetry is now namespaced. |
| Conflict resolution was a weapon | `bandwidth_hz` is a required key, so every action "touched" what the capacity floor reads and the victim was chosen purely by priority — an agent could breach an aggregate and watch a rival be dropped. Culpability now outranks priority. |
| Nine envelopes refused the whole control plane | A flood under one id exhausted the round budget. One request per principal; rounds scale with the bundle. |
| `committed=True` with no members and no refusals | Resolution could empty the bundle, find the baseline satisfactory, and report unqualified success for a transaction that emitted nothing. |
| A non-ASCII auth tag *raised* instead of refusing | `hmac.compare_digest` rejects non-ASCII `str`; the exception escaped the gate. An unauthenticated party could break the fail-closed discipline with 64 accented characters. |
| One bad packet vetoed every decision | The verdict folded in per-record checks, making the authenticated-records filter unreachable. Anyone able to put a packet on the bus could silence the domain. |
| A sequence of 2^63 silenced a source permanently | Forward jumps are now bounded. |
| The corroboration merge was attacker-controlled | It took the lexicographically-first source, and this repository's own test claimed that was "the only rule an attacker cannot influence by changing its measurement". False: the attacker changes its *name*, not its measurement. Registering as `aaa-sensor` won every contested field, for free, forever. Corroborated fields now take the median. |
| A bundle could allocate 160% of the cell's PRBs | `ProtectedSliceFloorInvariant` asks whether one slice has enough; nothing asked whether the cell had that much to give. |
| `AggregateEirpBudget` manufactured 3 dB | It summed bundle members, so a second agent *declaring* the power it was not changing added phantom transmit power. Aggregates now group by `resource_id`. |

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
- **Nothing batches the agents.** This is the most important limitation and it
  is not fixed. `SafetyTransaction.evaluate` is a pure function over a bundle
  someone else assembled: there is no queue, no epoch, no submission window and
  no state between calls. Two agents submitting *separately* each commit, and
  the 12.5 MHz outcome in the table above happens anyway — through the very
  component built to prevent it. Every result here is conditional on a
  serialisation point that does not exist yet, and building one is the
  precondition for both a system-derived baseline and any real emission path.
- **No emission path.** Nothing carries a committed transaction to A1 or E2.
  Worse, the existing wire gates would not help if one were added naively:
  both inspect the *per-action* certificate, and every member of a refused
  bundle carries a clean one — that is the premise of the whole design. The
  structural property the main package is proudest of, that a Shield-refused
  action cannot become an E2 control message, has no analogue at the
  transaction layer yet.
- **"Commit" means "return a tuple".** No two-phase protocol, no rollback, no
  idempotency key, no acknowledgement from the network.
- **No wire format.** The schemas under `schemas/` describe the envelope and
  the certificate, and a drift test keeps them honest against the dataclasses,
  but there is no serialiser, no parser, no ingress and no key-distribution
  story for `AgentRegistry`. An external agent cannot construct an envelope
  except by importing the Python dataclass in-process.
- **No runnable demonstration**, only a gate and tests.
- **Not in the proposal.** See the first paragraph.
