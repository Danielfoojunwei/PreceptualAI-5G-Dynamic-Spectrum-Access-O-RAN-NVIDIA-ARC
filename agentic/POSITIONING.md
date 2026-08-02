# Positioning: from a safety wrapper to an action-control plane

The category this work belongs in:

> **Horizon-RIC: the deterministic safety plane for agentic telecom networks —
> starting with licensed-spectrum enforcement in the RAN.**

The one-line claim that goes with it:

> Any vendor's AI agent may propose network actions. Only operator-compliant
> actions reach the network, with signed evidence of every correction, refusal
> and execution.

That is a stronger position than "we make reinforcement learning safer",
because it does not depend on the planner being a learner, on the learner being
ours, or on there being only one of them. It also happens to describe what the
code already does more accurately than the old framing did: the Shield never
reads the planner, so "one planner's safety wrapper" was always an
understatement of the mechanism.

## What changed in the industry, and what is actually evidenced

Three claims underpin the reposition. They are separated here by how well each
is sourced, because a positioning document that launders an unchecked claim
into a footnote is the exact failure this repository's claim ledger exists to
prevent.

### Confirmed against primary sources

**Automation platforms are extending past the RAN, and the app ecosystem is
following.** In June 2026 Ericsson announced that its Intelligent Automation
Platform — which runs rApps — will also run **cApps**, automation applications
for the *core* network, presenting itself as the first vendor to introduce the
concept. The announcement describes unified management and automation of RAN
and core from a single open platform, "a single source of truth for network
topology and resource data through the central resource layer", and low-latency
data from ENM's streaming capability. Ericsson states its rApp ecosystem has
more than 100 applications from over 100 members and says it will replicate
that model for cApps.
([press release](https://www.ericsson.com/en/press-releases/2026/6/ericsson-intelligent-automation-platform-expands-to-support-core-network-automation),
[technical blog](https://www.ericsson.com/en/blog/2026/6/intelligent-core-automation-with-eiap-and-capps))

The consequence for us is specific, and it is not "we should build a cApp". It
is that the *number of independent automation applications acting on one
operator's network, from one platform, across more than one domain* is going up
by design — and every one of them is a source of proposed actions.

**Governance of agent interactions is being standardised, and its scope is
identity and communication rather than physical action safety.** TM Forum
launched the first three projects of its AI-Native Blueprint — Model as a
Service, Data Products Lifecycle Management, and **Agentic Interactions
Security** — described as defining critical guardrails for agentic AI and
delivering "a common policy language and ontology for governance and assurance
at scale and in a machine-readable format, without a human in the loop", with
membership including AT&T, Verizon, T-Mobile, Telstra, Telenor, China Telecom
and others.
([TM Forum newsroom](https://www.tmforum.org/news-insight/newsroom/tm-forum-advances-ai-native-blueprint-with-launch-of-first-core-operational-projects-for-ai-at-scale),
[AI-Native Blueprint project](https://www.tmforum.org/ai-native-blueprint-project/))

This is the single most important item for strategy, and it argues for
alignment rather than competition — see [`STANDARDS_MAP.md`](STANDARDS_MAP.md).

### Found but not read

**Trade analysis arguing that agents constitute a new control plane.** A Light
Reading piece — *"Beyond agent mania – the architecture shift reshaping the
telco industry"* — exists at the expected URL under their 6G section, and the
title is consistent with the argument. **The page returned HTTP 403 to an
automated fetch, so its body was not read.** No characterisation of its
argument and no quotation from it appears in this repository, and none should
be added by anyone who has not opened it.

Two cautions for whoever does. First, the claim as usually stated has two
halves — *agents are becoming a control plane* and *regulated actions still
need deterministic boundaries* — and trade coverage often carries the first as
vendor enthusiasm while the second appears only as an analyst aside. Second, a
publication reporting what a vendor executive said is not the publication
making the argument; the attribution has to follow the source.

Nothing in the reposition depends on this article. The Ericsson and TM Forum
sources carry the argument on their own, and both are primary.

## What actually changes technically

Not the core mechanism. Projection onto operator-programmed invariants,
composed to a fixed point, with a fail-closed guard chain and hash-chained
signed evidence, is unchanged and unweakened — every action in a bundle still
passes through the same unmodified Shield.

What changes is the *unit* of enforcement and the *inputs* it is allowed to
trust.

| Industry shift | What it requires | Where it lands |
|---|---|---|
| Many apps and agents collaborating | Identity, delegated authority, target domain, and which keys that authority may change — enforced, not logged | `envelope.py` |
| Agents acting across domains | Invariants over an action *bundle*, not one radio command | `aggregate.py` |
| One agent's action conflicting with another's | Deterministic resolution on an operator-programmed ordering | `conflict.py` |
| Cross-domain actions succeeding or failing together | Validate, project and commit the whole plan, or refuse it as a unit | `bundle.py` |
| Shared telemetry feeding every agent | Freshness, provenance, integrity, plausibility and corroboration before state can influence an enforced action | `telemetry_trust.py` |
| Agents remaining non-deterministic | Deterministic projection and fail-closed refusal, unchanged | `horizon_ric.shield` |

## The weakness this exposes, stated plainly

The proposal's argument assumed trusted telemetry, and that assumption was
load-bearing in a way that was easy to miss because projection is about
outputs. Output projection still prevents an explicitly bounded violation — a
lying sensor cannot make the Shield emit 41 dBm against a 33 dBm ceiling. What
it can do is make the Shield emit the *wrong safe action*: correct with respect
to the invariants, computed from a world that does not exist, and signed.

With one planner reading one operator's telemetry, the blast radius of that was
one loop. With many agents reading one shared cross-domain data layer, a single
stale, replayed or misattributed stream steers several closed loops at once,
and each produces a valid certificate. The evidence chain would faithfully
record a set of decisions nobody could defend.

`telemetry_trust.py` does not fix this — nothing can prove telemetry truthful
from inside the receiver. It removes the *unconditional* assumption and
replaces it with a stated one: authenticated source, bounded staleness in both
directions, replay detection, declared physical bounds, corroboration by an
independent source for nominated critical fields, and refusal when required
evidence is absent. A source that passes all six can still be wrong. It can no
longer be *unexamined*.

## What this does not license us to say

- **Not a working cApp, and not cross-domain enforcement.** Nothing here
  touches a core network. `target_domain` admits the possibility; no core
  action has been built or run. For the AI-RAN Alliance submission the scope
  stays RAN enforcement, with RAN–core transactions described as the expansion
  path and nothing more.
- **Not a replacement for O-RAN conflict mitigation.** See
  [`STANDARDS_MAP.md`](STANDARDS_MAP.md).
- **Not an operator pilot, vendor onboarding, or live deployment.** The
  measured results this repository does have remain the 4658-of-4658 blocking
  result and the gates around it. The multi-agent work adds gate G6, which is
  an offline result against the repository's own Shield, and it should be
  described as exactly that.
