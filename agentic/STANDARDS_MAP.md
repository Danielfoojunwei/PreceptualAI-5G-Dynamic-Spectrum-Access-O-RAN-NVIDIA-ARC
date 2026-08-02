# Where this sits relative to the standards bodies

The short version: **TM Forum and O-RAN govern how agents identify themselves,
communicate, and avoid stepping on each other. This enforces whether the
physical network action they requested is safe and lawful to execute.** Those
are different questions, and the second one is not answered by the first.

Getting this wrong in either direction is costly. Claiming to solve what a
standards body already specifies invites a one-line rebuttal. Ignoring the
specification produces something no member can adopt.

## TM Forum — Agentic Interactions Security

**What exists.** TM Forum launched the first three projects of its AI-Native
Blueprint: Model as a Service, Data Products Lifecycle Management, and Agentic
Interactions Security. The last is described as a step towards an agentic
security architecture, defining guardrails for agentic AI and agentic
interactions, and delivering a **common policy language and ontology for
governance and assurance at scale, in a machine-readable format, without a
human in the loop**. Participants include AT&T, Verizon, T-Mobile, Telstra,
Telenor, China Telecom, China Unicom and others.
([newsroom](https://www.tmforum.org/news-insight/newsroom/tm-forum-advances-ai-native-blueprint-with-launch-of-first-core-operational-projects-for-ai-at-scale),
[project page](https://www.tmforum.org/ai-native-blueprint-project/))

**The relationship.** A machine-readable governance and policy language is
precisely what `envelope.py` needs and deliberately does not invent well.
`AgentActionEnvelope` and `AuthorityGrant` are a minimal internal vocabulary —
agent identity, delegation chain, scopes, target domain, operator priority — of
exactly the kind TM Forum is standardising. The right move is to express them
in that language once it stabilises, not to compete with it.

What TM Forum's scope does **not** appear to cover, and what this adds: whether
an authorised, well-formed, correctly-governed request is *physically and
legally safe to execute on licensed spectrum*. A policy language can say agent
A may request a bandwidth change. It does not evaluate the resulting emission
against a spectral mask, an EIRP ceiling, or a protected slice's capacity
commitment, and it does not project a non-compliant request onto the compliant
set. That is the seam.

**Action.** Track the Agentic Interactions Security deliverables. When the
policy language and ontology are published, map `AuthorityGrant`,
`AgentActionEnvelope` and the scope-to-key mapping onto it and treat any
mismatch as a defect here rather than there. Offer the aggregate-invariant and
atomic-transaction concepts as an assurance-layer contribution.

> **Not yet done.** No TM Forum deliverable has been read in detail, no mapping
> table exists, and no contribution has been made. This section states an
> intent and a rationale, not an accomplished alignment.

## O-RAN — conflict mitigation

**What exists.** O-RAN has published work on conflict between applications
acting on the same network. On the Non-RT RIC side this includes analysis and
service-level recommendations for detection and avoidance of A1 policy conflict
in the rApp environment; the WG2 Non-RT RIC specifications describe use cases
and the requirements they impose on the Non-RT RIC architecture and the A1 and
R1 interfaces. There is also a body of work on conflict mitigation in the
Near-RT RIC, including a published framework and detection scheme
([arXiv:2305.07117](https://arxiv.org/pdf/2305.07117)) and an industry white
paper on RIC-apps conflict management
([i14y Lab](https://www.i14y-lab.com/file/show/1145/5a39e9/Conflict_Mitigation_WhitePaper_final.pdf)).
O-RAN's own release notes record continued specification activity through 2025
([O-RAN blog](https://www.o-ran.org/blog/60-new-or-updated-o-ran-technical-documents-released-since-march-2025)).

> **Precision limit.** Exact specification designations and version numbers
> have not been pinned to primary O-RAN documents here. Anyone citing this in a
> submission must read the actual specification and quote its designation —
> naming a spec number from a search summary is how a wrong citation gets into
> a proposal.

**The relationship, and the honest part.** O-RAN got here first, and this is
not a conflict-mitigation framework. The distinction that makes the two
complementary rather than redundant:

- **Detection and avoidance** — noticing that two rApps' policies interact, and
  arranging that they do not — is what the O-RAN work addresses, largely at the
  policy and coordination layer.
- **Bounded outcome** — guaranteeing that whatever the coordination layer
  concludes, the resulting action still cannot breach a regulatory or service
  limit, and that a plan which would is refused as a unit — is what this adds.

They compose in one direction: a conflict-mitigation scheme that resolves two
rApps into a combined policy still has to emit something, and that something
should pass through enforcement. Conversely, this package's `conflict.py` is
deliberately thin — priority ordering and drop-only resolution — precisely
because sophisticated conflict *resolution* is O-RAN's problem and not one to
duplicate badly.

**What would be a lie.** Saying that O-RAN has no conflict handling, that this
invents multi-agent arbitration for O-RAN, or that `AbsoluteSliceCapacityFloor`
is a novel category of invariant. It is an ordinary capacity floor; what is
novel — modestly — is evaluating it over a bundle in a system whose enforcement
was otherwise per-action.

## AI-RAN Alliance

The proposal under `docs/proposal/` engages AI-RAN Alliance workstreams
(Data-for-AI, Test Methodology, AI-for-RAN, the endorsed labs) and is where the
Alliance-facing argument lives. **Nothing in `agentic/` is claimed there**, and
the multi-agent work should not be added to that submission: its evidence is
offline and one gate old, and the submission's strength is that every number in
it is measured.

The natural route for this work, if it earns one, is as a later contribution
once there is a result on a member's real planner — not as an additional claim
attached to an existing proposal.

## Summary table

| Body | Governs | This package |
|---|---|---|
| TM Forum Agentic Interactions Security | Agent identity, interaction governance, machine-readable policy language | Should express its envelope in that language; adds physical-action safety |
| O-RAN (Non-RT / Near-RT RIC conflict work) | Detecting and avoiding conflict between rApps/xApps | Adds a bounded outcome and atomic refusal beneath whatever coordination decides |
| O-RAN A1 / E2 | Policy and control interfaces | Already carried in `horizon_ric` — A1 assurance envelope, E2SM-RC construction |
| AI-RAN Alliance | AI-RAN research, data, test methodology | Engaged by the proposal; this work is not claimed there |
