"""Who is asking, under whose authority, and which keys that authority permits.

``horizon_ric.assurance.planner`` defines what a *well-formed* action looks
like — the required keys, the JSON-primitive rule, the sentinel keys a planner
must not set. It deliberately says nothing about *who* proposed it: a single
planner behind a single Shield needs no identity, because there is nobody to
distinguish it from.

With several agents on one control plane that assumption inverts. "The action
is well-formed" no longer implies "this agent was allowed to ask for it". An
energy agent that writes ``prb_allocation``, or a slice agent that moves
``frequency_hz``, produces a perfectly valid action outside its remit — and the
invariant chain will happily project it, because invariants bound *physics and
regulation*, not authority.

So authority is enforced here, at admission, before anything reaches the
Shield. Two properties carry the weight:

**Scopes bound writable keys.** A grant names scopes; each scope names the
action keys it may write. Anything else in the action is a refusal, including
the output sentinels — an agent that could set ``shield_fallback_to`` could
forge the appearance of a Shield-ordered fallback.

**Delegation narrows, never widens.** A chain of principals is valid only if
each step holds a subset of its delegator's scopes and domains. Without that
rule a delegation chain is decoration: any agent could name a powerful
delegator and inherit its reach.

Neither property is novel as security design. What matters is that both are
*checked*, and that the check sits in front of the enforcement path rather than
beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from horizon_ric.assurance.planner import OUTPUT_SENTINEL_KEYS

__all__ = [
    "SCOPE_ACTION_KEYS",
    "ALWAYS_WRITABLE",
    "AgentActionEnvelope",
    "AuthorityGrant",
    "AuthorityPolicy",
    "AuthorityVerdict",
    "authorize",
]

# Which action keys each scope permits. Derived from the real key contract in
# ``horizon_ric.assurance.planner``; ``test_scopes_cover_the_real_key_contract``
# fails if that contract grows a key no scope accounts for, so a new action
# field cannot become silently writable by everyone.
SCOPE_ACTION_KEYS: Mapping[str, frozenset[str]] = {
    "spectrum": frozenset(
        {"frequency_hz", "bandwidth_hz", "tx_power_dBm", "antenna_gain_dBi"}
    ),
    "slice": frozenset({"prb_allocation"}),
    "phy": frozenset(
        {
            "constellation_order",
            "papr_dB",
            "predicted_tbler",
            "baseline_tbler",
            "demap_confidence",
        }
    ),
    "ntn": frozenset({"ntn", "slant_range_m", "sat_antenna_gain_dBi"}),
}

# ``block`` names which decision block the action belongs to. Every agent must
# state it for its action to be routable at all, so it is not scope-gated.
ALWAYS_WRITABLE: frozenset[str] = frozenset({"block"})


@dataclass(frozen=True)
class AgentActionEnvelope:
    """An action, plus the claim of authority under which it is proposed.

    Everything here is a *claim*. :func:`authorize` is what turns a claim into
    an admission, by checking it against the operator's policy — the envelope
    itself proves nothing.
    """

    agent_id: str
    agent_version: str
    target_domain: str
    granted_scopes: frozenset[str]
    delegation_chain: tuple[str, ...]
    requested_action: Mapping[str, Any]
    mutates: frozenset[str] = frozenset()
    # Which physical resource the action addresses — a cell, sector or carrier.
    # Every aggregate limit that matters in RAN is scoped to one: without this,
    # two agents describing the SAME cell are indistinguishable from two
    # carriers, and a summing invariant manufactures power that does not exist
    # while a pair-wise one reports two co-sited carriers as overlapping.
    resource_id: str = ""
    # Identity. `agent_id` alone is an assertion anyone can make; these are what
    # bind it to a key the operator registered, and what stop a captured
    # envelope from being replayed. Verified by
    # `horizon_agentic.identity.EnvelopeAuthenticator`, which
    # `SafetyTransaction` runs before authority — deciding what a principal may
    # do presupposes knowing which principal it is.
    nonce: str = ""
    issued_at: float = 0.0
    signature: str = ""
    signing_key_fingerprint: str = ""

    @property
    def written_keys(self) -> frozenset[str]:
        return frozenset(self.requested_action) - ALWAYS_WRITABLE

    @property
    def declared_only(self) -> frozenset[str]:
        """Keys the action carries but does not claim to change."""
        return self.written_keys - self.mutates

    def to_dict(self) -> dict[str, Any]:
        """The wire form, matching `schemas/agent-action-envelope.schema.json`.

        Sets become sorted lists so two processes serialising the same envelope
        produce identical JSON — `frozenset` iteration order is not stable, and
        an unstable serialisation would break the signature that covers it.
        """
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "target_domain": self.target_domain,
            "granted_scopes": sorted(self.granted_scopes),
            "delegation_chain": list(self.delegation_chain),
            "requested_action": dict(self.requested_action),
            "mutates": sorted(self.mutates),
            "resource_id": self.resource_id,
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "signature": self.signature,
            "signing_key_fingerprint": self.signing_key_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentActionEnvelope":
        """Parse a wire envelope, refusing anything the schema does not declare.

        Unknown keys are a refusal rather than an ignore. The schema is
        `additionalProperties: false`, and silently dropping a field a sender
        believed it had set is how two peers come to disagree about what was
        requested — with a valid signature over the sender's version.
        """
        known = {
            "agent_id", "agent_version", "target_domain", "granted_scopes",
            "delegation_chain", "requested_action", "mutates", "resource_id",
            "nonce", "issued_at", "signature", "signing_key_fingerprint",
        }
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"envelope carries fields this version does not define: {unknown}"
            )
        missing = sorted(
            {"agent_id", "agent_version", "target_domain", "requested_action"}
            - set(payload)
        )
        if missing:
            raise ValueError(f"envelope is missing required fields: {missing}")
        return cls(
            agent_id=str(payload["agent_id"]),
            agent_version=str(payload["agent_version"]),
            target_domain=str(payload["target_domain"]),
            granted_scopes=frozenset(payload.get("granted_scopes") or ()),
            delegation_chain=tuple(payload.get("delegation_chain") or ()),
            requested_action=dict(payload["requested_action"]),
            mutates=frozenset(payload.get("mutates") or ()),
            resource_id=str(payload.get("resource_id", "")),
            nonce=str(payload.get("nonce", "")),
            issued_at=float(payload.get("issued_at", 0.0)),
            signature=str(payload.get("signature", "")),
            signing_key_fingerprint=str(payload.get("signing_key_fingerprint", "")),
        )


@dataclass(frozen=True)
class AuthorityGrant:
    """What the operator has granted one principal.

    ``priority`` is the operator's ordering, not the agent's: it is read from
    this policy during conflict resolution and never from the envelope, so an
    agent cannot promote itself by asserting a number.
    """

    principal_id: str
    domains: frozenset[str]
    scopes: frozenset[str]
    priority: int


@dataclass(frozen=True)
class AuthorityPolicy:
    """The operator's grants, and the root principal a chain must descend from."""

    grants: Mapping[str, AuthorityGrant]
    root_principal: str

    def __post_init__(self) -> None:
        if self.root_principal not in self.grants:
            raise ValueError(
                f"root principal {self.root_principal!r} has no grant; "
                "no delegation chain could ever be valid"
            )
        unknown = {
            scope
            for grant in self.grants.values()
            for scope in grant.scopes
            if scope not in SCOPE_ACTION_KEYS
        }
        if unknown:
            raise ValueError(
                f"policy grants scopes with no key mapping: {sorted(unknown)}"
            )

    def priority_of(self, agent_id: str) -> int:
        grant = self.grants.get(agent_id)
        if grant is None:
            raise KeyError(f"no grant for {agent_id!r}")
        return grant.priority


@dataclass(frozen=True)
class AuthorityVerdict:
    """Whether the envelope may proceed, and every reason it may not."""

    authorized: bool
    agent_id: str
    problems: tuple[str, ...]

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.authorized


def _chain_problems(env: AgentActionEnvelope, policy: AuthorityPolicy) -> list[str]:
    problems: list[str] = []
    chain = env.delegation_chain
    if not chain:
        return [f"agent {env.agent_id!r} presented no delegation chain"]
    if env.agent_id == policy.root_principal:
        # The root delegates; it does not act. Without this, a one-element
        # chain `(ROOT,)` from `agent_id=ROOT` satisfies every structural check
        # — starts at root, ends at the presenting agent, no repeats — and the
        # narrowing loop `zip(chain, chain[1:])` is *empty*, so no narrowing is
        # ever evaluated. The attacker inherits every scope the root holds.
        problems.append(
            f"the operator root {policy.root_principal!r} may not present "
            "envelopes; it exists to delegate authority, not to exercise it"
        )
    if len(chain) < 2:
        problems.append(
            f"delegation chain {list(chain)} has no delegation step in it; "
            "a single-element chain is never narrowed by anything"
        )
    if chain[0] != policy.root_principal:
        problems.append(
            f"delegation chain starts at {chain[0]!r}, not the operator root "
            f"{policy.root_principal!r}"
        )
    if chain[-1] != env.agent_id:
        problems.append(
            f"delegation chain ends at {chain[-1]!r} but the envelope is from "
            f"{env.agent_id!r}"
        )
    if len(set(chain)) != len(chain):
        problems.append(f"delegation chain repeats a principal: {list(chain)}")

    for principal in chain:
        if principal not in policy.grants:
            problems.append(f"delegation chain names ungranted principal {principal!r}")

    # Narrowing: each step must hold no more than its delegator. This is the
    # rule that makes the chain load-bearing rather than decorative.
    for delegator, delegate in zip(chain, chain[1:]):
        up = policy.grants.get(delegator)
        down = policy.grants.get(delegate)
        if up is None or down is None:
            continue
        widened_scopes = down.scopes - up.scopes
        if widened_scopes:
            problems.append(
                f"delegation {delegator!r} -> {delegate!r} widens scopes by "
                f"{sorted(widened_scopes)}"
            )
        widened_domains = down.domains - up.domains
        if widened_domains:
            problems.append(
                f"delegation {delegator!r} -> {delegate!r} widens domains by "
                f"{sorted(widened_domains)}"
            )
    return problems


def authorize(
    env: AgentActionEnvelope,
    policy: AuthorityPolicy,
    *,
    baseline: Mapping[str, Any] | None = None,
) -> AuthorityVerdict:
    """Decide whether ``env`` may be admitted under ``policy``.

    ``baseline`` is the network state the action is proposed against. It is
    what makes ``mutates`` enforceable: every key the action carries but does
    not declare as mutated must match the baseline. Without that check,
    ``mutates`` would be a self-report — an agent could change ``bandwidth_hz``
    while declaring it mutates only ``prb_allocation``, and key-scoping would
    wave it through.

    Passing no baseline checks scoping but not smuggling, which is the right
    behaviour for a caller that genuinely has no current-state view, and is
    why :class:`~horizon_agentic.bundle.SafetyTransaction` supplies one.

    Every problem is collected rather than short-circuited: a refusal that
    names one of four violations invites a caller to fix that one and retry,
    which turns admission control into an oracle.
    """
    problems: list[str] = []

    grant = policy.grants.get(env.agent_id)
    if grant is None:
        return AuthorityVerdict(
            False, env.agent_id, (f"agent {env.agent_id!r} holds no grant",)
        )

    if env.target_domain not in grant.domains:
        problems.append(
            f"agent {env.agent_id!r} targets domain {env.target_domain!r}, "
            f"granted {sorted(grant.domains)}"
        )

    claimed_beyond_grant = env.granted_scopes - grant.scopes
    if claimed_beyond_grant:
        problems.append(
            f"agent {env.agent_id!r} claims scopes it was not granted: "
            f"{sorted(claimed_beyond_grant)}"
        )

    unknown_scopes = env.granted_scopes - set(SCOPE_ACTION_KEYS)
    if unknown_scopes:
        problems.append(f"envelope claims unknown scopes: {sorted(unknown_scopes)}")

    problems.extend(_chain_problems(env, policy))

    # Effective scopes are the intersection: a claim cannot exceed the grant,
    # so an agent gains nothing by over-claiming and is not punished for
    # under-claiming.
    effective = (env.granted_scopes & grant.scopes) - unknown_scopes
    writable: set[str] = set()
    for scope in effective:
        writable |= SCOPE_ACTION_KEYS[scope]

    sentinels = env.written_keys & frozenset(OUTPUT_SENTINEL_KEYS)
    if sentinels:
        problems.append(
            f"agent {env.agent_id!r} set Shield output sentinel(s) {sorted(sentinels)}; "
            "only the Shield may write these"
        )

    # An action must carry every REQUIRED_ACTION_KEY to be well-formed at all,
    # so key-presence cannot be the test for authority: a slice agent has to
    # state the frequency and bandwidth it is allocating within. What it may
    # not do is *change* them. `mutates` draws that line, and the baseline
    # check below is what stops it being a self-report.
    out_of_scope = env.mutates - writable - frozenset(OUTPUT_SENTINEL_KEYS)
    if out_of_scope:
        problems.append(
            f"agent {env.agent_id!r} mutates keys outside its scopes "
            f"{sorted(effective)}: {sorted(out_of_scope)}"
        )

    undeclared = env.mutates - frozenset(env.requested_action)
    if undeclared:
        problems.append(
            f"agent {env.agent_id!r} declares mutations it does not carry: "
            f"{sorted(undeclared)}"
        )

    if baseline is None:
        # Without a baseline there is nothing to compare declared state
        # against, so `mutates` is a self-report: an agent can carry a changed
        # key, omit it from `mutates`, and no scope check will look at it.
        # Refusing unless the action mutates everything it carries keeps the
        # degraded mode sound instead of silently permissive.
        undeclared_keys = sorted(env.declared_only)
        if undeclared_keys:
            problems.append(
                f"agent {env.agent_id!r} carries undeclared state {undeclared_keys} "
                "and no baseline was supplied to check it against; supply a "
                "baseline or declare every key the action carries"
            )
    else:
        smuggled = sorted(
            key
            for key in env.declared_only
            if key in baseline and baseline[key] != env.requested_action[key]
        )
        if smuggled:
            problems.append(
                f"agent {env.agent_id!r} changes {smuggled} without declaring them "
                "as mutations; declared state must match the baseline"
            )
        invented = sorted(
            key for key in env.declared_only if key not in baseline
        )
        if invented:
            problems.append(
                f"agent {env.agent_id!r} asserts state absent from the baseline "
                f"without declaring it a mutation: {invented}"
            )

    return AuthorityVerdict(not problems, env.agent_id, tuple(problems))


def authorize_all(
    envelopes: Sequence[AgentActionEnvelope],
    policy: AuthorityPolicy,
    *,
    baseline: Mapping[str, Any] | None = None,
) -> tuple[AuthorityVerdict, ...]:
    """Authorize a whole bundle, preserving order."""
    return tuple(authorize(env, policy, baseline=baseline) for env in envelopes)
