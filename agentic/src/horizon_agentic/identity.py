"""Authenticating the envelope, because identity was previously just a claim.

``envelope.py`` decides what an agent is *allowed* to do. It has nothing to say
about whether the agent is who it says it is. As first written, `agent_id` was
a string in a dataclass: any process able to submit a bundle could set it to
`slice-agent` and inherit that grant. Delegation chains had the same problem —
carefully checked for narrowing, and trivially fabricated, because nothing bound
the chain to a key.

That is not a subtle hole, and it makes the authority model decorative on its
own. Scope checking answers "may this principal do this?"; it presupposes an
answer to "which principal is this?" that did not exist.

Two mechanisms here, and both are needed:

**Signature.** The envelope is signed with the agent's Ed25519 key over its
canonical bytes, and verified against an operator-held registry mapping agent
id to public key. Ed25519 rather than an HMAC because the evidence chain has to
be verifiable by someone who cannot forge entries — a shared secret gives
integrity but not non-repudiation, and an auditor holding the verification key
would be able to manufacture the very records they are auditing.

**Nonce.** A signature alone permits replay: capture a legitimate envelope,
resubmit it later, and it verifies perfectly. Every envelope carries a nonce
that must not have been seen from that agent before, and an ``issued_at`` that
bounds how long a captured envelope stays useful. Freshness is what keeps the
nonce set from growing without limit — nonces older than the acceptance window
can be evicted, because staleness refuses them anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from horizon_agentic.envelope import AgentActionEnvelope
from horizon_ric.shield.signing import key_fingerprint

__all__ = [
    "canonical_envelope_bytes",
    "sign_envelope",
    "AgentRegistry",
    "IdentityVerdict",
    "EnvelopeAuthenticator",
]


def canonical_envelope_bytes(env: AgentActionEnvelope) -> bytes:
    """Stable bytes over everything an agent asserts.

    Every field that carries meaning is included — notably ``mutates`` and
    ``delegation_chain``. Omitting either would let an attacker take a
    legitimately signed envelope and rewrite the part that authority actually
    checks, which would make the signature worse than useless by lending
    credibility to a modified request.

    Sets are sorted so that two runs of the same envelope produce identical
    bytes; ``frozenset`` iteration order is not stable across processes.
    """
    payload: dict[str, Any] = {
        "agent_id": env.agent_id,
        "agent_version": env.agent_version,
        "target_domain": env.target_domain,
        "granted_scopes": sorted(env.granted_scopes),
        "delegation_chain": list(env.delegation_chain),
        "mutates": sorted(env.mutates),
        "resource_id": env.resource_id,
        "requested_action": env.requested_action,
        "nonce": env.nonce,
        "issued_at": env.issued_at,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_envelope(
    env: AgentActionEnvelope, private_key: Ed25519PrivateKey
) -> AgentActionEnvelope:
    """Return a copy of ``env`` stamped with its signature and key fingerprint."""
    import dataclasses

    return dataclasses.replace(
        env,
        signature=private_key.sign(canonical_envelope_bytes(env)).hex(),
        signing_key_fingerprint=key_fingerprint(private_key.public_key()),
    )


@dataclass(frozen=True)
class AgentRegistry:
    """Operator-held mapping of agent id to verification key.

    Held by the operator, never supplied by the agent. An agent that could
    register its own key could authenticate as itself under any grant, which
    reduces to no authentication at all.
    """

    keys: Mapping[str, Ed25519PublicKey]

    def key_for(self, agent_id: str) -> Ed25519PublicKey | None:
        return self.keys.get(agent_id)


@dataclass(frozen=True)
class IdentityVerdict:
    """Whether the envelope is from who it claims, and every reason it is not."""

    authenticated: bool
    agent_id: str
    problems: tuple[str, ...] = ()


class EnvelopeAuthenticator:
    """Verifies envelope signatures and refuses replays.

    Stateful, like the telemetry gate and for the same reason: replay detection
    without memory is not detection. One instance per trust domain.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        clock: Callable[[], float],
        max_age_s: float = 30.0,
        max_skew_s: float = 1.0,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self._max_age_s = max_age_s
        self._max_skew_s = max_skew_s
        self._seen: dict[str, dict[str, float]] = {}

    def _evict(self, agent_id: str, now: float) -> None:
        """Drop nonces that staleness would refuse anyway.

        This is what bounds the memory. A nonce older than the acceptance
        window cannot be replayed successfully — the freshness check refuses it
        before the nonce check is consulted — so remembering it buys nothing
        and costs unbounded growth in a long-running control plane.
        """
        horizon = now - self._max_age_s
        seen = self._seen.get(agent_id)
        if not seen:
            return
        self._seen[agent_id] = {n: t for n, t in seen.items() if t >= horizon}

    def authenticate(self, env: AgentActionEnvelope) -> IdentityVerdict:
        problems: list[str] = []
        now = self._clock()

        public_key = self._registry.key_for(env.agent_id)
        if public_key is None:
            return IdentityVerdict(
                False, env.agent_id, (f"no registered key for agent {env.agent_id!r}",)
            )

        if not env.signature:
            problems.append(f"envelope from {env.agent_id!r} is unsigned")
        else:
            try:
                signature = bytes.fromhex(env.signature)
            except ValueError:
                signature = b""
                problems.append(f"envelope from {env.agent_id!r} has a malformed signature")
            if signature:
                try:
                    public_key.verify(signature, canonical_envelope_bytes(env))
                except Exception:
                    problems.append(
                        f"signature on envelope from {env.agent_id!r} does not verify"
                    )

        expected_fp = key_fingerprint(public_key)
        if env.signing_key_fingerprint and env.signing_key_fingerprint != expected_fp:
            # Not a security boundary on its own — the signature already binds
            # the key — but a mismatch means the agent believes it signed with
            # a different key, which is a misconfiguration worth refusing on
            # rather than papering over.
            problems.append(
                f"envelope from {env.agent_id!r} names key fingerprint "
                f"{env.signing_key_fingerprint[:12]}..., registry holds "
                f"{expected_fp[:12]}..."
            )

        age = now - env.issued_at
        if age > self._max_age_s:
            problems.append(
                f"envelope from {env.agent_id!r} is {age:.3f}s old, limit "
                f"{self._max_age_s:.3f}s"
            )
        elif age < -self._max_skew_s:
            problems.append(
                f"envelope from {env.agent_id!r} is {-age:.3f}s in the future"
            )

        if not env.nonce:
            problems.append(
                f"envelope from {env.agent_id!r} carries no nonce; a valid "
                "signature would otherwise be replayable indefinitely"
            )
        else:
            self._evict(env.agent_id, now)
            if env.nonce in self._seen.get(env.agent_id, {}):
                problems.append(
                    f"nonce {env.nonce!r} from {env.agent_id!r} has been seen before"
                )

        verdict = IdentityVerdict(not problems, env.agent_id, tuple(problems))
        if verdict.authenticated:
            # Record only on success, for the same reason the telemetry gate
            # advances its high-water mark only on an admitted batch: a refused
            # envelope that burned its nonce would let an attacker lock a
            # legitimate agent out by replaying its traffic first.
            self._seen.setdefault(env.agent_id, {})[env.nonce] = env.issued_at
        return verdict

    def authenticate_all(
        self, envelopes: Sequence[AgentActionEnvelope]
    ) -> tuple[IdentityVerdict, ...]:
        return tuple(self.authenticate(env) for env in envelopes)

    def seen_count(self, agent_id: str) -> int:
        """Exposed so the eviction bound can be asserted by a test."""
        return len(self._seen.get(agent_id, {}))


def registry_from_keys(
    pairs: Iterable[tuple[str, Ed25519PrivateKey]]
) -> AgentRegistry:
    """Convenience for tests and demos: build a registry from private keys."""
    return AgentRegistry({name: key.public_key() for name, key in pairs})
