"""Bind an NVCF-hosted inference function to a governed agent identity.

The problem
-----------
An AI-RAN agent deployed as an NVCF function is a black box behind an HTTPS
endpoint. It autoscales from zero across multi-region GPU clusters, it can be
redeployed under the same ``functionId`` at any moment, and nothing about the
response tells you which model produced it. That is exactly the principal
:class:`~horizon_agentic.envelope.AgentActionEnvelope` was designed for: it
assumes the proposer is untrusted and makes the *claim* of authority
explicit and checkable.

What this module adds is the binding between the two, and it is stricter than
it looks:

**Identity is the version, not the function.** ``functionId`` survives a
redeploy; ``versionId`` does not. So the agent id is
``nvcf:<functionId>/<versionId>``, and :class:`NvcfHostedAgent` refuses to
build an envelope unless the response came from the version the operator
authorised. A function silently rolled forward to a new model is a *different
principal* and has to be granted authority again.

**A refusal is not a proposal.** Every non-fulfilled disposition —
throttled, refused, still pending at the deadline, an undeclared status —
yields ``None``. There is no fallback action. A control plane that emits
something when its planner is unreachable is worse than one that emits
nothing.

**The response is parsed, never trusted.** The function's output is
attacker-controlled by assumption. Keys outside the granted scopes are
dropped before the envelope is built, non-finite numbers are refused, and
the envelope is signed by *us* with a key the operator registered — the
function has no key and cannot sign anything. NVCF's own transport
authentication proves we reached NVIDIA, not that the model is honest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from horizon_agentic.envelope import SCOPE_ACTION_KEYS, AgentActionEnvelope

from horizon_nvcf.client import NvcfClient
from horizon_nvcf.protocol import Disposition, InvocationOutcome

__all__ = ["AGENT_ID_PREFIX", "NvcfAgentBinding", "NvcfHostedAgent", "agent_id_for"]

AGENT_ID_PREFIX = "nvcf:"

# Keys an NVCF response may carry that are metadata, not action. Present so a
# well-behaved function can annotate its output without those annotations
# being mistaken for a request to change something.
_METADATA_KEYS = frozenset({"model", "model_version", "confidence", "explanation"})


def agent_id_for(function_id: str, version_id: str) -> str:
    """The agent identity for one immutable NVCF function version."""
    if not function_id or not version_id:
        raise ValueError(
            "both functionId and versionId are required: a version-less "
            "identity would survive a redeploy, which is the whole thing "
            "this binding exists to prevent"
        )
    return f"{AGENT_ID_PREFIX}{function_id}/{version_id}"


@dataclass(frozen=True)
class NvcfAgentBinding:
    """The operator's authorisation of one NVCF function version."""

    function_id: str
    version_id: str
    target_domain: str
    granted_scopes: frozenset[str]
    mutates: frozenset[str]
    resource_id: str
    delegation_chain: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.version_id:
            raise ValueError(
                "version_id is required — an NVCF functionId is stable across "
                "redeployments, so binding authority to it would grant a new "
                "model the previous model's privileges"
            )
        unknown = set(self.granted_scopes) - set(SCOPE_ACTION_KEYS)
        if unknown:
            raise ValueError(
                f"unknown scope(s) {sorted(unknown)}; known scopes are "
                f"{sorted(SCOPE_ACTION_KEYS)}"
            )
        if not self.mutates <= self.writable_keys:
            raise ValueError(
                f"binding declares it mutates {sorted(self.mutates - self.writable_keys)}, "
                "which its granted scopes do not cover"
            )

    @property
    def agent_id(self) -> str:
        return agent_id_for(self.function_id, self.version_id)

    @property
    def writable_keys(self) -> frozenset[str]:
        """Union of the action keys the granted scopes permit."""
        keys: set[str] = set()
        for scope in self.granted_scopes:
            keys |= set(SCOPE_ACTION_KEYS[scope])
        return frozenset(keys)


@dataclass(frozen=True)
class ProposalRejection:
    """Why an NVCF response did not become an envelope."""

    reason: str
    outcome: InvocationOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "outcome": self.outcome.to_dict() if self.outcome else None,
        }


class NvcfHostedAgent:
    """One NVCF function version, presented as a governed proposer."""

    def __init__(self, client: NvcfClient, binding: NvcfAgentBinding) -> None:
        if client.cfg.function_id != binding.function_id:
            raise ValueError(
                f"client targets function {client.cfg.function_id!r} but the "
                f"binding authorises {binding.function_id!r}"
            )
        if client.cfg.version_id != binding.version_id:
            raise ValueError(
                f"client targets version {client.cfg.version_id!r} but the "
                f"binding authorises {binding.version_id!r}; invoking the "
                "unversioned path would let a redeploy change the model "
                "behind the operator's back"
            )
        self._client = client
        self.binding = binding

    # ── proposal ─────────────────────────────────────────────────────────
    def propose(
        self, observation: Mapping[str, Any], *, block: str
    ) -> tuple[AgentActionEnvelope | None, ProposalRejection | None]:
        """Ask the function for an action; return an envelope or a rejection.

        Never returns both, and never returns neither.
        """
        outcome = self._client.invoke(dict(observation))
        if outcome.disposition != Disposition.FULFILLED or outcome.payload is None:
            return None, ProposalRejection(
                reason=f"NVCF invocation did not yield a result: {outcome.detail}",
                outcome=outcome,
            )
        return self.envelope_from(outcome, block=block)

    def envelope_from(
        self, outcome: InvocationOutcome, *, block: str
    ) -> tuple[AgentActionEnvelope | None, ProposalRejection | None]:
        """Convert a fulfilled outcome into an envelope, or say why not."""
        payload = outcome.payload
        if payload is None:
            return None, ProposalRejection("outcome carries no payload", outcome)

        # The function may echo which version answered. If it does and it
        # disagrees with the authorised version, that is a redeploy we did not
        # sanction — refuse, do not "helpfully" accept the newer model.
        echoed = payload.get("function_version_id") or payload.get("versionId")
        if echoed and str(echoed) != self.binding.version_id:
            return None, ProposalRejection(
                f"response declares function version {echoed!r} but the "
                f"operator authorised {self.binding.version_id!r}; a "
                "redeployed function is a different principal",
                outcome,
            )

        raw = payload.get("action")
        if not isinstance(raw, Mapping):
            return None, ProposalRejection(
                "response has no `action` object; nothing to propose", outcome
            )

        action, dropped, bad = self._sanitise(raw)
        if bad:
            return None, ProposalRejection(
                f"response action carries unusable value(s) for {sorted(bad)}; "
                "refusing rather than coercing an untrusted number",
                outcome,
            )
        if not action:
            return None, ProposalRejection(
                "response action contains nothing this binding is scoped to "
                f"write (dropped {sorted(dropped)})",
                outcome,
            )

        action["block"] = block
        return (
            AgentActionEnvelope(
                agent_id=self.binding.agent_id,
                agent_version=self.binding.version_id,
                target_domain=self.binding.target_domain,
                granted_scopes=self.binding.granted_scopes,
                delegation_chain=self.binding.delegation_chain,
                requested_action=action,
                mutates=self.binding.mutates,
                resource_id=self.binding.resource_id,
            ),
            None,
        )

    # ── sanitisation ─────────────────────────────────────────────────────
    def _sanitise(
        self, raw: Mapping[str, Any]
    ) -> tuple[dict[str, Any], set[str], set[str]]:
        """Keep in-scope keys with usable values; report the rest.

        Returns ``(action, dropped, unusable)``. Dropping an out-of-scope key
        is not leniency — the envelope's ``written_keys`` is what
        :func:`~horizon_agentic.envelope.authorize` grades, so passing a key
        the binding has no scope for would make the whole envelope refuse.
        Refusing the *whole proposal* because a model added a stray annotation
        is the wrong failure. An out-of-*range* value is different, and is
        never dropped: that is the model asking for something, and the Shield
        is what must answer.
        """
        writable = self.binding.writable_keys
        action: dict[str, Any] = {}
        dropped: set[str] = set()
        unusable: set[str] = set()

        for key, value in raw.items():
            if key in _METADATA_KEYS:
                continue
            if key not in writable:
                dropped.add(key)
                continue
            if isinstance(value, bool):
                unusable.add(key)
                continue
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    unusable.add(key)
                    continue
                action[key] = float(value)
            elif isinstance(value, (str, Mapping, list)):
                # `ntn` is a structured key; strings and containers pass
                # through untouched for the Shield to grade.
                action[key] = value
            else:
                unusable.add(key)

        return action, dropped, unusable
