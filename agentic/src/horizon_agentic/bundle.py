"""Validate, project and commit a whole multi-agent plan — or refuse it as a unit.

A cross-domain plan that half-executes is worse than one that does not execute
at all. If an energy action lands and the slice action it was paired with does
not, the network is left in a state no agent asked for and no certificate
describes. So the unit of enforcement here is the transaction, not the action.

The order of the stages is the argument:

1. **Telemetry trust.** Untrusted state refuses everything, before any agent's
   action is even read. Projection computed from false state produces a safe
   action for a world that does not exist, and signs it.
2. **Authority.** Every envelope must be admissible. One agent reaching outside
   its scope refuses the transaction rather than just its own member — a bundle
   is a joint proposal, and a joint proposal containing an unauthorised request
   is an unauthorised proposal.
3. **Per-action projection.** Each action goes through the *existing*
   ``horizon_ric`` Shield, unmodified. Nothing in this package weakens or
   replaces that path; a member the Shield blocks refuses the transaction.
4. **Aggregate invariants.** Only now, over actions already individually safe,
   are the limits that are properties of the combination evaluated.
5. **Conflict resolution.** Deterministic, operator-ordered, drop-only.
6. **Commit or refuse.** All admitted members, or none.

Stage 3 is worth stating plainly: the per-action guarantee is not relaxed to
make room for the aggregate one. An action that the single-action chain refuses
is still refused, whatever the bundle says. The aggregate layer can only ever
subtract from what is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from horizon_agentic.aggregate import AggregateInvariant
from horizon_agentic.conflict import ConflictResolution, resolve_conflicts
from horizon_agentic.envelope import (
    AgentActionEnvelope,
    AuthorityPolicy,
    AuthorityVerdict,
    authorize_all,
)
from horizon_agentic.telemetry_trust import (
    TelemetryRecord,
    TelemetryTrustGate,
    TrustVerdict,
)
from horizon_ric.shield.certificate import InvariantCheck, SafetyCertificate
from horizon_ric.shield.shield import Shield

__all__ = [
    "TransactionResult",
    "SafetyTransaction",
    "TransactionRefused",
]


class TransactionRefused(RuntimeError):
    """Raised when a refused transaction's actions are read as though committed."""


@dataclass(frozen=True)
class CommittedMember:
    """One agent's action as it would actually be emitted, with its certificate."""

    agent_id: str
    action: Mapping[str, Any]
    certificate: SafetyCertificate


@dataclass(frozen=True)
class TransactionResult:
    """Everything the transaction decided, and the actions only if it committed."""

    committed: bool
    refusals: tuple[str, ...]
    authority: tuple[AuthorityVerdict, ...]
    trust: TrustVerdict | None
    per_action_blocked: tuple[str, ...]
    aggregate_checks: tuple[InvariantCheck, ...]
    resolution: ConflictResolution | None
    _members: tuple[CommittedMember, ...] = ()

    @property
    def members(self) -> tuple[CommittedMember, ...]:
        """The actions to emit. Raises unless the transaction committed.

        Mirrors ``TrustVerdict.state``: there is no way to obtain the part of a
        refused plan that happened to pass, because emitting it is exactly the
        partial application the transaction exists to prevent.
        """
        if not self.committed:
            raise TransactionRefused(
                "transaction was refused; no action may be emitted. reasons: "
                + "; ".join(self.refusals)
            )
        return self._members


class SafetyTransaction:
    """Enforcement over a bundle of agent-proposed actions.

    ``shield`` is an ordinary ``horizon_ric`` Shield and is used exactly as a
    single-planner deployment would use it. ``trust_gate`` is optional only so
    that the aggregate behaviour can be tested in isolation; a deployment
    without one has not removed the unconditional-trust assumption and should
    say so.
    """

    def __init__(
        self,
        shield: Shield,
        policy: AuthorityPolicy,
        aggregates: Sequence[AggregateInvariant],
        *,
        trust_gate: TelemetryTrustGate | None = None,
        max_resolution_rounds: int = 8,
    ) -> None:
        self._shield = shield
        self._policy = policy
        self._aggregates = list(aggregates)
        self._trust = trust_gate
        self._max_rounds = max_resolution_rounds

    def evaluate(
        self,
        envelopes: Sequence[AgentActionEnvelope],
        *,
        telemetry: Sequence[TelemetryRecord] | None = None,
        context: Mapping[str, Any] | None = None,
        decision_id: str = "",
    ) -> TransactionResult:
        refusals: list[str] = []
        ctx: dict[str, Any] = dict(context or {})

        # 1. Telemetry trust ------------------------------------------------
        trust_verdict: TrustVerdict | None = None
        if self._trust is not None:
            trust_verdict = self._trust.admit(list(telemetry or ()))
            if not trust_verdict.trusted:
                for check in trust_verdict.refusals:
                    refusals.append(f"telemetry.{check.check_id}: {check.detail}")
                return TransactionResult(
                    False, tuple(refusals), (), trust_verdict, (), (), None
                )
            ctx.update(dict(trust_verdict.state))
        elif telemetry:
            refusals.append(
                "telemetry supplied but this transaction has no trust gate; "
                "refusing rather than trusting it implicitly"
            )
            return TransactionResult(False, tuple(refusals), (), None, (), (), None)

        if not envelopes:
            refusals.append("empty bundle: nothing to authorize or enforce")
            return TransactionResult(
                False, tuple(refusals), (), trust_verdict, (), (), None
            )

        # 2. Authority ------------------------------------------------------
        # The baseline is the network state the bundle is proposed against.
        # Supplying it is what turns `mutates` from a self-report into a
        # checkable claim; see `envelope.authorize`.
        baseline = ctx.get("baseline_action")
        verdicts = authorize_all(
            envelopes,
            self._policy,
            baseline=baseline if isinstance(baseline, Mapping) else None,
        )
        # Refuse on the verdict, not on whether it happened to carry text.
        # An earlier version keyed off `problems` being non-empty, which is the
        # same thing today and silently would not be if `authorize` ever
        # reported an advisory. Falsifying the G6 authority check found it:
        # forcing `authorized=True` left the gate green, because the problems
        # list was still populated and still triggered the refusal.
        unauthorized = [v for v in verdicts if not v.authorized]
        for verdict in unauthorized:
            if verdict.problems:
                for problem in verdict.problems:
                    refusals.append(f"authority.{verdict.agent_id}: {problem}")
            else:
                refusals.append(
                    f"authority.{verdict.agent_id}: refused without a stated reason"
                )
        if unauthorized:
            return TransactionResult(
                False, tuple(refusals), verdicts, trust_verdict, (), (), None
            )

        # 3. Per-action projection through the existing Shield --------------
        safe_actions: list[Mapping[str, Any]] = []
        certificates: list[SafetyCertificate] = []
        blocked: list[str] = []
        for position, env in enumerate(envelopes):
            disposition = self._shield.dispose(
                dict(env.requested_action),
                ctx,
                decision_id=f"{decision_id}:{env.agent_id}" if decision_id else env.agent_id,
            )
            safe_actions.append(disposition.safe_action)
            certificates.append(disposition.certificate)
            if disposition.certificate.emit_blocked:
                blocked.append(
                    f"{env.agent_id} (position {position}): "
                    f"violated {disposition.certificate.violated_ids}"
                )
        if blocked:
            refusals.extend(f"per_action_blocked: {entry}" for entry in blocked)
            return TransactionResult(
                False,
                tuple(refusals),
                verdicts,
                trust_verdict,
                tuple(blocked),
                (),
                None,
            )

        # 4-5. Aggregate invariants, then deterministic resolution ----------
        agent_ids = [env.agent_id for env in envelopes]
        resolution = resolve_conflicts(
            safe_actions,
            agent_ids,
            self._aggregates,
            self._policy,
            ctx,
            max_rounds=self._max_rounds,
        )
        if not resolution.resolved:
            for check in resolution.checks:
                if not check.satisfied:
                    refusals.append(f"aggregate.{check.invariant_id}: {check.detail}")
            if not refusals:
                refusals.append(
                    "conflict resolution exhausted its rounds without satisfying "
                    "the aggregate invariants"
                )
            return TransactionResult(
                False,
                tuple(refusals),
                verdicts,
                trust_verdict,
                (),
                resolution.checks,
                resolution,
            )

        # 6. Commit ---------------------------------------------------------
        members = tuple(
            CommittedMember(
                agent_id=agent_ids[i],
                action=safe_actions[i],
                certificate=certificates[i],
            )
            for i in resolution.admitted
        )
        return TransactionResult(
            True,
            (),
            verdicts,
            trust_verdict,
            (),
            resolution.checks,
            resolution,
            members,
        )
