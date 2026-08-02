"""Validate, project and commit a whole multi-agent plan — or refuse it as a unit.

A cross-domain plan that half-executes is worse than one that does not execute
at all. If an energy action lands and the slice action it was paired with does
not, the network is left in a state no agent asked for and no certificate
describes. So the unit of enforcement here is the transaction, not the action.

The order of the stages is the argument:

1. **Telemetry trust.** Untrusted state refuses everything, before any agent's
   action is even read. Projection computed from false state produces a safe
   action for a world that does not exist, and signs it.
2. **Identity.** Who is asking. Authority decides what a principal may do,
   which presupposes knowing which principal it is; an unauthenticated
   ``agent_id`` is a string anyone can set.
3. **Authority.** Every envelope must be admissible. One agent reaching outside
   its scope refuses the transaction rather than just its own member — a bundle
   is a joint proposal, and a joint proposal containing an unauthorised request
   is an unauthorised proposal.
4. **Per-action projection.** Each action goes through the *existing*
   ``horizon_ric`` Shield, unmodified. Nothing in this package weakens or
   replaces that path; a member the Shield blocks refuses the transaction.
5. **Aggregate invariants.** Only now, over actions already individually safe,
   are the limits that are properties of the combination evaluated.
6. **Conflict resolution.** Deterministic, operator-ordered, drop-only.
7. **Commit or refuse — and certify either way.**

Stage 4 is worth stating plainly: the per-action guarantee is not relaxed to
make room for the aggregate one. An action that the single-action chain refuses
is still refused, whatever the bundle says. The aggregate layer can only ever
subtract from what is emitted.

Stage 7 is the one that was missing. Every evaluated transaction now yields a
:class:`~horizon_agentic.evidence.TransactionCertificate` — committed or
refused — optionally signed and appended to a hash chain. Refusals need this
more than commits do: a commit leaves a trace in the network, a refusal leaves
nothing behind unless something writes it down, and the refusal is what an
operator will later be asked to justify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from horizon_agentic.aggregate import AggregateInvariant
from horizon_agentic.conflict import ConflictResolution, resolve_conflicts
from horizon_agentic.envelope import (
    AgentActionEnvelope,
    AuthorityPolicy,
    AuthorityVerdict,
    authorize_all,
)
from horizon_agentic.evidence import (
    ChainEntry,
    TransactionCertificate,
    TransactionEvidenceChain,
    check_dict,
    digest_of,
    signed_transaction,
    utc_now_iso,
)
from horizon_agentic.identity import EnvelopeAuthenticator, IdentityVerdict
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
    "CommittedMember",
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
    identity: tuple[IdentityVerdict, ...]
    authority: tuple[AuthorityVerdict, ...]
    trust: TrustVerdict | None
    per_action_blocked: tuple[str, ...]
    aggregate_checks: tuple[InvariantCheck, ...]
    resolution: ConflictResolution | None
    certificate: TransactionCertificate
    chain_entry: ChainEntry | None = None
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
    single-planner deployment would use it.

    ``trust_gate`` and ``authenticator`` are optional only so that the
    aggregate behaviour can be tested in isolation. A deployment without a
    trust gate has not removed the unconditional-trust assumption on telemetry,
    and a deployment without an authenticator is treating ``agent_id`` as
    self-asserted; both should say so rather than imply otherwise.

    ``signing_key`` and ``chain`` control evidence. A certificate is produced
    either way — the key adds a signature, the chain adds tamper localisation.
    """

    def __init__(
        self,
        shield: Shield,
        policy: AuthorityPolicy,
        aggregates: Sequence[AggregateInvariant],
        *,
        trust_gate: TelemetryTrustGate | None = None,
        authenticator: EnvelopeAuthenticator | None = None,
        signing_key: Ed25519PrivateKey | None = None,
        chain: TransactionEvidenceChain | None = None,
        max_resolution_rounds: int = 8,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self._shield = shield
        self._policy = policy
        self._aggregates = list(aggregates)
        self._trust = trust_gate
        self._auth = authenticator
        self._key = signing_key
        self._chain = chain
        self._max_rounds = max_resolution_rounds
        self._clock = clock

    # ── evidence ─────────────────────────────────────────────────────────
    def _certify(
        self,
        transaction_id: str,
        *,
        committed: bool,
        refusals: Sequence[str],
        trust: TrustVerdict | None,
        identity: Sequence[IdentityVerdict],
        authority: Sequence[AuthorityVerdict],
        aggregate_checks: Sequence[InvariantCheck],
        resolution: ConflictResolution | None,
        members: Sequence[CommittedMember],
    ) -> tuple[TransactionCertificate, ChainEntry | None]:
        authenticated_by = {v.agent_id: v.authenticated for v in identity}
        if authority:
            principals = tuple(
                {
                    "agent_id": v.agent_id,
                    "authenticated": authenticated_by.get(v.agent_id),
                    "authorized": v.authorized,
                    "problems": list(v.problems),
                }
                for v in authority
            )
        else:
            # Identity and telemetry refusals happen before authority runs.
            # Without this branch the certificate for those cases would name no
            # agent at all, which is precisely the record an auditor needs.
            principals = tuple(
                {
                    "agent_id": v.agent_id,
                    "authenticated": v.authenticated,
                    "authorized": None,
                    "problems": list(v.problems),
                }
                for v in identity
            )

        certificate = TransactionCertificate(
            transaction_id=transaction_id,
            issued_at=self._clock(),
            committed=committed,
            refusals=tuple(refusals),
            telemetry_checks=tuple(
                check_dict(c) for c in (trust.checks if trust else ())
            ),
            authority_verdicts=principals,
            aggregate_checks=tuple(check_dict(c) for c in aggregate_checks),
            dropped=tuple(
                {
                    "agent_id": d.agent_id,
                    "priority": d.priority,
                    "violated_id": d.violated_id,
                    "detail": d.detail,
                }
                for d in (resolution.dropped if resolution else ())
            ),
            admitted=tuple(
                {
                    "agent_id": m.agent_id,
                    "action_digest": digest_of(dict(m.action)),
                    "certificate_digest": digest_of(m.certificate.to_dict()),
                }
                for m in members
            ),
            resolution_rounds=resolution.rounds if resolution else 0,
        )
        if self._chain is not None:
            entry = self._chain.append(certificate, private_key=self._key)
            return entry.certificate, entry
        if self._key is not None:
            return signed_transaction(certificate, self._key), None
        return certificate, None

    # ── main entry point ─────────────────────────────────────────────────
    def evaluate(
        self,
        envelopes: Sequence[AgentActionEnvelope],
        *,
        telemetry: Sequence[TelemetryRecord] | None = None,
        context: Mapping[str, Any] | None = None,
        transaction_id: str = "",
        decision_id: str = "",
    ) -> TransactionResult:
        refusals: list[str] = []
        ctx: dict[str, Any] = dict(context or {})
        identity: tuple[IdentityVerdict, ...] = ()
        authority: tuple[AuthorityVerdict, ...] = ()
        trust_verdict: TrustVerdict | None = None
        decision_id = decision_id or transaction_id

        def refuse(
            *,
            blocked: Sequence[str] = (),
            aggregate_checks: Sequence[InvariantCheck] = (),
            resolution: ConflictResolution | None = None,
        ) -> TransactionResult:
            certificate, entry = self._certify(
                transaction_id,
                committed=False,
                refusals=refusals,
                trust=trust_verdict,
                identity=identity,
                authority=authority,
                aggregate_checks=aggregate_checks,
                resolution=resolution,
                members=(),
            )
            return TransactionResult(
                False,
                tuple(refusals),
                identity,
                authority,
                trust_verdict,
                tuple(blocked),
                tuple(aggregate_checks),
                resolution,
                certificate,
                entry,
            )

        # 1. Telemetry trust ------------------------------------------------
        if self._trust is not None:
            trust_verdict = self._trust.admit(list(telemetry or ()))
            if not trust_verdict.trusted:
                refusals.extend(
                    f"telemetry.{c.check_id}: {c.detail}" for c in trust_verdict.refusals
                )
                return refuse()
            # Namespaced, not merged. Sharing one flat namespace let a
            # telemetry field named `baseline_action` overwrite the baseline
            # with a float — `isinstance(baseline, Mapping)` then failed and
            # the anti-smuggling check switched off for the entire bundle.
            # Telemetry is input to a decision; it does not get to redefine the
            # decision's frame.
            ctx["telemetry"] = dict(trust_verdict.state)
        elif telemetry:
            refusals.append(
                "telemetry supplied but this transaction has no trust gate; "
                "refusing rather than trusting it implicitly"
            )
            return refuse()

        if not envelopes:
            refusals.append("empty bundle: nothing to authorize or enforce")
            return refuse()

        # One principal, one request per transaction. Without this a flood of
        # envelopes under one id — a rival's, since `agent_id` is only as good
        # as the authenticator — exhausts conflict resolution and refuses the
        # honest members along with the forged ones.
        ids = [e.agent_id for e in envelopes]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            refusals.append(
                f"bundle contains more than one request from {repeated}; each "
                "principal gets one request per transaction"
            )
            return refuse()

        # 2. Identity -------------------------------------------------------
        if self._auth is not None:
            identity = self._auth.authenticate_all(envelopes)
            unauthenticated = [v for v in identity if not v.authenticated]
            for verdict in unauthenticated:
                if verdict.problems:
                    refusals.extend(
                        f"identity.{verdict.agent_id}: {p}" for p in verdict.problems
                    )
                else:
                    refusals.append(
                        f"identity.{verdict.agent_id}: refused without a stated reason"
                    )
            if unauthenticated:
                return refuse()

        # 3. Authority ------------------------------------------------------
        # The baseline is the network state the bundle is proposed against.
        # Supplying it is what turns `mutates` from a self-report into a
        # checkable claim; see `envelope.authorize`.
        baseline = ctx.get("baseline_action")
        if not isinstance(baseline, Mapping) or not baseline:
            # Refuse rather than degrade. `authorize` can run without a
            # baseline, but only by requiring every carried key to be declared
            # — and a transaction that silently switched to that mode would
            # look identical to one that checked. Making the caller supply the
            # frame is the difference between fail-closed and fail-quiet.
            refusals.append(
                "no baseline_action in context: the network state the bundle is "
                "proposed against is required, or `mutates` cannot be checked "
                "against anything"
            )
            return refuse()
        authority = authorize_all(envelopes, self._policy, baseline=baseline)
        # Refuse on the verdict, not on whether it happened to carry text.
        # An earlier version keyed off `problems` being non-empty, which is the
        # same thing today and silently would not be if `authorize` ever
        # reported an advisory. Falsifying the G6 authority check found it:
        # forcing `authorized=True` left the gate green, because the problems
        # list was still populated and still triggered the refusal.
        unauthorized = [v for v in authority if not v.authorized]
        for denied in unauthorized:
            if denied.problems:
                refusals.extend(
                    f"authority.{denied.agent_id}: {p}" for p in denied.problems
                )
            else:
                refusals.append(
                    f"authority.{denied.agent_id}: refused without a stated reason"
                )
        if unauthorized:
            return refuse()

        # 4. Per-action projection through the existing Shield --------------
        safe_actions: list[Mapping[str, Any]] = []
        certificates: list[SafetyCertificate] = []
        blocked: list[str] = []
        for position, env in enumerate(envelopes):
            disposition = self._shield.dispose(
                dict(env.requested_action),
                ctx,
                decision_id=(
                    f"{decision_id}:{env.agent_id}" if decision_id else env.agent_id
                ),
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
            return refuse(blocked=blocked)

        # 5-6. Aggregate invariants, then deterministic resolution ----------
        agent_ids = [env.agent_id for env in envelopes]
        resolution = resolve_conflicts(
            safe_actions,
            agent_ids,
            self._aggregates,
            self._policy,
            ctx,
            resource_ids=[env.resource_id for env in envelopes],
            # Scale with the bundle: a fixed round budget means a large
            # legitimate bundle exhausts resolution and refuses everything,
            # which is a denial of service reachable by anyone able to add
            # members.
            max_rounds=max(self._max_rounds, len(envelopes)),
        )
        if resolution.resolved and not resolution.admitted:
            # Resolution can empty the bundle and then find the baseline alone
            # satisfies every aggregate — technically resolved, and reported as
            # a commit with no members and no refusals. A caller reading
            # `committed` sees unqualified success for a transaction that
            # emitted nothing and dropped everyone.
            refusals.append(
                "conflict resolution dropped every member: "
                + ", ".join(
                    f"{d.agent_id} (priority {d.priority}, {d.violated_id})"
                    for d in resolution.dropped
                )
            )
            return refuse(
                aggregate_checks=resolution.checks, resolution=resolution
            )

        if not resolution.resolved:
            refusals.extend(
                f"aggregate.{c.invariant_id}: {c.detail}"
                for c in resolution.checks
                if not c.satisfied
            )
            if not refusals:
                refusals.append(
                    "conflict resolution exhausted its rounds without satisfying "
                    "the aggregate invariants"
                )
            return refuse(aggregate_checks=resolution.checks, resolution=resolution)

        # 7. Commit ---------------------------------------------------------
        members = tuple(
            CommittedMember(
                agent_id=agent_ids[i],
                action=safe_actions[i],
                certificate=certificates[i],
            )
            for i in resolution.admitted
        )
        certificate, entry = self._certify(
            transaction_id,
            committed=True,
            refusals=(),
            trust=trust_verdict,
            identity=identity,
            authority=authority,
            aggregate_checks=resolution.checks,
            resolution=resolution,
            members=members,
        )
        return TransactionResult(
            True,
            (),
            identity,
            authority,
            trust_verdict,
            (),
            resolution.checks,
            resolution,
            certificate,
            entry,
            members,
        )
