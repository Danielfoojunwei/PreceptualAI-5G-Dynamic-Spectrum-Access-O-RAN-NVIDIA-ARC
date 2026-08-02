"""The serialisation point, without which none of the rest means anything.

An adversarial review put this first and it was right to. Everything the
aggregate layer does — detecting that two agents jointly breach a limit,
dropping the culpable one, refusing atomically — is conditional on both agents'
actions arriving in *one* evaluation. ``SafetyTransaction.evaluate`` is a pure
function over a bundle somebody else assembled. Nothing in the package caused
that bundle to exist, required it, or admitted the dependency.

The consequence was not academic. Two agents submitting separately each
committed, and the exact harm the README opens with — a protected slice cut
from 50 MHz to 12.5 MHz — happened anyway, through the component built to
prevent it. A control plane whose safety property holds only if callers
voluntarily batch is not a control plane.

:class:`TransactionCycle` is the missing component. Agents submit into an open
epoch and get a **receipt**, not a decision. Nothing is evaluated, nothing is
projected and nothing is emitted until the epoch closes, at which point every
request in it is decided together. An agent acting alone is simply an epoch
with one member, which is the correct semantics rather than a special case.

Two further things fall out of having a cycle, and both fix defects that had no
clean solution without one:

**The baseline stops being caller-asserted.** ``authorize``'s anti-smuggling
check compares declared state against the network's current state, and until
now that state arrived as raw caller input — unauthenticated, optional, and
load-bearing. The cycle reads it from a ``baseline_source`` at close time,
which makes it system-derived. An agent cannot supply the frame its own request
is checked against.

**Ordering becomes a property of the epoch, not of arrival.** Submissions are
sorted deterministically before evaluation, so two agents racing produce the
same decision either way, and the certificate can be replayed from its record.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from horizon_agentic.bundle import SafetyTransaction, TransactionResult
from horizon_agentic.envelope import AgentActionEnvelope

__all__ = [
    "SubmissionReceipt",
    "CycleClosed",
    "TransactionCycle",
]


class CycleClosed(RuntimeError):
    """Raised on submission to an epoch that has already been decided."""


@dataclass(frozen=True)
class SubmissionReceipt:
    """Acknowledgement that a request was *queued*. Not a decision.

    Returning a receipt rather than a result is the whole point of the type. An
    agent that received a decision on submission would have been decided in
    isolation, which is the failure this class exists to remove.
    """

    accepted: bool
    epoch: int
    agent_id: str
    position: int = -1
    reason: str = ""


@dataclass
class _Epoch:
    index: int
    opened_at: float
    envelopes: list[AgentActionEnvelope] = field(default_factory=list)
    closed: bool = False


class TransactionCycle:
    """Collects agent requests into epochs and decides each epoch as a unit.

    Thread-safe: a control plane taking submissions from several agents is
    concurrent by definition, and the epoch buffer is mutable shared state. The
    lock is held only around buffer mutation and the close handoff, not across
    evaluation, so a slow Shield does not block submissions to the next epoch.
    """

    def __init__(
        self,
        transaction: SafetyTransaction,
        *,
        clock: Callable[[], float],
        baseline_source: Callable[[], Mapping[str, Any]],
        window_s: float = 1.0,
        max_members: int = 32,
    ) -> None:
        self._txn = transaction
        self._clock = clock
        self._baseline = baseline_source
        self._window_s = window_s
        self._max_members = max_members
        self._lock = threading.Lock()
        self._epoch = _Epoch(index=0, opened_at=clock())
        self._history: list[TransactionResult] = []

    # ── submission ───────────────────────────────────────────────────────
    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch.index

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._epoch.envelopes)

    def submit(self, envelope: AgentActionEnvelope) -> SubmissionReceipt:
        """Queue a request into the open epoch.

        Refusals here are structural only — a full epoch, a second request from
        the same principal. Everything substantive (identity, authority,
        physics, aggregates) is decided at close, against the whole epoch,
        because deciding any of it at submission time would decide it in
        isolation.
        """
        with self._lock:
            epoch = self._epoch
            if epoch.closed:  # pragma: no cover - guarded by _rotate
                raise CycleClosed(f"epoch {epoch.index} is already decided")
            if len(epoch.envelopes) >= self._max_members:
                return SubmissionReceipt(
                    False,
                    epoch.index,
                    envelope.agent_id,
                    reason=(
                        f"epoch {epoch.index} is full at {self._max_members} "
                        "members; retry in the next epoch"
                    ),
                )
            if any(e.agent_id == envelope.agent_id for e in epoch.envelopes):
                # One request per principal per epoch. Enforced here as well as
                # in the transaction, because here it is a cheap structural
                # refusal that costs the flooder a round trip, rather than a
                # refusal that has already consumed an evaluation.
                return SubmissionReceipt(
                    False,
                    epoch.index,
                    envelope.agent_id,
                    reason=(
                        f"{envelope.agent_id!r} already has a request in epoch "
                        f"{epoch.index}"
                    ),
                )
            epoch.envelopes.append(envelope)
            return SubmissionReceipt(
                True, epoch.index, envelope.agent_id, position=len(epoch.envelopes) - 1
            )

    # ── closing ──────────────────────────────────────────────────────────
    def _rotate(self) -> tuple[int, list[AgentActionEnvelope]]:
        with self._lock:
            epoch = self._epoch
            epoch.closed = True
            members = list(epoch.envelopes)
            index = epoch.index
            self._epoch = _Epoch(index=index + 1, opened_at=self._clock())
        return index, members

    def close(
        self,
        *,
        telemetry: Sequence[Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> TransactionResult:
        """Decide the open epoch and start the next one.

        The baseline is read here, from the operator's source, not taken from
        the caller. That is what makes the anti-smuggling check in ``authorize``
        meaningful: an agent cannot supply the state its own declared keys are
        compared against.

        Submissions are ordered by ``(agent_id, nonce)`` rather than by arrival,
        so two agents racing produce the same decision either way and the
        certificate replays.
        """
        index, members = self._rotate()
        members.sort(key=lambda e: (e.agent_id, e.nonce))

        ctx: dict[str, Any] = dict(context or {})
        ctx["baseline_action"] = dict(self._baseline())

        result = self._txn.evaluate(
            members,
            telemetry=telemetry,
            context=ctx,
            transaction_id=f"epoch-{index}",
            epoch=index,
        )
        self._history.append(result)
        return result

    def tick(
        self,
        *,
        telemetry: Sequence[Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> TransactionResult | None:
        """Close the epoch if its window has elapsed, otherwise do nothing.

        Returns ``None`` while the window is open — including when the epoch is
        empty, because closing an empty epoch would append a certificate saying
        nothing happened, once per tick, forever.
        """
        with self._lock:
            elapsed = self._clock() - self._epoch.opened_at
            empty = not self._epoch.envelopes
        if elapsed < self._window_s or empty:
            return None
        return self.close(telemetry=telemetry, context=context)

    @property
    def history(self) -> tuple[TransactionResult, ...]:
        return tuple(self._history)
