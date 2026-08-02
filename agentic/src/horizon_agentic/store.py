"""Durable evidence, and the honest limit of what a log can prove about itself.

`evidence.py` builds the chain; this persists it. The split matters: the chain
is a data structure whose properties are provable, and durability is an
operational concern with a different failure model, so conflating them would
let a storage bug read as a cryptographic guarantee.

What the chain gives you on load is **tamper localisation within what is
present**. Alter record 41 and `verify_chain` names index 41. That property
survives a round trip through disk unchanged, because the file stores the same
bytes the hashes were computed over.

What it does **not** give you is truncation detection. Delete the last ten
lines of an append-only log and the remainder verifies perfectly — it is a
valid chain, just a shorter one. No hash-linked structure detects that from the
inside; the link points backwards, so removing the tail removes the only thing
that would have referred to it. Closing that needs an anchor kept somewhere the
attacker does not control: :meth:`JsonlChainStore.load` takes an
``expected_head`` for exactly this, and a deployment that does not supply one
has not solved truncation and should say so rather than assume the chain covers
it.

The repository already has `horizon_ric.evidence.store` for decision records.
This does not replace it or import it: the transaction chain has a different
shape and a different lifecycle, and coupling them would make the agentic
track's storage decisions the main package's problem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from horizon_agentic.evidence import (
    GENESIS,
    ChainBreak,
    ChainEntry,
    TransactionCertificate,
    TransactionEvidenceChain,
    verify_chain,
)

__all__ = ["JsonlChainStore", "StoreIntegrityError", "LoadedChain"]


class StoreIntegrityError(RuntimeError):
    """Raised when a persisted chain does not verify on load."""


@dataclass(frozen=True)
class LoadedChain:
    """What came back off disk, and whether it holds together."""

    entries: tuple[ChainEntry, ...]
    break_at: ChainBreak | None
    head: str

    @property
    def intact(self) -> bool:
        return self.break_at is None


class JsonlChainStore:
    """Append-only JSONL, one chain entry per line.

    JSONL rather than a single JSON array because appending to an array means
    rewriting the file, and a process killed mid-rewrite loses records that
    were already durable. One line per entry means a partial write damages at
    most the last line, which `load` reports rather than silently skipping.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, entry: ChainEntry) -> None:
        """Write one entry. Callers append in chain order."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "index": entry.index,
            "prev_hash": entry.prev_hash,
            "entry_hash": entry.entry_hash,
            "certificate": entry.certificate.to_dict(),
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    def append_chain(self, chain: TransactionEvidenceChain, *, from_index: int = 0) -> int:
        """Persist entries from ``from_index`` onward; returns how many."""
        written = 0
        for entry in chain.entries[from_index:]:
            self.append(entry)
            written += 1
        return written

    def load(
        self,
        *,
        public_key: Ed25519PublicKey | None = None,
        expected_head: str | None = None,
    ) -> LoadedChain:
        """Read the chain back and verify it.

        ``expected_head`` is the anchor. Supplied, it detects truncation: a
        shortened log verifies internally but will not end at the head the
        caller remembers. Omitted, truncation is undetectable — stated here
        rather than left for a reader to discover.
        """
        if not self._path.exists():
            return LoadedChain((), None, GENESIS)

        entries: list[ChainEntry] = []
        for line_no, line in enumerate(
            self._path.read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                # A torn final line is the expected shape of a crash. Report
                # where it is; do not skip it, because skipping turns a
                # detectable partial write into a silent gap.
                raise StoreIntegrityError(
                    f"{self._path}: line {line_no + 1} is not valid JSON ({exc}); "
                    "a partial write is likely and the chain cannot be verified "
                    "past this point"
                ) from exc
            entries.append(
                ChainEntry(
                    index=int(record["index"]),
                    prev_hash=str(record["prev_hash"]),
                    entry_hash=str(record["entry_hash"]),
                    certificate=_certificate_from(record["certificate"]),
                )
            )

        broken = verify_chain(entries, public_key=public_key)
        head = entries[-1].entry_hash if entries else GENESIS
        if expected_head is not None and broken is None and head != expected_head:
            broken = ChainBreak(
                len(entries),
                f"chain ends at {head[:12]}... but the anchor expects "
                f"{expected_head[:12]}...; records have been removed from the end",
            )
        return LoadedChain(tuple(entries), broken, head)

    def restore(
        self,
        *,
        public_key: Ed25519PublicKey | None = None,
        expected_head: str | None = None,
    ) -> TransactionEvidenceChain:
        """Load into a live chain, refusing to continue a broken one.

        Appending to a chain that does not verify would extend a record an
        auditor has to reject wholesale — better to stop at the point the
        damage is known than to bury it under new entries.
        """
        loaded = self.load(public_key=public_key, expected_head=expected_head)
        if loaded.break_at is not None:
            raise StoreIntegrityError(
                f"{self._path}: chain breaks at index {loaded.break_at.index}: "
                f"{loaded.break_at.reason}"
            )
        return TransactionEvidenceChain(entries=list(loaded.entries))


def _certificate_from(payload: dict[str, Any]) -> TransactionCertificate:
    """Rebuild a certificate, tolerating fields a newer version added.

    Unknown keys are dropped rather than refused, which is the opposite of the
    envelope's wire rule and deliberately so: an envelope is a *request* whose
    unknown field may change what is asked for, while a stored certificate is a
    *record* that a reader should still be able to verify the known parts of.
    A dropped field changes the recomputed hash, so `verify_chain` reports the
    entry as altered — the discrepancy surfaces rather than passing quietly.
    """
    fields = {
        "schema_version", "transaction_id", "issued_at", "committed",
        "baseline_digest", "aggregates", "epoch", "refusals",
        "telemetry_checks", "authority_verdicts", "aggregate_checks",
        "dropped", "admitted", "resolution_rounds", "signature",
        "signing_key_fingerprint",
    }
    kwargs: dict[str, Any] = {}
    for name in fields & set(payload):
        value = payload[name]
        kwargs[name] = tuple(value) if isinstance(value, list) else value
    return TransactionCertificate(**kwargs)


def entries_equal(a: Sequence[ChainEntry], b: Sequence[ChainEntry]) -> bool:
    """Structural comparison, for asserting a round trip lost nothing."""
    if len(a) != len(b):
        return False
    return all(
        x.index == y.index
        and x.prev_hash == y.prev_hash
        and x.entry_hash == y.entry_hash
        and x.certificate.to_dict() == y.certificate.to_dict()
        for x, y in zip(a, b)
    )
