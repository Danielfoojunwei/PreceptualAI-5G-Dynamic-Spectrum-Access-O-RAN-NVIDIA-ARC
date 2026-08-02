"""Durable evidence, key distribution, and an ingress that receives envelopes.

Three items disclosed as open in the README and closed here. Each was
disclosed rather than hidden, so each test says what the gap was.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
from horizon_agentic.aggregate import AbsoluteSliceCapacityFloor
from horizon_agentic.bundle import SafetyTransaction
from horizon_agentic.cycle import TransactionCycle
from horizon_agentic.envelope import (
    AgentActionEnvelope,
    AuthorityGrant,
    AuthorityPolicy,
)
from horizon_agentic.evidence import TransactionEvidenceChain, chain_hash
from horizon_agentic.identity import sign_envelope
from horizon_agentic.ingress import EnvelopeIngress, serve_line_delimited
from horizon_agentic.keyring import (
    KeyringError,
    load_registry,
    write_public_key,
)
from horizon_agentic.store import JsonlChainStore, StoreIntegrityError, entries_equal

from horizon_ric.shield import default_terrestrial_shield

ROOT = "operator-root"
NOW = 1_700_000_000.0
BASELINE = {
    "block": "ran_control",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 100e6,
    "tx_power_dBm": 20.0,
    "antenna_gain_dBi": 6.0,
    "prb_allocation": {"safety_critical": 0.50, "embb": 0.50},
}


def policy() -> AuthorityPolicy:
    return AuthorityPolicy(
        root_principal=ROOT,
        grants={
            ROOT: AuthorityGrant(ROOT, frozenset({"ran"}),
                                 frozenset({"spectrum", "slice"}), 0),
            "slice-agent": AuthorityGrant("slice-agent", frozenset({"ran"}),
                                          frozenset({"slice"}), 15),
            "energy-agent": AuthorityGrant("energy-agent", frozenset({"ran"}),
                                           frozenset({"spectrum"}), 20),
        },
    )


def slice_env(nonce="n1", share=0.25) -> AgentActionEnvelope:
    return AgentActionEnvelope(
        agent_id="slice-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"slice"}),
        delegation_chain=(ROOT, "slice-agent"),
        requested_action={
            **BASELINE,
            "prb_allocation": {"safety_critical": share, "embb": 1.0 - share},
        },
        mutates=frozenset({"prb_allocation"}),
        resource_id="cell-1",
        nonce=nonce,
        issued_at=NOW,
    )


def cycle(**kw) -> TransactionCycle:
    return TransactionCycle(
        SafetyTransaction(
            shield=default_terrestrial_shield(
                band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0
            ),
            policy=policy(),
            aggregates=[AbsoluteSliceCapacityFloor(min_capacity_hz=20e6)],
            clock=lambda: "2026-01-01T00:00:00Z",
            **kw,
        ),
        clock=lambda: NOW,
        baseline_source=lambda: dict(BASELINE),
        window_s=0.0,
    )


# ── durable evidence ─────────────────────────────────────────────────────
def test_a_chain_round_trips_through_disk(tmp_path) -> None:
    """Disclosed as open: the chain was in-memory only."""
    key = Ed25519PrivateKey.generate()
    chain = TransactionEvidenceChain()
    c = cycle(signing_key=key, chain=chain)
    for i in range(4):
        c.submit(slice_env(nonce=f"n{i}"))
        c.close()

    store = JsonlChainStore(tmp_path / "chain.jsonl")
    assert store.append_chain(chain) == 4

    loaded = store.load(public_key=key.public_key())
    assert loaded.intact, loaded.break_at
    assert entries_equal(loaded.entries, chain.entries)
    assert loaded.head == chain.head


def test_tampering_on_disk_is_localised(tmp_path) -> None:
    chain = TransactionEvidenceChain()
    c = cycle(chain=chain)
    for i in range(3):
        c.submit(slice_env(nonce=f"n{i}"))
        c.close()
    store = JsonlChainStore(tmp_path / "chain.jsonl")
    store.append_chain(chain)

    lines = (tmp_path / "chain.jsonl").read_text().splitlines()
    record = json.loads(lines[1])
    record["certificate"]["committed"] = False
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    (tmp_path / "chain.jsonl").write_text("\n".join(lines) + "\n")

    loaded = store.load()
    assert not loaded.intact
    assert loaded.break_at.index == 1


def test_truncation_needs_an_anchor_and_is_detected_with_one(tmp_path) -> None:
    """The honest limit, stated as a test.

    Delete the tail of an append-only log and the remainder verifies
    perfectly — it is a valid chain, just shorter. No hash-linked structure
    detects that from the inside, because the link points backwards. Only an
    externally held head detects it.
    """
    chain = TransactionEvidenceChain()
    c = cycle(chain=chain)
    for i in range(4):
        c.submit(slice_env(nonce=f"n{i}"))
        c.close()
    store = JsonlChainStore(tmp_path / "chain.jsonl")
    store.append_chain(chain)
    anchor = chain.head

    lines = (tmp_path / "chain.jsonl").read_text().splitlines()
    (tmp_path / "chain.jsonl").write_text("\n".join(lines[:2]) + "\n")

    # Without the anchor the truncated chain looks perfect.
    assert store.load().intact

    # With it, the removal is detected.
    anchored = store.load(expected_head=anchor)
    assert not anchored.intact
    assert "removed from the end" in anchored.break_at.reason


def test_a_torn_final_line_is_reported_not_skipped(tmp_path) -> None:
    """Skipping turns a detectable partial write into a silent gap."""
    chain = TransactionEvidenceChain()
    c = cycle(chain=chain)
    c.submit(slice_env())
    c.close()
    store = JsonlChainStore(tmp_path / "chain.jsonl")
    store.append_chain(chain)
    with (tmp_path / "chain.jsonl").open("a") as fh:
        fh.write('{"index": 1, "prev_ha')
    with pytest.raises(StoreIntegrityError, match="partial write"):
        store.load()


def test_restore_refuses_to_extend_a_broken_chain(tmp_path) -> None:
    """New entries on top of a broken chain bury the damage."""
    chain = TransactionEvidenceChain()
    c = cycle(chain=chain)
    for i in range(2):
        c.submit(slice_env(nonce=f"n{i}"))
        c.close()
    store = JsonlChainStore(tmp_path / "chain.jsonl")
    store.append_chain(chain)

    lines = (tmp_path / "chain.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["certificate"]["transaction_id"] = "forged"
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    (tmp_path / "chain.jsonl").write_text("\n".join(lines) + "\n")

    with pytest.raises(StoreIntegrityError, match="breaks at index"):
        store.restore()


def test_a_restored_chain_continues_correctly(tmp_path) -> None:
    chain = TransactionEvidenceChain()
    c = cycle(chain=chain)
    c.submit(slice_env(nonce="a"))
    c.close()
    store = JsonlChainStore(tmp_path / "chain.jsonl")
    store.append_chain(chain)

    revived = store.restore()
    assert revived.head == chain.head
    c2 = cycle(chain=revived)
    c2.submit(slice_env(nonce="b"))
    c2.close()
    assert revived.entries[-1].prev_hash == chain.entries[-1].entry_hash
    assert revived.verify() is None


def test_the_stored_hash_covers_the_certificate(tmp_path) -> None:
    """A record edited on disk must not recompute to its stored hash."""
    chain = TransactionEvidenceChain()
    c = cycle(chain=chain)
    c.submit(slice_env())
    c.close()
    entry = chain.entries[0]
    forged = dataclasses.replace(entry.certificate, epoch=99)
    assert chain_hash(entry.prev_hash, forged) != entry.entry_hash


# ── key distribution ─────────────────────────────────────────────────────
def test_a_registry_loads_from_a_directory(tmp_path) -> None:
    """Disclosed as open: 'construct it in Python' is not a distribution story."""
    keys = {name: Ed25519PrivateKey.generate()
            for name in ("slice-agent", "energy-agent")}
    for name, key in keys.items():
        write_public_key(tmp_path, name, key)

    registry = load_registry(tmp_path)
    assert set(registry.current) == set(keys)
    assert registry.as_registry().key_for("slice-agent") is not None
    assert registry.agents_still_on_retired_keys == ()


def test_only_the_public_half_is_written(tmp_path) -> None:
    key = Ed25519PrivateKey.generate()
    path = write_public_key(tmp_path, "slice-agent", key)
    text = path.read_bytes()
    assert b"PUBLIC KEY" in text
    assert b"PRIVATE" not in text


def test_rotation_accepts_both_keys_and_names_which_verified(tmp_path) -> None:
    """An agent cannot switch keys atomically with the operator.

    So the overlap has to exist — and the registry has to say which key
    actually verified, or a deployment cannot tell a completed rotation from a
    configured one.
    """
    old, new = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    write_public_key(tmp_path, "slice-agent", new)
    retired = tmp_path / "retired"
    retired.mkdir()
    (retired / "slice-agent.1.pub.pem").write_bytes(
        old.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    registry = load_registry(tmp_path)
    assert len(registry.keys_for("slice-agent")) == 2
    assert registry.agents_still_on_retired_keys == ("slice-agent",)

    from horizon_agentic.identity import canonical_envelope_bytes

    signed_old = sign_envelope(slice_env(), old)

    def verify(pub):
        try:
            pub.verify(bytes.fromhex(signed_old.signature),
                       canonical_envelope_bytes(signed_old))
            return True
        except Exception:
            return False

    ok, fingerprint = registry.verify_during_rotation("slice-agent", verify)
    assert ok
    from horizon_ric.shield.signing import key_fingerprint

    assert fingerprint == key_fingerprint(old.public_key())


def test_a_non_ed25519_key_is_refused_not_skipped(tmp_path) -> None:
    """A skipped agent surfaces later as a legitimate agent being refused."""
    write_public_key(tmp_path, "slice-agent", Ed25519PrivateKey.generate())
    rsa = generate_private_key(public_exponent=65537, key_size=2048)
    (tmp_path / "rsa-agent.pub.pem").write_bytes(
        rsa.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    with pytest.raises(KeyringError, match="not an Ed25519"):
        load_registry(tmp_path)


def test_a_malformed_key_file_is_refused(tmp_path) -> None:
    write_public_key(tmp_path, "slice-agent", Ed25519PrivateKey.generate())
    (tmp_path / "broken.pub.pem").write_text("not a key")
    with pytest.raises(KeyringError, match="not a readable PEM"):
        load_registry(tmp_path)


def test_an_agent_with_only_a_retired_key_is_refused(tmp_path) -> None:
    """A half-finished removal must not keep a decommissioned agent alive."""
    write_public_key(tmp_path, "kept", Ed25519PrivateKey.generate())
    retired = tmp_path / "retired"
    retired.mkdir()
    write_public_key(retired, "gone.1", Ed25519PrivateKey.generate())
    with pytest.raises(KeyringError, match="only retired keys"):
        load_registry(tmp_path)


def test_an_empty_directory_is_refused(tmp_path) -> None:
    with pytest.raises(KeyringError, match="no .* files"):
        load_registry(tmp_path)


# ── ingress ──────────────────────────────────────────────────────────────
def test_bytes_become_a_queued_submission() -> None:
    """Disclosed as open: nothing consumed a serialised envelope."""
    c = cycle()
    ingress = EnvelopeIngress(c)
    payload = json.dumps(slice_env().to_dict()).encode()
    result = ingress.submit_bytes(payload)
    assert result.accepted and result.epoch == 0
    assert c.pending == 1


def test_the_result_is_a_receipt_not_a_decision() -> None:
    """Deciding on submission would decide one agent in isolation."""
    result = EnvelopeIngress(cycle()).submit_bytes(
        json.dumps(slice_env().to_dict()).encode()
    )
    assert not hasattr(result, "committed")
    assert result.detail == "queued"


def test_an_oversized_payload_is_refused_before_parsing() -> None:
    """json.loads on unbounded attacker bytes is a memory exhaustion, and it
    happens before any authentication could run."""
    reasons: list[str] = []
    ingress = EnvelopeIngress(cycle(), max_bytes=128, on_reject=reasons.append)
    assert not ingress.submit_bytes(b"x" * 4096).accepted
    assert "exceeds the 128 limit" in reasons[0]


@pytest.mark.parametrize(
    "payload",
    [b"not json", b"[]", b'{"agent_id": 1}', b'{"surprise": true}', b"\xff\xfe"],
)
def test_malformed_payloads_are_refused(payload) -> None:
    assert not EnvelopeIngress(cycle()).submit_bytes(payload).accepted


def test_refusals_do_not_leak_which_check_failed() -> None:
    """An ingress distinguishing 'unknown agent' from 'bad signature' is a
    registry oracle. The operator gets the detail; the wire does not."""
    reasons: list[str] = []
    ingress = EnvelopeIngress(cycle(), on_reject=reasons.append)
    a = ingress.submit_bytes(b"not json")
    b = ingress.submit_bytes(b'{"surprise": true}')
    assert a.detail == b.detail == "envelope refused"
    assert reasons[0] != reasons[1], "the operator must still see the difference"


def test_a_structural_refusal_is_relayed_so_a_good_agent_can_retry() -> None:
    """A full epoch or a duplicate principal is not security-sensitive, and a
    flat refusal would leave a well-behaved agent with nothing to act on."""
    c = cycle()
    ingress = EnvelopeIngress(c)
    assert ingress.submit_bytes(json.dumps(slice_env("a").to_dict()).encode()).accepted
    second = ingress.submit_bytes(json.dumps(slice_env("b").to_dict()).encode())
    assert not second.accepted
    assert "already has a request" in second.detail


def test_a_signed_envelope_survives_the_ingress_and_authenticates() -> None:
    """End to end: an external agent signs, serialises, sends; we verify."""
    from horizon_agentic.identity import AgentRegistry, EnvelopeAuthenticator

    key = Ed25519PrivateKey.generate()
    signed = sign_envelope(slice_env(), key)
    c = cycle()
    ingress = EnvelopeIngress(c)
    assert ingress.submit_bytes(json.dumps(signed.to_dict()).encode()).accepted

    auth = EnvelopeAuthenticator(
        AgentRegistry({"slice-agent": key.public_key()}), clock=lambda: NOW
    )
    queued = c._epoch.envelopes[0]  # noqa: SLF001 - asserting on queued state
    assert auth.authenticate(queued).authenticated


def test_the_line_delimited_server_accepts_an_envelope() -> None:
    """The reference binding: newline-delimited JSON over TCP, stdlib only."""

    async def run() -> dict:
        c = cycle()
        ingress = EnvelopeIngress(c)
        server = await serve_line_delimited(ingress, port=0)
        host, port = server.sockets[0].getsockname()[:2]
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(json.dumps(slice_env().to_dict()).encode() + b"\n")
        await writer.drain()
        line = await reader.readline()
        writer.close()
        server.close()
        await server.wait_closed()
        return json.loads(line)

    reply = asyncio.run(run())
    assert reply["accepted"] is True
    assert reply["epoch"] == 0
