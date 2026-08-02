"""Every A1 dialect carries the assurance envelope to the wire, not just osc_a1.

Closes the last open item in `deploy/xapp-e2e/A1_ASSURANCE_WIRE_PROOF.md`'s
honest-scope section. That document proves the envelope crosses a real socket
under the `osc_a1` dialect against the official O-RAN-SC simulator, and then says
of the other four dialects:

    The schema and the builder are dialect-independent — the envelope is a key
    inside `policy_payload`, which every dialect transports verbatim — but only
    `osc_a1` was exercised here on a real socket.

That is an argument, not a result, and the argument is exactly the kind that is
usually right and occasionally not: `legacy` and `osc_a1` PUT a bare
`policy_payload`, `osc` nests it under `policy_data`, and `eiap` and `mantaray`
nest it under `policyData` with camelCase siblings. Five different body shapes,
each built by its own branch of the `_do_put` if-ladder in `a1_adapter.py`. A
branch that dropped or flattened the envelope would be invisible to the
single-dialect live proof.

So this pins it for all five, offline via `httpx.MockTransport` — no socket, no
simulator. What is asserted is the envelope arriving intact at the transport
boundary, byte for byte, wherever that dialect chooses to nest the payload.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from horizon_ric.rapp.a1_adapter import (
    A1Adapter,
    A1AdapterConfig,
    assurance_envelope,
)
from horizon_ric.shield import default_terrestrial_shield
from horizon_ric.shield.signing import (
    canonical_certificate_bytes,
    generate_signing_key,
    signed_certificate,
)

ALL_DIALECTS = ("legacy", "osc", "osc_a1", "eiap", "mantaray")

# Where each dialect puts the caller's policy_payload in the PUT body.
# None means the body IS the payload, with no wrapper.
PAYLOAD_LOCATION: dict[str, str | None] = {
    "legacy": None,
    "osc_a1": None,
    "osc": "policy_data",
    "eiap": "policyData",
    "mantaray": "policyData",
}

BAND_LO = 3.40e9
BAND_HI = 3.50e9
MAX_EIRP = 33.0

POLICY_TYPE = "horizon.qos.priority"
PAYLOAD: dict[str, Any] = {
    "scope": {"slice_id": "slice-dialect-check"},
    "qos_objectives": {"priority": 5},
    "rapp_metadata": {"decision_id": "dialect-check-0001"},
}


def _signed_certificate(tmp_path):
    """A real signed certificate from a real Shield disposition."""
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP
    )
    action = {
        "block": "policy_emit",
        "policy_type": POLICY_TYPE,
        "policy_payload": dict(PAYLOAD),
        "frequency_hz": 3.45e9,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": 6.0,
    }
    disposition = shield.dispose(action, {}, decision_id="dialect-check-0001")
    key = generate_signing_key(tmp_path / "signing.pem")
    return signed_certificate(disposition.certificate, key), key


async def _adapter(dialect: str, handler):
    cfg = A1AdapterConfig(dialect=dialect)
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )
    return adapter


def _extract_payload(dialect: str, body: dict[str, Any]) -> dict[str, Any]:
    where = PAYLOAD_LOCATION[dialect]
    return body if where is None else body[where]


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ALL_DIALECTS)
async def test_envelope_reaches_the_wire_for_every_dialect(dialect, tmp_path):
    certificate, _ = _signed_certificate(tmp_path)
    envelope = assurance_envelope(certificate)
    assert envelope, "no envelope to test with — the fixture is broken"

    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.content:
            captured.append(json.loads(request.content))
        return httpx.Response(202, json={})

    adapter = await _adapter(dialect, handler)
    try:
        await adapter.emit_policy(
            POLICY_TYPE,
            {**PAYLOAD, "assurance": envelope},
            policy_id="dialect-check",
            safety_certificate=certificate,
        )
    finally:
        await adapter.close()

    assert captured, f"{dialect}: no PUT body captured"
    payload = _extract_payload(dialect, captured[-1])
    assert "assurance" in payload, (
        f"{dialect}: the assurance envelope did not reach the wire; "
        f"payload keys were {sorted(payload)}"
    )
    # Byte-for-byte, not just present: a dialect that re-serialised the
    # envelope through a lossy path would still satisfy a presence check.
    assert json.dumps(payload["assurance"], sort_keys=True) == json.dumps(
        envelope, sort_keys=True
    ), f"{dialect}: envelope mutated in transit"


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ALL_DIALECTS)
async def test_digest_on_the_wire_still_covers_the_signed_bytes(dialect, tmp_path):
    """The whole point of the envelope: the digest must remain verifiable.

    Each dialect wraps the payload differently. This asserts the digest that
    arrives is still the SHA-256 of the exact bytes the signature was computed
    over, for every wrapper shape — so a receiver can verify regardless of which
    vendor surface delivered it.
    """
    import hashlib

    certificate, _ = _signed_certificate(tmp_path)
    envelope = assurance_envelope(certificate)
    expected = hashlib.sha256(canonical_certificate_bytes(certificate)).hexdigest()

    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.content:
            captured.append(json.loads(request.content))
        return httpx.Response(202, json={})

    adapter = await _adapter(dialect, handler)
    try:
        await adapter.emit_policy(
            POLICY_TYPE,
            {**PAYLOAD, "assurance": envelope},
            policy_id="dialect-check",
            safety_certificate=certificate,
        )
    finally:
        await adapter.close()

    payload = _extract_payload(dialect, captured[-1])
    assert payload["assurance"]["certificate_digest"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ALL_DIALECTS)
async def test_absent_envelope_leaves_the_body_unchanged(dialect):
    """Backward compatibility, per dialect.

    A deployment that has not adopted the envelope must see exactly the body it
    saw before. Asserted against each wrapper shape rather than once, because
    the wrappers are what a regression would land in.
    """
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.content:
            captured.append(json.loads(request.content))
        return httpx.Response(202, json={})

    adapter = await _adapter(dialect, handler)
    try:
        await adapter.emit_policy(POLICY_TYPE, dict(PAYLOAD), policy_id="no-env")
    finally:
        await adapter.close()

    payload = _extract_payload(dialect, captured[-1])
    assert "assurance" not in payload
    assert json.dumps(payload, sort_keys=True) == json.dumps(PAYLOAD, sort_keys=True)


def test_every_supported_dialect_is_covered() -> None:
    """Guard against a sixth dialect landing without a row here.

    `A1Adapter.__init__` validates against its own `supported_dialects` set. If
    someone adds one, this fails until it is added to ALL_DIALECTS and
    PAYLOAD_LOCATION — otherwise the cross-dialect claim silently stops covering
    the whole surface.
    """
    import inspect
    import re

    src = inspect.getsource(A1Adapter.__init__)
    match = re.search(r"supported_dialects\s*=\s*\{([^}]*)\}", src)
    assert match, "could not find supported_dialects in A1Adapter.__init__"
    declared = set(re.findall(r'"([a-z_0-9]+)"', match.group(1)))
    assert declared == set(ALL_DIALECTS), (
        f"dialect coverage drifted: adapter supports {sorted(declared)}, "
        f"this module covers {sorted(ALL_DIALECTS)}"
    )
    assert set(PAYLOAD_LOCATION) == set(ALL_DIALECTS)
