"""Live NETCONF integration test for O1Adapter.

This test is gated by the ``integration`` pytest marker. It is skipped by
the default suite and runs only with::

    pytest -m integration tests/test_o1_live.py

Requirements at run-time:
  * A real NETCONF/SSH server reachable at the host/port configured below.
    The repo ships a launcher (``deploy/start_netconf_server.sh``) that
    brings up a real ``yuma123 netconfd`` (BSD-3-clause) on 127.0.0.1:8830
    fronted by a user-space OpenSSH ``sshd`` running the NETCONF subsystem.
  * An SSH private key authorized by the server (default location below).

If neither of those is present the test is auto-skipped (not faked).

The test exercises a real protocol round-trip — Hello capability exchange,
``<get-config>``, ``<edit-config>`` on the candidate datastore, ``<commit>``
and ``<create-subscription>`` (RFC 5277) — and asserts the server responded
with real XML, not invented payloads.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from horizon_ric.rapp.o1_adapter import O1Adapter, O1AdapterConfig

_HOST = os.getenv("O1_TEST_HOST", "127.0.0.1")
_PORT = int(os.getenv("O1_TEST_PORT", "8830"))
_USER = os.getenv("O1_TEST_USER", os.getenv("USER", "netconf"))
_KEY = os.getenv("O1_TEST_KEY", "/tmp/nc-server/client_ed25519")


def _server_reachable() -> bool:
    try:
        with socket.create_connection((_HOST, _PORT), timeout=1.0):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _server_reachable(),
        reason=(
            f"No NETCONF server at {_HOST}:{_PORT}. "
            "Bring one up with deploy/start_netconf_server.sh or set "
            "O1_TEST_HOST/O1_TEST_PORT."
        ),
    ),
    pytest.mark.skipif(
        not Path(_KEY).is_file(),
        reason=f"SSH key {_KEY!r} not present (set O1_TEST_KEY).",
    ),
]


def _cfg() -> O1AdapterConfig:
    return O1AdapterConfig(
        host=_HOST,
        port=_PORT,
        username=_USER,
        key_filename=_KEY,
        hostkey_verify=False,
        look_for_keys=False,
        allow_agent=False,
    )


async def test_real_netconf_get_config() -> None:
    adapter = O1Adapter(_cfg())
    async with adapter.session():
        xml = await adapter.get_config(source="running")
    assert "<rpc-reply" in xml
    assert "<data" in xml


async def test_real_netconf_edit_config_candidate_commit() -> None:
    adapter = O1Adapter(_cfg())
    edit_xml = (
        '<config xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0">'
        '<system xmlns="urn:ietf:params:xml:ns:yang:ietf-system">'
        "<contact>horizon-ric@integration</contact>"
        "<hostname>horizon-ric-itest</hostname>"
        "</system></config>"
    )
    async with adapter.session():
        # ietf-system in yuma123 is candidate-only — match real semantics.
        # Reset candidate first — prior runs may have left edits.
        await adapter.discard_changes()
        await adapter.lock(target="candidate")
        try:
            reply = await adapter.edit_config(edit_xml, target="candidate")
            assert "<ok/>" in reply
            commit_reply = await adapter.commit()
            assert "<ok/>" in commit_reply
        finally:
            await adapter.unlock(target="candidate")

        running_xml = await adapter.get_config(source="running")
        assert "<rpc-reply" in running_xml


async def test_real_netconf_create_subscription() -> None:
    adapter = O1Adapter(_cfg())
    async with adapter.session():
        reply = await adapter.create_subscription()
        assert "<ok/>" in reply
