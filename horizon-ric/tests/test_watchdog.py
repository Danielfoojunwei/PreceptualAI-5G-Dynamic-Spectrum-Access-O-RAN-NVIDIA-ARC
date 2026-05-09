"""Tests for the systemd-style sd_notify watchdog.

Binds a real AF_UNIX datagram socket and asserts that the WATCHDOG=1
message arrives on the wire — no mocks of the socket layer.
"""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import pytest

from horizon_ric.runtime.watchdog import (
    notify_ready,
    notify_stopping,
    notify_watchdog,
    watchdog_loop,
)


def _bind_unix_dgram(path: Path) -> socket.socket:
    if path.exists():
        path.unlink()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(str(path))
    s.settimeout(1.0)
    return s


def test_notify_watchdog_sends_message(tmp_path, monkeypatch):
    sock_path = tmp_path / "notify.sock"
    server = _bind_unix_dgram(sock_path)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", str(sock_path))
        ok = notify_watchdog()
        assert ok is True
        data, _ = server.recvfrom(1024)
        assert data == b"WATCHDOG=1"
    finally:
        server.close()


def test_notify_ready_sends_message(tmp_path, monkeypatch):
    sock_path = tmp_path / "notify.sock"
    server = _bind_unix_dgram(sock_path)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", str(sock_path))
        assert notify_ready() is True
        data, _ = server.recvfrom(1024)
        assert data == b"READY=1"
    finally:
        server.close()


def test_notify_stopping_sends_message(tmp_path, monkeypatch):
    sock_path = tmp_path / "notify.sock"
    server = _bind_unix_dgram(sock_path)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", str(sock_path))
        assert notify_stopping() is True
        data, _ = server.recvfrom(1024)
        assert data == b"STOPPING=1"
    finally:
        server.close()


def test_notify_no_socket_is_noop(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    # Should not raise, returns False.
    assert notify_watchdog() is False
    assert notify_ready() is False


def test_notify_dead_socket_doesnt_crash(tmp_path, monkeypatch):
    """If systemd is down (socket gone), notify must log + return False."""
    sock_path = tmp_path / "missing.sock"
    monkeypatch.setenv("NOTIFY_SOCKET", str(sock_path))
    # No bind — the socket doesn't exist.
    assert notify_watchdog() is False


@pytest.mark.asyncio
async def test_watchdog_loop_pings_periodically(tmp_path, monkeypatch):
    sock_path = tmp_path / "notify.sock"
    server = _bind_unix_dgram(sock_path)
    server.settimeout(2.0)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", str(sock_path))
        # 0.1s interval keeps the test fast.
        task = asyncio.create_task(watchdog_loop(interval_s=0.1))
        try:
            # We expect at least 3 pings within ~0.5s.
            received = []
            for _ in range(3):
                data, _ = await asyncio.get_running_loop().run_in_executor(
                    None, server.recvfrom, 1024
                )
                received.append(data)
            assert all(d == b"WATCHDOG=1" for d in received)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # On cancel we should have seen STOPPING=1 too.
        try:
            data, _ = server.recvfrom(1024)
            assert data == b"STOPPING=1"
        except socket.timeout:
            pass
    finally:
        server.close()


def test_abstract_socket_path(monkeypatch):
    """Abstract sockets (path starts with @) are translated to NUL prefix.

    We can't bind an abstract socket portably in test, so we just
    confirm that the path translation logic doesn't blow up on an
    unbound abstract path — should fail gracefully and return False.
    """
    monkeypatch.setenv("NOTIFY_SOCKET", "@horizon-test-abstract-noexist")
    assert notify_watchdog() is False
