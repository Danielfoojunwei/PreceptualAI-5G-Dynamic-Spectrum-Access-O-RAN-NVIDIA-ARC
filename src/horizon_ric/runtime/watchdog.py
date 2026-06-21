"""systemd watchdog notifier (sd_notify).

Implements the `sd_notify(3)` protocol: writes "WATCHDOG=1" to the
AF_UNIX datagram socket whose path is in `$NOTIFY_SOCKET`. The systemd
service unit declares `WatchdogSec=30` (and `Type=notify`); systemd
restarts the unit if it doesn't see a ping within that window.

We use a dependency-free implementation rather than the `systemd-python`
PyPI package because:

  * `systemd-python` requires libsystemd-dev headers at build time and
    is fiddly on Jetson / Orin Nano edge images. The protocol is a
    9-line datagram send, which is far less risky than a C build.
  * `cysystemd` has the same C-build issue.

The protocol itself is the real one — same wire format systemd
expects, verified by `tests/test_watchdog.py` against a real bound
AF_UNIX socket.

If `$NOTIFY_SOCKET` is unset (e.g. running outside systemd, or in a
unit test that hasn't bound a socket), `notify_watchdog()` is a no-op
and emits a debug event. This is the documented sd_notify behaviour.

Stable structlog event names:

    horizon.watchdog.ready    — startup READY=1 sent
    horizon.watchdog.notified — periodic WATCHDOG=1 ping sent
    horizon.watchdog.skipped  — NOTIFY_SOCKET unset; nothing sent
    horizon.watchdog.failed   — socket send failed
    horizon.watchdog.stopped  — periodic task stopped (shutdown)
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Final

import structlog

logger = structlog.get_logger(__name__)

# 15 s = half of WatchdogSec=30 — gives systemd two windows of slack so
# a transient GC pause doesn't trigger a restart.
DEFAULT_WATCHDOG_INTERVAL_S: Final[float] = 15.0


def _send_notify(message: bytes) -> bool:
    """Send a raw sd_notify message.

    Returns True if the message was sent, False if NOTIFY_SOCKET is unset.
    Raises any underlying socket error so the caller can decide whether
    to log+continue or escalate.
    """
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return False
    # systemd uses a leading NUL for abstract sockets per sd_notify(3).
    if sock_path.startswith("@"):
        sock_path = "\0" + sock_path[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
        s.sendto(message, sock_path)
    return True


def notify_watchdog() -> bool:
    """Send `WATCHDOG=1` to systemd. Returns True if delivered."""
    try:
        delivered = _send_notify(b"WATCHDOG=1")
    except OSError as exc:
        logger.error("horizon.watchdog.failed", error=str(exc))
        return False
    if delivered:
        logger.debug("horizon.watchdog.notified")
    else:
        logger.debug("horizon.watchdog.skipped", reason="NOTIFY_SOCKET unset")
    return delivered


def notify_ready() -> bool:
    """Send `READY=1` once boot is complete. Returns True if delivered."""
    try:
        delivered = _send_notify(b"READY=1")
    except OSError as exc:
        logger.error("horizon.watchdog.failed", error=str(exc), kind="ready")
        return False
    if delivered:
        logger.info("horizon.watchdog.ready")
    return delivered


def notify_stopping() -> bool:
    """Send `STOPPING=1` on graceful shutdown."""
    try:
        return _send_notify(b"STOPPING=1")
    except OSError as exc:
        logger.error("horizon.watchdog.failed", error=str(exc), kind="stopping")
        return False


async def watchdog_loop(interval_s: float = DEFAULT_WATCHDOG_INTERVAL_S) -> None:
    """Background task: ping the watchdog every `interval_s` seconds.

    Cancellation-safe: if the task is cancelled (shutdown), we send
    STOPPING=1 and exit cleanly. Any individual ping failure is logged
    but does not stop the loop — a transient EAGAIN should not kill the
    rApp, only systemd's WatchdogSec timer should.
    """
    logger.info("horizon.watchdog.loop_start", interval_s=interval_s)
    try:
        while True:
            notify_watchdog()
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        notify_stopping()
        logger.info("horizon.watchdog.stopped")
        raise


__all__ = [
    "DEFAULT_WATCHDOG_INTERVAL_S",
    "notify_ready",
    "notify_stopping",
    "notify_watchdog",
    "watchdog_loop",
]
