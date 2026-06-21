"""Pure-Python sd_notify implementation.

Avoids a dependency on the ``systemd-python`` C extension (which pulls in
libsystemd-dev at build time and complicates the slim image). Speaks the
plain UDS protocol described in ``sd_notify(3)``.

When ``NOTIFY_SOCKET`` is unset (i.e. not running under systemd), every
call is a no-op so the same code path works in Docker, k8s, or direct
foreground invocation.
"""

from __future__ import annotations

import os
import socket
from typing import Final

_ENV_VAR: Final[str] = "NOTIFY_SOCKET"


def sd_notify(state: str) -> bool:
    """Send a state line to systemd's notify socket.

    Returns True on success, False if the socket is not configured or the
    send failed. Common state strings:

      - ``READY=1``          — boot complete; wired into Type=notify
      - ``WATCHDOG=1``       — heartbeat for ``WatchdogSec=``
      - ``STOPPING=1``       — graceful shutdown in progress
      - ``STATUS=<text>``    — human-readable status line
      - ``RELOADING=1``      — reload in progress (with MONOTONIC_USEC)
    """
    addr = os.environ.get(_ENV_VAR)
    if not addr:
        return False

    # systemd uses an abstract namespace socket when the path begins with '@'.
    if addr.startswith("@"):
        addr = "\0" + addr[1:]

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC)
    except OSError:
        return False
    try:
        sock.sendto(state.encode("utf-8"), addr)
        return True
    except OSError:
        return False
    finally:
        sock.close()


def watchdog_enabled() -> int:
    """Return WatchdogSec in microseconds (per ``WATCHDOG_USEC`` env), 0 if off."""
    raw = os.environ.get("WATCHDOG_USEC", "")
    try:
        return int(raw)
    except ValueError:
        return 0


__all__ = ["sd_notify", "watchdog_enabled"]
