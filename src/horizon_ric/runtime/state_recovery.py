"""Crash-resilient state checkpointing.

The rApp keeps a small piece of running state (rApp lifecycle phase,
A1-emitted counter, last decision id, evidence-chain head hash) that
must survive a crash so the daemon can rejoin its previous identity
on restart. This module provides the disk side of that contract:

  * `save_state(state, path)` — atomic write (write-temp + rename + fsync)
  * `load_state(path)` — returns None if missing or corrupt (logged)
  * `with_periodic_checkpoint(provider, path, interval)` — async context
    manager that snapshots `provider()` every interval

Atomicity is real:
  * the temporary file is fsync()'d before rename
  * `os.rename` is atomic on POSIX within the same filesystem
  * on a crash mid-write, the old file is intact (test verifies this)

Corruption recovery: if the JSON is malformed or the file is empty,
`load_state` returns None and emits `horizon.state.corrupt` so an
operator can investigate. The rApp boots from default state instead
of crash-looping.

Stable structlog event names:

    horizon.state.saved          — successful checkpoint write
    horizon.state.save_failed    — write failed (retryable)
    horizon.state.loaded         — successful restore on boot
    horizon.state.missing        — no checkpoint file present
    horizon.state.corrupt        — file exists but is unreadable
    horizon.state.checkpointer_started / stopped
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_CHECKPOINT_INTERVAL_S = 30.0


def save_state(state: dict[str, Any], path: Path) -> None:
    """Atomically persist `state` to `path`.

    Writes to a sibling temp file, fsync()s it, then rename()s into place.
    The directory entry is also fsynced so the rename survives a crash
    of the OS (not just the process).
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Use a temp file in the same dir (so rename is atomic — same FS).
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=parent)
    tmp = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, sort_keys=True, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX & Windows
        # Fsync the parent dir so the rename is durable.
        try:
            dir_fd = os.open(str(parent), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):  # pragma: no cover — non-POSIX
            pass
        logger.debug("horizon.state.saved", path=str(path), bytes=path.stat().st_size)
    except Exception as exc:
        logger.error("horizon.state.save_failed", path=str(path), error=str(exc))
        # Best-effort cleanup of the dangling temp.
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def load_state(path: Path) -> dict[str, Any] | None:
    """Load state from `path`. Returns None if missing or unreadable.

    Corruption is treated as a non-fatal condition: the operator gets a
    structured log event and the rApp continues with default state. This
    is correct for our RPO (30 s acceptable A1 lag) — re-emitting a
    handful of policies on recovery is cheaper than crashing.
    """
    path = Path(path)
    if not path.exists():
        logger.info("horizon.state.missing", path=str(path))
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            logger.error("horizon.state.corrupt", path=str(path), reason="empty")
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.error(
                "horizon.state.corrupt",
                path=str(path),
                reason=f"top-level not dict: {type(data).__name__}",
            )
            return None
        logger.info("horizon.state.loaded", path=str(path), keys=sorted(data.keys()))
        return data
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "horizon.state.corrupt",
            path=str(path),
            error=str(exc),
            exc_type=type(exc).__name__,
        )
        return None


@contextlib.asynccontextmanager
async def with_periodic_checkpoint(
    state_provider: Callable[[], dict[str, Any]] | Callable[[], Awaitable[dict[str, Any]]],
    path: Path,
    interval_s: float = DEFAULT_CHECKPOINT_INTERVAL_S,
) -> AsyncIterator[asyncio.Task[None]]:
    """Run a background checkpointer for the duration of the block.

    `state_provider` is called every `interval_s` (or its awaitable
    version). The result is saved via `save_state`. Failures are logged
    but do not stop the loop.

    On exit, one final snapshot is written so the most recent state is
    always on disk after a clean shutdown.
    """

    path = Path(path)

    async def _run() -> None:
        logger.info(
            "horizon.state.checkpointer_started",
            path=str(path),
            interval_s=interval_s,
        )
        try:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    state = state_provider()
                    if asyncio.iscoroutine(state):
                        state = await state
                    save_state(state, path)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "horizon.state.save_failed",
                        path=str(path),
                        error=str(exc),
                    )
        except asyncio.CancelledError:
            logger.info("horizon.state.checkpointer_stopped", path=str(path))
            raise

    task = asyncio.create_task(_run(), name="horizon.state.checkpointer")
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # Final snapshot on clean shutdown.
        try:
            state = state_provider()
            if asyncio.iscoroutine(state):
                state = await state
            save_state(state, path)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "horizon.state.save_failed",
                path=str(path),
                error=str(exc),
                phase="shutdown",
            )


__all__ = [
    "DEFAULT_CHECKPOINT_INTERVAL_S",
    "load_state",
    "save_state",
    "with_periodic_checkpoint",
]
