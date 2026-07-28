"""Tests for atomic state checkpointing and recovery."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from horizon_ric.runtime.state_recovery import (
    load_state,
    save_state,
    with_periodic_checkpoint,
)


def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    state = {
        "rapp_state": "running",
        "a1_emitted_count": 42,
        "last_decision_id": "abc-123",
        "evidence_chain_head": "deadbeef",
    }
    save_state(state, p)
    loaded = load_state(p)
    assert loaded == state


def test_load_missing_returns_none(tmp_path):
    assert load_state(tmp_path / "nope.json") is None


def test_load_corrupt_json_returns_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{this is not valid json}")
    assert load_state(p) is None


def test_load_empty_file_returns_none(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("")
    assert load_state(p) is None


def test_load_non_dict_returns_none(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]")
    assert load_state(p) is None


def test_atomicity_old_value_survives_failed_write(tmp_path, monkeypatch):
    """Simulate a crash mid-write: replace os.replace with a raise.

    The temp file may or may not exist; what matters is that the *real*
    target file still contains the old value. We don't get to see a
    partially-written state.json.
    """
    p = tmp_path / "state.json"
    save_state({"v": 1}, p)
    assert load_state(p) == {"v": 1}

    real_replace = os.replace

    def boom(*a, **kw):
        # Simulate a crash — temp file written and fsynced, but the
        # rename never happened.
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        save_state({"v": 2}, p)
    monkeypatch.setattr(os, "replace", real_replace)

    # Old value is intact. No partial file.
    assert load_state(p) == {"v": 1}


def test_atomicity_no_partial_file_visible(tmp_path):
    """save_state must never leave a half-written state.json on disk."""
    p = tmp_path / "state.json"
    save_state({"v": 1}, p)

    # The directory should contain only state.json (no .tmp) afterwards.
    leftover = [f.name for f in tmp_path.iterdir() if not f.name.endswith(".json")]
    assert leftover == [], f"unexpected leftover files: {leftover}"


def test_save_creates_parent_dir(tmp_path):
    nested = tmp_path / "sub" / "dir" / "state.json"
    save_state({"x": 1}, nested)
    assert nested.exists()
    assert load_state(nested) == {"x": 1}


# Wait on the checkpointer's own progress rather than on wall-clock. Sleeping
# for N intervals and asserting N ticks happened assumes the scheduler keeps up;
# a loaded CI runner can deliver one tick where the test expected five, which is
# a false failure about the runner rather than about the checkpointer.
TICK_TIMEOUT_S = 10.0


@pytest.mark.asyncio
async def test_periodic_checkpoint_writes_snapshots(tmp_path):
    p = tmp_path / "state.json"
    counter = {"i": 0}
    ticked = asyncio.Event()

    def provider():
        counter["i"] += 1
        if counter["i"] >= 3:
            ticked.set()
        return {"i": counter["i"]}

    async with with_periodic_checkpoint(provider, p, interval_s=0.01) as task:
        await asyncio.wait_for(ticked.wait(), timeout=TICK_TIMEOUT_S)
        assert not task.done()

    # After exit a final snapshot is written. Should reflect a high i.
    loaded = load_state(p)
    assert loaded is not None
    assert loaded["i"] >= 3


@pytest.mark.asyncio
async def test_periodic_checkpoint_provider_failure_doesnt_kill_loop(tmp_path):
    p = tmp_path / "state.json"
    state = {"v": 0}
    survived_failure = asyncio.Event()

    def provider():
        state["v"] += 1
        if state["v"] == 2:
            raise RuntimeError("transient")
        if state["v"] >= 3:
            # Reached only if the loop kept running past the raise above.
            survived_failure.set()
        return {"v": state["v"]}

    async with with_periodic_checkpoint(provider, p, interval_s=0.01) as task:
        await asyncio.wait_for(survived_failure.wait(), timeout=TICK_TIMEOUT_S)
        assert not task.done(), "checkpoint loop must survive provider failure"

    # The most recent successful snapshot is on disk.
    loaded = load_state(p)
    assert loaded is not None and loaded["v"] >= 3


@pytest.mark.asyncio
async def test_periodic_checkpoint_async_provider(tmp_path):
    p = tmp_path / "state.json"

    async def provider():
        return {"async": True}

    async with with_periodic_checkpoint(provider, p, interval_s=0.05):
        await asyncio.sleep(0.15)

    assert load_state(p) == {"async": True}
