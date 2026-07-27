"""Runtime-defect regression tests for the packaged daemon
(:mod:`horizon_ric.rapp.daemon`, shimmed by ``scripts/run_horizon_rapp.py``).

Covers checkpoint resume (real 2-run sequence over a tmp file), malformed-line
tolerance, the missing-telemetry-file policy (hard error in daemon mode, loud
synthetic fallback in --once), the explicit ``source.type: synthetic``,
error-vs-emit_failed report bookkeeping, prompt SIGTERM shutdown while blocked
on a slow source, and sd_notify READY/WATCHDOG integration.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from horizon_ric.rapp.daemon import run_daemon
from tests.test_pipeline_wiring import _build_a1_app, _event, _new_state, _serve

_ENV_TO_CLEAR = (
    "A1_CLIENT_TOKEN",
    "HORIZON_A1_OAUTH_TOKEN_URL",
    "HORIZON_A1_OAUTH_CLIENT_ID",
    "HORIZON_A1_OAUTH_CLIENT_SECRET",
    "HORIZON_A1_CLIENT_CERT_PATH",
    "HORIZON_A1_CLIENT_KEY_PATH",
    "HORIZON_A1_DIALECT",
    "HORIZON_A1_DRY_RUN",
    "HORIZON_A1_REQUIRE_CERT",
    "HORIZON_CERT_SIGNING_KEY_PATH",
    "HORIZON_ONCE_REQUIRE_ACCEPTED",
    "HORIZON_SHIELD_MAX_EIRP_DBM",
    "HORIZON_DECISION_BUDGET_MS",
    "NOTIFY_SOCKET",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    # Fast, deterministic polling and fast unreachable-SMO boot.
    monkeypatch.setenv("HORIZON_A1_STATUS_POLL_INTERVAL_S", "0.05")
    monkeypatch.setenv("HORIZON_SMO_URL", "http://127.0.0.1:1")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _closed_port() -> int:
    # Bind-then-close: nothing listens afterwards → instant ConnectError.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_config(tmp_path: Path, source: dict, name: str = "daemon.yaml") -> Path:
    cfg = tmp_path / name
    cfg.write_text(
        json.dumps(  # JSON is valid YAML
            {
                "source": source,
                "sink": {"audit_path": str(tmp_path / "audit.jsonl")},
                "health": {"port": _free_port()},
                "metrics": {"port": _free_port()},
                "state": {
                    "checkpoint_path": str(tmp_path / "state.json"),
                    "interval_seconds": 30,
                },
            }
        )
    )
    return cfg


def _write_events(path: Path, event_ids: list[str], risk: float = 0.05) -> None:
    path.write_text(
        "\n".join(
            _event(event_id=eid, sla_risk=risk).model_dump_json() for eid in event_ids
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# Checkpoint resume — a restart must not re-emit duplicate policies
# ---------------------------------------------------------------------------
def test_checkpoint_resume_skips_already_processed_events(tmp_path: Path, monkeypatch):
    events_file = tmp_path / "replay.jsonl"
    _write_events(events_file, ["evt-r0", "evt-r1", "evt-r2"])
    config = _write_config(tmp_path, {"type": "file", "path": str(events_file)})

    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{port}")

        report1 = tmp_path / "report1.json"
        rc1 = run_daemon(source_config=str(config), once=True, report_json=str(report1))
        assert rc1 == 0
        r1 = json.loads(report1.read_text())
        assert r1["events"] == 3
        assert r1["accepted"] == 3
        assert len(state["policies"]) == 3

        # Second run over the SAME file with the SAME checkpoint: everything
        # up to and including the checkpointed event_id is skipped — zero
        # duplicate policies reach the RIC.
        report2 = tmp_path / "report2.json"
        rc2 = run_daemon(source_config=str(config), once=True, report_json=str(report2))
        assert rc2 == 0
        r2 = json.loads(report2.read_text())
        assert r2["events"] == 0
        assert r2["accepted"] == 0
        assert len(state["policies"]) == 3  # unchanged — no re-emits

        # Appending new events after the checkpointed id resumes mid-file.
        with events_file.open("a") as f:
            f.write(_event(event_id="evt-r3", sla_risk=0.05).model_dump_json() + "\n")
        report3 = tmp_path / "report3.json"
        rc3 = run_daemon(source_config=str(config), once=True, report_json=str(report3))
        assert rc3 == 0
        r3 = json.loads(report3.read_text())
        assert r3["events"] == 1
        assert len(state["policies"]) == 4

    checkpoint = json.loads((tmp_path / "state.json").read_text())
    assert checkpoint["last_event_id"] == "evt-r3"


def test_checkpoint_resume_changed_file_processes_all_with_warning(
    tmp_path: Path, monkeypatch, capsys
):
    events_file = tmp_path / "replay.jsonl"
    _write_events(events_file, ["evt-a0", "evt-a1"])
    config = _write_config(tmp_path, {"type": "file", "path": str(events_file)})

    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{port}")
        rc1 = run_daemon(source_config=str(config), once=True)
        assert rc1 == 0
        assert len(state["policies"]) == 2

        # Replace the file: the checkpointed event id no longer exists —
        # the daemon must warn and process ALL events (no silent skip).
        _write_events(events_file, ["evt-b0", "evt-b1", "evt-b2"])
        report2 = tmp_path / "report2.json"
        rc2 = run_daemon(source_config=str(config), once=True, report_json=str(report2))
        assert rc2 == 0
        r2 = json.loads(report2.read_text())
        assert r2["events"] == 3
        assert len(state["policies"]) == 5

    out = capsys.readouterr().out
    assert "daemon.resume.checkpoint_event_not_found" in out


# ---------------------------------------------------------------------------
# Malformed source lines must not kill the daemon
# ---------------------------------------------------------------------------
def test_malformed_lines_are_skipped_not_fatal(tmp_path: Path, monkeypatch, capsys):
    events_file = tmp_path / "replay.jsonl"
    good0 = _event(event_id="evt-m0").model_dump_json()
    good1 = _event(event_id="evt-m1").model_dump_json()
    events_file.write_text(
        "\n".join(
            [
                good0,
                "{ this is not json",  # unparseable line
                json.dumps({"event_id": "evt-bad", "modality": "kpm_5g"}),  # schema-invalid
                good1,
            ]
        )
        + "\n"
    )
    config = _write_config(tmp_path, {"type": "file", "path": str(events_file)})

    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{port}")
        report = tmp_path / "report.json"
        rc = run_daemon(source_config=str(config), once=True, report_json=str(report))

    assert rc == 0
    r = json.loads(report.read_text())
    # Both good events (including the one AFTER the bad lines) processed.
    assert r["events"] == 2
    assert r["accepted"] == 2
    assert r["malformed"] == 2
    assert [p["policy_id"] for p in state["policies"]]  # policies flowed
    out = capsys.readouterr().out
    assert "daemon.event.malformed" in out


# ---------------------------------------------------------------------------
# Missing telemetry file policy
# ---------------------------------------------------------------------------
def test_missing_file_hard_error_in_daemon_mode(tmp_path: Path, monkeypatch, capsys):
    config = _write_config(
        tmp_path, {"type": "file", "path": str(tmp_path / "nope.jsonl")}
    )
    monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{_closed_port()}")
    rc = run_daemon(source_config=str(config), once=False)
    assert rc == 2  # loud startup failure, nonzero exit — no synthetic stream
    captured = capsys.readouterr()
    assert "telemetry source file not found" in captured.err


def test_missing_file_once_mode_falls_back_to_synthetic_loudly(
    tmp_path: Path, monkeypatch, capsys
):
    config = _write_config(
        tmp_path, {"type": "file", "path": str(tmp_path / "nope.jsonl")}
    )
    # RIC unreachable: every emit fails (ConnectError, then open breaker) —
    # the point here is ONLY that --once still drains a synthetic batch.
    monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{_closed_port()}")
    report = tmp_path / "report.json"
    rc = run_daemon(source_config=str(config), once=True, report_json=str(report))
    assert rc == 0
    r = json.loads(report.read_text())
    assert r["events"] == 10
    assert r["accepted"] == 0
    assert r["emit_failed"] == 10
    out = capsys.readouterr().out
    assert "daemon.source.file_missing_synthetic_fallback" in out


def test_unknown_source_type_fails_loudly(tmp_path: Path, monkeypatch, capsys):
    config = _write_config(tmp_path, {"type": "carrier-pigeon"})
    monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{_closed_port()}")
    rc = run_daemon(source_config=str(config), once=True)
    assert rc == 2
    assert "unknown source type" in capsys.readouterr().err


def test_kafka_source_without_aiokafka_fails_loudly(tmp_path: Path, monkeypatch, capsys):
    import importlib.util

    if importlib.util.find_spec("aiokafka") is not None:
        pytest.skip("aiokafka installed — missing-dep startup path not reachable")
    config = _write_config(
        tmp_path,
        {
            "type": "kafka",
            "bootstrap_servers": "localhost:9092",
            "topic": "horizon.telemetry",
            "group_id": "horizon-rapp",
        },
    )
    monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{_closed_port()}")
    rc = run_daemon(source_config=str(config), once=True)
    assert rc == 2
    err = capsys.readouterr().err
    assert "aiokafka" in err and "horizon-ric[kafka]" in err


def test_kafka_source_invalid_config_fails_loudly(tmp_path: Path, monkeypatch, capsys):
    config = _write_config(
        tmp_path,
        {"type": "kafka", "topic": "t", "group_id": "g"},  # no bootstrap_servers
    )
    monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{_closed_port()}")
    rc = run_daemon(source_config=str(config), once=True)
    assert rc == 2
    assert "bootstrap_servers" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Explicit synthetic source (demo installs)
# ---------------------------------------------------------------------------
def test_synthetic_source_type_with_risk_cycle(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        {"type": "synthetic", "count": 5, "interval_seconds": 0, "risk_profile": "cycle"},
    )
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{port}")
        report = tmp_path / "report.json"
        rc = run_daemon(source_config=str(config), once=True, report_json=str(report))

    assert rc == 0
    r = json.loads(report.read_text())
    assert r["events"] == 5
    assert r["accepted"] == 5
    # The 0.05→0.9 risk cycle crosses all planner bands: the first five
    # values (0.05, 0.2, 0.35, 0.5, 0.65) hit QoS, steering AND admission.
    seen_types = {p["policy_type_id"] for p in state["policies"]}
    assert seen_types == {20001, 20002, 20003}


def test_synthetic_source_bounded_count_stops(tmp_path: Path, monkeypatch):
    # count=3 with --once max 10: the synthetic generator itself must stop
    # after 3 events (bounded count), not the --once cap.
    config = _write_config(
        tmp_path, {"type": "synthetic", "count": 3, "interval_seconds": 0}
    )
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{port}")
        report = tmp_path / "report.json"
        rc = run_daemon(source_config=str(config), once=True, report_json=str(report))
    assert rc == 0
    assert json.loads(report.read_text())["events"] == 3


# ---------------------------------------------------------------------------
# Report bookkeeping: unexpected pipeline exceptions are `errors`
# ---------------------------------------------------------------------------
def test_unexpected_pipeline_exception_counts_as_error_not_emit_failed(
    tmp_path: Path, monkeypatch
):
    from horizon_ric.rapp.pipeline import DecisionPipeline

    events_file = tmp_path / "replay.jsonl"
    _write_events(events_file, ["evt-e0", "evt-e1", "evt-e2"])
    config = _write_config(tmp_path, {"type": "file", "path": str(events_file)})

    orig = DecisionPipeline.process_event

    async def flaky(self, event):
        if event.event_id == "evt-e1":
            raise RuntimeError("injected pipeline fault")
        return await orig(self, event)

    monkeypatch.setattr(DecisionPipeline, "process_event", flaky)

    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{port}")
        report = tmp_path / "report.json"
        rc = run_daemon(source_config=str(config), once=True, report_json=str(report))

    assert rc == 0
    r = json.loads(report.read_text())
    assert r["events"] == 3
    assert r["accepted"] == 2
    assert r["errors"] == 1
    assert r["emit_failed"] == 0  # an internal fault is NOT an emit failure


# ---------------------------------------------------------------------------
# SIGTERM while blocked on a slow source — prompt shutdown (stop-race)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_sigterm_prompt_shutdown_while_blocked_on_slow_source(tmp_path: Path):
    # Unbounded synthetic source with a 60 s interval: after the first
    # event the daemon blocks inside the source sleep. SIGTERM must still
    # shut it down promptly via the anext-vs-stop race.
    config = _write_config(
        tmp_path, {"type": "synthetic", "count": 0, "interval_seconds": 60}
    )
    env = dict(os.environ)
    env.pop("HORIZON_ONCE_REQUIRE_ACCEPTED", None)
    env["HORIZON_NEAR_RT_RIC_URL"] = f"http://127.0.0.1:{_closed_port()}"
    env["HORIZON_SMO_URL"] = "http://127.0.0.1:1"
    env["HORIZON_A1_STATUS_POLL_INTERVAL_S"] = "0.05"
    proc = subprocess.Popen(
        [sys.executable, "-m", "horizon_ric.rapp.daemon", "--source-config", str(config)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Give it time to boot and enter the blocked source read.
        time.sleep(4.0)
        assert proc.poll() is None, f"daemon exited early:\n{proc.stdout.read()}"
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=20.0)
        elapsed = time.monotonic() - t0
    finally:
        if proc.poll() is None:  # pragma: no cover — cleanup on failure
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, out
    assert elapsed < 15.0, f"shutdown took {elapsed:.1f}s (blocked source not raced)"
    assert "daemon.shutdown" in out


# ---------------------------------------------------------------------------
# sd_notify: READY=1 + WATCHDOG=1 + STOPPING=1 for systemd Type=notify
# ---------------------------------------------------------------------------
def test_sd_notify_ready_watchdog_and_stopping(tmp_path: Path, monkeypatch):
    sock_path = str(tmp_path / "notify.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(sock_path)
    server.setblocking(False)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", sock_path)
        monkeypatch.setenv(
            "HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{_closed_port()}"
        )
        config = _write_config(
            tmp_path, {"type": "synthetic", "count": 2, "interval_seconds": 0}
        )
        rc = run_daemon(source_config=str(config), once=True)
        assert rc == 0

        messages: list[bytes] = []
        while True:
            try:
                messages.append(server.recv(4096))
            except BlockingIOError:
                break
    finally:
        server.close()

    assert b"READY=1" in messages, messages
    assert b"WATCHDOG=1" in messages, messages  # first ping fires at task start
    assert b"STOPPING=1" in messages, messages  # watchdog_loop cancel hook
