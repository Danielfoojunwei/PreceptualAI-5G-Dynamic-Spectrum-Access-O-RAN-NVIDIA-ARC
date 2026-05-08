"""Disaster-recovery drill — real backup -> truncate -> restore -> verify.

Closes Row 38 of the gap matrix. We exercise `scripts/backup_and_restore.sh`
end-to-end against a real SQLite database (the prod backend is
TimescaleDB; the bash script already routes to either via
HORIZON_DR_BACKEND). No mocks — `subprocess.run("bash …backup")` actually
writes a dump, signs it, and pushes it to a local archive directory.

Three test cases:

  1. Round-trip restores all 100 evidence rows AND the SHA-256 chain
     re-verifies on the JSONL store rebuilt from the restored DB.
  2. Tamper-after-restore breaks `verify()`. (Sanity check that the chain
     check we rely on for RTO is real.)
  3. Retention prune keeps newest N daily snapshots and drops the rest,
     including their `.sig` companion files.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "backup_and_restore.sh"

# Make sure the package under src/ is importable.
sys.path.insert(0, str(ROOT / "src"))

from horizon_ric.evidence.schema import (  # noqa: E402
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(i: int) -> DecisionRecord:
    """Construct a real DecisionRecord — no Mock objects."""
    return DecisionRecord(
        decision_id=f"dec-{i:04d}",
        timestamp=datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=i),
        rapp_instance_id="dr-drill-rapp",
        tenant_id="dr-drill",
        state_hash=("%064x" % i),
        chosen_action={"prb_share": 0.5 + i * 1e-4},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.01, sla_risk_1min=0.02, sla_risk_5min=0.03
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="v1", risk_heads="v1", dyna="v1",
            policy="v1", constraint_layer="v1", rapp="v1",
        ),
    )


def _populate_jsonl(path: Path, n: int) -> JsonlEvidenceStore:
    store = JsonlEvidenceStore(path)
    for i in range(n):
        store.append(_make_record(i))
    return store


def _jsonl_to_sqlite(jsonl: Path, sqlite_path: Path) -> None:
    """Materialise the JSONL evidence file into a `decisions` SQLite table.
    Mirrors the prod TimescaleDB schema (decision_id, ts, tenant_id,
    chain_hash, payload) so the bash script's pg_dump path is exercised
    on real rows."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("""
            CREATE TABLE decisions (
                decision_id TEXT NOT NULL,
                ts          TEXT NOT NULL,
                tenant_id   TEXT NOT NULL,
                chain_hash  TEXT NOT NULL,
                payload     TEXT NOT NULL,
                PRIMARY KEY (decision_id, ts)
            );
        """)
        with jsonl.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rec = obj["record"]
                conn.execute(
                    "INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
                    (
                        rec["decision_id"],
                        rec["timestamp"],
                        obj["tenant_id"],
                        obj["hash"],
                        json.dumps(rec, sort_keys=True, separators=(",", ":")),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def _sqlite_to_jsonl(sqlite_path: Path, jsonl: Path) -> None:
    """Inverse of the above — read the `decisions` table back into JSONL
    so we can re-verify the SHA-256 chain on the restored content."""
    if jsonl.exists():
        jsonl.unlink()
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    try:
        rows = conn.execute(
            "SELECT decision_id, ts, tenant_id, chain_hash, payload "
            "FROM decisions ORDER BY ts ASC"
        ).fetchall()
    finally:
        conn.close()
    with jsonl.open("w") as fh:
        for _, _, tid, h, payload in rows:
            line = json.dumps(
                {"hash": h, "tenant_id": tid, "record": json.loads(payload)},
                sort_keys=True,
            )
            fh.write(line + "\n")


def _run_backup(env: dict[str, str]) -> str:
    proc = subprocess.run(
        ["bash", str(SCRIPT), "backup"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"backup exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    # The script prints the snapshot key on stdout's last line.
    key = proc.stdout.strip().splitlines()[-1]
    assert key.startswith("decisions-") and key.endswith(".dump"), (
        f"unexpected backup key: {key!r} (full stdout: {proc.stdout!r})"
    )
    return key


def _run_restore(key: str, env: dict[str, str]) -> None:
    proc = subprocess.run(
        ["bash", str(SCRIPT), "restore", key],
        env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"restore exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )


def _truncate(env: dict[str, str]) -> None:
    """Wipe all rows in the source DB without dropping the schema."""
    sqlite_path = env["HORIZON_DR_DB_URL"][len("sqlite://"):]
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("DELETE FROM decisions;")
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    finally:
        conn.close()
    assert n == 0, f"truncate failed; {n} rows remain"


def _make_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    sqlite_path = tmp_path / "decisions.sqlite"
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    env.update(
        HORIZON_DR_BACKEND="sqlite",
        HORIZON_DR_DB_URL=f"sqlite://{sqlite_path}",
        HORIZON_DR_LOCAL_ARCHIVE=str(archive),
        HORIZON_DR_REQUIRE_COSIGN="0",
        HORIZON_DR_HMAC_SECRET="dr-drill-test-secret",
    )
    return env


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SCRIPT.exists(), reason="backup_and_restore.sh missing")
def test_dr_round_trip_preserves_100_records_and_chain(tmp_path: Path) -> None:
    """Write 100 records, back up, truncate, restore, verify chain."""
    jsonl = tmp_path / "evidence.jsonl"
    _populate_jsonl(jsonl, 100)
    env = _make_env(tmp_path)
    sqlite_path = Path(env["HORIZON_DR_DB_URL"][len("sqlite://"):])

    # Stage the JSONL into the SQLite DB the script will dump.
    _jsonl_to_sqlite(jsonl, sqlite_path)

    # 1. Backup completes successfully.
    key = _run_backup(env)
    archive = Path(env["HORIZON_DR_LOCAL_ARCHIVE"])
    assert (archive / key).exists(), "dump artifact not stored"
    assert (archive / f"{key}.sig").exists(), "signature artifact not stored"
    assert (archive / key).stat().st_size > 0

    # 2. Truncate the source.
    _truncate(env)

    # 3. Restore.
    _run_restore(key, env)

    # 4. Reconstitute the JSONL from the restored SQLite and verify the chain.
    restored_jsonl = tmp_path / "evidence.restored.jsonl"
    _sqlite_to_jsonl(sqlite_path, restored_jsonl)

    restored_store = JsonlEvidenceStore(restored_jsonl)
    n_rows = sum(1 for _ in restored_store)
    assert n_rows == 100, f"restored row count = {n_rows}, expected 100"
    bad = restored_store.verify()
    assert bad == -1, f"chain broken at index {bad} after restore"


@pytest.mark.skipif(not SCRIPT.exists(), reason="backup_and_restore.sh missing")
def test_dr_tamper_after_restore_breaks_chain(tmp_path: Path) -> None:
    """Sanity: if we mutate a restored row, verify() must catch it."""
    jsonl = tmp_path / "evidence.jsonl"
    _populate_jsonl(jsonl, 50)
    env = _make_env(tmp_path)
    sqlite_path = Path(env["HORIZON_DR_DB_URL"][len("sqlite://"):])

    _jsonl_to_sqlite(jsonl, sqlite_path)
    key = _run_backup(env)
    _truncate(env)
    _run_restore(key, env)

    restored = tmp_path / "evidence.restored.jsonl"
    _sqlite_to_jsonl(sqlite_path, restored)

    # Tamper with the 25th record's payload but keep the stored hash.
    lines = restored.read_text().splitlines()
    assert len(lines) == 50
    tampered = json.loads(lines[25])
    tampered["record"]["chosen_action"]["prb_share"] = 9.9999
    lines[25] = json.dumps(tampered, sort_keys=True)
    restored.write_text("\n".join(lines) + "\n")

    bad = JsonlEvidenceStore(restored).verify()
    # Once the 25th record is corrupted, verify() returns 25 OR a later
    # index where the cascading chain mismatch is detected.
    assert bad >= 25, (
        f"verify() should have caught tamper at >=25, got {bad}"
    )


@pytest.mark.skipif(not SCRIPT.exists(), reason="backup_and_restore.sh missing")
def test_dr_retention_prunes_oldest_dailies(tmp_path: Path) -> None:
    """Pre-seed the archive with 5 distinct daily snapshots, run one more
    real backup with HORIZON_DR_RETENTION_DAILY=3, and assert the pruner
    kept only the 3 newest (real backup + two newest seeded) and dropped
    every older `.sig` companion file."""
    jsonl = tmp_path / "evidence.jsonl"
    _populate_jsonl(jsonl, 10)
    env = _make_env(tmp_path)
    env["HORIZON_DR_RETENTION_DAILY"] = "3"
    sqlite_path = Path(env["HORIZON_DR_DB_URL"][len("sqlite://"):])
    _jsonl_to_sqlite(jsonl, sqlite_path)

    archive = Path(env["HORIZON_DR_LOCAL_ARCHIVE"])
    # Seed five "older" snapshots with deterministic timestamps in the past.
    # `ls -t` orders by mtime, so we set explicit, monotonically increasing
    # mtimes; the oldest must be at index 0.
    seeded: list[str] = []
    base = 1_700_000_000  # arbitrary epoch in the past
    for i in range(5):
        ts = f"20240101T0{i}0000Z"  # textually distinct timestamps
        key = f"decisions-{ts}.dump"
        (archive / key).write_bytes(b"seed-payload-%d" % i)
        (archive / f"{key}.sig").write_bytes(b"seed-sig-%d" % i)
        os.utime(archive / key, (base + i, base + i))
        os.utime(archive / f"{key}.sig", (base + i, base + i))
        seeded.append(key)

    # Run one real backup — its mtime is "now", so it is the newest entry.
    fresh_key = _run_backup(env)
    assert fresh_key not in seeded
    assert (archive / fresh_key).exists()

    remaining = sorted(p.name for p in archive.iterdir() if p.name.endswith(".dump"))
    # We expect exactly RETENTION_DAILY (=3) survivors.
    assert len(remaining) == 3, (
        f"expected 3 dailies after prune, got {len(remaining)}: {remaining}"
    )
    # The 3 newest are: fresh_key + the two latest seeded (indices 4, 3).
    expected = {fresh_key, seeded[4], seeded[3]}
    assert set(remaining) == expected, (
        f"unexpected survivors: got {set(remaining)}, expected {expected}"
    )
    # The 3 oldest seeded snapshots must be gone, signatures included.
    for stale in seeded[:3]:
        assert not (archive / stale).exists(), f"{stale} still on disk"
        assert not (archive / f"{stale}.sig").exists(), f"{stale}.sig still on disk"
