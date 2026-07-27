"""LIVE PostgreSQL/TimescaleDB tests for PostgresEvidenceStore. No mocks.

A module-scoped fixture boots a real ``timescale/timescaledb:2.16.1-pg16``
container (``--network=host``, server on port 55432 to avoid clashing with
any host postgres) and tears it down afterwards. Skips cleanly when the
docker daemon is unavailable.

Marked ``integration`` (requires a live external service).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from horizon_ric.evidence import (
    DecisionRecord,
    PostgresEvidenceStore,
    open_evidence_store,
)
from horizon_ric.security.tenant import TenantScope
from tests.test_evidence_store import _outcome, _record, _versions

pytestmark = pytest.mark.integration

PG_IMAGE = "timescale/timescaledb:2.16.1-pg16"
PG_PORT = 55432  # non-default; --network=host means ports are host-global
PG_PASSWORD = "horizon-e2e"
PG_DSN = f"postgresql://postgres:{PG_PASSWORD}@127.0.0.1:{PG_PORT}/decisions"


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=30
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0


def _tenant_record(decision_id: str, tenant_id: str) -> DecisionRecord:
    return DecisionRecord.new(
        decision_id=decision_id,
        rapp_instance_id="rapp-live-1",
        tenant_id=tenant_id,
        state_hash="deadbeef",
        chosen_action={"tx_dBm": 25.0, "gateway": "G3"},
        predicted_outcome_chosen=_outcome(),
        rejected_alternatives=[],
        model_versions=_versions(),
    )


@pytest.fixture(scope="module")
def pg_dsn():
    """Boot a real TimescaleDB container; yield its DSN; tear it down."""
    if not _docker_usable():
        pytest.skip("docker daemon unavailable; skipping live Postgres tests")
    name = f"horizon-evidence-pg-{uuid.uuid4().hex[:8]}"
    run = subprocess.run(
        [
            "docker", "run", "-d", "--network=host", "--name", name,
            "-e", f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-e", "POSTGRES_DB=decisions",
            PG_IMAGE,
            # --network=host: move the server off 5432 so we never clash
            # with a host postgres. Passed straight to the postmaster.
            "-c", f"port={PG_PORT}",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"cannot start {PG_IMAGE}: {run.stderr.strip()[:400]}")
    try:
        deadline = time.monotonic() + 120.0
        last_err: Exception | None = None
        while True:
            try:
                with psycopg.connect(PG_DSN, connect_timeout=3) as conn:
                    conn.execute("SELECT 1")
                break
            except Exception as exc:  # noqa: BLE001 - retry until deadline
                last_err = exc
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"postgres in {name} never became ready: {last_err}"
                    ) from exc
                time.sleep(1.0)
        yield PG_DSN
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture()
def store(pg_dsn):
    """A PostgresEvidenceStore over a freshly dropped evidence_records table."""
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS evidence_records")
    s = PostgresEvidenceStore(pg_dsn)
    yield s
    s.close()


class TestLiveChain:
    def test_append_across_two_tenants_verifies_intact(self, store, pg_dsn):
        tenants = ["tenant-a", "tenant-b", "tenant-a", "tenant-b", "tenant-a"]
        hashes = []
        for i, tid in enumerate(tenants):
            h = store.append(_tenant_record(f"d-{i}", tid))
            assert isinstance(h, str) and len(h) == 64
            hashes.append(h)
        assert len(set(hashes)) == 5
        assert len(store) == 5
        assert store.verify() == -1
        assert store.verify_tenant("tenant-a") == -1
        assert store.verify_tenant("tenant-b") == -1

    def test_hypertable_created_on_timescaledb(self, store, pg_dsn):
        store.append(_tenant_record("d-ht", "tenant-a"))
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            row = conn.execute(
                "SELECT count(*) FROM timescaledb_information.hypertables "
                "WHERE hypertable_name = 'evidence_records'"
            ).fetchone()
        assert row[0] == 1, "evidence_records should be a hypertable on TimescaleDB"

    def test_tamper_detected_at_exact_index(self, store, pg_dsn):
        tenants = ["tenant-a", "tenant-b", "tenant-a", "tenant-b", "tenant-a"]
        for i, tid in enumerate(tenants):
            store.append(_tenant_record(f"d-{i}", tid))
        assert store.verify() == -1
        # Forge the payload of the 3rd row (global index 2; tenant-a's 2nd)
        # directly in SQL — exactly what a malicious DBA would do.
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute(
                "UPDATE evidence_records "
                "SET record = jsonb_set(record, '{chosen_action,tx_dBm}', '99.9') "
                "WHERE decision_id = 'd-2'"
            )
        assert store.verify() == 2
        # Chains are per-tenant: tenant-b's chain must remain intact.
        assert store.verify_tenant("tenant-a") == 1
        assert store.verify_tenant("tenant-b") == -1

    def test_len_iter_round_trip_preserves_fields(self, store):
        from horizon_ric.evidence import (
            ConstraintCorrection,
            RejectedAlternative,
            RejectionReasonMachine,
        )
        from horizon_ric.evidence.store import _canonical_json

        original = DecisionRecord.new(
            decision_id="rt-1",
            rapp_instance_id="rapp-live-1",
            tenant_id="tenant-rt",
            state_hash="cafebabe",
            state_blob_uri="s3://horizon-evidence-archive/state/rt-1",
            chosen_action={"tx_dBm": 21.5, "beam": [1, 2, 3]},
            predicted_outcome_chosen=_outcome(0.07),
            rejected_alternatives=[
                RejectedAlternative(
                    rank=1,
                    action={"tx_dBm": 30.0},
                    predicted_outcome=_outcome(0.4),
                    rejection_reason_machine=RejectionReasonMachine(
                        primary_cause="gateway_overload",
                        primary_metric="gateway_load_G3",
                        predicted_value=0.97,
                        threshold=0.9,
                        horizon="30s",
                    ),
                    rejection_reason_human="G3 would exceed 90% load within 30s",
                    random_seed=1234,
                )
            ],
            constraint_corrections=[
                ConstraintCorrection(
                    constraint_id="emf-limit-01",
                    severity="hard",
                    margin_dB=1.5,
                    message="clamped tx power to EMF limit",
                )
            ],
            model_versions=_versions(),
            random_seed=42,
            sla_breach_context=[{"slo": "latency_p99", "value_ms": 12.7}],
        )
        appended_hash = store.append(original)
        assert len(store) == 1
        rows = list(store)
        assert len(rows) == 1
        round_tripped, stored_hash = rows[0]
        assert stored_hash == appended_hash
        # Byte-identical canonical JSON => every field survived JSONB.
        assert _canonical_json(round_tripped) == _canonical_json(original)
        assert round_tripped.decision_id == "rt-1"
        assert round_tripped.tenant_id == "tenant-rt"
        assert round_tripped.timestamp == original.timestamp
        assert round_tripped.rejected_alternatives[0].random_seed == 1234
        assert round_tripped.constraint_corrections[0].margin_dB == 1.5
        assert round_tripped.sla_breach_context == [
            {"slo": "latency_p99", "value_ms": 12.7}
        ]

    def test_tenant_scope_isolates_iteration_and_len(self, store):
        for i in range(3):
            store.append(_tenant_record(f"a-{i}", "tenant-a"))
        for i in range(2):
            store.append(_tenant_record(f"b-{i}", "tenant-b"))
        assert len(store) == 5
        with TenantScope("tenant-a"):
            assert len(store) == 3
            assert {r.decision_id for r, _ in store} == {"a-0", "a-1", "a-2"}
            assert store.verify() == -1
        with TenantScope("tenant-b"):
            assert len(store) == 2

    def test_unstamped_record_lands_in_scoped_tenant_chain(self, store):
        with TenantScope("tenant-scoped"):
            store.append(_record("scoped-1"))  # record.tenant_id is None
        rows = list(store)
        assert rows[0][0].tenant_id == "tenant-scoped"
        assert store.verify_tenant("tenant-scoped") == -1

    def test_concurrent_appends_keep_valid_chain(self, pg_dsn):
        """20 appends from 4 threads, each with its OWN connection.

        The client-side threading.Lock cannot help across store instances;
        only the pg advisory lock prevents two connections from reading the
        same prev-hash and forking the chain.
        """
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS evidence_records")
        n_threads, per_thread = 4, 5
        errors: list[Exception] = []

        def worker(worker_idx: int) -> None:
            s = PostgresEvidenceStore(pg_dsn)
            try:
                for j in range(per_thread):
                    tid = f"tenant-conc-{worker_idx % 2}"
                    s.append(_tenant_record(f"c-{worker_idx}-{j}", tid))
            except Exception as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)
            finally:
                s.close()

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"append failed under concurrency: {errors!r}"
        checker = PostgresEvidenceStore(pg_dsn)
        try:
            assert len(checker) == n_threads * per_thread
            assert checker.verify() == -1
            assert checker.verify_tenant("tenant-conc-0") == -1
            assert checker.verify_tenant("tenant-conc-1") == -1
        finally:
            checker.close()


class TestFactoryLive:
    def test_factory_opens_live_postgres(self, pg_dsn):
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS evidence_records")
        store = open_evidence_store(pg_dsn)
        assert isinstance(store, PostgresEvidenceStore)
        try:
            store.append(_tenant_record("fac-1", "tenant-f"))
            store.append(_tenant_record("fac-2", "tenant-f"))
            assert len(store) == 2
            assert store.verify() == -1
        finally:
            store.close()

    def test_factory_postgres_scheme_alias(self, pg_dsn):
        # libpq accepts both postgres:// and postgresql://
        alias = pg_dsn.replace("postgresql://", "postgres://")
        store = open_evidence_store(alias)
        assert isinstance(store, PostgresEvidenceStore)
        try:
            store.append(_tenant_record("fac-3", "tenant-f2"))
            assert store.verify_tenant("tenant-f2") == -1
        finally:
            store.close()
