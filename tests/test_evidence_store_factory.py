"""Unit tests for `open_evidence_store` — the HORIZON_EVIDENCE_DSN contract.

No docker / live database required: the Postgres store connects lazily,
so the factory must be able to construct it without a reachable server.
Live Postgres behaviour is covered in tests/test_evidence_postgres_live.py.
"""

from pathlib import Path

from horizon_ric.evidence import (
    EvidenceStore,
    JsonlEvidenceStore,
    PostgresEvidenceStore,
    SqliteEvidenceStore,
    open_evidence_store,
)
from tests.test_evidence_store import _record


class TestFactoryDispatch:
    def test_postgres_scheme(self):
        store = open_evidence_store(
            "postgres://horizon@timescaledb:5432/decisions"
        )
        assert isinstance(store, PostgresEvidenceStore)

    def test_postgresql_scheme(self):
        store = open_evidence_store(
            "postgresql://horizon:pw@127.0.0.1:55432/decisions"
        )
        assert isinstance(store, PostgresEvidenceStore)

    def test_postgres_construction_is_lazy(self):
        # Port 1 is never listening — construction must not connect.
        store = open_evidence_store("postgresql://x@127.0.0.1:1/nope")
        assert isinstance(store, PostgresEvidenceStore)

    def test_sqlite_url_scheme(self, tmp_path: Path):
        store = open_evidence_store(f"sqlite:///{tmp_path / 'ev.db'}")
        assert isinstance(store, SqliteEvidenceStore)

    def test_db_suffix(self, tmp_path: Path):
        store = open_evidence_store(str(tmp_path / "evidence.db"))
        assert isinstance(store, SqliteEvidenceStore)

    def test_sqlite_suffix_as_path_object(self, tmp_path: Path):
        store = open_evidence_store(tmp_path / "evidence.sqlite")
        assert isinstance(store, SqliteEvidenceStore)

    def test_jsonl_suffix(self, tmp_path: Path):
        store = open_evidence_store(str(tmp_path / "audit.jsonl"))
        assert isinstance(store, JsonlEvidenceStore)

    def test_plain_path_defaults_to_jsonl(self, tmp_path: Path):
        store = open_evidence_store(tmp_path / "audit_log")
        assert isinstance(store, JsonlEvidenceStore)

    def test_every_result_is_an_evidence_store(self, tmp_path: Path):
        for target in (
            "postgres://x@localhost:5432/d",
            str(tmp_path / "a.db"),
            str(tmp_path / "b.jsonl"),
        ):
            assert isinstance(open_evidence_store(target), EvidenceStore)


class TestFactoryStoresWork:
    """The non-Postgres stores the factory returns must be fully usable."""

    def test_sqlite_store_round_trip(self, tmp_path: Path):
        store = open_evidence_store(str(tmp_path / "ev.sqlite"))
        h = store.append(_record("f-sq-1"))
        assert isinstance(h, str) and len(h) == 64
        assert [r.decision_id for r, _ in store] == ["f-sq-1"]
        assert store.verify() == -1

    def test_jsonl_store_round_trip(self, tmp_path: Path):
        store = open_evidence_store(str(tmp_path / "ev.jsonl"))
        store.append(_record("f-jl-1"))
        store.append(_record("f-jl-2"))
        assert len(store) == 2
        assert store.verify() == -1
