"""Evidence store: hash-chain integrity + tamper detection."""

import json
from pathlib import Path

import pytest

from horizon_ric.evidence import (
    DecisionRecord,
    JsonlEvidenceStore,
    ModelVersions,
    PredictedOutcome,
    SqliteEvidenceStore,
)


def _versions() -> ModelVersions:
    return ModelVersions(
        encoder="enc-0.1.0",
        risk_heads="rh-0.1.0",
        dyna="dy-0.1.0",
        policy="pol-0.1.0",
        constraint_layer="cl-0.1.0",
        rapp="horizon-ric-0.1.0",
    )


def _outcome(risk: float = 0.05) -> PredictedOutcome:
    return PredictedOutcome(
        sla_risk_30s=risk,
        sla_risk_1min=risk,
        sla_risk_5min=risk,
    )


def _record(decision_id: str) -> DecisionRecord:
    return DecisionRecord.new(
        decision_id=decision_id,
        rapp_instance_id="rapp-1",
        state_hash="abc",
        chosen_action={"tx_dBm": 25.0},
        predicted_outcome_chosen=_outcome(),
        rejected_alternatives=[],
        model_versions=_versions(),
    )


class TestJsonlStore:
    def test_append_returns_hash_and_persists(self, tmp_path: Path):
        store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
        h = store.append(_record("d-1"))
        assert isinstance(h, str) and len(h) == 64
        assert (tmp_path / "ev.jsonl").exists()
        assert len(store) == 1

    def test_chain_intact(self, tmp_path: Path):
        store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
        for i in range(5):
            store.append(_record(f"d-{i}"))
        assert store.verify() == -1  # all intact

    def test_chain_detects_tamper(self, tmp_path: Path):
        path = tmp_path / "ev.jsonl"
        store = JsonlEvidenceStore(path)
        for i in range(3):
            store.append(_record(f"d-{i}"))
        # Tamper with the middle record's payload
        lines = path.read_text().splitlines()
        obj = json.loads(lines[1])
        obj["record"]["chosen_action"]["tx_dBm"] = 99.9  # forged
        lines[1] = json.dumps(obj, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        # Verify must catch the tamper at index 1.
        assert store.verify() == 1

    def test_distinct_records_distinct_hashes(self, tmp_path: Path):
        store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
        h1 = store.append(_record("d-1"))
        h2 = store.append(_record("d-2"))
        assert h1 != h2

    def test_empty_store_verifies(self, tmp_path: Path):
        store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
        assert store.verify() == -1


class TestSqliteStore:
    def test_append_and_iterate(self, tmp_path: Path):
        url = f"sqlite:///{tmp_path / 'ev.db'}"
        store = SqliteEvidenceStore(url)
        h1 = store.append(_record("s-1"))
        h2 = store.append(_record("s-2"))
        assert h1 != h2
        records = list(store)
        assert [r.decision_id for r, _ in records] == ["s-1", "s-2"]

    def test_chain_intact(self, tmp_path: Path):
        url = f"sqlite:///{tmp_path / 'ev.db'}"
        store = SqliteEvidenceStore(url)
        for i in range(4):
            store.append(_record(f"s-{i}"))
        assert store.verify() == -1

    def test_duplicate_decision_id_raises(self, tmp_path: Path):
        url = f"sqlite:///{tmp_path / 'ev.db'}"
        store = SqliteEvidenceStore(url)
        store.append(_record("only-one"))
        with pytest.raises(Exception):  # IntegrityError or wrapped
            store.append(_record("only-one"))
