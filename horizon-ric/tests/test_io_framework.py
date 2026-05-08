"""I/O framework tests: schemas, registry, file connector round-trip."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Trigger reference connector registration.
import horizon_ric.io.connectors  # noqa: F401
from horizon_ric.io import (
    ConnectorConfigError,
    PolicyAction,
    Sink,
    Source,
    TelemetryEvent,
    get_sink,
    get_source,
    list_connectors,
    register_sink,
    register_source,
    registry,
)
from horizon_ric.io.connectors.file_connector import FileSink, FileSource

# ─── Schemas ─────────────────────────────────────────────────────────────


class TestSchemas:
    def test_telemetry_event_round_trip(self):
        e = TelemetryEvent(
            event_id="e1",
            modality="kpm_5g",
            source_id="cell_42",
            ts_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            payload={"rsrp_dbm": -85},
        )
        blob = e.model_dump_json()
        e2 = TelemetryEvent.model_validate_json(blob)
        assert e == e2

    def test_naive_timestamp_rejected(self):
        with pytest.raises(Exception):
            TelemetryEvent(
                event_id="e1",
                modality="kpm_5g",
                source_id="src",
                ts_utc=datetime(2026, 1, 1),  # naive
            )

    def test_unknown_modality_rejected(self):
        with pytest.raises(Exception):
            TelemetryEvent(
                event_id="e1",
                modality="bogus_modality",  # not in Modality literal
                source_id="src",
                ts_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )

    def test_extra_fields_preserved_on_telemetry(self):
        e = TelemetryEvent(
            event_id="e1",
            modality="kpm_5g",
            source_id="src",
            ts_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            future_field="hello",
        )
        blob = e.model_dump()
        assert blob.get("future_field") == "hello"

    def test_policy_action_strict(self):
        # PolicyAction does not allow unknown fields by default per schemas.
        pa = PolicyAction(
            action_id="a1",
            policy_type="horizon.qos.priority",
            policy_payload={"priority": 5},
        )
        assert pa.policy_type == "horizon.qos.priority"


# ─── Registry ────────────────────────────────────────────────────────────


class TestRegistry:
    def test_default_connectors_present(self):
        names = list_connectors()
        assert "file" in names["sources"]
        assert "file" in names["sinks"]
        assert "http" in names["sources"]
        assert "http" in names["sinks"]

    def test_get_unknown_raises(self):
        with pytest.raises(ConnectorConfigError):
            get_source("does_not_exist", {})

    def test_get_with_invalid_config_raises(self):
        with pytest.raises(ConnectorConfigError):
            # FileSource requires `path`; this dict lacks it.
            get_source("file", {"name": "test"})

    def test_register_and_get(self):
        # A custom sink class for the registry round-trip.
        class _NoopSink(Sink):
            async def connect(self):
                self._state = self._state.STARTED
            async def close(self):
                self._state = self._state.STOPPED
            async def write(self, message):
                pass

        register_sink("noop_test", _NoopSink)
        sink = get_sink("noop_test", {"name": "test_instance"})
        assert isinstance(sink, _NoopSink)

    def test_duplicate_registration_idempotent(self):
        # Re-registering the SAME class is fine; different class raises.
        register_source("file", FileSource)  # already registered, same cls

        class _Other(Source):
            async def connect(self):
                pass
            async def close(self):
                pass
            async def stream(self):
                if False:
                    yield  # pragma: no cover

        with pytest.raises(ValueError):
            register_source("file", _Other)


# ─── File connector round-trip ───────────────────────────────────────────


class TestFileConnector:
    @pytest.mark.asyncio
    async def test_file_sink_then_source(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        sink = FileSink(FileSink.Config(name="sink", path=str(path)))
        await sink.connect()
        events = [
            TelemetryEvent(
                event_id=f"e{i}",
                modality="kpm_5g",
                source_id="cell_1",
                ts_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                payload={"rsrp_dbm": -80 - i},
            )
            for i in range(5)
        ]
        for e in events:
            await sink.write(e)
        await sink.close()

        # Verify file content.
        assert path.exists()
        lines = path.read_text().splitlines()
        assert len(lines) == 5
        first = json.loads(lines[0])
        assert first["schema"] == "telemetry"

    @pytest.mark.asyncio
    async def test_file_source_reads_telemetry_events(self, tmp_path: Path):
        path = tmp_path / "in.jsonl"
        # Create source data: one TelemetryEvent per line.
        events = [
            TelemetryEvent(
                event_id=f"e{i}",
                modality="thermal",
                source_id="edge_1",
                ts_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                payload={"tj_celsius": 70 + i},
            )
            for i in range(3)
        ]
        path.write_text("\n".join(e.model_dump_json() for e in events) + "\n")

        src = FileSource(FileSource.Config(name="src", path=str(path)))
        await src.connect()
        recovered: list[TelemetryEvent] = []
        async for e in src.stream():
            recovered.append(e)
        await src.close()
        assert len(recovered) == 3
        assert recovered[0].payload["tj_celsius"] == 70

    @pytest.mark.asyncio
    async def test_source_missing_file_raises(self, tmp_path: Path):
        src = FileSource(FileSource.Config(
            name="src", path=str(tmp_path / "absent.jsonl")
        ))
        with pytest.raises(Exception):
            await src.connect()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, tmp_path: Path):
        path = tmp_path / "ctx.jsonl"
        async with FileSink(FileSink.Config(name="ctx", path=str(path))) as sink:
            await sink.write(
                TelemetryEvent(
                    event_id="e",
                    modality="kpm_5g",
                    source_id="x",
                    ts_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
            )
        # File should be flushed and closed.
        assert path.exists()
        assert path.stat().st_size > 0
