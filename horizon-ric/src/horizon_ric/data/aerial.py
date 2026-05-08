"""NVIDIA Aerial / cuBB / pyAerial data adapters.

Real-data readers for NVIDIA-native 5G NR L2/L1 traces. These are the
production replacements for the synthetic maritime generators that
PreceptualAI used during the early prototype phase.

Sources covered here:

  * Aerial **FAPI** parquet (`fapi.parquet`) — 3GPP small-cell L2/L1
    message log, one row per PUSCH/PDSCH scheduling decision. Produces
    `kpm_5g` events with the per-row PHY parameters in the payload.
  * Aerial **front-haul** parquet (`fh.parquet`) — split 7.2x O-RAN
    front-haul samples; the `fhData` column carries int16 IQ data, so
    the modality is auto-detected as `spectrum_iq`.
  * Aerial / cuMAC **HDF5 test vectors** — official cuBB end-to-end PHY
    test vectors. Each top-level dataset becomes its own
    `TelemetryEvent` with shape and dtype embedded in the payload.

All readers are pure Python iterators that emit
`horizon_ric.io.schemas.TelemetryEvent`. No tensors, no Sionna. The
pipeline downstream is responsible for any feature extraction.

A small `AerialDataset(IterableDataset)` glues multiple readers together
for use with `torch.utils.data.DataLoader`, and offers a
`feature_frame_window(window_s)` helper that buckets events into
`FeatureFrame` windows aligned with the existing schema.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq

try:
    import h5py  # type: ignore
except ImportError as _h5_err:  # pragma: no cover - exercised in CI without h5py
    h5py = None  # type: ignore
    _H5_IMPORT_ERROR = _h5_err
else:
    _H5_IMPORT_ERROR = None

from torch.utils.data import IterableDataset

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorIOError,
    ConnectorState,
    Source,
)
from horizon_ric.io.schemas import FeatureFrame, Modality, TelemetryEvent

# Heuristic column names that suggest IQ-like front-haul data. If any of
# these appear in the parquet, we tag the modality as `spectrum_iq`,
# otherwise we fall back to `kpm_5g`.
_IQ_HINT_COLUMNS = {"fhData", "iq", "iq_samples", "samples", "I", "Q"}

_FAPI_TS_COLS = ("TsTaiNs", "TsSwNs", "ts_utc", "timestamp")
_SCHEMAS_PRINTED: set[str] = set()


# ─── helpers ─────────────────────────────────────────────────────────────


def _utc_from_ns(ns: int) -> datetime:
    """Convert nanoseconds-since-epoch to a tz-aware UTC datetime.

    The Aerial parquet stores timestamps with nanosecond precision; we
    floor to microseconds because Python `datetime` only supports
    microseconds. The lost precision is preserved separately in
    `monotonic_ns` on the resulting `TelemetryEvent`.
    """

    if ns is None:
        return datetime.now(tz=timezone.utc)
    secs, rem_ns = divmod(int(ns), 1_000_000_000)
    return datetime.fromtimestamp(secs, tz=timezone.utc) + timedelta(
        microseconds=rem_ns // 1000
    )


def _row_timestamp_ns(row: dict[str, Any]) -> int | None:
    for col in _FAPI_TS_COLS:
        v = row.get(col)
        if v is None:
            continue
        # pyarrow returns numpy.datetime64; coerce to int ns.
        if isinstance(v, np.datetime64):
            return int(v.astype("datetime64[ns]").astype("int64"))
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, datetime):
            ts = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
            return int(ts.timestamp() * 1e9)
    return None


def _coerce_payload_value(v: Any) -> Any:
    """JSON-friendly coercion for parquet/h5 cell values.

    Pydantic accepts these as `Any`, but we still want them serialisable
    when the event is later round-tripped through `model_dump_json()`.
    """

    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (bytes, bytearray)):
        # Aerial FAPI stores some pdu data as bytes; encode as hex for
        # JSON-safety. Length-only summary keeps payload small for big
        # arrays; the full bytes live in `payload['_pduData_hex']`.
        return v.hex()
    if isinstance(v, list):
        return [_coerce_payload_value(x) for x in v]
    return v


def _print_schema_once(tag: str, columns: dict[str, Any]) -> None:
    if tag in _SCHEMAS_PRINTED:
        return
    _SCHEMAS_PRINTED.add(tag)
    sys.stderr.write(f"[aerial:{tag}] schema discovery:\n")
    for name, info in columns.items():
        sys.stderr.write(f"  {name}: {info}\n")
    sys.stderr.flush()


# ─── parquet readers ─────────────────────────────────────────────────────


class AerialFAPIReader:
    """Read an Aerial FAPI parquet (`fapi.parquet`) row-by-row.

    Each row becomes one `TelemetryEvent(modality="kpm_5g")`. PHY-level
    columns (RNTI, MCS, RB allocation, CQI, ...) live verbatim in the
    payload after JSON-safe coercion.
    """

    modality: Modality = "kpm_5g"

    def __init__(self, parquet_path: str | Path, source_id: str = "aerial_fapi"):
        self.path = Path(parquet_path)
        if not self.path.exists():
            raise FileNotFoundError(f"FAPI parquet not found: {self.path}")
        self.source_id = source_id
        self._table = pq.read_table(self.path)
        self._columns: list[str] = self._table.schema.names
        cards = {
            n: f"dtype={self._table.schema.field(n).type}, "
               f"n={self._table.column(n).length()}, "
               f"nulls={self._table.column(n).null_count}"
            for n in self._columns
        }
        _print_schema_once(f"FAPI[{self.path.name}]", cards)
        # Sanity: the readers must FAIL LOUD when the data they expect
        # is absent rather than emit zero-valued events.
        if not self._columns:
            raise ConnectorIOError(
                f"FAPI parquet {self.path} has no columns — schema unknown"
            )

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def __iter__(self) -> Iterator[TelemetryEvent]:
        rows = self._table.to_pylist()
        for idx, row in enumerate(rows):
            ts_ns = _row_timestamp_ns(row)
            payload = {k: _coerce_payload_value(v) for k, v in row.items()}
            yield TelemetryEvent(
                event_id=f"fapi-{self.path.stem}-{idx}",
                modality=self.modality,
                source_id=self.source_id,
                ts_utc=_utc_from_ns(ts_ns) if ts_ns is not None else datetime.now(tz=timezone.utc),
                monotonic_ns=ts_ns,
                sequence=idx,
                payload=payload,
                tags={"source_file": self.path.name, "format": "fapi_parquet"},
            )


class AerialFrontHaulReader:
    """Read an Aerial front-haul parquet (`fh.parquet`).

    Modality is auto-detected: if any column name hints at IQ data
    (`fhData`, `iq`, ...) we emit `spectrum_iq` events; otherwise we
    fall back to `kpm_5g`.
    """

    def __init__(self, parquet_path: str | Path, source_id: str = "aerial_fh"):
        self.path = Path(parquet_path)
        if not self.path.exists():
            raise FileNotFoundError(f"front-haul parquet not found: {self.path}")
        self.source_id = source_id
        self._table = pq.read_table(self.path)
        self._columns: list[str] = self._table.schema.names
        if not self._columns:
            raise ConnectorIOError(
                f"front-haul parquet {self.path} has no columns — schema unknown"
            )
        # Modality detection.
        if any(c in _IQ_HINT_COLUMNS for c in self._columns):
            self.modality: Modality = "spectrum_iq"
        else:
            self.modality = "kpm_5g"
        cards = {
            n: f"dtype={self._table.schema.field(n).type}, "
               f"n={self._table.column(n).length()}, "
               f"nulls={self._table.column(n).null_count}"
            for n in self._columns
        }
        _print_schema_once(
            f"FH[{self.path.name}|modality={self.modality}]", cards
        )

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def __iter__(self) -> Iterator[TelemetryEvent]:
        rows = self._table.to_pylist()
        for idx, row in enumerate(rows):
            ts_ns = _row_timestamp_ns(row)
            payload = {k: _coerce_payload_value(v) for k, v in row.items()}
            yield TelemetryEvent(
                event_id=f"fh-{self.path.stem}-{idx}",
                modality=self.modality,
                source_id=self.source_id,
                ts_utc=_utc_from_ns(ts_ns) if ts_ns is not None else datetime.now(tz=timezone.utc),
                monotonic_ns=ts_ns,
                sequence=idx,
                payload=payload,
                tags={"source_file": self.path.name, "format": "fh_parquet"},
            )


# ─── HDF5 cuMAC test-vector reader ───────────────────────────────────────


class AerialH5TestVectorReader:
    """Walk a directory of cuMAC `.h5` test vectors.

    For every top-level dataset in every file we emit a
    `TelemetryEvent(modality="kpm_5g")` whose payload carries the
    dataset's `shape`, `dtype`, and a small numeric preview (first 16
    elements, JSON-safe). Bulk arrays are intentionally NOT serialised
    on the bus — downstream consumers can re-open the file by path.
    """

    modality: Modality = "kpm_5g"

    def __init__(
        self,
        directory: str | Path,
        source_id: str = "aerial_h5",
        glob: str = "*.h5",
        max_files: int | None = None,
        preview_len: int = 16,
    ):
        if h5py is None:
            raise ImportError(
                "h5py is required for AerialH5TestVectorReader; "
                f"original error: {_H5_IMPORT_ERROR}"
            )
        self.dir = Path(directory)
        if not self.dir.exists():
            raise FileNotFoundError(f"H5 test-vector directory not found: {self.dir}")
        self.source_id = source_id
        self._files: list[Path] = sorted(self.dir.rglob(glob))
        if max_files is not None:
            self._files = self._files[:max_files]
        if not self._files:
            raise ConnectorIOError(
                f"no H5 files matching {glob} under {self.dir}"
            )
        self.preview_len = preview_len

    @property
    def files(self) -> list[Path]:
        return list(self._files)

    def __iter__(self) -> Iterator[TelemetryEvent]:
        seq = 0
        ts0 = datetime.now(tz=timezone.utc)
        for f in self._files:
            with h5py.File(f, "r") as fh:
                for ds_name in fh.keys():
                    item = fh[ds_name]
                    if not isinstance(item, h5py.Dataset):
                        # Skip groups silently — top-level datasets only.
                        continue
                    arr = item[()]
                    flat = np.asarray(arr).reshape(-1)
                    preview = flat[: self.preview_len].tolist()
                    payload = {
                        "dataset": ds_name,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "n_elements": int(flat.size),
                        "preview": _coerce_payload_value(preview),
                        "file": str(f),
                    }
                    yield TelemetryEvent(
                        event_id=f"h5-{f.stem}-{ds_name}-{seq}",
                        modality=self.modality,
                        source_id=ds_name,
                        ts_utc=ts0,
                        sequence=seq,
                        payload=payload,
                        tags={
                            "source_file": f.name,
                            "format": "cumac_h5",
                            "tv_dir": f.parent.name,
                        },
                    )
                    seq += 1


# ─── IterableDataset glue ────────────────────────────────────────────────


# Type alias: a "reader spec" is (factory_callable, args_tuple, kwargs_dict).
ReaderSpec = tuple[type, tuple, dict]


class AerialDataset(IterableDataset):
    """`torch.utils.data.IterableDataset` over a list of Aerial readers.

    Each item in `readers` is a `(reader_cls, args, kwargs)` triple; the
    dataset re-instantiates the reader on each iteration so multi-worker
    DataLoader doesn't share file handles across processes.

    `feature_frame_window(window_s)` collects events into UTC time
    buckets and yields `FeatureFrame`s aligned with the canonical schema
    in `horizon_ric.io.schemas`.
    """

    def __init__(self, readers: Iterable[ReaderSpec]):
        super().__init__()
        self.readers: list[ReaderSpec] = list(readers)
        if not self.readers:
            raise ValueError("AerialDataset requires at least one reader spec")

    def _iter_events(self) -> Iterator[TelemetryEvent]:
        for cls, args, kwargs in self.readers:
            reader = cls(*args, **kwargs)
            yield from reader

    def __iter__(self) -> Iterator[TelemetryEvent]:
        return self._iter_events()

    def feature_frame_window(self, window_s: float) -> Iterator[FeatureFrame]:
        """Bucket events into fixed-width UTC windows.

        Events are streamed in source order; the first event opens a
        window of `window_s` seconds, and subsequent events that fall
        inside that window are appended. Anything past the boundary
        flushes the current frame and opens the next.
        """

        if window_s <= 0:
            raise ValueError("window_s must be positive")
        bucket: list[TelemetryEvent] = []
        bucket_start: datetime | None = None
        bucket_end: datetime | None = None

        for ev in self._iter_events():
            if bucket_start is None:
                bucket_start = ev.ts_utc
                bucket_end = bucket_start + timedelta(seconds=window_s)
            assert bucket_end is not None
            if ev.ts_utc >= bucket_end:
                yield self._make_frame(bucket, bucket_start, bucket_end)
                # Slide window forward to whichever boundary contains ev.
                while ev.ts_utc >= bucket_end:
                    bucket_start = bucket_end
                    bucket_end = bucket_start + timedelta(seconds=window_s)
                bucket = []
            bucket.append(ev)

        if bucket and bucket_start is not None and bucket_end is not None:
            yield self._make_frame(bucket, bucket_start, bucket_end)

    @staticmethod
    def _make_frame(
        events: list[TelemetryEvent], t0: datetime, t1: datetime
    ) -> FeatureFrame:
        modalities = sorted({e.modality for e in events})
        return FeatureFrame(
            frame_id=f"aerial-{uuid.uuid4().hex[:12]}",
            window_start_utc=t0,
            window_end_utc=t1,
            modalities_present=modalities,
            n_events=len(events),
            payload={
                "event_ids": [e.event_id for e in events],
                "source_ids": sorted({e.source_id for e in events}),
                "tags_seen": sorted({k for e in events for k in e.tags}),
            },
        )


# ─── Source-registry adapters ────────────────────────────────────────────


class _PathConfig(ConnectorConfig):
    path: str
    source_id_override: str | None = None


class _DirConfig(ConnectorConfig):
    directory: str
    glob: str = "*.h5"
    max_files: int | None = None


class _AerialReaderSource(Source):
    """Adapter base — wraps a synchronous reader as an async `Source`."""

    Config = _PathConfig
    cfg: _PathConfig
    _reader_cls: type
    _expects: str = "path"  # 'path' or 'directory'

    async def connect(self) -> None:
        if self._expects == "path":
            kwargs = {}
            if self.cfg.source_id_override:
                kwargs["source_id"] = self.cfg.source_id_override
            self._reader = self._reader_cls(self.cfg.path, **kwargs)
        else:  # directory
            self._reader = self._reader_cls(
                self.cfg.directory,
                glob=self.cfg.glob,
                max_files=self.cfg.max_files,
            )
        self._state = ConnectorState.STARTED

    async def close(self) -> None:
        self._state = ConnectorState.STOPPED

    async def stream(self):
        if self._state != ConnectorState.STARTED:
            raise ConnectorIOError(f"{type(self).__name__} not connected")
        for ev in self._reader:
            yield ev


class AerialFAPISource(_AerialReaderSource):
    Config = _PathConfig
    _reader_cls = AerialFAPIReader
    _expects = "path"


class AerialFrontHaulSource(_AerialReaderSource):
    Config = _PathConfig
    _reader_cls = AerialFrontHaulReader
    _expects = "path"


class AerialH5Source(_AerialReaderSource):
    Config = _DirConfig
    cfg: _DirConfig  # type: ignore[assignment]
    _reader_cls = AerialH5TestVectorReader
    _expects = "directory"


__all__ = [
    "AerialDataset",
    "AerialFAPIReader",
    "AerialFAPISource",
    "AerialFrontHaulReader",
    "AerialFrontHaulSource",
    "AerialH5Source",
    "AerialH5TestVectorReader",
    "ReaderSpec",
]
