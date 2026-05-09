"""CelesTrak TLE catalog reader + auto-registered Source connectors.

CelesTrak (https://celestrak.org) mirrors the public element sets that
space-track.org publishes for the active satellite catalog. We download
the per-group ``*.tle`` files via ``scripts/download_tles.py`` into
``data/orbital/celestrak/`` and read them here.

Three layers of API:

* :class:`CelesTrakReader` — iterate raw ``(line1, line2)`` TLE pairs,
  parse them into ``sgp4.api.Satrec`` instances, or propagate them to a
  given UTC instant via :func:`horizon_ric.planner.physics.orbital.sgp4_state`.
* The module side-effect on import auto-registers a no-op ``Source``
  subclass under ``"celestrak_{group}"`` for every group present in the
  on-disk manifest, so existing IO-registry consumers can discover the
  catalogs by name.
* If the on-disk manifest is missing, instantiating a reader for a group
  raises :class:`FileNotFoundError` with a clear pointer to the download
  script — we never silently fabricate data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from sgp4.api import WGS84, Satrec

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorState,
    Source,
)
from horizon_ric.io.registry import register_source
from horizon_ric.io.schemas import TelemetryEvent
from horizon_ric.planner.physics.orbital import OrbitalState, sgp4_state

logger = logging.getLogger(__name__)

# Default location — matches `scripts/download_tles.py`. Callers can
# override via the `data_dir` argument.
DEFAULT_DATA_DIR = Path(
    "/home/danielfoojunwei/Preceptualv1/data/orbital/celestrak"
)


@dataclass(frozen=True)
class TLEEntry:
    """One CelesTrak TLE record: human-readable name + the two TLE lines."""

    name: str
    line1: str
    line2: str


def _missing_file_msg(path: Path) -> str:
    return (
        f"CelesTrak TLE file not found: {path}\n"
        f"Run scripts/download_tles.py to populate "
        f"{DEFAULT_DATA_DIR} from celestrak.org "
        f"(this writes one .tle per group plus manifest.json)."
    )


def _parse_tle_blob(text: str) -> list[TLEEntry]:
    """Parse a CelesTrak 3LE blob into TLEEntry records.

    The CelesTrak TLE format is a strict 3-line repeating block:
        Line 0: satellite name (≤24 chars, often padded with spaces)
        Line 1: NORAD line 1 ('1 NNNNNU ...')
        Line 2: NORAD line 2 ('2 NNNNN  ...')
    Blank lines (defensive against trailing newlines) are dropped before
    grouping, so file metadata never shifts the 3-line frame.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(
            f"TLE blob has {len(lines)} non-blank lines, not a multiple of 3 — "
            "file may be truncated or malformed."
        )
    entries: list[TLEEntry] = []
    for i in range(0, len(lines), 3):
        name = lines[i].strip()
        l1 = lines[i + 1]
        l2 = lines[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            raise ValueError(
                f"TLE record at index {i // 3} not in 3LE form: "
                f"name={name!r} l1={l1[:20]!r} l2={l2[:20]!r}"
            )
        entries.append(TLEEntry(name=name, line1=l1, line2=l2))
    return entries


class CelesTrakReader:
    """Read a downloaded CelesTrak TLE file by group name."""

    def __init__(
        self,
        group_name: str,
        *,
        data_dir: str | Path | None = None,
    ) -> None:
        self.group_name = group_name
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self.path = self.data_dir / f"{group_name}.tle"
        if not self.path.exists():
            raise FileNotFoundError(_missing_file_msg(self.path))
        self._entries: list[TLEEntry] | None = None

    # ── manifest helpers ────────────────────────────────────────────────

    @classmethod
    def manifest_path(cls, data_dir: str | Path | None = None) -> Path:
        d = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        return d / "manifest.json"

    @classmethod
    def load_manifest(
        cls, data_dir: str | Path | None = None
    ) -> dict:
        path = cls.manifest_path(data_dir)
        if not path.exists():
            raise FileNotFoundError(_missing_file_msg(path))
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def available_groups(cls, data_dir: str | Path | None = None) -> list[str]:
        """List all groups present in the manifest (i.e., successfully downloaded)."""
        try:
            manifest = cls.load_manifest(data_dir)
        except FileNotFoundError:
            return []
        return [g["group"] for g in manifest.get("groups", [])]

    # ── core parsing ────────────────────────────────────────────────────

    def _load(self) -> list[TLEEntry]:
        if self._entries is None:
            self._entries = _parse_tle_blob(
                self.path.read_text(encoding="utf-8", errors="replace")
            )
        return self._entries

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Yield raw ``(line1, line2)`` TLE pairs, in catalog order."""
        for e in self._load():
            yield (e.line1, e.line2)

    def __len__(self) -> int:
        return len(self._load())

    def entries(self) -> list[TLEEntry]:
        """Return full TLEEntry records (name + line1 + line2)."""
        return list(self._load())

    def parse_satrecs(self) -> list[Satrec]:
        """Parse every TLE in the file into an ``sgp4.api.Satrec``."""
        out: list[Satrec] = []
        for e in self._load():
            try:
                out.append(Satrec.twoline2rv(e.line1, e.line2, WGS84))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "skipping malformed TLE %s in %s: %s",
                    e.name, self.path.name, exc,
                )
        return out

    def to_orbital_states(self, t_utc: datetime) -> list[OrbitalState]:
        """Propagate every TLE to ``t_utc`` via the project's SGP4 wrapper.

        ``t_utc`` must be timezone-aware (the wrapper enforces this). Records
        whose SGP4 propagation fails (e.g. decayed satellites) are skipped
        with a warning rather than aborting the batch.
        """
        states: list[OrbitalState] = []
        for e in self._load():
            try:
                states.append(sgp4_state(e.line1, e.line2, t_utc))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SGP4 failed for %s in %s: %s",
                    e.name, self.path.name, exc,
                )
        return states


# ─── Source connector adapter ────────────────────────────────────────────


class _CelesTrakSourceConfig(ConnectorConfig):
    """Configuration for a CelesTrak Source.

    The catalog group is fixed at registration time (``celestrak_{group}``);
    this config only carries override hooks: an alternative ``data_dir`` and
    the propagation epoch ``t_utc_iso`` (defaults to the connector's
    ``connect()`` wall-clock if absent).
    """

    data_dir: str | None = None
    t_utc_iso: str | None = None


class CelesTrakSource(Source):
    """Async-iterable Source over a CelesTrak group's propagated states.

    The Source emits one :class:`TelemetryEvent` per satellite at the
    chosen UTC instant, with payload containing the propagated ECI
    position/velocity. This is a one-shot snapshot — it isn't a live
    stream. Callers needing a re-propagation cycle should re-instantiate.
    """

    Config = _CelesTrakSourceConfig
    cfg: _CelesTrakSourceConfig
    GROUP: str = ""  # set per subclass at registration time

    async def connect(self) -> None:
        # Trigger file existence check eagerly so misconfigurations
        # surface at connect() time, not on first read.
        self._reader = CelesTrakReader(self.GROUP, data_dir=self.cfg.data_dir)
        self._state = ConnectorState.STARTED

    async def close(self) -> None:
        self._state = ConnectorState.STOPPED

    async def stream(self) -> AsyncIterator[TelemetryEvent]:
        if self._state != ConnectorState.STARTED:
            raise RuntimeError("CelesTrakSource not connected")
        if self.cfg.t_utc_iso is not None:
            t_utc = datetime.fromisoformat(self.cfg.t_utc_iso)
        else:
            from datetime import timezone
            t_utc = datetime.now(timezone.utc)
        for idx, (entry, state) in enumerate(
            zip(self._reader.entries(), self._reader.to_orbital_states(t_utc))
        ):
            # TelemetryEvent uses 'tle' as the canonical modality for
            # ephemeris updates; orbital state goes into the payload.
            yield TelemetryEvent(
                event_id=f"{self.cfg.name}:{idx}",
                modality="tle",
                source_id=self.cfg.name,
                ts_utc=state.epoch_utc,
                sequence=idx,
                payload={
                    "satellite_name": entry.name,
                    "tle_line1": entry.line1,
                    "tle_line2": entry.line2,
                    "r_eci_km": list(state.r_eci_km),
                    "v_eci_km_s": list(state.v_eci_km_s),
                    "altitude_km": state.altitude_km(),
                },
            )


def _make_source_class(group: str) -> type[CelesTrakSource]:
    cls = type(
        f"CelesTrakSource_{group.replace('-', '_')}",
        (CelesTrakSource,),
        {"GROUP": group, "__doc__": f"CelesTrak Source for group {group!r}."},
    )
    return cls


def _auto_register_sources(data_dir: str | Path | None = None) -> list[str]:
    """Register one Source subclass per group present on disk.

    Idempotent — re-registration of the same class is a no-op. Returns the
    list of registered names. Silent (logged) if the manifest is missing,
    so importing this module never breaks just because TLEs aren't on disk
    yet (callers get a clear FileNotFoundError later when they try to use
    a Reader directly).
    """
    registered: list[str] = []
    try:
        groups = CelesTrakReader.available_groups(data_dir)
    except Exception as exc:  # noqa: BLE001
        logger.debug("celestrak: could not enumerate groups: %s", exc)
        return registered
    for group in groups:
        name = f"celestrak_{group}"
        cls = _make_source_class(group)
        try:
            register_source(name, cls)
            registered.append(name)
        except ValueError:
            # already registered with a different class — skip honestly
            logger.debug("celestrak: %s already registered, skipping", name)
    return registered


# Side effect on import: register Source connectors for every group whose
# .tle file is on disk. Safe to call repeatedly (idempotent for same class).
_REGISTERED_AT_IMPORT: list[str] = _auto_register_sources()


__all__ = [
    "CelesTrakReader",
    "CelesTrakSource",
    "DEFAULT_DATA_DIR",
    "TLEEntry",
]
