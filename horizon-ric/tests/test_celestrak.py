"""Tests for the CelesTrak TLE reader + auto-registered Source connectors.

These tests assume ``scripts/download_tles.py`` has been run successfully
and at least the GEO + a LEO group are present on disk. Tests that need
specific groups skip rather than fail when the group is absent (so a
partial download still passes the suite).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sgp4.api import Satrec

from horizon_ric.data.celestrak import (
    DEFAULT_DATA_DIR,
    CelesTrakReader,
    CelesTrakSource,
)
from horizon_ric.io.registry import list_connectors

# Pick the first available LEO-ish group for the basic reader tests.
# Order matters: prefer larger ones for stronger signal, fall back as needed.
_PREFERRED_GROUPS = [
    "iridium-next", "globalstar", "oneweb", "starlink",
    "planet", "spire", "orbcomm", "amateur",
]


def _first_available_group() -> str | None:
    available = set(CelesTrakReader.available_groups())
    for g in _PREFERRED_GROUPS:
        if g in available:
            return g
    # Last resort: any group at all
    return next(iter(available), None)


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not CelesTrakReader.manifest_path().exists():
        pytest.skip(
            "manifest.json missing — run scripts/download_tles.py first"
        )
    return CelesTrakReader.load_manifest()


@pytest.fixture(scope="module")
def some_group() -> str:
    g = _first_available_group()
    if g is None:
        pytest.skip("no CelesTrak groups downloaded")
    return g


# ─── 1. Reader opens existing TLE file ───────────────────────────────────


def test_reader_opens_existing_file(some_group: str) -> None:
    reader = CelesTrakReader(some_group)
    assert reader.path.exists()
    assert reader.path.suffix == ".tle"
    assert len(reader) > 0, "expected at least one TLE in the file"


# ─── 2. parse_satrecs() returns valid Satrec instances ───────────────────


def test_parse_satrecs_returns_valid_satrecs(some_group: str) -> None:
    reader = CelesTrakReader(some_group)
    satrecs = reader.parse_satrecs()
    assert len(satrecs) >= 1
    for sat in satrecs[:5]:
        assert isinstance(sat, Satrec)
        # Satrec exposes a NORAD catalog number — sanity-check it.
        assert sat.satnum > 0


# ─── 3. to_orbital_states returns OrbitalState with non-zero r_eci_km ───


def test_to_orbital_states_propagates(some_group: str) -> None:
    reader = CelesTrakReader(some_group)
    t = datetime.now(timezone.utc)
    states = reader.to_orbital_states(t)
    assert len(states) >= 1
    # Position must be non-zero (a real orbit, not the origin).
    for s in states[:5]:
        x, y, z = s.r_eci_km
        radius = (x * x + y * y + z * z) ** 0.5
        # Earth radius is ~6371 km, so any real satellite is well above.
        assert radius > 6000.0
        assert s.epoch_utc.tzinfo is not None


# ─── 4. Manifest is valid JSON with ≥10 groups ───────────────────────────


def test_manifest_valid_with_enough_groups(manifest: dict) -> None:
    # round-trip JSON to confirm validity
    reserialised = json.dumps(manifest)
    assert json.loads(reserialised) == manifest

    assert "groups" in manifest
    assert isinstance(manifest["groups"], list)
    assert len(manifest["groups"]) >= 10, (
        f"only {len(manifest['groups'])} groups in manifest; "
        "expected at least 10 successful downloads."
    )

    # Every entry has the documented fields
    for g in manifest["groups"]:
        for key in (
            "group", "url", "n_satellites", "sha256",
            "downloaded_at_utc", "file_bytes",
        ):
            assert key in g, f"manifest entry missing {key}: {g}"
        assert g["n_satellites"] > 0
        assert len(g["sha256"]) == 64  # hex digest


# ─── 5. Bad group name raises FileNotFoundError ──────────────────────────


def test_bad_group_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        CelesTrakReader("definitely-not-a-real-group-xyz")
    msg = str(excinfo.value)
    # Error must point users to the download script.
    assert "download_tles.py" in msg


# ─── 6. GEO altitudes are ~35,786 km ─────────────────────────────────────


def test_geo_altitude_correctness() -> None:
    if "geo" not in CelesTrakReader.available_groups():
        pytest.skip("geo group not downloaded")
    reader = CelesTrakReader("geo")
    t = datetime.now(timezone.utc)
    states = reader.to_orbital_states(t)
    assert len(states) > 0
    altitudes = [s.altitude_km() for s in states]
    # GSO is ~35,786 km. Operational geostationary spacecraft drift in
    # inclined / eccentric orbits but stay within a few hundred km of the
    # nominal radius. The MEDIAN should be very close.
    altitudes.sort()
    median = altitudes[len(altitudes) // 2]
    assert 35_000.0 < median < 36_500.0, (
        f"GEO median altitude {median:.1f} km is outside the expected "
        f"35,786 km neighbourhood — TLE data may be corrupted."
    )

    # And the bulk (95th percentile band) should hug GSO too. Allow a
    # generous ±2,000 km to cover GTOs and supersynchronous drift orbits
    # that CelesTrak labels as GEO catalog members.
    p05 = altitudes[int(0.05 * len(altitudes))]
    p95 = altitudes[int(0.95 * len(altitudes))]
    assert 30_000.0 < p05, (
        f"5th-percentile GEO altitude {p05:.1f} km is too low"
    )
    assert p95 < 45_000.0, (
        f"95th-percentile GEO altitude {p95:.1f} km is too high"
    )


# ─── 7. Auto-registered Source connectors ────────────────────────────────


def test_celestrak_sources_auto_registered(manifest: dict) -> None:
    """Importing horizon_ric.data.celestrak registers per-group Sources."""
    sources = set(list_connectors()["sources"])
    for g in manifest["groups"]:
        expected = f"celestrak_{g['group']}"
        assert expected in sources, (
            f"expected {expected} to be auto-registered; "
            f"registered sources are {sorted(sources)}"
        )


# ─── 8. Reader is iterable as (line1, line2) pairs ───────────────────────


def test_reader_iterates_line_pairs(some_group: str) -> None:
    reader = CelesTrakReader(some_group)
    n = 0
    for pair in reader:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        l1, l2 = pair
        assert l1.startswith("1 ")
        assert l2.startswith("2 ")
        n += 1
        if n >= 3:
            break
    assert n >= 1
