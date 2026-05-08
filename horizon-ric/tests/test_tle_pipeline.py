"""Tests for ``horizon_ric.data.tle_pipeline``.

Covers:
    * TLE-file loading and Satrec construction.
    * ``at(t_utc)`` snapshot semantics (NGSOEmitter list shape + content).
    * ``visible_from`` does in fact reduce the count vs the full snapshot.
    * ``epfd_time_cdf_real`` returns a valid ``EPFDTimeCDF`` against a
      tiny synthetic stub TLE.
    * Bad path raises a clear ``FileNotFoundError``.
    * Real CelesTrak Starlink TLE smoke test — auto-skipped if the
      catalogue isn't on disk yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from horizon_ric.data.tle_pipeline import TLEConstellationPropagator
from horizon_ric.planner.physics.epfd import EPFDTimeCDF, NGSOEmitter

# ─── helpers ─────────────────────────────────────────────────────────────


# Two well-known ISS TLEs (epoch 2024-12-31). Stable enough for unit
# tests since SGP4 will accept them — even if propagation is years away
# from the epoch the propagator still returns a numeric state (it's the
# user's responsibility not to use ancient TLEs in production).
STUB_TLE = """\
ISS (ZARYA)
1 25544U 98067A   24366.50000000  .00016717  00000+0  10270-3 0  9001
2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.50190100456789
TIANGONG
1 48274U 21035A   24366.50000000  .00010000  00000+0  20000-3 0  9000
2 48274  41.4750 100.0000 0005000  90.0000 270.0000 15.60000000200000
"""


@pytest.fixture
def stub_tle_file(tmp_path: Path) -> Path:
    p = tmp_path / "stub.tle"
    p.write_text(STUB_TLE, encoding="utf-8")
    return p


T_UTC = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
ROTTERDAM = (51.9244, 4.4777)


# ─── tests ───────────────────────────────────────────────────────────────


def test_loading_produces_at_least_one_satrec(stub_tle_file: Path) -> None:
    prop = TLEConstellationPropagator(stub_tle_file)
    assert len(prop) >= 1
    assert "ISS (ZARYA)" in prop.names()


def test_at_returns_ngso_emitter_list(stub_tle_file: Path) -> None:
    prop = TLEConstellationPropagator(stub_tle_file)
    emitters = prop.at(T_UTC)
    assert isinstance(emitters, list)
    assert len(emitters) == len(prop)
    for em in emitters:
        assert isinstance(em, NGSOEmitter)
        assert em.name  # non-empty
        # ECEF magnitude should be > Earth radius (sat is in space).
        r2 = (
            em.position_ecef.x ** 2
            + em.position_ecef.y ** 2
            + em.position_ecef.z ** 2
        ) ** 0.5
        assert r2 > 6_000_000.0  # > 6000 km


def test_visible_from_reduces_count(stub_tle_file: Path) -> None:
    """At any one instant from a single ground station, NOT every sat in
    the catalog is above the horizon — visibility filtering is strictly
    a subset operation."""
    prop = TLEConstellationPropagator(stub_tle_file)
    # Sample several instants — at least one should yield a strict subset.
    saw_subset = False
    for offset_s in (0, 600, 1_200, 1_800, 2_400, 3_000):
        t = T_UTC + timedelta(seconds=offset_s)
        full = prop.at(t)
        visible = prop.visible_from(*ROTTERDAM, t, min_elevation_deg=10.0)
        assert len(visible) <= len(full)
        if len(visible) < len(full):
            saw_subset = True
            break
    assert saw_subset, "expected visibility to filter at least one snapshot"


def test_epfd_time_cdf_real_returns_valid_cdf(stub_tle_file: Path) -> None:
    prop = TLEConstellationPropagator(stub_tle_file)
    cdf = prop.epfd_time_cdf_real(
        es_lat_deg=ROTTERDAM[0],
        es_lon_deg=ROTTERDAM[1],
        wanted_gso_lon_deg=10.0,
        es_diameter_m=1.2,
        es_frequency_hz=12e9,
        t_start_utc=T_UTC,
        duration_s=600.0,
        step_s=60.0,
    )
    assert isinstance(cdf, EPFDTimeCDF)
    assert len(cdf.samples_dB) >= 2
    assert cdf.duration_s == 600.0
    assert cdf.step_s == 60.0
    # Percentile should be a real number (not -inf, not +inf).
    p50 = cdf.percentile(50.0)
    assert -400.0 < p50 < 100.0


def test_bad_path_raises_filenotfound(tmp_path: Path) -> None:
    bogus = tmp_path / "does-not-exist.tle"
    with pytest.raises(FileNotFoundError) as ei:
        TLEConstellationPropagator(bogus)
    # Message must reference the path and how to fix it.
    msg = str(ei.value)
    assert "does-not-exist.tle" in msg
    assert "download_tles" in msg


# ─── real CelesTrak smoke test (auto-skip if not downloaded) ────────────


REAL_STARLINK_PATH = Path(
    "/home/danielfoojunwei/Preceptualv1/data/orbital/celestrak/starlink.tle"
)


def test_real_starlink_tle_smoke() -> None:
    """If the CelesTrak Starlink TLEs are on disk, the pipeline must
    parse them and produce a non-empty visibility snapshot. Otherwise
    skip — the parallel download might not have finished yet."""
    if not REAL_STARLINK_PATH.exists():
        pytest.skip(
            f"{REAL_STARLINK_PATH} not present — "
            f"run scripts/download_tles.py first"
        )
    prop = TLEConstellationPropagator(REAL_STARLINK_PATH).slice(50)
    assert len(prop) >= 1
    now = datetime.now(timezone.utc)
    emitters = prop.at(now)
    assert len(emitters) >= 1
    visible = prop.visible_from(*ROTTERDAM, now, min_elevation_deg=10.0)
    assert len(visible) <= len(emitters)
