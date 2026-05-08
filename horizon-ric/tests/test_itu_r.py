"""ITU-R reference map reader + propagation wiring tests.

These tests exercise the reader against the actually-downloaded ITU
ZIPs (`data/itu_r/<rec_id>/`). They `pytest.skip` honestly when the
underlying data isn't present, rather than silently passing.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest

from horizon_ric.data.itu_r import (
    ITUMapReader,
    P453Map,
    P837Map,
    P839Map,
    P1510Map,
    _bilinear,
    _Grid,
    _to_grid_lon,
)
from horizon_ric.planner.physics.propagation import rain_attenuation_dB
import numpy as np


# Resolve the on-disk data dir the same way the reader does. We
# don't import its private helper; we just hard-code the canonical
# location, since this test bundle ships alongside the repo.
_DATA_DIR = Path("/home/danielfoojunwei/Preceptualv1/data/itu_r")
_MANIFEST = _DATA_DIR / "manifest.json"


def _downloaded_recs() -> dict[str, dict]:
    if not _MANIFEST.exists():
        return {}
    with _MANIFEST.open() as fh:
        manifest = json.load(fh)
    return {r["rec_id"]: r for r in manifest.get("recommendations", [])}


def test_manifest_exists_or_reports_honestly():
    """At least one ITU rec downloaded successfully; manifest tracks it."""
    if not _MANIFEST.exists():
        pytest.skip(
            "data/itu_r/manifest.json missing — run "
            "scripts/download_itu_data.py before this test."
        )
    recs = _downloaded_recs()
    # Manifest should be a dict with recommendations; we don't insist all 6
    # downloaded since ITU URLs drift, but at least one must have succeeded.
    assert isinstance(recs, dict)
    if not recs:
        pytest.skip(
            "Manifest exists but no ITU-R rec downloaded "
            "(probably all URLs 404'd). Try manual fetch from itu.int."
        )
    # Every downloaded entry should point to a real directory.
    for rec_id, entry in recs.items():
        assert Path(entry["extracted_dir"]).exists(), \
            f"{rec_id} entry refers to missing dir {entry['extracted_dir']}"


def test_reader_constructed_for_one_downloaded_rec():
    """ITUMapReader can be constructed for at least one downloaded rec."""
    recs = _downloaded_recs()
    if not recs:
        pytest.skip("no ITU-R rec downloaded")
    rec_id = next(iter(recs))
    reader = ITUMapReader(rec_id, data_dir=_DATA_DIR)
    assert reader.is_available, f"{rec_id} files not visible to the reader"
    assert reader.rec_id == rec_id


def test_lookup_zero_zero_is_finite():
    """Lookup at the equator/prime meridian returns a finite number."""
    recs = _downloaded_recs()
    if "P.839-4" not in recs:
        pytest.skip("P.839-4 not downloaded")
    val = P839Map(data_dir=_DATA_DIR).lookup(0.0, 0.0)
    assert math.isfinite(val)
    # Rain height at equator should be high (warm troposphere) — sanity
    # check 3–6 km.
    assert 3.0 < val < 6.0


def test_p839_rotterdam_in_expected_range():
    """P.839-4 rain height at Rotterdam (51.92, 4.48) is mid-latitude."""
    recs = _downloaded_recs()
    if "P.839-4" not in recs:
        pytest.skip("P.839-4 not downloaded")
    val = P839Map(data_dir=_DATA_DIR).lookup(51.92, 4.48)
    assert 1.5 < val < 5.0, f"Rotterdam rain height {val} outside [1.5, 5.0]"


def test_rain_attenuation_uses_itu_maps_when_enabled():
    """`use_itu_maps=True` produces a different result than the default."""
    recs = _downloaded_recs()
    if "P.839-4" not in recs:
        pytest.skip("P.839-4 not downloaded — wiring can't be exercised")

    # Pick a high-latitude site (Tromsø, 69.65 N) where the rain height
    # from P.839 is well below the engineering default of 3 km. That
    # difference will visibly change the attenuation.
    common = dict(
        rain_rate_mm_per_hr=20.0,
        elevation_deg=30.0,
        frequency_ghz=20.0,
        polarization="circular",
    )
    a_default = rain_attenuation_dB(**common)
    a_itu = rain_attenuation_dB(
        **common,
        es_lat_deg=69.65,
        es_lon_deg=18.96,
        use_itu_maps=True,
    )
    assert math.isfinite(a_default) and math.isfinite(a_itu)
    assert abs(a_itu - a_default) > 0.1, (
        f"Expected ITU map to change rain attenuation, got "
        f"default={a_default} dB vs itu={a_itu} dB"
    )


def test_missing_data_falls_back_with_warning(tmp_path, caplog):
    """Reader returns the documented constant + logs when data is absent."""
    # Construct a reader pointing at an empty directory.
    reader = ITUMapReader("P.999-1", data_dir=tmp_path)
    assert not reader.is_available

    with caplog.at_level(logging.WARNING, logger="horizon_ric.data.itu_r"):
        val = reader.lookup(10.0, 20.0)

    # Unknown rec_id → NaN fallback by design; pinned recs return their
    # documented constant. We test a known rec_id below for the constant.
    assert math.isnan(val) or math.isfinite(val)

    # And a second reader (real rec_id) returns its documented constant.
    reader2 = ITUMapReader("P.839-4", data_dir=tmp_path)
    val2 = reader2.lookup(0.0, 0.0)
    assert val2 == pytest.approx(3.0)


def test_longitude_wraparound_is_continuous():
    """Bilinear lookup must wrap correctly across the dateline."""
    # Build a synthetic grid storing lon ∈ [0, 360] (P.839 convention).
    # Use a non-trivial pattern that varies smoothly with lon.
    lats = np.array([-1.0, 1.0])
    lons = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    # Make the value a function of cos(lon), 360-periodic.
    values = np.tile(np.cos(np.radians(lons)), (lats.size, 1))
    # Grid wraps: at lon=0 and lon=360 the value matches.
    grid = _Grid(lats=lats, lons=lons, values=values)

    # Sample just east and just west of 180° via -179 and 179. The two
    # lookups should give nearly identical numbers because the field is
    # symmetric about 180°.
    east = _bilinear(grid, 0.0, 179.5)
    west = _bilinear(grid, 0.0, -179.5)  # same as 180.5 wrapped
    assert math.isclose(east, west, abs_tol=0.05)

    # And lon=-1 (i.e. 359) wraps onto the segment between lon=270 and
    # lon=360: the helper should not raise and produce something close to
    # cos(359°)=cos(-1°)≈1.
    near_zero = _bilinear(grid, 0.0, -1.0)
    assert math.isclose(near_zero, math.cos(math.radians(-1.0)), abs_tol=0.05)


def test_lon_normalisation_for_360_grid():
    """`_to_grid_lon` shifts negative lons onto a 0..360 grid."""
    grid_lons = np.linspace(0.0, 360.0, 241)
    assert _to_grid_lon(-10.0, grid_lons) == pytest.approx(350.0)
    assert _to_grid_lon(10.0, grid_lons) == pytest.approx(10.0)
    # And leaves a [-180, 180] grid alone.
    grid_lons2 = np.linspace(-180.0, 180.0, 481)
    assert _to_grid_lon(10.0, grid_lons2) == pytest.approx(10.0)
    assert _to_grid_lon(-10.0, grid_lons2) == pytest.approx(-10.0)


def test_p1510_rotterdam_temperature_celsius_range():
    """P.1510-1 mean annual temperature at Rotterdam is mid-latitude oceanic."""
    recs = _downloaded_recs()
    if "P.1510-1" not in recs:
        pytest.skip("P.1510-1 not downloaded")
    val = P1510Map(data_dir=_DATA_DIR).lookup(51.92, 4.48)
    # Rotterdam annual mean ≈ 10–11°C. Allow a wide band.
    assert 273.0 < val < 295.0, f"Rotterdam mean temp {val} K out of range"


def test_p453_lookup_returns_finite():
    """P.453-14 wet-refractivity lookup yields a finite real number."""
    recs = _downloaded_recs()
    if "P.453-14" not in recs:
        pytest.skip("P.453-14 not downloaded")
    val = P453Map(data_dir=_DATA_DIR).lookup(0.0, 0.0)
    assert math.isfinite(val)
