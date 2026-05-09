"""Tests for the NVIDIA Aerial / DeepMIMO data adapters.

Exercise the real on-disk datasets at `data/aerial/` and `data/deepmimo/`.
We intentionally hit small file sizes (one of the ~5 MB cuMAC H5s, the
36 KB FAPI parquet, the 3 MB FH parquet) so the suite runs in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from horizon_ric.data import (
    AerialDataset,
    AerialFAPIReader,
    AerialFrontHaulReader,
    AerialH5TestVectorReader,
    DeepMIMOScenarioReader,
)
from horizon_ric.io.registry import registry
from horizon_ric.io.schemas import FeatureFrame, TelemetryEvent

# Repository root is two levels up from this test file
# (horizon-ric/tests/test_data_aerial.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
AERIAL_ROOT = REPO_ROOT / "data" / "aerial"
DEEPMIMO_ROOT = REPO_ROOT / "data" / "deepmimo" / "asu_campus_3p5_dyn"

FAPI_PARQUET = AERIAL_ROOT / "parquet" / "fapi.parquet"
FH_PARQUET = AERIAL_ROOT / "parquet" / "fh.parquet"
H5_DIR = (
    AERIAL_ROOT / "cumac_test_vectors" / "testVectors" / "4T4R_type0Alloc"
)
SMALLEST_H5 = H5_DIR / "TV_cumac_F08-MC-CC-8PC_DL.h5"

# Skip the whole module if data fixtures are missing — keeps the test
# suite green on machines without the NVIDIA bundle.
pytestmark = pytest.mark.skipif(
    not FAPI_PARQUET.exists() or not FH_PARQUET.exists(),
    reason="Aerial data fixtures not present on this host",
)


def test_fapi_first_event_is_valid_telemetry_event():
    reader = AerialFAPIReader(FAPI_PARQUET)
    assert reader.columns, "FAPI parquet must expose at least one column"
    it = iter(reader)
    ev = next(it)
    assert isinstance(ev, TelemetryEvent)
    assert ev.modality == "kpm_5g"
    assert ev.source_id == "aerial_fapi"
    assert ev.ts_utc.tzinfo is not None
    # FAPI columns must be present in payload (no silent zero-fill).
    assert "rnti" in ev.payload or "RNTI" in ev.payload or "SFN" in ev.payload
    assert ev.tags["format"] == "fapi_parquet"


def test_fh_modality_detected_and_schema_survives():
    # Open with a single, isolated reader and ensure modality gets picked
    # from column hints rather than hard-coded.
    reader = AerialFrontHaulReader(FH_PARQUET)
    assert "fhData" in reader.columns, "fh.parquet must contain fhData column"
    assert reader.modality == "spectrum_iq"
    events = list(reader)
    assert events, "FH parquet must yield at least one event"
    first = events[0]
    assert first.modality == "spectrum_iq"
    assert first.tags["source_file"] == FH_PARQUET.name
    # The IQ list must have non-zero length — i.e. real data, not a stub.
    iq = first.payload.get("fhData")
    assert isinstance(iq, list) and len(iq) > 0


@pytest.mark.skipif(
    not SMALLEST_H5.exists(),
    reason="cuMAC H5 test vector not on disk",
)
def test_h5_reader_iterates_real_test_vector():
    # Restrict to the single small file via max_files=1.
    reader = AerialH5TestVectorReader(H5_DIR, max_files=1)
    assert len(reader.files) == 1
    events = list(reader)
    assert events, "H5 reader must emit at least one dataset event"
    sample = events[0]
    assert isinstance(sample, TelemetryEvent)
    assert sample.modality == "kpm_5g"
    assert sample.payload["dataset"]
    assert sample.payload["shape"]
    assert sample.payload["dtype"]
    # source_id is the dataset name per the spec.
    assert sample.source_id == sample.payload["dataset"]


@pytest.mark.skipif(
    not DEEPMIMO_ROOT.exists(),
    reason="DeepMIMO bundle not on disk",
)
def test_deepmimo_lists_at_least_one_scenario():
    reader = DeepMIMOScenarioReader(DEEPMIMO_ROOT, max_scenes=2)
    assert len(reader.scenes) >= 1
    events = list(reader)
    assert events
    ev = events[0]
    assert ev.modality == "kpm_ntn"
    assert "params" in ev.payload
    assert isinstance(ev.payload["params"], dict)
    assert "channel_files" in ev.payload


def test_aerial_dataset_window_bucketing_yields_feature_frame():
    # Mix FAPI + FH: real timestamps from different epochs ensure the
    # bucketing logic sees more than one window in the general case.
    ds = AerialDataset(
        readers=[
            (AerialFAPIReader, (FAPI_PARQUET,), {}),
            (AerialFrontHaulReader, (FH_PARQUET,), {}),
        ]
    )
    # Use a wide window so all events at least collapse into 1+ frames.
    frames = list(ds.feature_frame_window(window_s=3600.0))
    assert frames, "expected at least one FeatureFrame from real data"
    f = frames[0]
    assert isinstance(f, FeatureFrame)
    assert f.n_events > 0
    assert f.modalities_present  # non-empty
    assert f.window_end_utc > f.window_start_utc


def test_registry_returns_aerial_and_deepmimo_connectors():
    fapi = registry.get_source(
        "aerial_fapi", {"name": "fapi-test", "path": str(FAPI_PARQUET)}
    )
    assert fapi.name == "fapi-test"

    if DEEPMIMO_ROOT.exists():
        dmm = registry.get_source(
            "deepmimo",
            {"name": "deepmimo-test", "scenarios_dir": str(DEEPMIMO_ROOT)},
        )
        assert dmm.name == "deepmimo-test"
    else:  # pragma: no cover
        pytest.skip("DeepMIMO bundle not on disk")
