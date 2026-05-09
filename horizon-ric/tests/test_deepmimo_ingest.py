"""Full-suite DeepMIMO ingest test (Row 22 of GAPS_TO_PILOT.md).

Loads every locally-present scenario via :class:`DeepMIMOScenarioReader`
and asserts shape consistency. Scenarios that are not on disk are
skipped gracefully — no synthetic / mocked data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from horizon_ric.data.deepmimo import (
    DeepMIMOScenarioReader,
    summarize_scenario,
)

DEEPMIMO_ROOT = Path("/home/danielfoojunwei/Preceptualv1/data/deepmimo")
MANIFEST = DEEPMIMO_ROOT / "manifest.json"


def _present_scenarios() -> list[str]:
    if not DEEPMIMO_ROOT.exists():
        return []
    out = []
    for spec in DeepMIMOScenarioReader.list_known_scenarios():
        if (DEEPMIMO_ROOT / spec.scenario_id).is_dir():
            out.append(spec.scenario_id)
    return out


def test_manifest_exists_and_is_valid():
    if not DEEPMIMO_ROOT.exists():
        pytest.skip("DeepMIMO root not present on this host")
    assert MANIFEST.exists(), "data/deepmimo/manifest.json missing"
    data = json.loads(MANIFEST.read_text())
    assert data["schema_version"] >= 2
    assert "scenarios" in data
    # Every known scenario must appear in the manifest (present or not).
    known_ids = {s.scenario_id for s in DeepMIMOScenarioReader.list_known_scenarios()}
    assert set(data["scenarios"].keys()) >= known_ids


@pytest.mark.parametrize("scenario_id", _present_scenarios())
def test_scenario_loads_and_shapes_consistent(scenario_id: str):
    """For every locally-present scenario:
    - reader yields >= 1 TelemetryEvent
    - first event payload carries `params`, `channel_files`, `scene_dir`
    - summarize_scenario reports n_ues > 0 and a non-empty channel
      tensor shape with 2 dims (n_ues, n_paths)
    """
    scenario_dir = DEEPMIMO_ROOT / scenario_id
    reader = DeepMIMOScenarioReader(scenario_dir, max_scenes=1)
    events = list(reader)
    assert len(events) >= 1, f"{scenario_id}: reader yielded no events"
    ev = events[0]
    payload = ev.payload
    assert "params" in payload
    assert "scene_dir" in payload
    assert payload["n_channel_files"] > 0

    summary = summarize_scenario(scenario_dir)
    assert summary["n_ues"] > 0, f"{scenario_id}: no UEs detected"
    assert summary["n_bs"] >= 1
    assert summary["n_antennas"] > 0
    shape = summary["channel_tensor_shape"]
    assert shape is not None and len(shape) == 2
    n_ues, n_paths = shape
    assert n_ues == summary["n_ues"]
    assert n_paths == summary["n_antennas"]
    # Channel pickle SHA must be present for any scenario with a power
    # tensor on disk (every present scenario should have one).
    assert summary["channel_pickle_sha256"] is not None
    assert len(summary["channel_pickle_sha256"]) == 64


def test_iterate_all_scenarios_yields_at_least_one_event():
    """The convenience iterator must yield at least one event per
    present scenario when the on-disk data exists."""
    if not DEEPMIMO_ROOT.exists():
        pytest.skip("DeepMIMO root not present")
    seen_scenarios: set[str] = set()
    for ev in DeepMIMOScenarioReader.iterate_all_scenarios(
        root=DEEPMIMO_ROOT, max_scenes_per_dataset=1
    ):
        assert ev.modality == "kpm_ntn"
        # source_id is "deepmimo:<scenario>"
        sid = ev.source_id.split(":", 1)[-1]
        seen_scenarios.add(sid)
    assert seen_scenarios, "no DeepMIMO scenarios produced events"
    # Every present scenario should have produced at least one event.
    assert seen_scenarios.issuperset(set(_present_scenarios()))
