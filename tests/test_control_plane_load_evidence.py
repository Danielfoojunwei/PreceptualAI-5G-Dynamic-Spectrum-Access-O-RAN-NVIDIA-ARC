from __future__ import annotations

import json
from pathlib import Path

RESULT = Path("benchmarks/results/control_plane_load.json")


def test_committed_load_result_is_substantial_and_scoped():
    evidence = json.loads(RESULT.read_text(encoding="utf-8"))
    assert evidence["parameters"]["iterations"] >= 50_000
    assert evidence["results"]["blocked_decisions"] == 0
    assert evidence["results"]["projected_decisions"] > 0
    assert evidence["results"]["decisions_per_second"] > 0
    assert evidence["results"]["latency_microseconds"]["p99"] > 0
    limitations = " ".join(evidence["scope"]["does_not_prove"]).lower()
    assert "carrier-scale" in limitations
    assert "24-hour" in limitations
