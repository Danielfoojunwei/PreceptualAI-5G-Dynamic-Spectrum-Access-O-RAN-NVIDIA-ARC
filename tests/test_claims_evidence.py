from __future__ import annotations

import json
from pathlib import Path

MATRIX = Path("docs/CLAIMS_EVIDENCE.json")


def test_claim_matrix_evidence_paths_exist():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    allowed = set(matrix["status_definitions"])
    for claim in matrix["claims"]:
        assert claim["status"] in allowed
        for evidence_path in claim["evidence"]:
            assert Path(evidence_path).exists(), (
                f"{claim['id']} references missing evidence {evidence_path}"
            )


def test_external_claims_remain_explicitly_unestablished():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    by_id = {claim["id"]: claim for claim in matrix["claims"]}
    required = {
        "live_operator_network",
        "commercial_vendor_interop",
        "complete_rf_safety",
        "formal_compliance",
        "unknown_attack_security",
        "carrier_scale",
        "global_novelty",
        "independent_validation",
    }
    assert required <= by_id.keys()
    for claim_id in required:
        assert by_id[claim_id]["status"] == "not_established"


def test_simulator_gate_is_not_mislabelled_as_vendor_deployment():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    by_id = {claim["id"]: claim for claim in matrix["claims"]}
    osc = by_id["osc_open_source_simulator"]
    assert osc["status"] == "ci_external_simulator"
    assert "commercial vendor" in osc["excludes"].lower()
