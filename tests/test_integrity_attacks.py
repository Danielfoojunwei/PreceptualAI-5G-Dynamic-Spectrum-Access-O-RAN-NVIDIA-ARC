"""End-to-end integrity / supply-chain / audit-chain / Shield-bypass attacks.

One test per adversarial probe, asserting the SHIPPED defense triggers
(IntegrityError / verify False / chain-break index / fallback / emit_blocked /
ValueError). No mocks — these drive the production code paths.

Companion battery: ``benchmarks/integrity_attack_suite.py``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.policy.li_constraint import LIConstraint, LIJurisdictionRule
from horizon_ric.provenance.signing import (
    ModelProvenance,
    ProvenanceError,
    manifest_digest,
    sign_model,
    verify_model,
    verify_or_raise,
)
from horizon_ric.runtime.artefact_vault import ArtefactVault, IntegrityError
from horizon_ric.security.hsm import InMemoryHSMBackend
from horizon_ric.shield import (
    NeuralRxEnvelopeInvariant,
    default_terrestrial_shield,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _versions() -> ModelVersions:
    return ModelVersions(
        encoder="enc-1", risk_heads="rh-1", dyna="dy-1",
        policy="pol-1", constraint_layer="cl-1", rapp="horizon-ric-1",
    )


def _outcome(risk: float = 0.05) -> PredictedOutcome:
    return PredictedOutcome(sla_risk_30s=risk, sla_risk_1min=risk, sla_risk_5min=risk)


def _record(decision_id: str, tx: float = 25.0) -> DecisionRecord:
    return DecisionRecord.new(
        decision_id=decision_id,
        timestamp=datetime.now(timezone.utc),
        rapp_instance_id="attack-rapp",
        state_hash="0" * 64,
        chosen_action={"tx_dBm": tx},
        predicted_outcome_chosen=_outcome(),
        rejected_alternatives=[],
        model_versions=_versions(),
    )


def _sign_trusted(weights: bytes, manifest: dict[str, Any]):
    hsm = InMemoryHSMBackend()
    prov = sign_model(
        weights, trainer_id="trainer-trusted", training_manifest=manifest,
        hsm=hsm, key_label="trusted-signer",
    )
    return prov, bytes.fromhex(prov.public_key_der_hex)


# ---------------------------------------------------------------------------
# Probe 1 — model-swap / untrusted signer
# ---------------------------------------------------------------------------
def test_probe1_model_swap_untrusted_signer_rejected():
    weights = b"\x01\x02\x03 trusted-model-weights " * 64
    manifest = {"dataset": "ds-A", "epochs": 10}
    trusted_prov, trusted_der = _sign_trusted(weights, manifest)

    # Genuine artefact verifies under the pinned trusted key.
    assert verify_model(weights, trusted_prov, trusted_public_key_der=trusted_der) is True

    # Attacker re-signs the SAME bytes (and spoofs the trainer id) with their key.
    attacker_hsm = InMemoryHSMBackend()
    attacker_prov = sign_model(
        weights, trainer_id="trainer-trusted", training_manifest=manifest,
        hsm=attacker_hsm, key_label="attacker-signer",
    )
    # Self-consistent (the attacker's signature verifies under the attacker key)...
    assert verify_model(weights, attacker_prov) is True
    # ...but rejected once pinned to the trusted public key.
    assert verify_model(weights, attacker_prov, trusted_public_key_der=trusted_der) is False
    with pytest.raises(ProvenanceError):
        verify_or_raise(weights, attacker_prov, trusted_public_key_der=trusted_der)


# ---------------------------------------------------------------------------
# Probe 2 — weight-byte tamper
# ---------------------------------------------------------------------------
def test_probe2_weight_byte_tamper(tmp_path: Path):
    weights = b"\x10\x20\x30 model-binary-content " * 128
    manifest = {"dataset": "ds-B", "epochs": 5}
    prov, trusted_der = _sign_trusted(weights, manifest)

    vault = ArtefactVault(tmp_path)
    sha = vault.store(weights, label="v1")
    # Untampered retrieve succeeds.
    assert vault.retrieve(sha) == weights

    # Flip a byte on disk.
    bin_path, _ = vault._paths_for(sha)
    raw = bytearray(bin_path.read_bytes())
    raw[0] ^= 0xFF
    bin_path.write_bytes(bytes(raw))

    # Vault must raise IntegrityError.
    with pytest.raises(IntegrityError):
        vault.retrieve(sha)

    # Provenance sha check must fail on the tampered bytes.
    assert verify_model(bytes(raw), prov, trusted_public_key_der=trusted_der) is False


# ---------------------------------------------------------------------------
# Probe 3 — manifest tamper
# ---------------------------------------------------------------------------
def test_probe3_manifest_tamper_rejected():
    weights = b"\xaa\xbb model-3 " * 100
    manifest = {"dataset": "ds-clean", "epochs": 20, "lr": 0.001}
    prov, trusted_der = _sign_trusted(weights, manifest)
    assert verify_model(weights, prov, trusted_public_key_der=trusted_der) is True

    tampered_manifest = {"dataset": "ds-POISONED", "epochs": 20, "lr": 0.001}
    new_digest = manifest_digest(tampered_manifest)
    assert new_digest != prov.manifest_sha256

    forged = ModelProvenance(
        weights_sha256=prov.weights_sha256,
        manifest_sha256=new_digest,           # reflects the tampered manifest
        trainer_id=prov.trainer_id,
        sig_alg=prov.sig_alg,
        signature_hex=prov.signature_hex,     # but signature was over the clean digest
        public_key_der_hex=prov.public_key_der_hex,
        created_at=prov.created_at,
    )
    assert verify_model(weights, forged, trusted_public_key_der=trusted_der) is False


# ---------------------------------------------------------------------------
# Probe 4 — evidence-chain field tamper
# ---------------------------------------------------------------------------
def test_probe4_evidence_chain_tamper_detected(tmp_path: Path):
    path = tmp_path / "ev.jsonl"
    store = JsonlEvidenceStore(path)
    for i in range(4):
        store.append(_record(f"d-{i}"))
    assert store.verify() == -1

    lines = path.read_text().splitlines()
    obj = json.loads(lines[1])
    obj["record"]["chosen_action"]["tx_dBm"] = 99.9   # forge a field
    lines[1] = json.dumps(obj, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    # Break must be detected at the tampered index (and verify is non -1).
    assert store.verify() == 1


# ---------------------------------------------------------------------------
# Probe 5 — evidence reorder / replay
# ---------------------------------------------------------------------------
def test_probe5_evidence_reorder_detected(tmp_path: Path):
    path = tmp_path / "ev.jsonl"
    store = JsonlEvidenceStore(path)
    for i in range(4):
        store.append(_record(f"d-{i}"))
    assert store.verify() == -1

    lines = path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]   # swap two records
    path.write_text("\n".join(lines) + "\n")

    broken = store.verify()
    assert broken != -1
    assert broken == 1   # the swap first breaks at the earlier position


def test_probe5_evidence_replay_detected(tmp_path: Path):
    path = tmp_path / "ev.jsonl"
    store = JsonlEvidenceStore(path)
    for i in range(4):
        store.append(_record(f"d-{i}"))
    assert store.verify() == -1

    lines = path.read_text().splitlines()
    lines.append(lines[1])   # replay an earlier record verbatim
    path.write_text("\n".join(lines) + "\n")

    broken = store.verify()
    assert broken != -1
    assert broken == 4   # replayed row sits at index 4 and breaks the chain there


# ---------------------------------------------------------------------------
# Probe 6 — Shield self-report spoofing
# ---------------------------------------------------------------------------
def test_probe6_shield_self_report_spoof_falls_back():
    inv = NeuralRxEnvelopeInvariant(tolerance_dB=1.0)
    poisoned = {
        "block": "neural_rx_v2",
        "baseline_tbler": 1e-3,
        "predicted_tbler": 1e-9,     # self-reported spoof
        "demap_confidence": 0.999,
    }
    # Independent CRC/HARQ measurement says the slot is being corrupted.
    ctx = {"measured_tbler": 0.5}

    check = inv.evaluate(poisoned, ctx)
    assert check.satisfied is False   # spoof did NOT bypass the gate

    safe, viols = inv.project(poisoned, ctx)
    assert safe["block"].startswith("classical")
    assert len(viols) == 1
    assert viols[0].severity == "hard"


def test_probe6_shield_unverified_path_falls_back():
    inv = NeuralRxEnvelopeInvariant(tolerance_dB=1.0, require_measurement=True)
    action = {
        "block": "neural_rx_v2",
        "baseline_tbler": 1e-3,
        "predicted_tbler": 1e-9,
        "demap_confidence": 0.999,
    }
    # No measured_tbler in context → conservative fall back.
    check = inv.evaluate(action, {})
    assert check.satisfied is False
    assert "UNVERIFIED" in check.detail

    safe, viols = inv.project(action, {})
    assert safe["block"].startswith("classical")
    assert len(viols) == 1


# ---------------------------------------------------------------------------
# Probe 7 — illegal-emit attempt
# ---------------------------------------------------------------------------
@pytest.fixture()
def terrestrial_shield():
    return default_terrestrial_shield(
        band_lo_hz=3.30e9, band_hi_hz=3.80e9, max_eirp_dBm=33.0, max_papr_dB=8.5,
    )


def test_probe7_out_of_band_carrier_made_safe(terrestrial_shield):
    action = {
        "block": "ric_policy",
        "frequency_hz": 2.0e9,    # far below the band
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": 5.0,
    }
    disp = terrestrial_shield.dispose(action, {"measured_tbler": 1e-4})
    cert = disp.certificate
    # Either blocked or projected to a safe carrier — never an illegal emit.
    assert cert.emit_blocked or cert.safe
    if cert.safe:
        f = disp.safe_action["frequency_hz"]
        bw = disp.safe_action["bandwidth_hz"]
        assert (f - bw / 2.0) >= 3.30e9
        assert (f + bw / 2.0) <= 3.80e9


def test_probe7_over_eirp_clamped(terrestrial_shield):
    action = {
        "block": "ric_policy",
        "frequency_hz": 3.55e9,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 60.0,      # 60 + 20 = 80 dBm EIRP >> 33 ceiling
        "antenna_gain_dBi": 20.0,
    }
    disp = terrestrial_shield.dispose(action, {"measured_tbler": 1e-4})
    cert = disp.certificate
    assert not cert.emit_blocked and cert.safe
    eirp = disp.safe_action["tx_power_dBm"] + disp.safe_action["antenna_gain_dBi"]
    assert eirp <= 33.0 + 1e-6


def test_probe7_illegal_constellation_snapped(terrestrial_shield):
    action = {
        "block": "learned_constellation",
        "frequency_hz": 3.55e9,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": 5.0,
        "constellation_order": 11,    # illegal M-QAM order
        "papr_dB": 14.0,              # over the PAPR ceiling
    }
    disp = terrestrial_shield.dispose(action, {"measured_tbler": 1e-4})
    cert = disp.certificate
    assert not cert.emit_blocked and cert.safe
    assert disp.safe_action["constellation_order"] in (4, 16, 64, 256)
    assert disp.safe_action["papr_dB"] <= 8.5 + 1e-6
    assert cert.fallback_used


def test_probe7_zero_illegal_emits_across_battery(terrestrial_shield):
    illegal = [
        {"block": "ric_policy", "frequency_hz": 2.0e9, "bandwidth_hz": 20e6,
         "tx_power_dBm": 20.0, "antenna_gain_dBi": 5.0},
        {"block": "ric_policy", "frequency_hz": 3.55e9, "bandwidth_hz": 20e6,
         "tx_power_dBm": 60.0, "antenna_gain_dBi": 20.0},
        {"block": "learned_constellation", "frequency_hz": 3.55e9, "bandwidth_hz": 20e6,
         "tx_power_dBm": 20.0, "antenna_gain_dBi": 5.0,
         "constellation_order": 11, "papr_dB": 14.0},
    ]
    illegal_emits = 0
    for act in illegal:
        cert = terrestrial_shield.dispose(act, {"measured_tbler": 1e-4}).certificate
        # An illegal emit is one that is neither blocked nor actually safe.
        if not cert.emit_blocked and (not cert.safe or cert.violated_ids):
            illegal_emits += 1
    assert illegal_emits == 0


# ---------------------------------------------------------------------------
# Probe 8 — LI fail-closed bypass
# ---------------------------------------------------------------------------
def test_probe8_li_fail_closed_blocks_every_emit():
    li = LIConstraint(rules=[], fail_closed=True)
    action = {
        "block": "ric_policy",
        "frequency_hz": 3.55e9,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": 5.0,
        "affected_ue_ids": ["ue-1"],
        "target_jurisdiction": "DE",
    }
    viols = li.check_feasibility(action, {})
    assert any(v.constraint_id == "li_fail_closed" for v in viols)

    shield = default_terrestrial_shield(
        band_lo_hz=3.30e9, band_hi_hz=3.80e9, li_constraint=li,
    )
    disp = shield.dispose(action, {"measured_tbler": 1e-4})
    assert disp.certificate.emit_blocked is True
    assert "lawful_intercept" in disp.certificate.violated_ids


def test_probe8_populated_catalogue_allows_unrelated_ue():
    # Confirms the fail-closed block is the gate, not a blanket refusal.
    li = LIConstraint(
        rules=[LIJurisdictionRule(rule_id="w-1", protected_ue_ids={"ue-protected"})],
        fail_closed=True,
    )
    action = {"affected_ue_ids": ["ue-other"], "target_jurisdiction": "DE"}
    assert li.check_feasibility(action, {}) == []


def test_probe8_unaudited_fail_open_construction_raises():
    with pytest.raises(ValueError):
        LIConstraint(rules=[], fail_closed=False)


def test_probe8_audited_fail_open_construction_ok():
    li = LIConstraint(rules=[], fail_closed=False, deployment_audit_record="audit-entry-42")
    # With fail_closed=False and a recorded audit election, an unrelated action passes.
    assert li.check_feasibility({"affected_ue_ids": ["ue-x"]}, {}) == []
