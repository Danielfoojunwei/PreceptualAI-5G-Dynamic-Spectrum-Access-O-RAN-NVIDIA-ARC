"""Integrity / supply-chain / audit-chain / Shield-bypass attack battery.

Mounts REAL adversarial probes against the SHIPPED Horizon-RIC defenses and
records, per attack, whether the defense detected/blocked it and by which
mechanism. No mocks: every probe drives the production code paths in
``horizon_ric.provenance``, ``horizon_ric.security.hsm``,
``horizon_ric.runtime.artefact_vault``, ``horizon_ric.evidence``,
``horizon_ric.shield`` and ``horizon_ric.policy.li_constraint``.

Attack matrix
-------------
1. Model-swap / untrusted signer — attacker re-signs the same weights with
   their OWN HSM key; ``verify_model`` with a pinned ``trusted_public_key_der``
   must REJECT it.
2. Weight-byte tamper — flip bytes of a stored artefact; ``ArtefactVault.retrieve``
   must raise ``IntegrityError`` and ``verify_model`` must fail the sha check.
3. Manifest tamper — change the training manifest after signing; ``verify_model``
   must fail (signed manifest digest no longer matches).
4. Evidence-chain tamper — mutate one ``DecisionRecord`` field after append;
   ``verify`` must detect the break at that index.
5. Evidence reorder / replay — swap two records / re-append an old one; ``verify``
   must detect the chain inconsistency.
6. Shield self-report spoofing — a poisoned model reports a perfect
   ``predicted_tbler`` / high confidence but the INDEPENDENT measured TBLER (CRC)
   is terrible; ``NeuralRxEnvelopeInvariant`` must still fall back. Also the
   conservative-when-unverified path (no ``measured_tbler`` ⇒ fall back).
7. Illegal-emit attempt — craft fully-compromised actions (out-of-band, over-EIRP,
   illegal constellation); the Shield must project to safe / block — 0 illegal
   emits leave the envelope.
8. LI fail-closed bypass — empty LIMF catalogue with ``fail_closed=True`` must
   block every emit; constructing ``fail_closed=False`` without an audit record
   must raise.

Run::

    .venv/bin/python benchmarks/integrity_attack_suite.py

Writes ``benchmarks/results/integrity_attack_suite.json`` summarising defense
coverage. If ANY probe slips through a defense it is reported loudly (the run
exits non-zero and the ``bypasses`` array is non-empty).
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.policy.li_constraint import LIConstraint, LIJurisdictionRule
from horizon_ric.provenance.signing import (
    ModelProvenance,
    manifest_digest,
    sign_model,
    verify_model,
)
from horizon_ric.runtime.artefact_vault import ArtefactVault, IntegrityError
from horizon_ric.security.hsm import InMemoryHSMBackend
from horizon_ric.shield import (
    NeuralRxEnvelopeInvariant,
    Shield,
    default_terrestrial_shield,
)

_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "integrity_attack_suite.json"


# ---------------------------------------------------------------------------
# Helpers
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


def _trusted_provenance(weights: bytes, manifest: dict[str, Any]):
    """Sign ``weights`` with a trusted HSM key; return (provenance, trusted_der)."""
    hsm = InMemoryHSMBackend()
    prov = sign_model(
        weights, trainer_id="trainer-trusted", training_manifest=manifest,
        hsm=hsm, key_label="trusted-signer",
    )
    trusted_der = bytes.fromhex(prov.public_key_der_hex)
    return prov, trusted_der


# ---------------------------------------------------------------------------
# Probes — each returns (detected_or_blocked: bool, mechanism: str, detail: str)
# ---------------------------------------------------------------------------
def probe_1_model_swap_untrusted_signer() -> tuple[bool, str, str]:
    """Attacker re-signs identical weights with their OWN key; pinned verify rejects."""
    weights = b"\x01\x02\x03 trusted-model-weights " * 64
    manifest = {"dataset": "ds-A", "epochs": 10}
    trusted_prov, trusted_der = _trusted_provenance(weights, manifest)

    # Sanity: the genuine artefact verifies under the pinned trusted key.
    genuine_ok = verify_model(weights, trusted_prov, trusted_public_key_der=trusted_der)

    # Attacker: same bytes, same manifest, but signed with the ATTACKER's HSM key.
    attacker_hsm = InMemoryHSMBackend()
    attacker_prov = sign_model(
        weights, trainer_id="trainer-trusted",  # spoof the trainer id too
        training_manifest=manifest, hsm=attacker_hsm, key_label="attacker-signer",
    )
    # The attacker signature is internally valid (self-consistent)...
    attacker_self_consistent = verify_model(weights, attacker_prov)
    # ...but MUST be rejected when pinned to the trusted public key.
    rejected = not verify_model(
        weights, attacker_prov, trusted_public_key_der=trusted_der
    )

    detected = genuine_ok and attacker_self_consistent and rejected
    return (
        detected,
        "verify_model: pinned trusted_public_key_der != embedded signer key",
        f"genuine_ok={genuine_ok} attacker_self_consistent={attacker_self_consistent} "
        f"pinned_reject={rejected}",
    )


def probe_2_weight_byte_tamper() -> tuple[bool, str, str]:
    """Flip bytes of a stored artefact: vault retrieve raises; verify fails sha."""
    weights = b"\x10\x20\x30 model-binary-content " * 128
    manifest = {"dataset": "ds-B", "epochs": 5}
    prov, trusted_der = _trusted_provenance(weights, manifest)

    with tempfile.TemporaryDirectory() as td:
        vault = ArtefactVault(Path(td))
        sha = vault.store(weights, label="v1")

        # Tamper the on-disk artefact bytes directly (flip one byte).
        bin_path, _ = vault._paths_for(sha)
        raw = bytearray(bin_path.read_bytes())
        raw[0] ^= 0xFF
        bin_path.write_bytes(bytes(raw))

        vault_detected = False
        try:
            vault.retrieve(sha)
        except IntegrityError:
            vault_detected = True

        # The tampered bytes also fail the provenance sha check.
        tampered = bytes(raw)
        verify_detected = not verify_model(tampered, prov, trusted_public_key_der=trusted_der)

    detected = vault_detected and verify_detected
    return (
        detected,
        "ArtefactVault.retrieve IntegrityError + verify_model sha256 mismatch",
        f"vault_IntegrityError={vault_detected} verify_sha_fail={verify_detected}",
    )


def probe_3_manifest_tamper() -> tuple[bool, str, str]:
    """Change the training manifest after signing: verify must fail."""
    weights = b"\xaa\xbb model-3 " * 100
    manifest = {"dataset": "ds-clean", "epochs": 20, "lr": 0.001}
    prov, trusted_der = _trusted_provenance(weights, manifest)

    # Attacker swaps the manifest (e.g. hides a poisoned dataset) but keeps the
    # original signature & weights. Recompute the digest the verifier expects.
    tampered_manifest = {"dataset": "ds-POISONED", "epochs": 20, "lr": 0.001}
    new_digest = manifest_digest(tampered_manifest)

    # Forge a provenance whose manifest_sha256 reflects the tampered manifest but
    # keeps the original signature (which signed the clean digest).
    forged = ModelProvenance(
        weights_sha256=prov.weights_sha256,
        manifest_sha256=new_digest,
        trainer_id=prov.trainer_id,
        sig_alg=prov.sig_alg,
        signature_hex=prov.signature_hex,
        public_key_der_hex=prov.public_key_der_hex,
        created_at=prov.created_at,
    )
    # Signature was over the clean manifest digest → must fail.
    forged_rejected = not verify_model(weights, forged, trusted_public_key_der=trusted_der)
    # Sanity: the untouched provenance still verifies.
    clean_ok = verify_model(weights, prov, trusted_public_key_der=trusted_der)

    detected = forged_rejected and clean_ok and (new_digest != prov.manifest_sha256)
    return (
        detected,
        "verify_model: RSA-PSS signature covers manifest_sha256 — tamper invalidates it",
        f"clean_ok={clean_ok} forged_rejected={forged_rejected}",
    )


def probe_4_evidence_chain_tamper() -> tuple[bool, str, str]:
    """Mutate one DecisionRecord field after append; chain break detected at index."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ev.jsonl"
        store = JsonlEvidenceStore(path)
        for i in range(4):
            store.append(_record(f"d-{i}"))
        assert store.verify() == -1, "baseline chain should be intact"

        # Forge the chosen_action of record index 1.
        lines = path.read_text().splitlines()
        obj = json.loads(lines[1])
        obj["record"]["chosen_action"]["tx_dBm"] = 99.9
        lines[1] = json.dumps(obj, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")

        broken_idx = store.verify()

    detected = broken_idx == 1
    return (
        detected,
        "EvidenceStore.verify: sha256(prev || canonical_json(record)) chain break",
        f"first_broken_index={broken_idx} (expected 1)",
    )


def probe_5_evidence_reorder_replay() -> tuple[bool, str, str]:
    """Swap two records / re-append an old one; verify detects the inconsistency."""
    # 5a — reorder: swap records 1 and 2.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ev.jsonl"
        store = JsonlEvidenceStore(path)
        for i in range(4):
            store.append(_record(f"d-{i}"))
        assert store.verify() == -1
        lines = path.read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        path.write_text("\n".join(lines) + "\n")
        reorder_idx = store.verify()
    reorder_detected = reorder_idx != -1

    # 5b — replay: re-append a copy of an old line at the end.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ev.jsonl"
        store = JsonlEvidenceStore(path)
        for i in range(4):
            store.append(_record(f"d-{i}"))
        assert store.verify() == -1
        lines = path.read_text().splitlines()
        lines.append(lines[1])  # replay an earlier record verbatim
        path.write_text("\n".join(lines) + "\n")
        replay_idx = store.verify()
    replay_detected = replay_idx != -1

    detected = reorder_detected and replay_detected
    return (
        detected,
        "EvidenceStore.verify: per-tenant chain order/replay inconsistency",
        f"reorder_break_idx={reorder_idx} replay_break_idx={replay_idx}",
    )


def probe_6_shield_self_report_spoof() -> tuple[bool, str, str]:
    """Poisoned model spoofs predicted_tbler; measured CRC says it's bad → fallback.

    Also tests the conservative path: no measured_tbler ⇒ fall back.
    """
    inv = NeuralRxEnvelopeInvariant(tolerance_dB=1.0)

    # Poisoned model: claims a perfect TBLER + high confidence.
    poisoned_action = {
        "block": "neural_rx_v2",
        "baseline_tbler": 1e-3,        # classical LMMSE baseline
        "predicted_tbler": 1e-9,       # self-reported (the spoof)
        "demap_confidence": 0.999,
    }
    # Independent CRC/HARQ measurement: the slot is actually being corrupted.
    spoof_ctx = {"measured_tbler": 0.5}  # 50% block errors — terrible

    check_spoof = inv.evaluate(poisoned_action, spoof_ctx)
    proj_action, viols = inv.project(poisoned_action, spoof_ctx)
    spoof_fell_back = (
        not check_spoof.satisfied
        and proj_action.get("block", "").startswith("classical")
        and len(viols) > 0
    )

    # Conservative-when-unverified: no measured_tbler supplied at all.
    check_unverified = inv.evaluate(poisoned_action, {})
    proj_unv, viols_unv = inv.project(poisoned_action, {})
    unverified_fell_back = (
        not check_unverified.satisfied
        and proj_unv.get("block", "").startswith("classical")
        and len(viols_unv) > 0
    )

    detected = spoof_fell_back and unverified_fell_back
    return (
        detected,
        "NeuralRxEnvelopeInvariant: grades on independent measured_tbler, not self-report; "
        "fail-closed when unverified",
        f"spoof_fell_back={spoof_fell_back} unverified_fell_back={unverified_fell_back} "
        f"spoof_detail={check_spoof.detail!r}",
    )


def probe_7_illegal_emit_attempt() -> tuple[bool, str, str]:
    """Fully compromised model crafts illegal actions; Shield projects safe / blocks."""
    band_lo, band_hi = 3.30e9, 3.80e9  # n78-ish 500 MHz licensed channel
    shield = default_terrestrial_shield(
        band_lo_hz=band_lo, band_hi_hz=band_hi, max_eirp_dBm=33.0, max_papr_dB=8.5,
    )

    illegal_actions = [
        # Out-of-band carrier (centre far below the band).
        {
            "block": "ric_policy",
            "frequency_hz": 2.0e9,
            "bandwidth_hz": 20e6,
            "tx_power_dBm": 20.0,
            "antenna_gain_dBi": 5.0,
        },
        # Over-EIRP emit (60 dBm tx + 20 dBi >> 33 dBm ceiling).
        {
            "block": "ric_policy",
            "frequency_hz": 3.55e9,
            "bandwidth_hz": 20e6,
            "tx_power_dBm": 60.0,
            "antenna_gain_dBi": 20.0,
        },
        # Illegal constellation order (e.g. 11-QAM) + excess PAPR.
        {
            "block": "learned_constellation",
            "frequency_hz": 3.55e9,
            "bandwidth_hz": 20e6,
            "tx_power_dBm": 20.0,
            "antenna_gain_dBi": 5.0,
            "constellation_order": 11,
            "papr_dB": 14.0,
        },
    ]

    illegal_emits = 0
    details: list[str] = []
    for act in illegal_actions:
        disp = shield.dispose(act, {"measured_tbler": 1e-4})
        cert = disp.certificate
        safe_action = disp.safe_action
        # An emit is illegal iff the Shield certified it safe AND did NOT block
        # AND the final action still violates a hard invariant.
        final_violated = bool(cert.violated_ids)
        if not cert.emit_blocked and not cert.safe:
            illegal_emits += 1
        if not cert.emit_blocked and final_violated:
            illegal_emits += 1
        details.append(
            f"block={act.get('block')} emit_blocked={cert.emit_blocked} "
            f"safe={cert.safe} violated={cert.violated_ids} "
            f"safe_freq={safe_action.get('frequency_hz')} "
            f"safe_tx={safe_action.get('tx_power_dBm')} "
            f"safe_order={safe_action.get('constellation_order')}"
        )

    detected = illegal_emits == 0
    return (
        detected,
        "Shield.dispose: project-to-safe-set / fail-closed (emit_blocked) — 0 illegal emits",
        f"illegal_emits={illegal_emits}; " + " | ".join(details),
    )


def probe_8_li_fail_closed_bypass() -> tuple[bool, str, str]:
    """Empty LIMF + fail_closed=True blocks every emit; fail_closed=False w/o audit raises."""
    # 8a — fail-closed default: empty catalogue rejects every action, both via the
    # raw constraint and via the Shield (LawfulInterceptInvariant chain).
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
    raw_viols = li.check_feasibility(action, {})
    raw_blocks = any(v.constraint_id == "li_fail_closed" for v in raw_viols)

    shield = default_terrestrial_shield(
        band_lo_hz=3.30e9, band_hi_hz=3.80e9, li_constraint=li,
    )
    disp = shield.dispose(action, {"measured_tbler": 1e-4})
    shield_blocks = disp.certificate.emit_blocked

    # Sanity: a populated catalogue does NOT block an unrelated UE (so the block
    # above is genuinely the fail-closed gate, not a blanket refusal of everything).
    li_populated = LIConstraint(
        rules=[LIJurisdictionRule(rule_id="w-1", protected_ue_ids={"ue-protected"})],
        fail_closed=True,
    )
    populated_allows = len(li_populated.check_feasibility(action, {})) == 0

    # 8b — opting out of fail-closed with empty rules and NO audit record must raise.
    construct_raised = False
    try:
        LIConstraint(rules=[], fail_closed=False)
    except ValueError:
        construct_raised = True

    # And the legitimate opt-out (with an audit-record id) must succeed.
    audited_ok = False
    try:
        LIConstraint(rules=[], fail_closed=False, deployment_audit_record="audit-entry-42")
        audited_ok = True
    except ValueError:
        audited_ok = False

    detected = (
        raw_blocks and shield_blocks and populated_allows
        and construct_raised and audited_ok
    )
    return (
        detected,
        "LIConstraint fail-closed gate (li_fail_closed) + ValueError on unaudited opt-out",
        f"raw_blocks={raw_blocks} shield_emit_blocked={shield_blocks} "
        f"populated_allows={populated_allows} unaudited_optout_raised={construct_raised} "
        f"audited_optout_ok={audited_ok}",
    )


# ---------------------------------------------------------------------------
# Battery driver
# ---------------------------------------------------------------------------
_PROBES: list[tuple[str, str, Callable[[], tuple[bool, str, str]]]] = [
    ("1_model_swap_untrusted_signer", "model-swap / untrusted signer", probe_1_model_swap_untrusted_signer),
    ("2_weight_byte_tamper", "weight-byte tamper", probe_2_weight_byte_tamper),
    ("3_manifest_tamper", "training-manifest tamper", probe_3_manifest_tamper),
    ("4_evidence_chain_tamper", "evidence-chain field tamper", probe_4_evidence_chain_tamper),
    ("5_evidence_reorder_replay", "evidence reorder / replay", probe_5_evidence_reorder_replay),
    ("6_shield_self_report_spoof", "shield self-report spoofing", probe_6_shield_self_report_spoof),
    ("7_illegal_emit_attempt", "illegal-emit attempt", probe_7_illegal_emit_attempt),
    ("8_li_fail_closed_bypass", "LI fail-closed bypass", probe_8_li_fail_closed_bypass),
]


def run_battery() -> dict[str, Any]:
    attacks: list[dict[str, Any]] = []
    bypasses: list[str] = []
    for attack_id, name, fn in _PROBES:
        attempted = True
        try:
            detected, mechanism, detail = fn()
        except Exception as exc:  # a probe that errors out is NOT a passing defense
            detected, mechanism, detail = False, f"probe raised {type(exc).__name__}", str(exc)
        attacks.append(
            {
                "attack_id": attack_id,
                "name": name,
                "attempted": attempted,
                "detected_or_blocked": bool(detected),
                "mechanism": mechanism,
                "detail": detail,
            }
        )
        if not detected:
            bypasses.append(attack_id)

    total = len(attacks)
    blocked = sum(1 for a in attacks if a["detected_or_blocked"])
    summary = {
        "suite": "integrity_attack_suite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_attacks": total,
        "detected_or_blocked": blocked,
        "bypassed": len(bypasses),
        "all_defended": len(bypasses) == 0,
        "bypasses": bypasses,
        "attacks": attacks,
    }
    return summary


def main() -> int:
    summary = run_battery()
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"\nIntegrity attack battery — {summary['detected_or_blocked']}/"
          f"{summary['total_attacks']} probes detected/blocked")
    print(f"{'attack':<34} {'blocked':<9} mechanism")
    print("-" * 100)
    for a in summary["attacks"]:
        flag = "YES" if a["detected_or_blocked"] else ">>> NO <<<"
        print(f"{a['attack_id']:<34} {flag:<9} {a['mechanism']}")

    if summary["bypasses"]:
        print("\n!!! DEFENSE BYPASS DETECTED !!!")
        for b in summary["bypasses"]:
            print(f"  - {b} slipped through its defense — INVESTIGATE")
        return 1
    print("\nAll probes detected/blocked across the board.")
    print(f"Results written to {_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
