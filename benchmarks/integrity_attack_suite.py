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

Probes 1-8 are self-contained: they attack the shipped defenses with adversary
behaviour we construct, which is the correct methodology for a cryptographic or
fail-closed gate (there is nothing external to measure — a signature check is a
property of the code, not of a dataset).

Probes 9-11 are driven by REAL CAPTURED ATTACK TRAFFIC
-------------------------------------------------------
Source: 5GAD-2022 (Idaho National Laboratory), MIT licensed, see
``datasets/5gad_inl/``. These are over-the-wire packet captures of ten attacks
actually executed against a real free5GC 5G standalone core on a physical test
bench — real attacker tooling, real target, real responses.

9.  Captured forged-NF-profile injection — the 398 forged AMF registration
    request bodies that the FakeAMFInsert attack actually put on the wire, byte
    for byte, and the responses the real NRF actually returned (398/398
    ``HTTP/1.1 200 OK`` — the real core accepted every one). Those exact bytes
    are then presented to Horizon's artefact-admission gate.
10. Captured malformed / fuzzed control-plane input — the real request lines
    from CrashNRF (empty required discovery parameters, which is what took the
    real NRF down), randomDataDump and randomAMFInsert (attacker-generated
    random field values). The Shield must fail closed, not panic.
11. Evidence chain at real captured volume and ordering — one DecisionRecord per
    real captured control-plane event, in real captured order.

Honest scoping for 9-11: 5GAD is a *core-network* capture. It contains no
Horizon RIC artefacts, records or radio actions, so probes 10 and 11 take the
real attacker-supplied strings / the real event sequence and drive Horizon's own
objects with them; the mapping is stated per probe. Probe 9 is the strongest of
the three because it feeds the captured bytes through unchanged. None of these
is a claim that Horizon defends free5GC.

Run::

    python datasets/5gad_inl/build.py      # once, ~24 MB, no login
    python benchmarks/integrity_attack_suite.py

Writes ``benchmarks/results/integrity_attack_suite.json`` summarising defense
coverage. If ANY probe slips through a defense it is reported loudly (the run
exits non-zero and the ``bypasses`` array is non-empty). If the 5GAD features
have not been built, probes 9-11 are reported as ``skipped`` — never as passing.
"""

from __future__ import annotations

import base64
import hashlib
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "integrity_attack_suite.json"
_DATASET_DIR = _REPO_ROOT / "datasets" / "5gad_inl"


# ---------------------------------------------------------------------------
# Real captured-attack corpus (5GAD-2022, MIT). Absent => probes 9-11 skip.
# ---------------------------------------------------------------------------
class RealCorpusMissing(RuntimeError):
    """The 5GAD-derived features have not been built."""


def _load_real_corpus() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = _DATASET_DIR / "manifest.json"
    events_path = _DATASET_DIR / "generated" / "control_plane_events.jsonl"
    profiles_path = _DATASET_DIR / "generated" / "nf_profile_writes.jsonl"
    missing = [p for p in (manifest_path, events_path, profiles_path) if not p.is_file()]
    if missing:
        raise RealCorpusMissing(
            "5GAD captured-attack features not built ("
            + ", ".join(str(p) for p in missing)
            + "). Build with: python datasets/5gad_inl/build.py"
        )

    def _rows(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    return json.loads(manifest_path.read_text(encoding="utf-8")), _rows(events_path), _rows(
        profiles_path
    )


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
# Probes driven by REAL captured attack traffic (5GAD-2022, MIT)
# ---------------------------------------------------------------------------
def probe_9_captured_forged_nf_injection() -> tuple[bool, str, str]:
    """Replay the REAL forged AMF NF-profile registrations through the artefact gate.

    The FakeAMFInsert attack in 5GAD PUTs a forged AMF NF profile
    (``nfInstanceId=b01dface-bead-cafe-bade-cabledfabled``) to the NRF's
    ``/nnrf-nfm/v1/nf-instances/{id}`` endpoint. The full capture records what
    the real free5GC NRF did about it: it answered ``HTTP/1.1 200 OK`` with a
    ``Location`` header for the forged instance, 398 times out of 398. The core
    admitted every forged registration, because nothing in its NF-management
    path pins a signer.

    This probe takes those captured request bodies BYTE FOR BYTE and presents
    them to Horizon's artefact-admission gate as an untrusted registrant would:
    signed with the attacker's own HSM key, internally self-consistent, and
    pinned-verified against the operator's trusted public key. Every one must be
    rejected. It additionally stores each captured body in an ``ArtefactVault``
    and flips a byte to confirm the content-addressed retrieval catches it.

    Scope: this is NOT a claim that Horizon protects free5GC's NRF. It is a
    measured comparison on identical bytes — what a real production-grade 5G
    core accepted, versus what Horizon's own gate does with the same input.
    """
    _manifest, _events, profiles = _load_real_corpus()
    forged = [
        row
        for row in profiles
        if row["capture"] == "FakeAMFInsert_full" and row["method"] == "PUT"
    ]
    if not forged:
        raise RuntimeError("no captured FakeAMFInsert NF-profile writes in the corpus")

    real_core_accepted = sum(1 for r in forged if 200 <= r["observed_response_status"] < 300)

    # One trusted operator signer, pinned. The attacker has their own HSM.
    trusted_hsm = InMemoryHSMBackend()
    attacker_hsm = InMemoryHSMBackend()
    reference = sign_model(
        b"operator-reference-artefact",
        trainer_id="operator",
        training_manifest={"role": "reference"},
        hsm=trusted_hsm,
        key_label="operator-signer",
    )
    trusted_der = bytes.fromhex(reference.public_key_der_hex)

    pinned_rejected = 0
    self_consistent = 0
    digest_matched = 0
    vault_detected = 0
    with tempfile.TemporaryDirectory() as td:
        for index, row in enumerate(forged):
            body = base64.b64decode(row["body_b64"])
            # The derived row's recorded digest must match the bytes we replay,
            # otherwise the corpus and the probe have drifted apart.
            if hashlib.sha256(body).hexdigest() == row["body_sha256"]:
                digest_matched += 1

            attacker_prov = sign_model(
                body,
                trainer_id="amf",  # the forged profile claims to be an AMF
                training_manifest={
                    "nf_instance_id": row.get("nfInstanceId", ""),
                    "nf_type": row.get("nfType", ""),
                    "path": row["path"],
                },
                hsm=attacker_hsm,
                key_label="rogue-nf-signer",
            )
            if verify_model(body, attacker_prov):
                self_consistent += 1
            if not verify_model(body, attacker_prov, trusted_public_key_der=trusted_der):
                pinned_rejected += 1

            # Byte-tamper the stored artefact for a sample of the corpus (a full
            # sweep would be 398 vault round-trips for no extra information).
            # Each trial gets its OWN vault: the captured forged registrations
            # are byte-identical repeats, and a content-addressed store
            # deduplicates them, so a shared vault would silently re-flip the
            # same byte back and halve the detections.
            if index % 50 == 0:
                trial_vault = ArtefactVault(Path(td) / f"trial-{index}")
                sha = trial_vault.store(body, label=f"captured-{index}")
                bin_path, _ = trial_vault._paths_for(sha)
                raw = bytearray(bin_path.read_bytes())
                raw[0] ^= 0xFF
                bin_path.write_bytes(bytes(raw))
                try:
                    trial_vault.retrieve(sha)
                except IntegrityError:
                    vault_detected += 1

    total = len(forged)
    tamper_trials = len(range(0, total, 50))
    detected = (
        total > 0
        and digest_matched == total
        and self_consistent == total
        and pinned_rejected == total
        and vault_detected == tamper_trials
    )
    return (
        detected,
        "verify_model pinned trusted_public_key_der + ArtefactVault content addressing, "
        "applied to the verbatim captured forged NF-profile bodies",
        f"captured_forged_registrations={total} "
        f"real_free5gc_nrf_accepted_2xx={real_core_accepted}/{total} "
        f"horizon_pinned_reject={pinned_rejected}/{total} "
        f"attacker_signature_self_consistent={self_consistent}/{total} "
        f"replayed_bytes_match_recorded_digest={digest_matched}/{total} "
        f"vault_IntegrityError={vault_detected}/{tamper_trials}",
    )


def probe_10_captured_malformed_input_failclosed() -> tuple[bool, str, str]:
    """Drive the Shield with the REAL malformed / fuzzed attacker strings.

    Real inputs, verbatim from the captures:

    * ``CrashNRF`` — 398 x ``GET /nnrf-disc/v1/nf-instances?requester-nf-type=
      &target-nf-type=``. Both required discovery parameters are EMPTY. This is
      the request that takes the real free5GC NRF down.
    * ``randomDataDump`` — 397 discovery requests whose ``requester-nf-type``
      carries attacker-generated random junk.
    * ``randomAMFInsert`` — 799 NF-management writes against 399 random
      instance UUIDs.

    Mapping (stated because it is not measured): 5GAD contains no Horizon
    actions, so each captured request drives two arms.

    Arm A — hostile text. The captured request line is placed into the
    STRING-typed fields of an otherwise legal Shield action (block name, target
    jurisdiction, affected-UE list). The attacker-controlled text is real; where
    it lands is our construction.

    Arm B — the real attack's SEMANTICS. CrashNRF works by sending a request
    whose REQUIRED parameters are present-but-empty, which the real NRF then
    dereferences and dies on. Arm B reproduces that shape against Horizon: the
    action's required fields are present-but-empty in exactly the way the
    captured request left ``requester-nf-type`` and ``target-nf-type`` empty.
    The Shield must fail closed on this, not emit and not raise.

    Pass condition: zero unhandled exceptions across both arms for every
    captured request; zero illegal emits (an emit neither blocked nor certified
    safe, or one leaving a hard invariant violated); and every Arm-B action
    either blocked or projected — never emitted unchanged, because an action
    with empty required fields must never reach the air.
    """
    _manifest, events, _profiles = _load_real_corpus()
    hostile = [
        row
        for row in events
        if row["capture"] in {"CrashNRF", "randomDataDump", "randomAMFInsert"}
    ]
    if not hostile:
        raise RuntimeError("no captured malformed/fuzzed requests in the corpus")

    shield = default_terrestrial_shield(
        band_lo_hz=3.30e9, band_hi_hz=3.80e9, max_eirp_dBm=33.0, max_papr_dB=8.5
    )

    crashes: list[str] = []
    illegal_emits = 0
    arm_a_handled = 0
    arm_b_contained = 0
    arm_b_total = 0

    def _dispose(action: dict[str, Any], decision_id: str):
        nonlocal illegal_emits
        try:
            disposition = shield.dispose(
                action, {"measured_tbler": 1e-4}, decision_id=decision_id
            )
        except Exception as exc:  # noqa: BLE001 - any escape is a finding
            crashes.append(f"{decision_id}: {type(exc).__name__}: {exc}")
            return None
        certificate = disposition.certificate
        if not certificate.emit_blocked and not certificate.safe:
            illegal_emits += 1
        if not certificate.emit_blocked and certificate.violated_ids:
            illegal_emits += 1
        return certificate

    for index, row in enumerate(hostile):
        # Arm A — attacker-controlled strings, verbatim from the wire, in an
        # otherwise legal action.
        certificate = _dispose(
            {
                "block": row["path"],
                "target_jurisdiction": row["method"],
                "affected_ue_ids": [row["path"]],
                "frequency_hz": 3.55e9,
                "bandwidth_hz": 20e6,
                "tx_power_dBm": 20.0,
                "antenna_gain_dBi": 5.0,
            },
            f"captured-text-{index}",
        )
        if certificate is not None:
            arm_a_handled += 1

        # Arm B — the captured attack's own shape: required parameters present
        # but empty. Only the requests that actually carried empty required
        # parameters on the wire drive this arm.
        if "requester-nf-type=&" in row["path"] or row["path"].endswith("target-nf-type="):
            arm_b_total += 1
            certificate = _dispose(
                {
                    "block": row["path"],
                    "frequency_hz": None,
                    "bandwidth_hz": None,
                    "tx_power_dBm": None,
                    "antenna_gain_dBi": None,
                },
                f"captured-empty-{index}",
            )
            if certificate is not None and (
                certificate.emit_blocked or certificate.projected
            ):
                arm_b_contained += 1

    detected = (
        not crashes
        and illegal_emits == 0
        and arm_a_handled == len(hostile)
        and arm_b_total > 0
        and arm_b_contained == arm_b_total
    )
    sample = crashes[0] if crashes else "none"
    return (
        detected,
        "Shield.dispose is total over attacker-controlled text and over the captured "
        "attack's present-but-empty required fields: no unhandled exception, "
        "project-to-safe-set / fail-closed, 0 illegal emits",
        f"captured_malformed_requests={len(hostile)} "
        f"armA_hostile_text_handled={arm_a_handled}/{len(hostile)} "
        f"armB_empty_required_fields_contained={arm_b_contained}/{arm_b_total} "
        f"unhandled_exceptions={len(crashes)} illegal_emits={illegal_emits} "
        f"first_exception={sample}",
    )


def probe_11_evidence_chain_at_captured_volume() -> tuple[bool, str, str]:
    """Hash-chain integrity at the real captured event count and ordering.

    One ``DecisionRecord`` per control-plane event actually captured on the real
    core, appended in the real captured order. The chain must verify clean; a
    single field mutated at the index of the first captured NF-profile write
    must then be detected at exactly that index.

    Real: the number of events (and therefore the chain length), and their
    order. Not real: the records themselves — 5GAD contains no Horizon decision
    records, so Horizon records carry the captured events' identifiers.
    """
    _manifest, events, _profiles = _load_real_corpus()
    ordered = sorted(events, key=lambda r: (r["capture"], r["t_offset_ns"], r["path"]))
    # Tamper at the first captured NF-profile write, i.e. the first point in the
    # real event stream at which an attacker actually mutated core state.
    tamper_index = next(
        (i for i, r in enumerate(ordered) if r["method"] == "PUT" and r["body_sha256"]),
        1,
    )

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "captured.jsonl"
        store = JsonlEvidenceStore(path)
        for index, row in enumerate(ordered):
            store.append(_record(f"{row['capture']}-{index}", tx=25.0))
        clean_index = store.verify()

        lines = path.read_text().splitlines()
        obj = json.loads(lines[tamper_index])
        obj["record"]["chosen_action"]["tx_dBm"] = 99.9
        lines[tamper_index] = json.dumps(obj, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        broken_index = store.verify()

    detected = clean_index == -1 and broken_index == tamper_index
    return (
        detected,
        "EvidenceStore.verify: sha256(prev || canonical_json(record)) chain break, "
        "at the real captured event count and ordering",
        f"chain_length={len(ordered)} (= captured control-plane events) "
        f"clean_verify={clean_index} tamper_index={tamper_index} "
        f"first_broken_index={broken_index}",
    )


# ---------------------------------------------------------------------------
# Battery driver
# ---------------------------------------------------------------------------
_REAL_DATA_PROBES = frozenset(
    {
        "9_captured_forged_nf_injection",
        "10_captured_malformed_input_failclosed",
        "11_evidence_chain_at_captured_volume",
    }
)

_PROBES: list[tuple[str, str, Callable[[], tuple[bool, str, str]]]] = [
    ("1_model_swap_untrusted_signer", "model-swap / untrusted signer", probe_1_model_swap_untrusted_signer),
    ("2_weight_byte_tamper", "weight-byte tamper", probe_2_weight_byte_tamper),
    ("3_manifest_tamper", "training-manifest tamper", probe_3_manifest_tamper),
    ("4_evidence_chain_tamper", "evidence-chain field tamper", probe_4_evidence_chain_tamper),
    ("5_evidence_reorder_replay", "evidence reorder / replay", probe_5_evidence_reorder_replay),
    ("6_shield_self_report_spoof", "shield self-report spoofing", probe_6_shield_self_report_spoof),
    ("7_illegal_emit_attempt", "illegal-emit attempt", probe_7_illegal_emit_attempt),
    ("8_li_fail_closed_bypass", "LI fail-closed bypass", probe_8_li_fail_closed_bypass),
    (
        "9_captured_forged_nf_injection",
        "captured forged NF-profile injection (5GAD FakeAMFInsert)",
        probe_9_captured_forged_nf_injection,
    ),
    (
        "10_captured_malformed_input_failclosed",
        "captured malformed/fuzzed control-plane input (5GAD CrashNRF/randomDataDump/randomAMFInsert)",
        probe_10_captured_malformed_input_failclosed,
    ),
    (
        "11_evidence_chain_at_captured_volume",
        "evidence chain at real captured volume and ordering (5GAD, all captures)",
        probe_11_evidence_chain_at_captured_volume,
    ),
]


def _real_data_provenance() -> dict[str, Any]:
    """Provenance block for the captured-attack corpus, or an honest absence."""
    try:
        manifest, events, profiles = _load_real_corpus()
    except RealCorpusMissing as exc:
        return {
            "available": False,
            "reason": str(exc),
            "consequence": "probes 9-11 are reported as skipped, not as passing",
        }
    return {
        "available": True,
        "dataset": manifest["dataset"],
        "data_kind": manifest["data_kind"],
        "repo_url": manifest["repo_url"],
        "repo_commit": manifest["repo_commit"],
        "repo_doi": manifest["repo_doi"],
        "paper_doi": manifest["paper_doi"],
        "licence": manifest["licensing"]["source_repository"],
        "attribution": manifest["licensing"]["attribution_required"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "events_sha256": manifest["events_sha256"],
        "features_sha256": manifest["features_sha256"],
        "ci_reproducible": manifest["ci_reproducible"],
        "captured_control_plane_events": len(events),
        "captured_nf_profile_writes": len(profiles),
        "claim": (
            "Probes 9-11 replay traffic captured over the wire while ten real "
            "attacks were executed against a real free5GC 5G standalone core. "
            "Probe 9 feeds the captured forged NF-profile request bodies through "
            "Horizon's artefact gate BYTE FOR BYTE. Probes 10 and 11 use the real "
            "attacker-supplied strings and the real event sequence to drive "
            "Horizon's own objects, because 5GAD contains no Horizon artefacts."
        ),
        "not_claimed": [
            "that Horizon defends free5GC or any 5G core network function",
            "that probes 1-8 use captured data — they are cryptographic and "
            "fail-closed gates, which are properties of the code, not of a dataset",
        ],
    }


def run_battery() -> dict[str, Any]:
    attacks: list[dict[str, Any]] = []
    bypasses: list[str] = []
    skipped: list[str] = []
    for attack_id, name, fn in _PROBES:
        status = "run"
        try:
            detected, mechanism, detail = fn()
        except RealCorpusMissing as exc:
            # The captured-attack corpus is not built. Report the probe as
            # skipped and say so loudly; never let a missing dataset read as a
            # passing defense.
            status = "skipped_dataset_absent"
            detected, mechanism, detail = False, "skipped: real corpus absent", str(exc)
        except Exception as exc:  # a probe that errors out is NOT a passing defense
            detected, mechanism, detail = False, f"probe raised {type(exc).__name__}", str(exc)
        attacks.append(
            {
                "attack_id": attack_id,
                "name": name,
                "attempted": status == "run",
                "status": status,
                "data_provenance": (
                    "real_captured_attack_traffic"
                    if attack_id in _REAL_DATA_PROBES
                    else "constructed_adversary_against_shipped_defense"
                ),
                "detected_or_blocked": bool(detected),
                "mechanism": mechanism,
                "detail": detail,
            }
        )
        if status == "skipped_dataset_absent":
            skipped.append(attack_id)
        elif not detected:
            bypasses.append(attack_id)

    run_attacks = [a for a in attacks if a["status"] == "run"]
    blocked = sum(1 for a in run_attacks if a["detected_or_blocked"])
    summary = {
        "suite": "integrity_attack_suite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_probes": len(attacks),
        "total_attacks": len(run_attacks),
        "detected_or_blocked": blocked,
        "bypassed": len(bypasses),
        "skipped": len(skipped),
        "skipped_ids": skipped,
        "all_defended": len(bypasses) == 0 and not skipped,
        "bypasses": bypasses,
        "real_data_probes": sorted(_REAL_DATA_PROBES),
        "real_data_provenance": _real_data_provenance(),
        "attacks": attacks,
    }
    return summary


def main() -> int:
    summary = run_battery()
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"\nIntegrity attack battery — {summary['detected_or_blocked']}/"
          f"{summary['total_attacks']} probes detected/blocked "
          f"({summary['skipped']} skipped)")
    provenance = summary["real_data_provenance"]
    if provenance["available"]:
        print(f"  real captured-attack corpus: {provenance['dataset']} "
              f"({provenance['licence']})")
        print(f"  {provenance['captured_control_plane_events']} captured control-plane "
              f"events, {provenance['captured_nf_profile_writes']} captured NF-profile writes")
    else:
        print(f"  real captured-attack corpus UNAVAILABLE: {provenance['reason']}")
    print()
    print(f"{'attack':<40} {'source':<10} {'blocked':<9} mechanism")
    print("-" * 118)
    for a in summary["attacks"]:
        if a["status"] != "run":
            flag = "SKIPPED"
        else:
            flag = "YES" if a["detected_or_blocked"] else ">>> NO <<<"
        source = "REAL" if a["attack_id"] in _REAL_DATA_PROBES else "probe"
        print(f"{a['attack_id']:<40} {source:<10} {flag:<9} {a['mechanism']}")

    if summary["skipped_ids"]:
        print("\n!!! REAL-DATA PROBES SKIPPED — these are NOT passing results !!!")
        for s in summary["skipped_ids"]:
            print(f"  - {s}: build the corpus with 'python datasets/5gad_inl/build.py'")

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
