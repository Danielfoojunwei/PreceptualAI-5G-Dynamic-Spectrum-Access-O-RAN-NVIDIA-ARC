#!/usr/bin/env python3
"""Standalone, OFFLINE verifier for a Horizon RIC evidence export.

A spectrum authority / regulator can run this with nothing installed beyond
the Python standard library, the ``cryptography`` package and ``asn1tools``
(RFC-3161 parsing). It NEVER imports ``horizon_ric``.

Usage::

    python audit/verify_evidence.py <evidence.jsonl>
                                    [--pubkey <ed25519_pub.pem|hex>]
                                    [--anchor <anchor.json>]
                                    [--tsa-cert <tsa_cert.pem>]
                                    [--json]

Exit codes:
    0  every enabled check passed (chain intact; any certificates/anchor OK)
    1  a check failed (first broken chain line, bad certificate, bad anchor)
    2  usage / input error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``horizon_audit`` importable when run as a bare script, without
# requiring an install. This adds ONLY this repo's audit/src — never the
# horizon_ric source tree.
_AUDIT_SRC = Path(__file__).resolve().parent / "src"
if str(_AUDIT_SRC) not in sys.path:
    sys.path.insert(0, str(_AUDIT_SRC))

from horizon_audit import certificate as cert_mod  # noqa: E402
from horizon_audit import timestamp as ts_mod  # noqa: E402
from horizon_audit.chain import parse_jsonl, verify_chain  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_evidence.py",
        description="Offline verifier for a Horizon RIC evidence export.",
    )
    p.add_argument("evidence", help="path to the evidence .jsonl export")
    p.add_argument(
        "--pubkey",
        help="Ed25519 public key (PEM/DER file or 64-char raw hex) to verify "
        "any embedded SafetyCertificate signatures against.",
    )
    p.add_argument(
        "--anchor",
        help="path to an RFC-3161 anchor JSON to verify offline (imprint "
        "binding + CMS signature self-consistency).",
    )
    p.add_argument(
        "--tsa-cert",
        dest="tsa_cert",
        help="TSA certificate (PEM) to establish the anchor's trust anchor "
        "offline. Without it, trust_anchored is reported as null.",
    )
    p.add_argument("--json", action="store_true", help="emit a JSON report")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report: dict = {"evidence": args.evidence}
    failed = False

    # ── 1. Hash chain ───────────────────────────────────────────────────
    ev_path = Path(args.evidence)
    if not ev_path.exists():
        _fail_usage(f"evidence file not found: {ev_path}", args.json)
        return 2
    try:
        lines = parse_jsonl(ev_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail_usage(f"could not read/parse evidence: {exc}", args.json)
        return 2

    chain = verify_chain(lines)
    report["chain"] = chain.as_dict()
    if not chain.intact:
        failed = True

    # ── 2. Embedded Ed25519 certificates (optional) ─────────────────────
    if args.pubkey:
        cert_report: dict = {"checked": 0, "verified": 0, "failed": 0, "items": []}
        try:
            pub = cert_mod.load_public_key(args.pubkey)
        except ValueError as exc:
            _fail_usage(str(exc), args.json)
            return 2
        full_found = False
        for i, obj in enumerate(lines):
            rec = obj.get("record") if isinstance(obj, dict) else None
            if rec is None:
                continue
            for emb in cert_mod.find_embedded_certificates(rec, _prefix=f"line[{i}].record"):
                if not emb.full:
                    cert_report["items"].append(
                        {
                            "path": emb.path,
                            "full": False,
                            "verified": None,
                            "note": "signature-digest reference only; the full "
                            "certificate document is required to verify it",
                        }
                    )
                    continue
                full_found = True
                ok = cert_mod.verify_certificate(emb.cert, pub)
                fp_ok = cert_mod.fingerprint_matches(emb.cert, pub)
                cert_report["checked"] += 1
                cert_report["verified" if ok else "failed"] += 1
                cert_report["items"].append(
                    {
                        "path": emb.path,
                        "full": True,
                        "verified": ok,
                        "fingerprint_matches_pubkey": fp_ok,
                    }
                )
                if not ok:
                    failed = True
        if not full_found:
            cert_report["note"] = (
                "no embedded (full, self-verifiable) certificate found in the "
                "export; nothing to verify with --pubkey"
            )
        report["certificates"] = cert_report

    # ── 3. RFC-3161 anchor (optional) ───────────────────────────────────
    if args.anchor:
        try:
            anchor = ts_mod.load_anchor(args.anchor)
        except (OSError, json.JSONDecodeError) as exc:
            _fail_usage(f"could not read/parse anchor: {exc}", args.json)
            return 2
        tsa_cert_pem = None
        if args.tsa_cert:
            try:
                tsa_cert_pem = Path(args.tsa_cert).read_bytes()
            except OSError as exc:
                _fail_usage(f"could not read --tsa-cert: {exc}", args.json)
                return 2
        result = ts_mod.verify_anchor(anchor, tsa_cert_pem=tsa_cert_pem)
        anchor_report = result.as_dict()
        # Cross-check: does the anchored head actually appear in this export?
        stored_hashes = {obj.get("hash") for obj in lines if isinstance(obj, dict)}
        anchor_report["chain_head_present_in_export"] = (
            result.chain_head_hex in stored_hashes
        )
        report["anchor"] = anchor_report
        if not result.ok:
            failed = True

    report["result"] = "FAIL" if failed else "PASS"

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)

    return 1 if failed else 0


def _print_human(report: dict) -> None:
    chain = report["chain"]
    print(f"evidence : {report['evidence']}")
    print(
        f"chain    : {chain['line_count']} line(s) across "
        f"{chain['tenant_count']} tenant(s)"
    )
    if chain["intact"]:
        print("           INTACT — every per-tenant hash chain verifies")
    else:
        print(
            f"           BROKEN — first broken line index "
            f"{chain['first_broken_index']} (tenant "
            f"{chain['first_broken_tenant']!r})"
        )
        for t in chain["tenants"]:
            if not t["intact"]:
                print(
                    f"             tenant {t['tenant_id']!r}: broken at global "
                    f"index {t['first_broken_global_index']} "
                    f"(per-tenant #{t['first_broken_tenant_index']}) — {t['detail']}"
                )
        for e in chain["errors"]:
            print(f"             error: {e}")

    if "certificates" in report:
        c = report["certificates"]
        if c.get("note") and c["checked"] == 0:
            print(f"certs    : {c['note']}")
        else:
            print(
                f"certs    : checked {c['checked']}, verified {c['verified']}, "
                f"failed {c['failed']}"
            )
            for it in c["items"]:
                if it["full"]:
                    print(
                        f"             {it['path']}: "
                        f"{'OK' if it['verified'] else 'INVALID'}"
                    )
                else:
                    print(f"             {it['path']}: {it['note']}")

    if "anchor" in report:
        a = report["anchor"]
        print(f"anchor   : tsa={a.get('tsa')!r} imprint={a['imprint_alg']}")
        print(
            f"           imprint_match={a['imprint_match']} "
            f"signature_verified={a['signature_verified']} "
            f"trust_anchored={a['trust_anchored']}"
        )
        print(
            f"           chain_head_present_in_export="
            f"{a['chain_head_present_in_export']}  ok={a['ok']}"
        )
        for n in a["notes"]:
            print(f"             note: {n}")

    print(f"result   : {report['result']}")


def _fail_usage(msg: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"result": "ERROR", "error": msg}))
    else:
        print(f"error: {msg}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
