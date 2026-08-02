#!/usr/bin/env python3
"""Gate — Horizon's E2SM-RC control is readable by somebody else's decoder.

The claim under gate:

    The E2SM-RC Style 2 Action 6 payload Horizon constructs is decoded, in
    full and with the correct values, by an INDEPENDENT implementation:
    FlexRIC's asn1c-generated C codec. Not by our own encoder in reverse.

Why this is a distinct gate from G8
-----------------------------------
G8 proves Horizon's message carries the RAN Parameter IDs OCUDU declares,
nested in the order OCUDU's parser walks, and that it round-trips through
*our* encoder — Python ``asn1tools`` over the vendored O-RAN ASN.1.

Round-tripping through the encoder that produced the bytes proves the encoder
is self-consistent. A private format is also self-consistent. What makes bytes
a *control message* is that a different implementation can read them, and this
gate is the only thing in the repository that establishes that: same ASN.1
source text, different tool (asn1c), different language (C), different
project (FlexRIC).

Where it sits relative to G4
----------------------------
G4 is "a RIC Control Request derived from a signed certificate is accepted by
an E2 node". That has two halves — the bytes being *readable*, and the request
being *delivered and executed*. This gate closes the first half against a real
third-party codec. The second half remains blocked: OCUDU registers the
Style 2 Action 6 executor only in ``e2_du_factory.cpp``, and the DU-side E2
agent has never joined (see ``ocudu/e2/du_e2_probe.py``). This gate does not
claim otherwise.

The negative control is load-bearing
------------------------------------
"FlexRIC decoded it" means nothing without evidence that FlexRIC would refuse
something else. It nearly meant nothing here: FlexRIC's asn1c wrapper does not
raise on a malformed buffer — flipping one byte yields Format 1 with **zero**
RAN parameters and no error, so the first version of the checker exited 0 on
corrupt input. ``flexric_decode_check.c`` now treats an empty recovery as a
failed decode, and ``corrupted_payload_is_rejected`` pins it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (REPO / "src", HERE / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from horizon_ocudu.rc_slice_quota import (  # noqa: E402
    SliceQuota,
    build_control_message,
    encode_control_message,
    visit_order,
)

FLEXRIC = REPO / "third_party" / "flexric"
CHECKER_SRC = HERE / "e2" / "flexric_decode_check.c"

# The quota under test. Values chosen so every recovered integer is distinct,
# so a decoder that mixed two fields up would be visible rather than lucky.
QUOTA_NO_SD = SliceQuota(
    plmn=bytes.fromhex("00f110"), sst=1, min_ratio=20, max_ratio=80, dedicated_ratio=10
)
QUOTA_WITH_SD = SliceQuota(
    plmn=bytes.fromhex("00f110"),
    sst=1,
    sd=66051,  # 0x010203 — three distinct octets
    min_ratio=20,
    max_ratio=80,
    dedicated_ratio=10,
)


class Check:
    def __init__(self, cid: str, passed: bool, detail: str) -> None:
        self.id, self.passed, self.detail = cid, passed, detail

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "passed": self.passed, "detail": self.detail}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_checker(out: Path) -> tuple[bool, str]:
    """Compile the C checker against FlexRIC's built RC service model."""
    lib = FLEXRIC / "build" / "src" / "sm" / "rc_sm" / "librc_sm_static.a"
    if not lib.exists():
        return False, f"FlexRIC RC SM static library not built at {lib}"
    cmd = [
        "gcc", "-O2", "-o", str(out), str(CHECKER_SRC),
        f"-I{FLEXRIC / 'src'}", f"-I{FLEXRIC}", f"-I{FLEXRIC / 'build'}",
        str(lib), "-lm", "-lpthread", "-lsctp",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"compile failed: {proc.stderr[-500:]}"
    return True, " ".join(cmd)


def decode(checker: Path, payload: bytes) -> tuple[int, dict[str, Any] | None, str]:
    proc = subprocess.run(
        [str(checker)], input=payload, capture_output=True, timeout=60
    )
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")
    try:
        tree = json.loads(stdout) if stdout.strip() else None
    except json.JSONDecodeError:
        tree = None
    return proc.returncode, tree, stderr.strip()


def _flatten(tree: dict[str, Any]) -> list[tuple[int, Any]]:
    """``(id, leaf_value_or_None)`` in visit order, matching visit_order()."""
    out: list[tuple[int, Any]] = []

    def walk_value(value: dict[str, Any]) -> None:
        if "struct" in value:
            for item in value["struct"]:
                walk_seq(item)
        elif "list" in value:
            for item in value["list"]:
                for inner in item["struct"]:
                    walk_seq(inner)

    def walk_seq(seq: dict[str, Any]) -> None:
        value = seq["value"]
        leaf = value.get("int", value.get("octets", value.get("real")))
        out.append((seq["id"], leaf))
        walk_value(value)

    for top in tree["ran_parameters"]:
        walk_seq(top)
    return out


def run_checks(checker: Path) -> list[Check]:
    checks: list[Check] = []

    def add(cid: str, passed: bool, detail: str) -> None:
        checks.append(Check(cid, passed, detail))

    payload = encode_control_message([QUOTA_NO_SD])
    rc, tree, err = decode(checker, payload)

    # ── 1. an independent codec reads it at all ─────────────────────────
    add(
        "flexric_codec_decodes_horizon_payload",
        rc == 0 and tree is not None,
        f"Horizon encoded {len(payload)} octets with Python asn1tools; "
        f"FlexRIC's asn1c-generated C codec decoded them (exit {rc}). {err}",
    )
    if tree is None:
        return checks  # nothing further is meaningful

    # ── 2. the ids it recovers are exactly the ones we sent ─────────────
    recovered = _flatten(tree)
    recovered_ids = [i for i, _ in recovered]
    expected_ids = visit_order(build_control_message([QUOTA_NO_SD]))
    add(
        "recovered_ids_equal_our_visit_order",
        recovered_ids == expected_ids,
        f"FlexRIC recovered ids {recovered_ids}; horizon_ocudu.visit_order() "
        f"says {expected_ids}. These are OCUDU's own declared RAN Parameter "
        "IDs (G8 extracts them from its source), so agreement here is three "
        "implementations concurring, not two.",
    )

    # ── 3. and the VALUES survived, not just the shape ──────────────────
    leaves = {i: v for i, v in recovered if v is not None}
    expected_leaves = {
        7: "00f110",  # PLMN identity, 3 octets
        9: "01",  # SST
        11: 20,  # Min PRB Policy Ratio
        12: 80,  # Max
        13: 10,  # Dedicated
    }
    add(
        "recovered_values_equal_what_we_encoded",
        all(leaves.get(k) == v for k, v in expected_leaves.items()),
        f"recovered leaves {leaves} vs encoded {expected_leaves}. The three "
        "ratios are distinct on purpose — a decoder that transposed two "
        "fields would show here rather than pass by coincidence.",
    )

    # ── 4. the optional SD really is optional, both ends agreeing ───────
    payload_sd = encode_control_message([QUOTA_WITH_SD])
    rc_sd, tree_sd, _ = decode(checker, payload_sd)
    ids_sd = [i for i, _ in _flatten(tree_sd)] if tree_sd else []
    add(
        "optional_sd_appears_in_both_encodings",
        rc_sd == 0
        and 10 in ids_sd
        and 10 not in recovered_ids
        and len(payload_sd) > len(payload),
        f"without SD: {len(payload)} octets, ids {recovered_ids}. "
        f"with SD: {len(payload_sd)} octets, ids {ids_sd}. Parameter 10 (SD) "
        "is present in one and absent from the other, and FlexRIC's decoder "
        "agrees with our encoder about which.",
    )

    # ── 5. THE NEGATIVE CONTROL ─────────────────────────────────────────
    corrupted = bytearray(payload)
    corrupted[5] ^= 0xFF
    rc_bad, _, err_bad = decode(checker, bytes(corrupted))
    rc_empty, _, _ = decode(checker, b"")
    add(
        "corrupted_payload_is_rejected",
        rc_bad != 0 and rc_empty != 0,
        f"one byte flipped -> exit {rc_bad}; empty input -> exit {rc_empty}. "
        "This check nearly passed for the wrong reason: FlexRIC's asn1c "
        "wrapper does NOT raise on a malformed buffer — it returned Format 1 "
        "with zero RAN parameters and exit 0. The checker now treats an empty "
        f"recovery as a failed decode. Detail: {err_bad[:120]}",
    )

    # ── 6. the two codecs are genuinely different implementations ───────
    add(
        "the_two_codecs_are_independent",
        (FLEXRIC / "src" / "sm" / "rc_sm" / "dec" / "rc_dec_asn.h").exists(),
        "encoder: Python asn1tools over "
        "src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn. decoder: "
        "FlexRIC third_party/flexric/src/sm/rc_sm (asn1c-generated C). "
        "Different tool, language and project, generated from the same O-RAN "
        "ASN.1 source text — which is what makes the agreement evidence "
        "rather than a tautology.",
    )

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the result JSON here")
    parser.add_argument(
        "--checker",
        help="prebuilt flexric_decode_check binary (compiled here if omitted)",
    )
    args = parser.parse_args(argv)

    if args.checker:
        checker = Path(args.checker)
        build_detail = f"prebuilt: {checker}"
        ok = checker.exists()
    else:
        import tempfile

        checker = Path(tempfile.mkdtemp()) / "flexric_decode_check"
        ok, build_detail = build_checker(checker)

    if not ok:
        print(f"SKIP: {build_detail}", file=sys.stderr)
        result = {
            "gate": "G4-codec-interop",
            "skipped": True,
            "reason": build_detail,
            "passed": False,
        }
        if args.out:
            Path(args.out).write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return 3

    checks = run_checks(checker)
    passed = all(c.passed for c in checks)
    result = {
        "gate": "G4-codec-interop",
        "claim": (
            "Horizon's E2SM-RC Style 2 Action 6 payload is decoded in full, "
            "with correct values, by FlexRIC's independent asn1c-generated C "
            "codec."
        ),
        "closes": "the readability half of G4",
        "does_not_close": (
            "the delivery half of G4: OCUDU registers the Style 2 Action 6 "
            "executor only in e2_du_factory.cpp and the DU E2 agent has not "
            "been observed to join"
        ),
        "source_digests": {
            "ocudu/e2/flexric_decode_check.c": _sha256(CHECKER_SRC),
            "ocudu/src/horizon_ocudu/rc_slice_quota.py": _sha256(
                HERE / "src" / "horizon_ocudu" / "rc_slice_quota.py"
            ),
            "src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn": _sha256(
                REPO / "src" / "horizon_ric" / "e2" / "asn1"
                / "e2sm_rc_v1_03_standard.asn"
            ),
        },
        "checks": [c.to_dict() for c in checks],
        "passed": passed,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.id}", file=sys.stderr)
    print(
        f"G4-codec-interop: {sum(c.passed for c in checks)}/{len(checks)} passed",
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
