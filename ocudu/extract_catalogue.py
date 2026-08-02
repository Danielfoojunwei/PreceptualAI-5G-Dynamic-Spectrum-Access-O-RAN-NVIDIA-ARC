#!/usr/bin/env python3
"""Extract OCUDU's E2SM control-action catalogue from its source, not from a spec PDF.

Horizon's E2SM-RC encoder previously carried **placeholder** RAN Parameter IDs,
flagged amber in the conformance table with the note that E2SM-RC assigns them
per node via ``RANFunctionDefinition-Control-Action-Item``. That was the honest
position given what we had: the spec defines the *shape*, and each RAN
implementation declares the actual identifiers it will parse.

OCUDU is a real gNB implementation whose source we have pinned. It declares
those identifiers, in code, in the executors that parse incoming control
requests. Reading them from there turns a placeholder into a fact about a
specific interoperating implementation.

Extraction rather than transcription is the point. Hand-copying thirteen
parameter IDs into a Python dict would work exactly once and then rot silently
the next time the submodule moves. This parses the executors, records the
OCUDU commit and the SHA-256 of every file it read, and
``verify_g8_ocudu_conformance.py`` re-runs it and compares — so a change in
OCUDU that moves an identifier fails the gate rather than producing a control
message a gNB will reject at runtime.

What is extracted:

* **E2SM-RC** control actions from the CU and DU executors — action id, action
  name, and the full RAN-parameter table each one declares.
* **E2SM-CCC** control styles, and the ``O-RRMPolicyRatio`` attributes the DU
  executor accepts, which is the same slice-quota control expressed as JSON
  rather than ASN.1 PER.

Run it with ``--out`` to regenerate ``catalogue/ocudu-e2sm-catalogue.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OCUDU = REPO / "third_party" / "ocudu"

RC_DIR = OCUDU / "lib/e2/e2sm/e2sm_rc"
CCC_DIR = OCUDU / "lib/e2/e2sm/e2sm_ccc"

SOURCES = [
    RC_DIR / "e2sm_rc_control_action_du_executor.cpp",
    RC_DIR / "e2sm_rc_control_action_cu_executor.cpp",
    CCC_DIR / "e2sm_ccc_control_service_impl.h",
    CCC_DIR / "e2sm_ccc_asn1_packer.cpp",
]

# `e2sm_rc_control_action_2_6_du_executor::e2sm_rc_control_action_2_6_du_executor(`
# encodes style 2, action 6 in the type name — which is where OCUDU keeps that
# association, so it is where we read it from.
CTOR_RE = re.compile(
    r"e2sm_rc_control_action_(\d+)_(\d+)_(cu|du)_executor::"
    r"e2sm_rc_control_action_\1_\2_\3_executor\b"
)
ACTION_NAME_RE = re.compile(r'action_name\s*=\s*"([^"]*)"')
PARAM_RE = re.compile(r'action_params\.insert\(\{\s*(\d+)\s*,\s*"([^"]*)"\s*\}\)')
# The parse switch is what actually decides which ids do something, and how.
PARSE_ID_RE = re.compile(r"ran_param_id\s*==\s*(\d+)")
CCC_STYLE_RE = re.compile(r'\{\s*(\d+),\s*"([^"]+)",\s*(\d+),\s*(\d+),\s*(\d+)\s*\}')
CCC_ATTR_RE = re.compile(r'attribute_name\.from_string\("([^"]+)"\)')
CCC_STRUCT_RE = re.compile(r'ran_cfg_structure_name\.from_string\("([^"]+)"\)')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ocudu_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(OCUDU), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - submodule not initialised
        return ""


def _split_constructors(text: str) -> list[tuple[str, str, str, str]]:
    """Slice the file into per-constructor bodies, in source order."""
    matches = list(CTOR_RE.finditer(text))
    out: list[tuple[str, str, str, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), m.group(2), m.group(3), text[m.start() : end]))
    return out


def extract_rc(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    actions: list[dict[str, Any]] = []
    for style, action, node, body in _split_constructors(text):
        name = ACTION_NAME_RE.search(body)
        params = {int(pid): pname for pid, pname in PARAM_RE.findall(body)}
        if not params:
            # A constructor that declares no parameters is a base or a stub;
            # recording it as an empty action would imply we can address it.
            continue
        actions.append(
            {
                "service_model": "E2SM-RC",
                "node": node.upper(),
                "style_id": int(style),
                "action_id": int(action),
                "action_name": name.group(1) if name else "",
                "ran_parameters": [
                    {"id": pid, "name": params[pid]} for pid in sorted(params)
                ],
                "source_file": str(path.relative_to(REPO)),
            }
        )
    return actions


# A single arm matching several ids — `if (ran_param_id == 1 or ran_param_id == 3
# or ...)` — is how OCUDU spells "these are pure structure". Detecting it is what
# separates a parameter that carries a value from one that only opens a nesting
# level.
NOOP_ARM_RE = re.compile(
    r"if\s*\((?P<ids>ran_param_id\s*==\s*\d+(?:\s+or\s+ran_param_id\s*==\s*\d+)+)\)"
    r"\s*\{\s*//\s*No need to parse"
)


def extract_parsed_ids(path: Path, style: int, action: int, node: str) -> list[int]:
    """Which parameter ids the executor's parser actually branches on.

    Declaring a parameter and *doing something with it* are different things,
    and the difference matters: OCUDU declares thirteen for the slice-quota
    action but treats 1, 3, 5 and 8 as pure structure. An encoder that omitted
    a declared-but-structural id would still be parsed; one that omitted a
    load-bearing id would silently lose a field.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = f"e2sm_rc_control_action_{style}_{action}_{node.lower()}_executor::" \
             "parse_action_ran_parameter_value"
    start = text.find(marker)
    if start < 0:
        return []
    nxt = text.find("\n}\n", start)
    body = text[start : nxt if nxt > 0 else len(text)]
    return sorted({int(i) for i in PARSE_ID_RE.findall(body)})


def extract_structural_ids(path: Path, style: int, action: int, node: str) -> list[int]:
    """Ids the executor explicitly declines to parse — nesting, not values."""
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = f"e2sm_rc_control_action_{style}_{action}_{node.lower()}_executor::" \
             "parse_action_ran_parameter_value"
    start = text.find(marker)
    if start < 0:
        return []
    nxt = text.find("\n}\n", start)
    body = text[start : nxt if nxt > 0 else len(text)]
    out: set[int] = set()
    for m in NOOP_ARM_RE.finditer(body):
        out.update(int(i) for i in PARSE_ID_RE.findall(m.group("ids")))
    return sorted(out)


def extract_ccc() -> dict[str, Any]:
    styles_src = (CCC_DIR / "e2sm_ccc_control_service_impl.h").read_text(
        encoding="utf-8", errors="replace"
    )
    packer_src = (CCC_DIR / "e2sm_ccc_asn1_packer.cpp").read_text(
        encoding="utf-8", errors="replace"
    )
    styles = [
        {
            "style_id": int(sid),
            "style_name": name,
            "ric_action_format": int(a),
            "ric_control_header_format": int(b),
            "ric_control_message_format": int(c),
        }
        for sid, name, a, b, c in CCC_STYLE_RE.findall(styles_src)
    ]
    return {
        "service_model": "E2SM-CCC",
        "styles": styles,
        "ran_configuration_structures": sorted(set(CCC_STRUCT_RE.findall(packer_src))),
        "writable_attributes": sorted(set(CCC_ATTR_RE.findall(packer_src))),
    }


def build() -> dict[str, Any]:
    missing = [str(p.relative_to(REPO)) for p in SOURCES if not p.exists()]
    if missing:
        raise SystemExit(
            "OCUDU sources not present — is the submodule initialised?\n  "
            + "\n  ".join(missing)
        )

    rc_actions: list[dict[str, Any]] = []
    for path in (SOURCES[0], SOURCES[1]):
        for entry in extract_rc(path):
            entry["parsed_parameter_ids"] = extract_parsed_ids(
                path, entry["style_id"], entry["action_id"], entry["node"]
            )
            structural = extract_structural_ids(
                path, entry["style_id"], entry["action_id"], entry["node"]
            )
            entry["structural_parameter_ids"] = structural
            entry["value_parameter_ids"] = sorted(
                set(entry["parsed_parameter_ids"]) - set(structural)
            )
            rc_actions.append(entry)
    rc_actions.sort(key=lambda a: (a["node"], a["style_id"], a["action_id"]))

    return {
        "generated_by": "ocudu/extract_catalogue.py",
        "ocudu_commit": ocudu_commit(),
        "source_digests": {
            str(p.relative_to(REPO)): sha256(p) for p in SOURCES
        },
        "e2sm_rc_control_actions": rc_actions,
        "e2sm_ccc": extract_ccc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    catalogue = build()
    text = json.dumps(catalogue, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text, end="")

    for action in catalogue["e2sm_rc_control_actions"]:
        print(
            f"  {action['node']} E2SM-RC style {action['style_id']} "
            f"action {action['action_id']}: {action['action_name']!r} "
            f"({len(action['ran_parameters'])} params, "
            f"{len(action['value_parameter_ids'])} carry values, "
            f"{len(action['structural_parameter_ids'])} structural)",
            file=sys.stderr,
        )
    ccc = catalogue["e2sm_ccc"]
    print(
        f"  E2SM-CCC: {len(ccc['styles'])} styles, structures "
        f"{ccc['ran_configuration_structures']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
