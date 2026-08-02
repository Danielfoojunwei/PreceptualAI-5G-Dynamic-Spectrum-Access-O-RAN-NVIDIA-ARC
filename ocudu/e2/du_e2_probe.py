#!/usr/bin/env python3
"""Does the OCUDU DU-side E2 agent join the RIC? Settle it on the binaries.

Why this exists
---------------
Gate G4 needs "a RIC Control Request derived from a signed certificate
accepted by an E2 node". Horizon encodes E2SM-RC **Style 2, Action 6**
(slice-level PRB quota). Reading OCUDU's factories:

* ``lib/e2/common/e2_cu_cp_factory.cpp`` registers control **style 3**
  (``e2sm_rc_control_action_3_1_cu_executor`` — handover control).
* ``lib/e2/common/e2_du_factory.cpp`` registers control **style 2, action 6**
  (``e2sm_rc_control_action_2_6_du_executor``).

So the executor for the action Horizon encodes lives in the **DU**, and
``OCUDU_E2_JOIN_PROOF.md`` records that only the CU-CP association was ever
observed. Whether the DU agent joins therefore decides whether G4 is
attemptable at all on this stack.

Static source reading has produced a wrong answer about this subsystem twice
(once about ``add_subcommand``, once about whether the DU E2 schema is
registered at all — both were withdrawn). This probe answers it by running
the real binaries and reading what each end says, which is the only evidence
that settles it.

It reports what it sees. It does not assume a DU join is possible, and it does
not treat its absence as a failure of this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

DEFAULT_RIC = REPO / "third_party" / "flexric" / "build" / "examples" / "ric" / "nearRT-RIC"
DEFAULT_GNB = REPO / "third_party" / "ocudu" / "build" / "apps" / "gnb" / "gnb"
DEFAULT_CFG = REPO / "deploy" / "e2-companion" / "ocudu-gnb-e2.yml"

# The RIC prints one of these per E2 node that completes setup.
_SETUP_RX = re.compile(
    r"E2 SETUP-REQUEST rx from PLMN\s+(?P<plmn>[\d.\s]+)\s+Node ID (?P<node>\d+)"
    r"\s+RAN type (?P<rantype>\S+)"
)
_ACCEPT_FN = re.compile(r"Accepting RAN function ID (?P<id>\d+) with def = (?P<def>\S+)")
# The gNB logs one line per E2 agent that establishes a connection.
_GNB_CONN = re.compile(r"\[(?P<unit>E2-[A-Z-]+)\].*E2 Setup procedure successful")
_GNB_ADDED = re.compile(
    r"\[(?P<unit>E2-[A-Z-]+)\].*Added supported RAN function with id (?P<id>\d+) "
    r"and OID (?P<oid>[\d.]+)"
)
_GNB_TRY = re.compile(r"\[(?P<unit>E2-[A-Z-]+)\].*Trying to establish E2 connection")


def _popen(cmd: list[str], log: Path, shim: Path | None) -> subprocess.Popen:
    env = dict(os.environ)
    if shim is not None:
        env["LD_PRELOAD"] = str(shim)
    fh = log.open("w", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )


def _kill(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001 — best-effort teardown
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def analyse(ric_log: str, gnb_log: str) -> dict[str, Any]:
    """What each end says happened."""
    nodes = [m.groupdict() for m in _SETUP_RX.finditer(ric_log)]
    functions = [m.groupdict() for m in _ACCEPT_FN.finditer(ric_log)]

    gnb_units_connected = sorted({m.group("unit") for m in _GNB_CONN.finditer(gnb_log)})
    gnb_units_tried = sorted({m.group("unit") for m in _GNB_TRY.finditer(gnb_log)})
    gnb_functions = [m.groupdict() for m in _GNB_ADDED.finditer(gnb_log)]

    ran_types = sorted({n["rantype"] for n in nodes})
    # A DU association shows up either as a distinct RAN type at the RIC or as
    # a second E2 unit at the gNB. Both are checked because the RIC's RAN-type
    # string is a FlexRIC rendering, not a wire field we control.
    du_at_ric = any("du" in t.lower() for t in ran_types)
    du_at_gnb = any("DU" in u for u in gnb_units_connected)

    return {
        "ric": {
            "e2_setup_requests": len(nodes),
            "ran_types": ran_types,
            "nodes": nodes,
            "accepted_ran_functions": functions,
        },
        "gnb": {
            "e2_units_that_tried": gnb_units_tried,
            "e2_units_that_connected": gnb_units_connected,
            "ran_functions_added": gnb_functions,
        },
        "du_agent_joined": bool(du_at_ric or du_at_gnb),
        "du_evidence": {
            "distinct_du_ran_type_at_ric": du_at_ric,
            "du_unit_connected_at_gnb": du_at_gnb,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ric", type=Path, default=DEFAULT_RIC)
    p.add_argument("--gnb", type=Path, default=DEFAULT_GNB)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--shim", type=Path, default=None, help="LD_PRELOAD SCTP shim")
    p.add_argument("--settle-s", type=float, default=25.0)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--out", type=Path, help="write the result JSON here")
    args = p.parse_args(argv)

    for name, path in (("RIC", args.ric), ("gNB", args.gnb), ("config", args.config)):
        if not path.exists():
            print(f"error: {name} not found at {path}", file=sys.stderr)
            return 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    ric_log = args.workdir / "ric.log"
    gnb_log = args.workdir / "gnb.log"

    ric = gnb = None
    try:
        ric = _popen([str(args.ric)], ric_log, args.shim)
        time.sleep(4.0)
        if ric.poll() is not None:
            print("error: the RIC exited immediately", file=sys.stderr)
            print(ric_log.read_text(errors="replace")[-2000:], file=sys.stderr)
            return 3

        gnb = _popen([str(args.gnb), "-c", str(args.config)], gnb_log, args.shim)

        # Wait until both ends stop producing new E2 lines, or the budget runs
        # out. A fixed sleep would either be slow or race the second agent.
        deadline = time.monotonic() + args.settle_s
        last = ""
        stable_for = 0.0
        while time.monotonic() < deadline:
            time.sleep(1.0)
            now = ric_log.read_text(errors="replace") + gnb_log.read_text(
                errors="replace"
            )
            if now == last:
                stable_for += 1.0
                if stable_for >= 6.0:
                    break
            else:
                stable_for = 0.0
            last = now

        result = analyse(
            ric_log.read_text(errors="replace"), gnb_log.read_text(errors="replace")
        )
    finally:
        _kill(gnb)
        _kill(ric)

    result["probe"] = "ocudu_du_e2_agent_join"
    result["binaries"] = {
        "ric": str(args.ric),
        "gnb": str(args.gnb),
        "config": str(args.config),
        "sctp_shim": str(args.shim) if args.shim else None,
    }
    result["logs"] = {"ric": str(ric_log), "gnb": str(gnb_log)}

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)

    r = result["ric"]
    print(
        f"\nRIC saw {r['e2_setup_requests']} E2 setup(s), RAN types {r['ran_types']}; "
        f"gNB connected units {result['gnb']['e2_units_that_connected']}; "
        f"DU agent joined = {result['du_agent_joined']}",
        file=sys.stderr,
    )
    # Exit 0 either way: the probe's job is to report, not to pass.
    return 0


if __name__ == "__main__":
    if shutil.which("gcc") is None:  # pragma: no cover - informational only
        pass
    raise SystemExit(main())
