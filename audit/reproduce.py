#!/usr/bin/env python3
"""One-command reproduction front door for Horizon-RIC's enforcement gates.

A third party should not have to read five CI workflow files to learn which of
~15 gate scripts to run, in what order, against which committed result. This is
the front door. It reads audit/MANIFEST.json — the machine-readable catalogue of
every verification gate invoked in CI — and drives them.

    python audit/reproduce.py            run every OFFLINE gate (requirement
                                         "none"): pure stdlib + committed data,
                                         no Horizon install, Docker, submodule
                                         or network. Prints a PASS/SKIP/FAIL
                                         table; exits non-zero if any RAN gate
                                         failed.
    python audit/reproduce.py --list     print the manifest as a table and exit
                                         0, running nothing.
    python audit/reproduce.py --all      attempt every gate, discovering at
                                         runtime whether each requirement is
                                         met on this host and SKIPPING (loudly,
                                         with a reason) those that are not.
    python audit/reproduce.py --only ID ...   run only the named gate ids.

Requirement discovery is done at runtime, never assumed:

    none                 always runnable.
    horizon              `import horizon_ric` must succeed.
    docker               a `docker` binary AND a reachable daemon.
    submodule:<name>     third_party/<name> must be a non-empty checkout.
    network              never attempted automatically (a gate in this class
                         needs a multi-GB dataset (re)download); always skipped
                         with the manual command stated in README_reproduce.md.

A skip is always loud and always carries a reason. The runner never turns an
unmet requirement into a silent pass, and never reports success on a red gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MANIFEST_PATH = HERE / "MANIFEST.json"

# Gates that live outside src/ need their package roots on PYTHONPATH, exactly
# as the CI workflows set them. Recorded here so a standalone run matches CI.
EXTRA_PYTHONPATH = ("src", "agentic/src", "ocudu/src")


# --------------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------------- #

REQUIRED_KEYS = (
    "id",
    "title",
    "requirement",
    "command",
    "result_file",
    "ci_workflow",
    "notes",
)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "gates" not in data:
        raise ValueError(f"{path} is not a valid reproduce manifest")
    return data


def gates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest["gates"])


# --------------------------------------------------------------------------- #
# Requirement discovery (runtime, never assumed)
# --------------------------------------------------------------------------- #


def _horizon_importable() -> bool:
    # Probe in a child process so the parent's import state and PYTHONPATH
    # tweaks cannot mask a missing install.
    env = _child_env()
    proc = subprocess.run(
        [sys.executable, "-c", "import horizon_ric"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
    )
    return proc.returncode == 0


def _docker_available() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "no docker binary on PATH"
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker present but daemon unreachable ({exc})"
    if proc.returncode != 0:
        return False, "docker present but daemon not reachable (docker info failed)"
    return True, "docker daemon reachable"


def _submodule_path(name: str) -> Path:
    return REPO_ROOT / "third_party" / name


def _submodule_checked_out(name: str) -> tuple[bool, str]:
    path = _submodule_path(name)
    if not path.is_dir():
        return False, f"submodule third_party/{name} not present"
    # A registered-but-uninitialised submodule is an empty directory.
    if not any(path.iterdir()):
        return False, (
            f"submodule third_party/{name} empty "
            f"(run: git submodule update --init third_party/{name})"
        )
    return True, f"submodule third_party/{name} checked out"


def requirement_status(requirement: str) -> tuple[bool, str]:
    """Return (satisfied, human reason) for a requirement string."""
    if requirement == "none":
        return True, "no external requirement"
    if requirement == "horizon":
        if _horizon_importable():
            return True, "horizon_ric importable"
        return False, "horizon_ric not importable (run: pip install -e .)"
    if requirement == "docker":
        return _docker_available()
    if requirement.startswith("submodule:"):
        return _submodule_checked_out(requirement.split(":", 1)[1])
    if requirement == "network":
        return False, (
            "network-class gate: requires a dataset (re)download; not attempted "
            "automatically (see README_reproduce.md for the manual command)"
        )
    return False, f"unknown requirement class {requirement!r}"


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class Outcome:
    gate_id: str
    status: str  # PASS / FAIL / SKIP
    reason: str
    detail: str = ""


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT / p) for p in EXTRA_PYTHONPATH]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _resolve_argv(argv: list[str], tmp: Path) -> list[str]:
    # {FRESH}  -> a fresh output FILE inside the per-gate temp dir.
    # {TMPDIR} -> the per-gate temp DIRECTORY (for gates emitting several files).
    fresh = str(tmp / "fresh.json")
    resolved: list[str] = []
    for token in argv:
        resolved.append(token.replace("{FRESH}", fresh).replace("{TMPDIR}", str(tmp)))
    return resolved


def _missing_probe_imports(modules: list[str], env: dict[str, str]) -> str | None:
    """Return the first import that fails in a child process, or None."""
    for module in modules:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
        )
        if proc.returncode != 0:
            return module
    return None


def _script_of(command: list[str]) -> Path:
    return REPO_ROOT / command[0]


def run_gate(gate: dict[str, Any]) -> Outcome:
    gate_id = gate["id"]
    requirement = gate["requirement"]

    satisfied, reason = requirement_status(requirement)
    if not satisfied:
        return Outcome(gate_id, SKIP, reason)

    script = _script_of(gate["command"])
    if not script.exists():
        note = "companion artifact built separately" if gate.get("companion") else ""
        detail = f" ({note})" if note else ""
        return Outcome(
            gate_id, SKIP, f"script not present on disk: {gate['command'][0]}{detail}"
        )

    env = _child_env()

    probe = gate.get("probe_imports")
    if probe:
        missing = _missing_probe_imports(probe, env)
        if missing is not None:
            return Outcome(
                gate_id,
                SKIP,
                f"runtime module not importable: {missing} "
                f"(install this gate's runtime; see MANIFEST notes)",
            )

    with tempfile.TemporaryDirectory(prefix=f"repro-{gate_id}-") as td:
        tmp = Path(td)

        regen = gate.get("regen")
        if regen:
            regen_argv = [sys.executable, *_resolve_argv(regen, tmp)]
            regen_proc = subprocess.run(
                regen_argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True
            )
            if regen_proc.returncode != 0:
                return Outcome(
                    gate_id,
                    FAIL,
                    "regeneration step failed",
                    detail=(regen_proc.stderr or regen_proc.stdout).strip(),
                )

        argv = [sys.executable, *_resolve_argv(gate["command"], tmp)]
        proc = subprocess.run(
            argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True
        )
        if proc.returncode == 0:
            summary = _last_line(proc.stdout) or _last_line(proc.stderr)
            return Outcome(gate_id, PASS, "verified", detail=summary)
        return Outcome(
            gate_id,
            FAIL,
            f"exit {proc.returncode}",
            detail=(proc.stderr or proc.stdout).strip(),
        )


def _last_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


# --------------------------------------------------------------------------- #
# Selection and reporting
# --------------------------------------------------------------------------- #


def select_gates(
    manifest: dict[str, Any], *, mode: str, only: list[str] | None
) -> list[dict[str, Any]]:
    all_gates = gates(manifest)
    if only:
        by_id = {g["id"]: g for g in all_gates}
        missing = [gid for gid in only if gid not in by_id]
        if missing:
            raise SystemExit(f"unknown gate id(s): {', '.join(missing)}")
        return [by_id[gid] for gid in only]
    if mode == "all":
        return all_gates
    # default: offline set == requirement "none"
    return [g for g in all_gates if g["requirement"] == "none"]


def _fmt_row(cols: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cols, widths))


def print_list(manifest: dict[str, Any]) -> None:
    rows = [
        [g["id"], g["requirement"], g.get("ci_workflow") or "(not in CI)", _cmd_str(g)]
        for g in gates(manifest)
    ]
    header = ["ID", "REQUIREMENT", "CI WORKFLOW", "COMMAND"]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))
    ]
    print(_fmt_row(header, widths))
    print(_fmt_row(["-" * w for w in widths], widths))
    for row in rows:
        print(_fmt_row(row, widths))
    print(f"\n{len(rows)} gates catalogued in {MANIFEST_PATH.relative_to(REPO_ROOT)}")


def _cmd_str(gate: dict[str, Any]) -> str:
    prefix = "python " if gate["command"][0].endswith(".py") else ""
    regen = ""
    if gate.get("regen"):
        regen = "python " + " ".join(gate["regen"]) + " ; "
    return regen + prefix + " ".join(gate["command"])


def print_report(outcomes: list[Outcome]) -> None:
    header = ["GATE", "STATUS", "DETAIL"]
    rows = [[o.gate_id, o.status, o.reason] for o in outcomes]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))
    ]
    print(_fmt_row(header, widths))
    print(_fmt_row(["-" * w for w in widths], widths))
    for o, row in zip(outcomes, rows):
        print(_fmt_row(row, widths))
    n_pass = sum(o.status == PASS for o in outcomes)
    n_fail = sum(o.status == FAIL for o in outcomes)
    n_skip = sum(o.status == SKIP for o in outcomes)
    print()
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nFailures:")
        for o in outcomes:
            if o.status == FAIL:
                print(f"  - {o.gate_id}: {o.reason}")
                if o.detail:
                    tail = o.detail.splitlines()[-6:]
                    for line in tail:
                        print(f"      {line}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list", action="store_true", help="print the manifest and exit (runs nothing)"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="attempt every gate, skipping those whose requirements are unmet",
    )
    group.add_argument(
        "--only", nargs="+", metavar="ID", help="run only the named gate id(s)"
    )
    args = parser.parse_args(argv)

    manifest = load_manifest()

    if args.list:
        print_list(manifest)
        return 0

    mode = "all" if args.all else "offline"
    selected = select_gates(manifest, mode=mode, only=args.only)

    if mode == "offline" and not args.only:
        print(
            f"Reproducing {len(selected)} offline gate(s) "
            "(requirement=none: stdlib + committed data). "
            "Use --all to attempt every gate, --list to see the catalogue.\n"
        )

    outcomes = [run_gate(g) for g in selected]
    print_report(outcomes)

    # Exit non-zero if any gate that actually RAN failed. Skips never fail.
    return 1 if any(o.status == FAIL for o in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
