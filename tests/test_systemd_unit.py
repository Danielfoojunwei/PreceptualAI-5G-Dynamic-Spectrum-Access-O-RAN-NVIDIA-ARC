"""Verify the Jetson Orin Nano systemd unit + timer + soak script.

Row 26 of GAPS_TO_PILOT.md: "systemd_unit_runs_on_jetson_orin_nano_for_24h".

We can't boot a real systemd on the test runner, so this test parses the unit
files and asserts that the load-bearing constraints are present and correct:

  * `Type=notify` + `WatchdogSec=30s`              — sd_notify watchdog wired up.
  * `CPUAffinity=0 1`                              — 2-core Orin Nano envelope.
  * `MemoryMax=8G`                                 — Orin Nano RAM ceiling.
  * `User=horizon` (NOT root)                      — non-root.
  * `NoNewPrivileges=true`, `ProtectSystem=strict` — hardening.
  * `ReadWritePaths` does NOT include `/`, `/etc`, `/usr` — minimal write surface.
  * Timer fires daily and is persistent.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
SERVICE_FILE = SYSTEMD_DIR / "horizon-ric-orin.service"
TIMER_FILE = SYSTEMD_DIR / "horizon-ric-orin-soak.timer"
SOAK_SERVICE_FILE = SYSTEMD_DIR / "horizon-ric-orin-soak.service"
SOAK_SCRIPT = SYSTEMD_DIR / "horizon-soak-1h.sh"


def _parse_unit(path: Path) -> configparser.ConfigParser:
    """systemd unit files are INI-ish but allow duplicate keys (e.g. OnCalendar)."""
    cp = configparser.ConfigParser(strict=False, allow_no_value=True,
                                   interpolation=None)
    cp.optionxform = str  # preserve case
    cp.read(path, encoding="utf-8")
    return cp


# ---------- service unit ----------

def test_service_file_exists() -> None:
    assert SERVICE_FILE.is_file(), f"missing {SERVICE_FILE}"


def test_service_watchdog_30s() -> None:
    cfg = _parse_unit(SERVICE_FILE)
    assert cfg["Service"]["Type"].strip() == "notify"
    assert cfg["Service"]["WatchdogSec"].strip() == "30s"


def test_service_cpu_affinity_two_cores() -> None:
    cfg = _parse_unit(SERVICE_FILE)
    affinity = cfg["Service"]["CPUAffinity"].strip()
    # Accept "0 1", "0,1", or "0-1" — all valid systemd forms for 2 cores.
    assert affinity in {"0 1", "0,1", "0-1"}, f"unexpected CPUAffinity={affinity!r}"


def test_service_memory_max_8g() -> None:
    cfg = _parse_unit(SERVICE_FILE)
    assert cfg["Service"]["MemoryMax"].strip() == "8G"
    assert cfg["Service"]["MemoryHigh"].strip() == "7G"


def test_service_runs_as_horizon_not_root() -> None:
    cfg = _parse_unit(SERVICE_FILE)
    user = cfg["Service"]["User"].strip()
    assert user == "horizon"
    assert user != "root"


def test_service_hardening_flags() -> None:
    cfg = _parse_unit(SERVICE_FILE)
    s = cfg["Service"]
    assert s["NoNewPrivileges"].strip().lower() == "true"
    assert s["ProtectSystem"].strip() == "strict"
    assert s["ProtectHome"].strip().lower() in {"yes", "true"}
    assert s["PrivateTmp"].strip().lower() in {"yes", "true"}


def test_service_readwritepaths_minimal() -> None:
    """Writable paths must NOT include the root, /etc, or /usr."""
    cfg = _parse_unit(SERVICE_FILE)
    rwp = cfg["Service"]["ReadWritePaths"].strip()
    paths = rwp.split()
    for forbidden in ("/", "/etc", "/etc/", "/usr", "/usr/"):
        assert forbidden not in paths, (
            f"ReadWritePaths must not include {forbidden!r}; got {paths}")
    assert "/var/lib/horizon-ric" in paths
    assert "/var/log/horizon-ric" in paths


def test_service_install_target() -> None:
    cfg = _parse_unit(SERVICE_FILE)
    assert cfg["Install"]["WantedBy"].strip() == "multi-user.target"


def test_service_exec_uses_horizon_ric_cli() -> None:
    text = SERVICE_FILE.read_text(encoding="utf-8")
    assert "ExecStart=/usr/bin/python3 -m horizon_ric.cli serve --bind 127.0.0.1:8083" in text


# ---------- timer ----------

def test_timer_fires_daily_and_persistent() -> None:
    text = TIMER_FILE.read_text(encoding="utf-8")
    # Either "OnCalendar=daily" or a daily wall-clock spec must appear.
    assert "OnCalendar=daily" in text or "OnCalendar=*-*-*" in text
    assert "Persistent=true" in text
    cfg = _parse_unit(TIMER_FILE)
    assert cfg["Install"]["WantedBy"].strip() == "timers.target"


# ---------- soak script ----------

def test_soak_script_pins_to_two_cores() -> None:
    assert SOAK_SCRIPT.is_file()
    body = SOAK_SCRIPT.read_text(encoding="utf-8")
    assert "taskset -c 0-1" in body
    assert "scripts/soak_24h.py" in body
    assert "--duration-min 60" in body
    assert "--speedup 24" in body
