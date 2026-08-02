"""Positive control for the DU E2 join probe.

The probe's headline result is a NEGATIVE: the OCUDU DU-side E2 agent never
attempts a connection. A negative from an instrument that cannot detect the
positive is worth nothing — it is the same trap the codec gate fell into,
where a corrupted payload "decoded successfully" because exit status carried
no information.

So this pins that the probe *would* have seen a DU join had one happened,
using the exact logger id OCUDU itself assigns.

That id is not guessed. ``apps/helpers/e2/e2_config_translators.h`` selects it
by SCTP PPID::

    const std::string logger_id = (ppid == E2_DU_PPID)   ? "E2-DU"
                                  : (ppid == E2_UP_PPID) ? "E2-CU-UP"
                                  : (ppid == E2_CP_PPID) ? "E2-CU-CP"
                                                         : "E2";

and ``test_du_logger_id_matches_ocudu_source`` reads it back out of that file,
so if upstream renames the logger this fails rather than silently making the
probe blind again.

Run with::

    PYTHONPATH=ocudu/e2 /home/user/venv/bin/python -m pytest \\
        ocudu/tests/test_du_e2_probe.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "ocudu" / "e2"
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))

from du_e2_probe import analyse  # noqa: E402

TRANSLATORS = (
    REPO / "third_party" / "ocudu" / "apps" / "helpers" / "e2"
    / "e2_config_translators.h"
)

# ocudulog pads logger names to a fixed width; [GNB     ] and [E2-CU-CP] are
# both 8 characters inside the brackets. A DU agent therefore logs as
# "[E2-DU   ]", and an unpadded regex would never see it.
DU_LINES = """2026-08-02T00:00:00.000000 [E2-DU   ] [D] Trying to establish E2 connection to Near-RT RIC (configured addrs 127.0.0.1, port 36421)...
2026-08-02T00:00:00.100000 [E2-DU   ] [I] E2 Setup procedure successful.
2026-08-02T00:00:00.100001 [E2-DU   ] [I] Added supported RAN function with id 3 and OID 1.3.6.1.4.1.53148.1.1.2.3
2026-08-02T00:00:00.200000 [E2-CU-CP] [I] E2 Setup procedure successful."""

RIC_DU_SETUP = (
    "[E2AP]: E2 SETUP-REQUEST rx from PLMN   1. 1 Node ID 411 "
    "RAN type ngran_gNB_DU"
)


@pytest.mark.skipif(
    not TRANSLATORS.exists(),
    reason="third_party/ocudu submodule not checked out",
)
def test_du_logger_id_matches_ocudu_source():
    """The name this test asserts on is OCUDU's, not one we invented."""
    text = TRANSLATORS.read_text(encoding="utf-8")
    assert re.search(r'ppid == E2_DU_PPID\)\s*\?\s*"E2-DU"', text), (
        "OCUDU no longer names the DU E2 logger 'E2-DU'; the probe's regexes "
        "and this test must be updated together or the probe goes blind"
    )


# ── the positive control ────────────────────────────────────────────────────


def test_probe_detects_a_du_join_when_one_happens():
    r = analyse(RIC_DU_SETUP, DU_LINES)
    assert r["du_agent_joined"] is True
    assert "E2-DU" in r["gnb"]["e2_units_that_tried"]
    assert "E2-DU" in r["gnb"]["e2_units_that_connected"]


def test_padding_is_tolerated():
    """The specific bug this guards: an unpadded regex misses every unit."""
    assert "E2-DU" in analyse("", DU_LINES)["gnb"]["e2_units_that_connected"]
    # And the same line without padding must still parse, so the regex is
    # tolerant rather than merely retuned for one width.
    unpadded = DU_LINES.replace("[E2-DU   ]", "[E2-DU]")
    assert "E2-DU" in analyse("", unpadded)["gnb"]["e2_units_that_connected"]


def test_ric_side_alone_is_sufficient_evidence():
    """Either witness can establish the join; they are not ANDed."""
    r = analyse(RIC_DU_SETUP, "")
    assert r["du_evidence"]["distinct_du_ran_type_at_ric"] is True
    assert r["du_agent_joined"] is True


def test_gnb_side_alone_is_sufficient_evidence():
    r = analyse("", DU_LINES)
    assert r["du_evidence"]["du_unit_connected_at_gnb"] is True
    assert r["du_agent_joined"] is True


# ── the negative it is used to support ──────────────────────────────────────


def test_cu_cp_only_reports_no_du_join():
    """The committed OCUDU_E2_JOIN_PROOF outcome, reproduced."""
    gnb = (
        "[E2-CU-CP] [D] Trying to establish E2 connection to Near-RT RIC "
        "(configured addrs 127.0.0.1, port 36421)...\n"
        "[E2-CU-CP] [I] E2 Setup procedure successful."
    )
    ric = "[E2AP]: E2 SETUP-REQUEST rx from PLMN   1. 1 Node ID 411 RAN type ngran_gNB"
    r = analyse(ric, gnb)
    assert r["du_agent_joined"] is False
    assert r["gnb"]["e2_units_that_tried"] == ["E2-CU-CP"]


def test_cu_up_is_not_mistaken_for_a_du():
    """CU-UP joins in the real runs; it must not be counted as the DU."""
    gnb = "[E2-CU-UP] [I] E2 Setup procedure successful."
    ric = (
        "[E2AP]: E2 SETUP-REQUEST rx from PLMN   1. 1 Node ID 411 "
        "RAN type ngran_gNB_CUUP"
    )
    r = analyse(ric, gnb)
    assert "E2-CU-UP" in r["gnb"]["e2_units_that_connected"]
    assert r["du_agent_joined"] is False, (
        "ngran_gNB_CUUP contains no 'du' and E2-CU-UP contains no 'DU' — if "
        "this ever passes, the DU test has become a substring accident"
    )
