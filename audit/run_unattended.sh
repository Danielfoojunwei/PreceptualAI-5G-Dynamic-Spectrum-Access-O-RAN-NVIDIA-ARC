#!/bin/bash
# One-command launcher for the §VII "runs unattended" demonstration.
#
# This is the single local command the proposal implies exists. Upstream the
# three steps are spelled out separately in deploy/xapp-e2e/README.md; this
# wrapper chains them so a reviewer runs ONE thing. It is a thin, honest
# orchestrator — it builds nothing itself, it delegates to the committed
# run_stack.sh and the committed proof scripts.
#
# WHAT IT DOES (live), in order:
#   1. deploy/xapp-e2e/run_stack.sh — clone+build the pinned O-RAN-SC RMR C
#      library, the Go A1 mediator, and the hw-python reference xApp from
#      source, start redis + mediator + xApp, wait for the mediator northbound.
#   2. scripts/run_horizon_rapp.py — drive the full Horizon pipeline
#      (telemetry -> planner -> Shield -> A1 emit) for 12 replay events against
#      the live mediator, writing a pipeline report.
#   3. scripts/xapp_e2e_proof.py — cross-check three independent witnesses
#      (pipeline report, mediator enforceStatus, xApp RMR ACK log) and emit the
#      proof JSON.
#
# HONEST PREREQUISITES (this DOES fail closed if they are absent):
#   * Network egress to https://gerrit.o-ran-sc.org (clones RMR / a1 / hw-python).
#   * Build toolchain: gcc, cmake, make, Go >= 1.21.
#   * redis-server on PATH.
#   * A Python 3.11 interpreter for the xApp venv ONLY (hiredis' setup.py
#     imports the stdlib `imp`, removed in 3.12). Point XAPP_PYTHON at it; the
#     Horizon pipeline itself runs on this script's PYTHON (3.12+ is fine).
#   * Free TCP ports: 10000 (mediator northbound), 4560/4562 (RMR), 6379 (redis).
#   * NO Docker is required for THIS path — run_stack.sh builds from source.
#     (Docker is only used by the separate osc-a1 simulator job, not here.)
#
# WHAT THIS WRAPPER DOES NOT DO:
#   * It does not exercise the "Shield-corrected over-power proposal refused"
#     step. That is a SEPARATE proof — deploy/xapp-e2e/a1_assurance_proof.py —
#     which runs against the OSC A1 simulator, not this mediator+xApp path.
#     See audit/UNATTENDED.md for how to run it and where its committed
#     evidence lives.
#
# If you CANNOT satisfy the prerequisites, do not fake it: read the committed
# proof at deploy/xapp-e2e/results/xapp-e2e-proof.json and gate it with
#   /home/user/venv/bin/python audit/check_unattended.py
# which verifies the committed evidence WITHOUT building anything.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/horizon-unattended}"
WORK="${XAPP_E2E_WORK:-$HOME/oran-deps}"

mkdir -p "$OUT_DIR"
echo ">> repo:    $REPO_ROOT"
echo ">> python:  $PYTHON ($($PYTHON --version 2>&1))"
echo ">> work:    $WORK"
echo ">> output:  $OUT_DIR"

echo ">> [1/3] Building + starting the real O-RAN-SC stack (run_stack.sh)…"
XAPP_E2E_WORK="$WORK" bash "$REPO_ROOT/deploy/xapp-e2e/run_stack.sh"

echo ">> [2/3] Driving the Horizon pipeline against the live mediator…"
HORIZON_A1_DIALECT=legacy \
HORIZON_NEAR_RT_RIC_URL=http://127.0.0.1:10000 \
HORIZON_ONCE_REQUIRE_ACCEPTED=1 \
"$PYTHON" "$REPO_ROOT/scripts/run_horizon_rapp.py" \
    --source-config "$REPO_ROOT/deploy/xapp-e2e/source-replay.yaml" \
    --once --once-max-events 12 \
    --report-json "$OUT_DIR/pipeline-report.json"

echo ">> [3/3] Cross-checking the three witnesses (xapp_e2e_proof.py)…"
"$PYTHON" "$REPO_ROOT/scripts/xapp_e2e_proof.py" \
    --a1-base-url http://127.0.0.1:10000 \
    --report "$OUT_DIR/pipeline-report.json" \
    --mediator-log "$WORK/a1mediator.log" \
    --xapp-log "$WORK/hwxapp.log" \
    --out "$OUT_DIR/xapp-e2e-proof.json"

echo ">> DONE. Fresh proof at: $OUT_DIR/xapp-e2e-proof.json"
echo ">> Compare against the committed proof:"
echo ">>   deploy/xapp-e2e/results/xapp-e2e-proof.json"
