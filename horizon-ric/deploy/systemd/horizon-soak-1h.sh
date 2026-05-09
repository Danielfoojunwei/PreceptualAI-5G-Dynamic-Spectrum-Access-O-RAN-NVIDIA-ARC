#!/usr/bin/env bash
# horizon-soak-1h.sh — 1h wall × 24x speed-up = 24 simulated hours of soak.
#
# Called by horizon-ric-orin-soak.service (oneshot, fired daily by the timer).
# Pinned to CPUs 0-1 via taskset to mirror the main service's 2-core Orin
# Nano envelope, even when the unit's CPUAffinity= isn't enforced (e.g. when
# invoked by hand for verification).

set -euo pipefail

REPO_ROOT="${HORIZON_REPO_ROOT:-/opt/horizon}"
PYTHON="${HORIZON_PYTHON:-/usr/bin/python3}"
LOG_DIR="${HORIZON_SOAK_LOG_DIR:-/var/log/horizon-ric}"
STAMP="$(date +%Y%m%d)"
OUT="${LOG_DIR}/soak-${STAMP}.json"

mkdir -p "${LOG_DIR}"

echo "horizon-soak-1h: starting soak; output=${OUT}"

# 60 min wall × 24x speed-up = 24 simulated hours.
# taskset -c 0-1 pins to the same 2 cores as the main service.
exec taskset -c 0-1 "${PYTHON}" "${REPO_ROOT}/scripts/soak_24h.py" \
    --duration-min 60 \
    --speedup 24 \
    --output "${OUT}"
