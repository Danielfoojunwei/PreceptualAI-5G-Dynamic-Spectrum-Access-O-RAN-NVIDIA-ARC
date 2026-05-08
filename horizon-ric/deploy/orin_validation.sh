#!/usr/bin/env bash
# deploy/orin_validation.sh — Row 26 delivery-time validation gate.
#
# Detects whether running on real Jetson Orin Nano (via /proc/device-tree/model)
# or on a substitute aarch64 host. Runs the 24-h-equivalent soak + edge
# benchmark in the appropriate envelope, parses the 5 Row 26 acceptance bars
# from the resulting JSON, and exits 0 iff all 5 pass.
#
# Usage:
#   sudo bash deploy/orin_validation.sh                  # full real run
#        bash deploy/orin_validation.sh --dry-run        # parse-only against
#                                                       # deploy/SOAK_24H_PROOF.md
#   sudo bash deploy/orin_validation.sh --soak-json X.json --edge-json Y.json
set -uo pipefail

# Acceptance bar thresholds (Orin envelope, per deploy/SLO.md row 2a)
A1_SUCCESS_MIN=99.5            # %
P99_LATENCY_MAX=250.0          # ms
WATCHDOG_SILENCE_MAX=30.0      # s
AUDIT_CHAIN_REQUIRED=1440      # verifies, all-pass

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
SOAK_JSON=""
EDGE_JSON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=1; shift;;
        --soak-json)       SOAK_JSON="$2"; shift 2;;
        --edge-json)       EDGE_JSON="$2"; shift 2;;
        -h|--help)         sed -n '2,12p' "$0"; exit 0;;
        *)                 echo "unknown arg: $1" >&2; exit 2;;
    esac
done

# Substrate detection
SUBSTRATE="substitute"
if [[ -r /proc/device-tree/model ]]; then
    MODEL="$(tr -d '\0' < /proc/device-tree/model)"
    if [[ "$MODEL" == *"Jetson Orin Nano"* ]] || [[ "$MODEL" == *"tegra"* ]]; then
        SUBSTRATE="real-orin"
    fi
fi
echo "[orin-validate] substrate=$SUBSTRATE  dry_run=$DRY_RUN"

# Run the soak unless we already have JSON
if [[ -z "$SOAK_JSON" && $DRY_RUN -eq 0 ]]; then
    SOAK_JSON="/tmp/orin_validate_soak.json"
    if [[ "$SUBSTRATE" == "real-orin" ]]; then
        command -v jetson_clocks >/dev/null 2>&1 && sudo jetson_clocks || true
        command -v nvpmodel       >/dev/null 2>&1 && sudo nvpmodel -m 0  || true
        "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/soak_24h.py" \
            --duration-min 12 --speedup 120 --json-out "$SOAK_JSON"
    else
        taskset -c 0-1 prlimit --as=8589934592 \
            "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/soak_24h.py" \
            --duration-min 12 --speedup 120 --json-out "$SOAK_JSON"
    fi
fi

# Parser: extract the 5 bars from JSON OR from the canonical proof markdown
parse_and_check() {
    local input="$1"
    "$REPO_ROOT/.venv/bin/python" - "$input" <<'PY'
import json, re, sys
path = sys.argv[1]
text = open(path).read()
data = {}
try:
    data = json.loads(text)
except Exception:
    pass

def num(pattern, default=None):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

a1   = data.get("a1_success_pct")          or num(r"A1 emit success rate[^\d]+([\d.]+)\s*%")
p99  = data.get("p99_decision_latency_ms") or num(r"p99[^\d]+([\d.]+)\s*ms")
chain= data.get("audit_verifies_passed")   or num(r"(\d{3,5})\s*/\s*\1\s*verify=True")
wdog = data.get("max_watchdog_silence_s")  or num(r"watchdog silence[^\d]+([\d.]+)\s*s")
unh  = data.get("unhandled_exceptions",    0)

bars = [
    ("audit_chain_integrity",  chain is not None and chain >= 1440,                      f"{int(chain) if chain else '??'}/1440"),
    ("a1_success_pct",          a1 is not None and a1 >= 99.5,                            f"{a1:.2f}%"     if a1 else "??"),
    ("p99_latency_ms",          p99 is not None and p99 <= 250.0,                         f"{p99:.2f}ms"   if p99 else "??"),
    ("watchdog_silence_s",      wdog is not None and wdog <= 30.0,                        f"{wdog:.2f}s"   if wdog else "??"),
    ("zero_unhandled_exc",      int(unh) == 0,                                            f"{int(unh)}"),
]
all_ok = all(ok for _, ok, _ in bars)
for name, ok, val in bars:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} = {val}")
print("OVERALL:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
PY
}

# Choose what to feed the parser
if [[ $DRY_RUN -eq 1 ]]; then
    SOURCE="${SOAK_JSON:-$REPO_ROOT/deploy/SOAK_24H_PROOF.md}"
else
    SOURCE="$SOAK_JSON"
fi

echo "[orin-validate] parsing acceptance bars from: $SOURCE"
parse_and_check "$SOURCE"
RC=$?

if [[ $RC -eq 0 ]]; then
    echo "[orin-validate] Row 26 acceptance: PASS  (substrate=$SUBSTRATE)"
else
    echo "[orin-validate] Row 26 acceptance: FAIL  (substrate=$SUBSTRATE) — see docs/HARDWARE_PROCUREMENT.md §rollback"
fi
exit $RC
