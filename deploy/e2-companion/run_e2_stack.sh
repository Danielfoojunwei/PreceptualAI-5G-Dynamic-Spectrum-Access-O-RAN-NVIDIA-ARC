#!/bin/bash
# Build and run the real FlexRIC near-RT RIC E2 stack (nearRT-RIC + emulated
# gNB E2 agent + stock E2SM-KPM monitor xApp), capture live E2AP traffic, and
# drive one REAL E2SM-KPM RIC Indication through the Horizon bridge →
# DecisionPipeline → a real A1 endpoint.
#
# Components (pinned):
#   flexric            ef6d722f22191eea74089966983da1f5ec1fedd4 (gitlab.eurecom.fr/mosaic5g/flexric, dev)
#   mouse07410/asn1c   940dd5fa9f3917913fd487b13dfddfacd0ded06e (the fork FlexRIC's own CI pins)
#
# Transport disclosure: on kernels without SCTP (EPROTONOSUPPORT), the stack
# runs under the LD_PRELOAD shim sctp_udp_shim.c, which maps FlexRIC's
# one-to-many SCTP calls onto loopback UDP datagrams 1:1. All E2AP/E2SM bytes
# are produced/parsed by unmodified FlexRIC code. On a kernel WITH SCTP,
# leave HORIZON_E2_SHIM empty to run standards-conformant SCTP.
#
# Prereqs: gcc-13/cmake/make, libsctp-dev, libpcre2-dev, autotools+bison+flex
# (for asn1c), python venv with horizon-ric[oran] installed, root (sniffer).
set -euo pipefail

WORK="${E2_COMPANION_WORK:-$HOME/oran-deps}"
HARNESS="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="${HORIZON_VENV_PY:-$HOME/venv/bin/python}"
FLEXRIC_PIN=ef6d722f22191eea74089966983da1f5ec1fedd4
ASN1C_PIN=940dd5fa9f3917913fd487b13dfddfacd0ded06e
A1_URL="${HORIZON_NEAR_RT_RIC_URL:-http://127.0.0.1:10000}"
A1_DIALECT="${HORIZON_A1_DIALECT:-legacy}"
XAPP_SECONDS="${XAPP_SECONDS:-20}"
MAKE_J="${MAKE_J:-2}"

mkdir -p "$WORK"

# 1. asn1c — the fork+commit FlexRIC's own Dockerfile pins (needed for the
#    NR RRC decoders linked into the xApp SDK).
if [ ! -x /opt/asn1c/bin/asn1c ]; then
    git clone https://github.com/mouse07410/asn1c "$WORK/asn1c" 2>/dev/null || true
    git -C "$WORK/asn1c" checkout --detach "$ASN1C_PIN"
    ( cd "$WORK/asn1c" && autoreconf -iv && ./configure --prefix /opt/asn1c/ \
      && make -j"$MAKE_J" && sudo make install )
fi

# 2. FlexRIC, pinned, built for E2SM-KPM v3.00 (matching the committed
#    ASN.1 spec in src/horizon_ric/e2/asn1/).
if [ ! -d "$WORK/flexric" ]; then
    git clone https://gitlab.eurecom.fr/mosaic5g/flexric.git "$WORK/flexric"
fi
git -C "$WORK/flexric" checkout --detach "$FLEXRIC_PIN"
mkdir -p "$WORK/flexric/build"
( cd "$WORK/flexric/build" \
  && cmake -DKPM_VERSION=KPM_V3_00 -DCMAKE_BUILD_TYPE=Release \
           -DASN1C_EXEC=/opt/asn1c/bin/asn1c .. \
  && make -j"$MAKE_J" \
  && sudo make install ) 2>&1 | tee "$WORK/flexric-build.log" | tail -2

# 3. FlexRIC's own KPM self-tests (no transport needed).
( cd "$WORK/flexric/build" && ctest -R "KPM" --output-on-failure )

# 4. Transport shim (only used when the kernel lacks SCTP).
gcc -shared -fPIC -O2 "$HARNESS/sctp_udp_shim.c" -o "$WORK/sctp_udp_shim.so" -ldl
if "$VENV_PY" -c "import socket; socket.socket(socket.AF_INET, socket.SOCK_STREAM, 132)" 2>/dev/null; then
    SHIM=""
    echo "kernel has SCTP: running standards-conformant transport"
else
    SHIM="$WORK/sctp_udp_shim.so"
    echo "kernel lacks SCTP: using UDP transport shim (disclosed substitution)"
fi

# 5. Start the stack: sniffer -> nearRT-RIC -> E2 agent -> KPM monitor xApp.
pkill -9 -f 'nearRT-RIC' 2>/dev/null || true
pkill -9 -f 'emu_agent_gnb' 2>/dev/null || true
pkill -9 -f 'sniff_e2ap' 2>/dev/null || true
sleep 1
rm -f "$WORK/e2ap_capture.jsonl"
nohup "$VENV_PY" "$HARNESS/sniff_e2ap.py" "$WORK/e2ap_capture.jsonl" \
    > "$WORK/sniffer.log" 2>&1 &
sleep 1
( cd "$WORK/flexric/build" && LD_PRELOAD="$SHIM" nohup stdbuf -oL -eL \
    ./examples/ric/nearRT-RIC > "$WORK/flexric-nearrt-ric.log" 2>&1 & )
sleep 2
( cd "$WORK/flexric/build" && LD_PRELOAD="$SHIM" nohup stdbuf -oL -eL \
    ./examples/emulator/agent/emu_agent_gnb > "$WORK/flexric-emu-agent.log" 2>&1 & )
sleep 3
grep -q "E2 SETUP-REQUEST rx" "$WORK/flexric-nearrt-ric.log" \
    || { echo "E2 Setup did not complete"; exit 1; }
( cd "$WORK/flexric/build" && XAPP_DURATION="$XAPP_SECONDS" LD_PRELOAD="$SHIM" \
    timeout $((XAPP_SECONDS + 20)) stdbuf -oL -eL \
    ./examples/xApp/c/monitor/xapp_kpm_moni > "$WORK/flexric-kpm-xapp.log" 2>&1 ) || true
N_IND=$(grep -c "KPM ind_msg latency" "$WORK/flexric-kpm-xapp.log" || true)
echo "KPM monitor xApp received $N_IND indication(s)"
[ "$N_IND" -ge 1 ] || { echo "no KPM indications received"; exit 1; }

# 6. Bridge one live-captured indication into the Horizon pipeline → A1.
"$VENV_PY" "$HARNESS/bridge_capture_to_a1.py" \
    --e2ap-capture "$WORK/e2ap_capture.jsonl" \
    --a1-url "$A1_URL" --dialect "$A1_DIALECT" \
    --evidence "$WORK/e2-companion-evidence.jsonl" \
    --report-json "$WORK/e2-companion-report.json"

pkill -9 -f 'sniff_e2ap' 2>/dev/null || true
echo "done: report at $WORK/e2-companion-report.json"
