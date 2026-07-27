#!/bin/bash
# Build and run the real O-RAN-SC near-RT RIC A1 stack + official hw-python
# xApp that the Horizon-RIC rApp drives over A1 in the xApp E2E proof.
#
# Components (all official O-RAN-SC source, pinned):
#   ric-plt/lib/rmr      8b9a214906a40b338def981d5c16b4ea247176fb  (RMR 4.9.4, C)
#   ric-plt/a1           09a757b4fd63198d8690d50b52bfd04552d47f1f  (Go A1 mediator)
#   ric-app/hw-python    a6d00525aa2f62f2457d84731da2b7d89e0b1013  (reference xApp)
#   ricxappframe         2.2.0 (PyPI, pinned by hw-python's setup.py)
#
# The mediator carries one local patch (patches/a1mediator-policy-resp-type.patch):
# upstream serialises policy_type_id as a JSON string in A1_POLICY_REQ
# (pkg/rmr/messages.go builds map[string]string) but Consume() asserts
# float64 on the echoed A1_POLICY_RESP and panics — the stock binary
# cannot parse the ACK to its own request. The patch accepts both
# encodings on receive. hw-python needs one config addition: its
# A1PolicyHandler reads config["xapp_name"], which the in-repo
# init/config-file.json does not define (the RIC helm configmap injects
# it in cluster deployments) — add `"xapp_name": "hw-python"`.
#
# Prereqs: gcc/cmake/make, Go >= 1.21, python3 venv, redis-server.
set -euo pipefail

WORK="${XAPP_E2E_WORK:-$HOME/oran-deps}"
HARNESS="$(cd "$(dirname "$0")" && pwd)"
RMR_PIN=8b9a214906a40b338def981d5c16b4ea247176fb
A1_PIN=09a757b4fd63198d8690d50b52bfd04552d47f1f
HW_PIN=a6d00525aa2f62f2457d84731da2b7d89e0b1013

mkdir -p "$WORK/rt" "$WORK/bin"
cp "$HARNESS/a1.rt" "$WORK/rt/a1.rt"
cp "$HARNESS/a1-frame-config.json" "$WORK/rt/a1-frame-config.json"

clone_pin() { # repo dir pin
    if [ ! -d "$WORK/$2" ]; then
        git clone "https://gerrit.o-ran-sc.org/r/$1" "$WORK/$2"
    fi
    git -C "$WORK/$2" fetch origin "$3" 2>/dev/null || git -C "$WORK/$2" fetch --unshallow origin || true
    git -C "$WORK/$2" checkout --detach "$3"
}

# 1. RMR (C library) — runtime lib + dev headers.
clone_pin ric-plt/lib/rmr ric-plt-lib-rmr "$RMR_PIN"
( cd "$WORK/ric-plt-lib-rmr" \
  && mkdir -p .build && cd .build \
  && cmake .. -DDEV_PKG=1 -DCMAKE_INSTALL_PREFIX=/usr/local && make -j"$(nproc)" install \
  && cd .. && mkdir -p .build-rt && cd .build-rt \
  && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && make -j"$(nproc)" install \
  && ldconfig )

# 2. Go A1 mediator (patched).
clone_pin ric-plt/a1 ric-plt-a1 "$A1_PIN"
( cd "$WORK/ric-plt-a1" \
  && git apply --check "$HARNESS/patches/a1mediator-policy-resp-type.patch" 2>/dev/null \
  && git apply "$HARNESS/patches/a1mediator-policy-resp-type.patch" || true \
  && go build -o "$WORK/bin/a1mediator" ./cmd/a1.go )

# 3. hw-python xApp (+ pinned ricxappframe from PyPI).
clone_pin ric-app/hw-python ric-app-hw-python "$HW_PIN"
python3 -m venv "$WORK/venv-xapp"
"$WORK/venv-xapp/bin/pip" install -q -U pip
"$WORK/venv-xapp/bin/pip" install -q -e "$WORK/ric-app-hw-python"
python3 - "$WORK/ric-app-hw-python/init/config-file.json" <<'EOF'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
cfg.setdefault("xapp_name", "hw-python")
json.dump(cfg, open(p, "w"), indent=4)
EOF

# 4. SDL backend (redis) + mediator + xApp.
redis-server --daemonize yes --port 6379 --bind 127.0.0.1 || true

kill "$(cat "$WORK/a1mediator.pid" 2>/dev/null)" 2>/dev/null || true
setsid nohup env \
  A1_CONFIG_FILE="$WORK/ric-plt-a1/config/config.yaml" \
  CFG_FILE="$WORK/rt/a1-frame-config.json" \
  RMR_SEED_RT="$WORK/rt/a1.rt" \
  DBAAS_SERVICE_HOST=127.0.0.1 DBAAS_SERVICE_PORT=6379 \
  LD_LIBRARY_PATH=/usr/local/lib \
  "$WORK/bin/a1mediator" > "$WORK/a1mediator.log" 2>&1 < /dev/null &
echo $! > "$WORK/a1mediator.pid"

kill "$(cat "$WORK/hwxapp.pid" 2>/dev/null)" 2>/dev/null || true
( cd "$WORK/ric-app-hw-python" && setsid nohup env \
  CONFIG_FILE="$WORK/ric-app-hw-python/init/config-file.json" \
  RMR_SEED_RT="$WORK/rt/a1.rt" \
  DBAAS_SERVICE_HOST=127.0.0.1 DBAAS_SERVICE_PORT=6379 \
  LD_LIBRARY_PATH=/usr/local/lib PYTHONUNBUFFERED=1 \
  "$WORK/venv-xapp/bin/python" -m src.main > "$WORK/hwxapp.log" 2>&1 < /dev/null & \
  echo $! > "$WORK/hwxapp.pid" )

# 5. Wait for the mediator northbound.
for _ in $(seq 1 30); do
    curl -fsS http://127.0.0.1:10000/A1-P/v2/healthcheck >/dev/null 2>&1 && break
    sleep 2
done
curl -fsS http://127.0.0.1:10000/A1-P/v2/healthcheck >/dev/null
echo "A1 mediator northbound live on :10000; hw-python xApp on RMR :4560."
