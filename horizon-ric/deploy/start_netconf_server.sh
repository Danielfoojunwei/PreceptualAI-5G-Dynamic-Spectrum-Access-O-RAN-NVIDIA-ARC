#!/usr/bin/env bash
#
# Bring up a real NETCONF/YANG server in user-space, with no docker
# daemon access required.
#
# Why not docker compose?
#   The repo's CI/edge target (Jetson Orin Nano) is hardened so the
#   developer is not in the docker group and `sudo` requires a password.
#   We therefore extract the upstream Ubuntu `netconfd` (yuma123,
#   BSD-3-clause, https://yuma123.org) packages into /tmp/netconf-local
#   and front them with a non-privileged OpenSSH `sshd` on port 8830,
#   which invokes /usr/sbin/netconf-subsystem on each session.
#
# This is REAL upstream netconfd — not a mock. The same binary is used
# by Open vSwitch, ZeroTier and other OEM appliances. Wire format is
# RFC 6241 (NETCONF) over RFC 6242 (SSH); RPC reply XML is what the
# server actually emits, not what PreceptualAI pretends.
#
# Usage:  bash deploy/start_netconf_server.sh        # foreground (^C to stop)
#         bash deploy/start_netconf_server.sh -d     # detach
#
# Stop:   pkill -f netconfd; pkill -f 'sshd.*8830'
#
set -euo pipefail

ROOT="/tmp/netconf-local"
SRV="/tmp/nc-server"
PORT="${HORIZON_O1_PORT:-8830}"

# 1. Fetch real upstream packages (idempotent).
mkdir -p /tmp/_debs && cd /tmp/_debs
need_pkgs=(netconfd libyuma2t64 libyuma-base libnetconf2-2t64 sysrepo
           libsysrepo6t64 libyang2t64 libyang2-tools netconfd-module-ietf-system)
if ls netconfd_*.deb >/dev/null 2>&1; then
    echo "[start_netconf_server] debs already cached at $PWD"
else
    apt-get download "${need_pkgs[@]}"
fi

# 2. Extract into ROOT (no install, no sudo).
mkdir -p "$ROOT"
for d in *.deb; do dpkg-deb -x "$d" "$ROOT" 2>/dev/null || true; done

# 3. Symlink SIL plugins into a YUMA_RUNPATH directory.
mkdir -p /tmp/yuma-runpath/yuma
ln -sf "$ROOT/usr/lib/aarch64-linux-gnu/yuma/libietf-system.so" \
       /tmp/yuma-runpath/yuma/libietf-system.so 2>/dev/null || true
ln -sf "$ROOT/usr/lib/x86_64-linux-gnu/yuma/libietf-system.so" \
       /tmp/yuma-runpath/yuma/libietf-system.so 2>/dev/null || true

# 4. Bootstrap data dirs + SSH host & client keys.
mkdir -p /tmp/yuma-home/data "$SRV"/{etc,run,log,authorized_keys}
[ -f "$SRV/host_ed25519" ]   || ssh-keygen -t ed25519 -f "$SRV/host_ed25519"   -N '' -q
[ -f "$SRV/client_ed25519" ] || ssh-keygen -t ed25519 -f "$SRV/client_ed25519" -N '' -q
cp "$SRV/client_ed25519.pub" "$SRV/authorized_keys/authorized_keys"
chmod 700 "$SRV/authorized_keys"
chmod 600 "$SRV/authorized_keys/authorized_keys"

# 5. netconf-subsystem wrapper (sshd will exec this on session start).
cat > "$SRV/netconf-wrapper.sh" <<EOF
#!/bin/bash
LIB1="$ROOT/usr/lib/aarch64-linux-gnu"
LIB2="$ROOT/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="\$LIB1:\$LIB2:\$LIB1/yuma:\$LIB2/yuma"
export YUMA_HOME=/tmp/yuma-home
export YUMA_MODPATH="$ROOT/usr/share/yuma/modules:$ROOT/usr/share/yuma/modules/yuma123:$ROOT/usr/share/yuma/modules/ietf:$ROOT/usr/share/yuma/modules/ietf-derived:$ROOT/usr/share/yuma/modules/netconfcentral:$ROOT/usr/share/yuma/modules/ietf-draft:$ROOT/usr/share/yang/modules/libyang"
exec "$ROOT/usr/sbin/netconf-subsystem" 2>>"$SRV/log/subsys.log"
EOF
chmod +x "$SRV/netconf-wrapper.sh"

# 6. sshd config — user-space, non-root, public-key only.
cat > "$SRV/sshd_config" <<EOF
Port $PORT
ListenAddress 127.0.0.1
HostKey $SRV/host_ed25519
PidFile $SRV/run/sshd.pid
PermitRootLogin no
PubkeyAuthentication yes
PasswordAuthentication no
UsePAM no
AuthorizedKeysFile $SRV/authorized_keys/authorized_keys
StrictModes no
Subsystem netconf $SRV/netconf-wrapper.sh
LogLevel VERBOSE
EOF

# 7. Boot netconfd (the YANG datastore + RPC handler).
rm -f /tmp/ncxserver.sock /tmp/yuma-home/data/startup-cfg-txid.txt
cd "$ROOT"
LIB="$ROOT/usr/lib/aarch64-linux-gnu"
[ -d "$ROOT/usr/lib/x86_64-linux-gnu" ] && LIB="$ROOT/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$LIB:$LIB/yuma"
export YUMA_HOME=/tmp/yuma-home
export YUMA_RUNPATH=/tmp/yuma-runpath/yuma
export YUMA_MODPATH="$ROOT/usr/share/yuma/modules:$ROOT/usr/share/yuma/modules/yuma123:$ROOT/usr/share/yuma/modules/ietf:$ROOT/usr/share/yuma/modules/ietf-derived:$ROOT/usr/share/yuma/modules/netconfcentral:$ROOT/usr/share/yuma/modules/ietf-draft:$ROOT/usr/share/yang/modules/libyang"

if [[ "${1:-}" == "-d" ]]; then
    nohup "$ROOT/usr/sbin/netconfd" \
        --no-startup --superuser="$(whoami)" --access-control=off \
        --port="$PORT" --module=ietf-system \
        > "$SRV/log/netconfd.log" 2>&1 &
    NCPID=$!
    /usr/sbin/sshd -f "$SRV/sshd_config" -E "$SRV/log/sshd.log"
    sleep 1
    echo "netconfd  PID=$NCPID  log=$SRV/log/netconfd.log"
    echo "sshd      PID=$(cat "$SRV/run/sshd.pid")  log=$SRV/log/sshd.log"
    echo "listen    127.0.0.1:$PORT"
    echo "client_key $SRV/client_ed25519"
    echo "Connect:  ssh -p $PORT -i $SRV/client_ed25519 -s $(whoami)@127.0.0.1 netconf"
else
    "$ROOT/usr/sbin/netconfd" \
        --no-startup --superuser="$(whoami)" --access-control=off \
        --port="$PORT" --module=ietf-system &
    NCPID=$!
    /usr/sbin/sshd -f "$SRV/sshd_config" -E "$SRV/log/sshd.log"
    echo "netconfd PID=$NCPID, sshd on 127.0.0.1:$PORT — Ctrl-C to stop"
    trap 'kill $NCPID 2>/dev/null; kill $(cat "$SRV/run/sshd.pid") 2>/dev/null; exit 0' INT TERM
    wait $NCPID
fi
