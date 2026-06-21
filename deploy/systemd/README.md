# Horizon-RIC — Jetson Orin Nano systemd profile

> *Canonical-to-v3-trust-layer-wave: 2026-05-08.*


Production-grade `systemd` unit + daily soak timer for the **Jetson Orin Nano
8 GiB** edge profile. The constraints in `horizon-ric-orin.service` (2-CPU
affinity, 8 GiB memory ceiling, watchdog) reproduce the Orin envelope on any
aarch64 host (including the GB10 dev box used in CI), so the same bits ship
unchanged from CI to the device.

## What ships

| File                                | Purpose                                            |
|-------------------------------------|----------------------------------------------------|
| `horizon-ric-orin.service`          | Main rApp daemon (Type=notify, watchdog=30s).      |
| `horizon-ric-orin-soak.timer`       | Daily firing at 02:30 (Persistent=true).           |
| `horizon-ric-orin-soak.service`     | Oneshot called by the timer.                       |
| `horizon-soak-1h.sh`                | Wrapper: 60 min wall × 24× = 24 simulated hours.   |

## Install

```bash
# 1. Drop the unit files into /etc/systemd/system.
sudo cp horizon-ric-orin.service /etc/systemd/system/
sudo cp horizon-ric-orin-soak.{timer,service} /etc/systemd/system/

# 2. Install the soak wrapper.
sudo cp horizon-soak-1h.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/horizon-soak-1h.sh

# 3. Create the unprivileged service user + writable dirs.
sudo useradd -r -s /bin/false horizon
sudo mkdir -p /var/lib/horizon-ric /var/log/horizon-ric
sudo chown horizon:horizon /var/lib/horizon-ric /var/log/horizon-ric

# 4. Reload + enable.
sudo systemctl daemon-reload
sudo systemctl enable --now horizon-ric-orin.service horizon-ric-orin-soak.timer

# 5. Verify.
sudo systemctl status horizon-ric-orin.service
sudo journalctl -u horizon-ric-orin.service -f
sudo systemctl list-timers horizon-ric-orin-soak.timer
```

## Orin Nano-specific tuning

These are device-side knobs, not part of the unit file:

* **Power mode.** Set to `MAXN` for steady-state inference:
  ```bash
  sudo nvpmodel -m 0
  sudo jetson_clocks
  ```
  Re-running `jetson_clocks` after every reboot is required — it does not
  persist. Add it to `/etc/rc.local` or a separate one-shot unit if you want
  it on every boot.

* **Kernel watchdog.** The Orin Nano stock kernel exposes `nvwdt` (NVIDIA
  Watchdog Timer). To make the systemd `WatchdogSec=` graceful on hard hang,
  also enable the hardware watchdog:
  ```bash
  echo "nvwdt" | sudo tee -a /etc/modules-load.d/horizon.conf
  ```

* **Memory.** Orin Nano 8 GiB has ~7.4 GiB usable after the kernel +
  driver footprint. `MemoryMax=8G` is therefore *cgroup-effective* —
  in practice the process will be killed by the kernel OOM well before
  it reaches that ceiling. `MemoryHigh=7G` produces the cgroup back-pressure
  signal we want (allocator slows, page reclaim runs) before the OOM.

* **CPU affinity.** Orin Nano has 6 Cortex-A78AE cores; we constrain to 2
  (`CPUAffinity=0 1`) to leave 4 cores for the OS, DLA orchestration, and
  the camera pipeline that customer-side workloads typically expect.

* **Boot tuning.** If you use `nvbootctrl` slot-based A/B updates, set
  `RestartSec=5s` (already in the unit) so the `Restart=on-failure` loop
  trips the bootloader's "rolling fail" detector and triggers a slot
  fallback — instead of looping forever on a broken slot.

## Verification

The unit is parsed and asserted in CI:

```bash
.venv/bin/python -m pytest tests/test_systemd_unit.py -v
```

The 11 test functions cover the service file existing, Type=notify,
WatchdogSec=30s, CPUAffinity=0 1, MemoryMax=8G, User=horizon (not root),
NoNewPrivileges + ProtectSystem=strict, restricted ReadWritePaths, the
install target, the ExecStart using the horizon_ric CLI, the timer's
`Persistent=true` + daily firing, and the soak script's `taskset -c 0-1` pin.

## Soak output

`horizon-soak-1h.sh` writes to `/var/log/horizon-ric/soak-YYYYMMDD.json`
(24 simulated hours' worth of metrics: A1 acks, latency p50/p95/p99,
watchdog pings, OOM events). The wrapper invokes `scripts/soak_24h.py`,
which is **not currently checked into this tree** and must be restored
before the soak timer will run end-to-end on the device. Long-term
retention is the operator's responsibility — `logrotate` config in
`deploy/logrotate.d/horizon-ric` (if present) handles weekly rotation.
