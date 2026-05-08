# Hardware Procurement Playbook — Jetson Orin Nano (Row 26)

_Document version: 1.0 · Issued: 2026-05-08_

## Honest framing

This is a **procurement timeline** issue, not a code-readiness issue. The
PreceptualAI code is delivery-ready: it boots, soaks, and self-validates on
aarch64. The substitute-envelope attestation in
`deploy/ORIN_HARDWARE_ATTESTATION.md` lets the customer's procurement
architect sign Row 26 today, with the validation script
(`deploy/orin_validation.sh`) as the post-delivery gate that promotes Row 26
from "TRUE (substitute)" to "TRUE (hardware-validated)" once the
physical box arrives.

Nothing in this playbook is a blocker for contract signature.

## Bill of materials

| Item | Quantity | Unit price (USD, 2026-05) | Vendor SKU |
|---|---:|---:|---|
| NVIDIA Jetson Orin Nano 8 GB Developer Kit | 1 | ~$499 | `945-13766-0005-000` |
| 64 GB UHS-I microSD (Samsung Pro Endurance or equivalent) | 1 | ~$15 | vendor-agnostic |
| USB-C 5V/3A power adapter (if not bundled) | 1 | ~$15 | vendor-agnostic |
| Ethernet cable (Cat 6, ≥ 1 m) | 1 | ~$5 | vendor-agnostic |
| **Total** | | **~$535** | |

Quantity-1 sourcing is sufficient for the post-delivery acceptance gate.
Pilot rollout (multiple sites) re-runs this playbook per site.

## Vendors and lead time

| Vendor | Typical lead time | Channel |
|---|---|---|
| NVIDIA Marketplace (direct) | 1–3 weeks | https://marketplace.nvidia.com/ |
| Arrow Electronics | 2–4 weeks | https://www.arrow.com/ — search SKU above |
| Mouser Electronics | 2–4 weeks | https://www.mouser.com/ — search SKU above |
| Seeed Studio | 1–2 weeks | https://www.seeedstudio.com/ |

Customer's preferred vendor wins; we do not insist. A 30-day acceptance
window (per `deploy/ORIN_HARDWARE_ATTESTATION.md` §9) starts on hardware
receipt — i.e. the longest plausible total path from PO to validation is
**~7 weeks** (4-week ship + 3-week setup + run buffer).

## First-boot checklist

1. Flash the developer kit with NVIDIA JetPack 5.1.2 or later
   (https://developer.nvidia.com/embedded/jetpack-sdk-512).
2. Connect via Ethernet, complete `oem-config` first-boot wizard.
3. Verify substrate self-identification:
   ```
   $ cat /proc/device-tree/model
   NVIDIA Jetson Orin Nano Developer Kit
   ```
4. Install runtime prerequisites (Python 3.11, systemd ≥ 245 — both ship
   with JetPack 5.1.2).
5. Clone the PreceptualAI repository, run `make venv` and `make install`.
6. Lock the clock and power profile for the acceptance run:
   ```
   sudo nvpmodel -m 0       # 15 W mode
   sudo jetson_clocks       # disable DVFS for stable measurement
   ```
7. Proceed to "Acceptance test" below.

## Acceptance test

```
sudo bash deploy/orin_validation.sh
```

This is the **single command** the customer's architect runs to promote
Row 26 from "TRUE (substitute)" to "TRUE (hardware-validated)". The
script (see `deploy/orin_validation.sh`):

1. Auto-detects the substrate is real Orin via `/proc/device-tree/model`.
2. Runs the 24-h-equivalent soak + edge benchmark.
3. Parses the five Row 26 acceptance bars from the resulting JSON.
4. Prints PASS/FAIL per bar.
5. Exits 0 iff all five bars pass.

The five bars are documented in `deploy/ORIN_HARDWARE_ATTESTATION.md`
§6:

| Bar | Threshold |
|---|---|
| Audit chain integrity | 1440 / 1440 verify=True |
| A1 emit success rate | ≥ 99.5 % |
| p99 decision latency | ≤ 250 ms |
| Max watchdog silence | ≤ 30 s |
| Unhandled exceptions | 0 |

Expected wall-clock: ~12 minutes for a 24-h-equivalent soak at 120×
speedup, plus ~2 minutes for the 10 K-step edge benchmark.

## Acceptance test failure (rollback path)

If `deploy/orin_validation.sh` exits non-zero on real Orin Nano hardware:

1. Capture the failing soak JSON at `/tmp/orin_validate_soak.json`.
2. File a P1 internal ticket attaching the JSON plus a histogram diff
   versus the §6 substitute numbers in
   `deploy/ORIN_HARDWARE_ATTESTATION.md`.
3. Hold deployment behind the operator's
   `enable_orin_substrate=false` configuration flag.
4. Root-cause the gap (likely candidates: thermal throttling under
   `nvpmodel 1` 7 W mode, DRAM bit-flip caught by audit chain, or
   BSP-specific watchdog timing).
5. Re-issue `deploy/ORIN_HARDWARE_ATTESTATION.md` with revised numbers,
   re-run the validation script, re-attempt sign-off.

The contract does **not** invalidate on first-attempt validation failure.
The 30-day acceptance window in
`deploy/ORIN_HARDWARE_ATTESTATION.md` §9 explicitly accommodates one
rollback iteration.
