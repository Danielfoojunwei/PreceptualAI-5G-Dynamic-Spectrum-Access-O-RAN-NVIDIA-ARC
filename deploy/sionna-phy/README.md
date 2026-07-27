# Sionna → Horizon PHY telemetry seam

Turns **real ray-traced MIMO channels** (DeepMIMO) into **real link-level PHY
KPIs** (NVIDIA Sionna) and drives them through the Horizon control loop to a
guarded A1 policy. This is the concrete, CPU-runnable realisation of the
neural-PHY seam Horizon declares in `pyproject.toml`; see
[`../../docs/ECOSYSTEM.md`](../../docs/ECOSYSTEM.md) for where it sits in the
open 6G stack and [`SIONNA_PHY_PROOF.md`](SIONNA_PHY_PROOF.md) for measured
results (12 sites, 12/12 ENFORCED, a real BLER waterfall driving graded
decisions).

## Components

| File | Role |
| --- | --- |
| [`../../src/horizon_ric/phy/sionna_bridge.py`](../../src/horizon_ric/phy/sionna_bridge.py) | `SionnaPhyBridge` (KPIs → `TelemetryEvent`, `phy_risk_v1` mapping) + `run_link_level_over_channels` (Sionna LDPC/QAM/LMMSE link) |
| [`run_phy_pipeline.py`](run_phy_pipeline.py) | End-to-end runner: DeepMIMO channels → Sionna PHY → pipeline → A1 |
| [`../../tests/test_sionna_phy_bridge.py`](../../tests/test_sionna_phy_bridge.py) | Pure-Python bridge tests (run everywhere) + link-level test (needs the `phy` extra) |

## Reproduce

```bash
# 1. Install the heavy extras into a dedicated venv (pulls PyTorch + DeepMIMO).
python3.11 -m venv .venv-phy
.venv-phy/bin/pip install -e '.[realdata,phy]'

# 2. Fetch the real DeepMIMO ASU 3.5 GHz scenario (checksum-gated, ~133 MB).
.venv-phy/bin/python datasets/deepmimo_asu_3p5/build.py \
    --cache-dir /tmp/dm --features /tmp/dm/features.jsonl --manifest /tmp/dm/manifest.json --samples 64

# 3. Bring up the real A1 side (mediator + hw-python xApp) — see deploy/xapp-e2e.
deploy/xapp-e2e/run_stack.sh

# 4. Run the seam end-to-end against the real mediator.
CUDA_VISIBLE_DEVICES="" .venv-phy/bin/python deploy/sionna-phy/run_phy_pipeline.py \
    --scenario-dir /tmp/dm/deepmimo_scenarios/asu_campus_3p5 \
    --a1-base-url http://127.0.0.1:10000 --dialect legacy \
    --sites 12 --out deploy/sionna-phy/results/phy-e2e-proof.json
```

Without `--a1-base-url` the runner starts a real loopback A1 endpoint (uvicorn,
ephemeral port) so the chain runs self-contained. `--n-tx 2` switches from the
default 1×4 single-layer config to 2-stream spatial multiplexing.

## CI note

The bridge's pure-Python half (measurement → telemetry → risk → pipeline) is
unit-tested in the normal suite. The link-level simulation and the DeepMIMO
fetch need the heavy `phy` / `realdata` extras, so those tests `importorskip`
and are skipped in the default CI environment — they run where the extras are
installed. Sionna/PyTorch and DeepMIMO are **not** added to the core runtime
dependencies.
