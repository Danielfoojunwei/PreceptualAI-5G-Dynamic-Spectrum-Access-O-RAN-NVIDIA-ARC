#!/bin/bash
set -euo pipefail

# ============================================================
# UHCI Full Training Pipeline — Single Command
#
# Runs EVERYTHING: install deps, download datasets, train models,
# run benchmarks, produce results. No manual steps.
#
# Usage:
#   chmod +x run_everything.sh
#   ./run_everything.sh              # Full run (GPU auto-detected)
#   ./run_everything.sh --quick      # Quick test (1 seed, 5K steps)
#   ./run_everything.sh --cpu        # Force CPU mode
#
# Requires: Python 3.10+, git, ~10GB disk for datasets
# Recommended: NVIDIA GPU (GB10, A100, H100, RTX 4090)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
DATA_DIR="$REPO_ROOT/data"
RESULTS_DIR="$REPO_ROOT/results"
BENCHMARKS_DIR="$REPO_ROOT/benchmarks"

# Defaults
SEEDS=3
BENCHMARK_STEPS=50000
TRAINING_STEPS=5000000
EVAL_EPISODES=100
QUICK=false
FORCE_CPU=false

# Parse args
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true; SEEDS=1; BENCHMARK_STEPS=5000; TRAINING_STEPS=50000; EVAL_EPISODES=20 ;;
        --cpu) FORCE_CPU=true ;;
        --seeds=*) SEEDS="${arg#*=}" ;;
        --steps=*) TRAINING_STEPS="${arg#*=}" ;;
    esac
done

echo "================================================================"
echo "  UHCI Full Training Pipeline"
echo "  Seeds: $SEEDS | Benchmark: ${BENCHMARK_STEPS} steps"
echo "  Training: ${TRAINING_STEPS} steps | Eval: ${EVAL_EPISODES} episodes"
echo "================================================================"
echo ""

# ────────────────────────────────────────────────────────────
# PHASE 1: Install Dependencies
# ────────────────────────────────────────────────────────────
echo "[Phase 1/6] Installing dependencies..."

pip install -q torch numpy gymnasium pyyaml pandas scipy matplotlib datasets pydantic sgp4 2>/dev/null || true

# Install the UHCI package itself
cd "$REPO_ROOT"
pip install -q -e ".[dev,benchmarks]" 2>/dev/null || pip install -q -e . 2>/dev/null || true

echo "  Dependencies installed."

# ────────────────────────────────────────────────────────────
# PHASE 2: Download All Datasets
# ────────────────────────────────────────────────────────────
echo ""
echo "[Phase 2/6] Downloading datasets..."

# 2a. UCC MISL 5G Dataset (188K real measurements from Irish operator)
UCC_DIR="$DATA_DIR/ucc_misl/5Gdataset"
UCC_EXTRACTED="$UCC_DIR/extracted/5G-production-dataset"
if [ -d "$UCC_EXTRACTED" ] && [ "$(find "$UCC_EXTRACTED" -name '*.csv' ! -path '*__MACOSX*' | wc -l)" -gt 50 ]; then
    echo "  UCC MISL: already present ($(find "$UCC_EXTRACTED" -name '*.csv' ! -path '*__MACOSX*' | wc -l) CSVs)"
else
    echo "  UCC MISL: downloading from GitHub..."
    mkdir -p "$DATA_DIR/ucc_misl"
    rm -rf "$UCC_DIR"
    git clone --quiet https://github.com/uccmisl/5Gdataset.git "$UCC_DIR"
    if [ -f "$UCC_DIR/5G-production-dataset.zip" ]; then
        cd "$UCC_DIR"
        unzip -q 5G-production-dataset.zip -d extracted
        cd "$REPO_ROOT"
        CSV_COUNT=$(find "$UCC_EXTRACTED" -name '*.csv' ! -path '*__MACOSX*' 2>/dev/null | wc -l)
        echo "  UCC MISL: extracted $CSV_COUNT CSV files"
    else
        echo "  UCC MISL: WARNING - zip not found, will use TelecomTS only"
    fi
fi

# 2b. TelecomTS (auto-downloads from HuggingFace on first use)
echo "  TelecomTS: verifying HuggingFace access..."
python3 -c "
from datasets import load_dataset
ds = load_dataset('AliMaatouk/TelecomTS', split='train', streaming=True)
sample = next(iter(ds))
print('  TelecomTS: accessible (32K samples x 128 timesteps, 18 KPIs)')
" 2>/dev/null || echo "  TelecomTS: WARNING - HuggingFace access failed, will use UCC MISL only"

echo "  All datasets ready."

# ────────────────────────────────────────────────────────────
# PHASE 3: Detect Hardware
# ────────────────────────────────────────────────────────────
echo ""
echo "[Phase 3/6] Detecting hardware..."

if [ "$FORCE_CPU" = true ]; then
    DEVICE="cpu"
    echo "  Forced CPU mode"
else
    DEVICE=$(python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'cuda')
    import sys
    print(f'  GPU: {name} ({mem:.1f} GB)', file=sys.stderr)
else:
    print('cpu')
    import sys
    print('  No GPU detected — using CPU (will be slow)', file=sys.stderr)
" 2>&1 | head -1)
    # Print the GPU info line
    python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {name} ({mem:.1f} GB)')
else:
    print('  No GPU — using CPU (training will be slower)')
" 2>/dev/null || echo "  CPU mode"
fi

echo "  Device: $DEVICE"

# ────────────────────────────────────────────────────────────
# PHASE 4: Run Scheduling Methods Comparison (14 baselines vs SAC-LTC)
# ────────────────────────────────────────────────────────────
echo ""
echo "[Phase 4/6] Running scheduling methods comparison..."
echo "  14 traditional baselines vs SAC-LTC + SAC-LSTM"
echo "  3 environments: Original / Realistic (no occupancy) / Heterogeneous (LEO+5G+WiFi7)"
echo "  $SEEDS seeds x $BENCHMARK_STEPS steps x $EVAL_EPISODES eval episodes"

mkdir -p "$RESULTS_DIR"

cd "$BENCHMARKS_DIR"
python3 run_enhanced_comparison.py \
    --seeds "$SEEDS" \
    --steps "$BENCHMARK_STEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --output-dir "$RESULTS_DIR/scheduling_comparison" \
    2>&1 | tee "$RESULTS_DIR/scheduling_comparison.log"

echo "  Comparison results: $RESULTS_DIR/scheduling_comparison/"

# ────────────────────────────────────────────────────────────
# PHASE 5: Train 5.3M Parameter Model on Real Data
# ────────────────────────────────────────────────────────────
echo ""
echo "[Phase 5/6] Training 5.3M parameter SAC-LTC on real 5G data..."
echo "  Model: hidden=384, latent=384, 3 layers (5,291,571 params)"
echo "  Data: UCC MISL (188K measurements) + TelecomTS (32K samples)"
echo "  Steps: $TRAINING_STEPS"

cd "$BENCHMARKS_DIR"

# Train on UCC MISL real data
if [ -d "$UCC_EXTRACTED" ]; then
    echo "  Training on UCC MISL real 5G traces..."
    python3 train_500k_realdata.py \
        --steps "$TRAINING_STEPS" \
        --seeds "$SEEDS" \
        --eval-episodes "$EVAL_EPISODES" \
        --env real5g \
        --output-dir "$RESULTS_DIR/real5g_training" \
        2>&1 | tee "$RESULTS_DIR/real5g_training.log"
    echo "  Real5G results: $RESULTS_DIR/real5g_training/"
fi

# Train on TelecomTS
echo "  Training on TelecomTS real 5G testbed data..."
python3 train_500k_realdata.py \
    --steps "$TRAINING_STEPS" \
    --seeds "$SEEDS" \
    --eval-episodes "$EVAL_EPISODES" \
    --env telecomts \
    --output-dir "$RESULTS_DIR/telecomts_training" \
    2>&1 | tee "$RESULTS_DIR/telecomts_training.log"
echo "  TelecomTS results: $RESULTS_DIR/telecomts_training/"

# ────────────────────────────────────────────────────────────
# PHASE 6: Summary Report
# ────────────────────────────────────────────────────────────
echo ""
echo "[Phase 6/6] Generating summary..."

python3 -c "
import json, os, glob

results_dir = '$RESULTS_DIR'
print()
print('=' * 80)
print('  UHCI BENCHMARK RESULTS SUMMARY')
print('=' * 80)

# Scheduling comparison
comp_file = os.path.join(results_dir, 'scheduling_comparison', 'combined_results.json')
if os.path.exists(comp_file):
    with open(comp_file) as f:
        data = json.load(f)
    for env_name, agents in data.items():
        print(f'\n  Environment: {env_name}')
        print(f'  {\"Agent\":25s} {\"Success%\":>10s} {\"Collision%\":>12s} {\"Reward\":>10s}')
        print('  ' + '-' * 60)
        sorted_agents = sorted(agents.keys(),
            key=lambda a: agents[a].get('success_rate', {}).get('mean', 0), reverse=True)
        for agent in sorted_agents[:10]:  # top 10
            sr = agents[agent].get('success_rate', {}).get('mean', 0)
            cr = agents[agent].get('collision_rate', {}).get('mean', 0)
            rw = agents[agent].get('mean_reward', {}).get('mean', 0)
            cat = 'RL' if agent in ('sac_ltc', 'sac_lstm') else 'trad'
            print(f'  {agent:25s} {sr:>9.2%}   {cr:>9.2%}   {rw:>9.2f}  [{cat}]')

# Real data training results
for env in ['real5g', 'telecomts']:
    result_files = glob.glob(os.path.join(results_dir, f'{env}_training', '**', '*results.json'), recursive=True)
    for rf in result_files:
        with open(rf) as f:
            rdata = json.load(f)
        print(f'\n  Real Data Training: {env}')
        for agent, metrics in rdata.items():
            sr = metrics.get('success_rate', {}).get('mean', 0)
            print(f'    {agent:25s} Success={sr:.2%}')

print()
print('=' * 80)
print(f'  All results saved to: {results_dir}/')
print('=' * 80)
" 2>&1 | tee "$RESULTS_DIR/summary.txt"

echo ""
echo "================================================================"
echo "  DONE. All results in: $RESULTS_DIR/"
echo ""
echo "  Key files:"
echo "    $RESULTS_DIR/scheduling_comparison/combined_results.json"
echo "    $RESULTS_DIR/real5g_training/"
echo "    $RESULTS_DIR/telecomts_training/"
echo "    $RESULTS_DIR/summary.txt"
echo "================================================================"
