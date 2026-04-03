# Troubleshooting

This guide provides a systematic way to diagnose common issues when working with the repository. The main principle is to troubleshoot from the **core installation outward**. In other words, verify the package and install profile first, then the benchmark or UHCI training path you are using, then export or serving, and only after that layered surfaces such as federated learning or accelerated deployment.[1]

| Symptom | Most likely layer |
|---|---|
| Import or dependency failures | Packaging and install profile |
| Training does not start | Environment, dependencies, or script usage |
| Benchmark run fails | Benchmark extras or config mismatch |
| Export fails | Checkpoint or export dependencies |
| Serving fails | Runtime config, protobuf generation, or server layer |
| UHCI fails | Optional dependencies or advanced data assumptions |

## Installation Problems

If imports fail immediately, inspect your install profile first. The project manifest uses optional extras for benchmarks, O-RAN, GPU runtime, and UHCI. Many failures come from trying to run a richer surface with only the default lightweight profile installed.[2]

## Training Problems

If training fails, return to the simplest path that matches the workflow you are testing: run a short training loop, inspect the output directory, and confirm that a checkpoint is written. This verifies that the environment, agent, and script surface are functioning together.

```bash
python scripts/train.py --num_steps 5000 --output_dir results/debug_run
```

## Benchmark Problems

If a benchmark run fails, verify that the benchmark extras are installed and that you are using the expected configuration file.

```bash
pip install -e ".[dev,benchmarks]"
python benchmarks/run_full_benchmark.py --config benchmarks/benchmark_config.yaml
```

## Serving Problems

If the model export or serving path fails, confirm that the checkpoint exists, the export dependencies are installed, and the serving config points to valid artifacts. Then inspect the protobuf and runtime surfaces rather than only the script wrapper.[3] [4]

## UHCI Problems

UHCI failures should usually be interpreted as **surface-specific dependency or configuration failures**, not as proof that the whole repository is broken. Confirm that the default install and the exact workflow you are targeting work before debugging graph-learning dependencies, dataset pathways, or heterogeneous-provider logic.

## Troubleshooting Order

| Order | Check |
|---|---|
| 1 | `pyproject.toml` and install profile |
| 2 | Default-installation tests |
| 3 | Short train/evaluate loop for the target workflow |
| 4 | Export or serving path |
| 5 | Benchmark suite |
| 6 | Federated or UHCI path |

## References

[1]: [Setup guide](SETUP_AND_QUICKSTART.md)
[2]: [Project manifest in `pyproject.toml`](../pyproject.toml)
[3]: [Inference protobuf in `proto/preceptualai.proto`](../proto/preceptualai.proto)
[4]: [Deployment guide](DEPLOYMENT_AND_XAPP_GUIDE.md)
