"""Edge inference agent.

Runs on the edge node (e.g. Jetson Orin Nano). Loads a SLARiskHead
checkpoint, ingests telemetry from stdin (or a polling source), and
emits per-horizon SLA-breach predictions to stdout for the upstream
rApp to consume.

The console script `horizon-edge-agent` is wired to
`horizon_ric.agent.inference:main` in pyproject.toml.
"""

from horizon_ric.agent.inference import EdgeAgentConfig, main, run

__all__ = ["EdgeAgentConfig", "main", "run"]
