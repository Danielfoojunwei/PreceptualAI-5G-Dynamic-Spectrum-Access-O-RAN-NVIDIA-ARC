"""Vulture whitelist — names vulture flags but that are NOT dead code.

Run with:
    .venv/bin/python -m vulture src/ .vulture_whitelist.py --min-confidence 90
"""

# Abstract-method parameters in DomainAdapter / MetricSuite contracts —
# vulture can't see that concrete subclasses bind these names.
raw_telemetry  # noqa: F821
action_logits  # noqa: F821
scenario_id  # noqa: F821
ground_truth  # noqa: F821

# SionnaChannelGenerator: legacy compatibility kwargs kept for backward
# call sites that still pass them; the constructor stores neither (the
# Sionna backend is the only path) but removing them would break
# downstream test fixtures that pass `require_sionna=True`.
use_standin  # noqa: F821
require_sionna  # noqa: F821
