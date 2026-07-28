"""Runtime environment stamped into every committed benchmark result.

Distinct from :mod:`horizon_ric.provenance`, which attests *models*
cryptographically. This module records the *interpreter environment* a
benchmark result was produced in, and asserts nothing.

Every Horizon benchmark result JSON is a reproduction target: CI rebuilds the
licence-gated DeepMIMO features, re-runs the loop and asserts the numbers still
match. That contract has a silent failure mode. All four closed-loop benchmarks
draw randomness from ``numpy.random.default_rng``, and numpy's own ``Generator``
documentation is explicit that it carries **no** bit-stream compatibility
guarantee across releases — "as better algorithms evolve the bit stream may
change". A numpy upgrade is therefore indistinguishable from the outside from a
reseed: the loops still run, the physics is unchanged, but every stochastic
field can move.

A 12-seed sweep measured how little room that leaves. The federated verifier's
``krum_rmse < baseline`` gate holds by 0.6501 dB against a seed-induced standard
deviation of 1.8687 dB, and ``robust_target_receiver`` reproduces at only 8 of
12 seeds. So a routine dependency bump has a real chance of presenting as an
unreproducible-science alarm rather than as the dependency change it is.

Stamping numpy and the interpreter into each result makes that case diagnosable
from one ``git diff`` of the result JSON. The recorded fields are deliberately
environment-only, and no verifier asserts them: these values are *expected* to
differ between the machine that commits a result and the CI runner that
reproduces it — today they legitimately do, because the ``realdata`` extra pulls
``deepmimo==4.0.0``, which caps numpy at ``<2.3``. See ``docs/REPRODUCIBILITY.md``.
"""

from __future__ import annotations

import platform
from typing import Any

import numpy as np

__all__ = ["runtime_environment", "stamp"]


def runtime_environment() -> dict[str, Any]:
    """Return the environment fields recorded alongside a benchmark result.

    Limited to values that can actually move a stochastic result: the numpy
    version (RNG bit stream) and the interpreter. Deliberately no timestamp and
    no hostname — those would churn the committed JSONs on every run while
    carrying no diagnostic signal.
    """
    return {
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "rng_stream_guarantee": (
            "numpy.random.Generator provides no cross-version bit-stream "
            "compatibility guarantee; a numpy change can act as a reseed"
        ),
    }


def stamp(result: dict[str, Any]) -> dict[str, Any]:
    """Attach :func:`runtime_environment` to ``result`` under ``runtime``."""
    result["runtime"] = runtime_environment()
    return result
