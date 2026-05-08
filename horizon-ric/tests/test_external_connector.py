"""External-connector entry-point pickup.

Verifies that ``examples/external_connector`` (an out-of-tree pip-installable
package) auto-registers via the ``horizon_ric.connectors`` entry-point
group when imported through :mod:`horizon_ric.io.registry`.
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from horizon_ric.io.registry import ConnectorRegistry, get_source, list_connectors


HAS_EXAMPLE = importlib.util.find_spec("random_telemetry") is not None


@pytest.mark.skipif(
    not HAS_EXAMPLE,
    reason="example connector not installed; run "
    "`pip install ./examples/external_connector`",
)
def test_external_connector_auto_registers():
    """Once the example wheel is installed, the registry must list it
    without any manual registration call."""
    fresh = ConnectorRegistry()
    found = fresh.list_connectors()
    assert "random_telemetry" in found["sources"], (
        f"random_telemetry not auto-loaded; sources: {found['sources']}"
    )


@pytest.mark.skipif(not HAS_EXAMPLE, reason="example connector not installed")
def test_external_connector_can_emit_events():
    """End-to-end: instantiate via registry, stream events."""
    src = get_source(
        "random_telemetry",
        {"name": "test", "rate_hz": 100.0, "n_events": 3, "seed": 7},
    )

    async def _drain():
        out = []
        async with src:
            async for ev in src.stream():
                out.append(ev)
        return out

    events = asyncio.run(_drain())
    assert len(events) == 3
    assert all(ev.source_id == "test" for ev in events)
    assert all(ev.modality == "ue_qos" for ev in events)


@pytest.mark.skipif(not HAS_EXAMPLE, reason="example connector not installed")
def test_module_level_list_connectors_includes_example():
    """The module-level helper must also show the new connector."""
    assert "random_telemetry" in list_connectors()["sources"]
