# Example external connector — `random_telemetry`

> *Canonical-to-v3-trust-layer-wave: 2026-05-08.*


A self-contained external Python package that ships a PreceptualAI
connector. Demonstrates the entry-point plug-in path: installing this
wheel makes `RandomTelemetrySource` discoverable by the core registry
without editing any PreceptualAI source.

## Install

```bash
.venv/bin/pip install ./examples/external_connector
```

## Verify

```python
from horizon_ric.io.registry import list_connectors

print(list_connectors())
# {'sources': [..., 'random_telemetry'], 'sinks': [...]}
```

## Use

```python
import asyncio
from horizon_ric.io.registry import get_source

async def main():
    src = get_source(
        "random_telemetry",
        {"name": "demo", "rate_hz": 5.0, "n_events": 3, "seed": 42},
    )
    async with src:
        async for event in src.stream():
            print(event.modality, event.payload)

asyncio.run(main())
```

## Plug-in mechanics

```toml
# pyproject.toml
[project.entry-points."horizon_ric.connectors"]
random_telemetry = "random_telemetry:RandomTelemetrySource"
```

The `horizon_ric.io.registry.ConnectorRegistry` lazily loads every entry
point under the `horizon_ric.connectors` group on first call to
`list_connectors()` / `get_source()` / `get_sink()`.

The same plug-in convention is used for:

| group                       | registry                                                |
| --------------------------- | ------------------------------------------------------- |
| `horizon_ric.connectors`    | `horizon_ric.io.registry`                               |
| `horizon_ric.encoders`      | `horizon_ric.core.encoder_registry`                     |
| `horizon_ric.verifiers`     | `horizon_ric.policy.verifier_registry`                  |
| `horizon_ric.guards`        | `horizon_ric.policy.guard_registry`                     |
