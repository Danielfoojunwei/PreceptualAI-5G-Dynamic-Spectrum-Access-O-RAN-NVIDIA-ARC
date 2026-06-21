# Horizon-RIC Modularity Inventory

*Date: 2026-05-08 (post-v3 trust-layer wave; canonical with 144 source modules / 30 751 LoC).*

Horizon-RIC is built around **named registries**. Anything that is likely
to vary between operators, deployments, regions, or model families is
swapped at config time, not patched into core. The same registry pattern
recurs across six surfaces.

## Pluggable surfaces

| Surface     | Registry module                                | Entry-point group           | Built-in names                                                                |
| ----------- | ---------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------- |
| Connectors  | `horizon_ric.io.registry`                      | `horizon_ric.connectors`    | `kafka`, `http`, `file`                                                       |
| Encoders    | `horizon_ric.core.encoder_registry`            | `horizon_ric.encoders`      | `identity`, `cfc`, `liquid_s4`, `latent_ode`                                  |
| Verifiers   | `horizon_ric.policy.verifier_registry`         | `horizon_ric.verifiers`     | `gso_pfd`, `itu_spectral_mask`, `edge_gpu`, `epfd`, `li_jurisdiction`         |
| Guards      | `horizon_ric.policy.guard_registry`            | `horizon_ric.guards`        | `head_pretrained`, `constraint_context_complete`, `decision_within_a1_budget`, `corrections_recorded` |
| A1 dialects | `horizon_ric.rapp.a1_adapter` (`A1_DIALECTS`)  | n/a (in-tree only)          | `osc_nonrtric`, `mantaray`, `eiap`                                            |
| Scenarios   | `horizon_ric.scenarios` (`SCENARIO_REGISTRY`)  | n/a (in-tree only)          | `maritime`, `aerial`, `coexistence`, `ntn_air_ran`                            |

Each registry follows the same triplet:

```python
register_X(name, factory_or_class)
get_X(name, **config)
list_X() -> list[str]
```

External packages register by declaring an entry point. The registry
lazily loads every entry point on the first `get_*` / `list_*` call.

## How to add each kind of plug-in

### Add a connector

```python
# my_pkg/sources.py
from horizon_ric.io.connector import Source, ConnectorConfig

class MySource(Source):
    Config = ConnectorConfig
    async def connect(self): ...
    async def close(self): ...
    def stream(self): ...
```

```toml
# my_pkg/pyproject.toml
[project.entry-points."horizon_ric.connectors"]
my_source = "my_pkg.sources:MySource"
```

`pip install ./my_pkg && python -c "from horizon_ric.io.registry import list_connectors; print(list_connectors())"` will show `my_source` under sources. See `examples/external_connector/` for a working end-to-end example.

### Add an encoder

```python
from horizon_ric.core.encoder_registry import register_encoder, TemporalEncoder

class MyEncoder(TemporalEncoder):
    def __init__(self, d_model: int = 64, **_):
        super().__init__()
        self.d_model = d_model
        # ...
    def forward(self, x):  # (B, T, D) -> (B, T, D)
        return ...

register_encoder("my_encoder", MyEncoder)
```

Or via entry point under `horizon_ric.encoders`. The world model picks the encoder by config name only — no callsite changes.

### Add a verifier

```python
from horizon_ric.policy.verifier_registry import register_verifier, Verifier

class MyVerifier(Verifier):
    constraint_id = "my_rule"
    def __init__(self, config=None): self.cfg = config
    def check(self, action, ctx): return [...]
    def project(self, action, ctx): return action, [...]

register_verifier("my_rule", MyVerifier)
```

Compose into the constraint layer:

```python
HorizonRICConstraintLayer(
    config,
    verifier_chain=["gso_pfd", "itu_spectral_mask", "my_rule"],
)
```

### Add a guard

```python
from horizon_ric.policy.guard_registry import register_guard, GuardContext
from horizon_ric.policy.emit_guards import GuardFailure

def my_guard(ctx: GuardContext) -> GuardFailure | None:
    if not ctx.extras.get("my_clearance"):
        return GuardFailure(guard_id="my_clearance_missing", message="...")
    return None

register_guard("my_clearance", my_guard)
```

`run_guard_chain(ctx)` will run every registered guard in registration order.

### Add an A1 dialect

```python
# horizon_ric/rapp/a1_adapter.py
A1_DIALECTS["my_vendor"] = MyDialect()
```

(See `tests/test_a1_*_dialect.py` for the required `serialise` / `validate` contract.)

### Add a scenario

```python
# horizon_ric/scenarios/__init__.py
SCENARIO_REGISTRY["my_scenario"] = MyScenarioBuilder
```

## Compatibility matrix

| Encoder      | World-model contract met? | Edge (Jetson) | Cloud  |
| ------------ | ------------------------- | ------------- | ------ |
| `identity`   | yes                       | yes           | yes    |
| `cfc`        | yes                       | yes           | yes    |
| `liquid_s4`  | yes                       | yes           | yes    |
| `latent_ode` | yes                       | yes (≤ 64-d)  | yes    |

| Verifier            | Used at training (Lagrangian) | Used at inference (projection) |
| ------------------- | ----------------------------- | ------------------------------ |
| `gso_pfd`           | yes                           | yes                            |
| `itu_spectral_mask` | no                            | yes                            |
| `edge_gpu`          | no                            | yes                            |
| `epfd`              | no                            | yes (refusal only)             |
| `li_jurisdiction`   | no (hard-only by ETSI)        | yes                            |

## Scaling evidence

`benchmarks/scaling.png` plots wall-clock per stage for
N ∈ {1, 10, 100, 1000, 5000} entities, exercising the real production
encoder front-door (SpatialPrior → EntityTokenizer → PerceiverFusion).
Latest run on GB10:

| N    | SpatialPrior (ms) | EntityTok (ms) | Perceiver (ms) |
| ---- | ----------------- | -------------- | -------------- |
| 1    | 0.0               | 3.3            | 20.2           |
| 10   | 0.3               | 17.5           | 23.5           |
| 100  | 19.2              | 170.4          | 20.0           |
| 1000 | 1939.8            | 1688.4         | 33.2           |
| 5000 | (skipped — O(N²)) | 8844.6         | 31.9           |

The Perceiver scales **sub-linearly**: 5000× more entities for ≈ 1.6× the wall-clock, because it cross-attends into a fixed-size latent.

## Edge distillation evidence

`scripts/distill_for_edge.py` distills the v0.4 SLA head
(latent_dim=128, hidden_dim=128) into an edge variant (32, 32):

* Teacher size: 282,829 bytes
* Edge size: 38,764 bytes (13.7% of teacher; under the 30% target)
* Max MAE on val (vs teacher): 0.0080 (under the 0.05 target)

The student stays inside the 8 GB Jetson Orin Nano memory + bandwidth
ceiling enforced by the `edge_gpu` verifier.
