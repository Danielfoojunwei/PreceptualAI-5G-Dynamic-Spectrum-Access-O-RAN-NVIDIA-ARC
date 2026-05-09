"""External-data adapters for PreceptualAI.

This package wraps real, on-disk NVIDIA-native datasets as
`horizon_ric.io.Source` connectors so the rest of the system can pull
them through the unified registry rather than via dataset-specific
glue. The pivot away from synthetic maritime traces lives here.

Currently registered sources (string names → connector class):

    aerial_fapi   → AerialFAPISource          (`fapi.parquet`)
    aerial_fh     → AerialFrontHaulSource     (`fh.parquet`, IQ heuristic)
    aerial_h5     → AerialH5Source            (cuMAC HDF5 test vectors)
    deepmimo      → DeepMIMOSource            (per-scene ray-traced runs)

Optional historical adapters (`AODTScenario`, `SionnaChannelGenerator`)
are imported lazily; they remain available if the underlying modules
exist but no longer block this package from loading when they don't.
"""

from horizon_ric.data.aerial import (
    AerialDataset,
    AerialFAPIReader,
    AerialFAPISource,
    AerialFrontHaulReader,
    AerialFrontHaulSource,
    AerialH5Source,
    AerialH5TestVectorReader,
)
from horizon_ric.data.deepmimo import DeepMIMOScenarioReader, DeepMIMOSource
from horizon_ric.data.space_track import (
    SpaceTrackClient,
    SpaceTrackSource,
    SpaceTrackSourceConfig,
)
from horizon_ric.io.registry import register_source

# Auto-register on import so configs can reference connectors by name.
register_source("aerial_fapi", AerialFAPISource)
register_source("aerial_fh", AerialFrontHaulSource)
register_source("aerial_h5", AerialH5Source)
register_source("deepmimo", DeepMIMOSource)
register_source("space_track", SpaceTrackSource)


# Best-effort imports for legacy adapters; missing modules are tolerated.
try:  # pragma: no cover - optional
    from horizon_ric.data.aodt import AODTScenario  # type: ignore
except ImportError:  # pragma: no cover
    AODTScenario = None  # type: ignore

try:  # pragma: no cover - optional
    from horizon_ric.data.sionna_channel import SionnaChannelGenerator  # type: ignore
except ImportError:  # pragma: no cover
    SionnaChannelGenerator = None  # type: ignore


__all__ = [
    "AODTScenario",
    "AerialDataset",
    "AerialFAPIReader",
    "AerialFAPISource",
    "AerialFrontHaulReader",
    "AerialFrontHaulSource",
    "AerialH5Source",
    "AerialH5TestVectorReader",
    "DeepMIMOScenarioReader",
    "DeepMIMOSource",
    "SionnaChannelGenerator",
    "SpaceTrackClient",
    "SpaceTrackSource",
    "SpaceTrackSourceConfig",
]
