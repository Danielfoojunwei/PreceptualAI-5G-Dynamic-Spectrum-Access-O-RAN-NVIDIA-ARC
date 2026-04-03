"""
Universal Connectivity Provider Registry.

Defines every connectivity provider type — terrestrial and non-terrestrial —
with their physics constraints, spectrum bands, and operational parameters.

The registry is the foundation of the Universal Heterogeneous Connectivity
Intelligence (UHCI) paradigm: treating LEO, MEO, GEO, HAPS, 5G FR1/FR3,
6G ISAC, and WiFi 7 as nodes in a unified interference graph.

Provider types form a hierarchy:
  NTN (Non-Terrestrial):
    - LEO (Low Earth Orbit,  200–2000 km)    — e.g. Starlink, OneWeb
    - MEO (Medium Earth Orbit, 2000–35786 km) — e.g. O3b mPOWER
    - GEO (Geostationary,   ~35786 km)        — e.g. SES, Intelsat
    - HAPS (High-Altitude Platform, ~20 km)   — e.g. Airbus Zephyr
  Terrestrial:
    - 5G_FR1 (sub-6 GHz, 410 MHz – 7.125 GHz)
    - 5G_FR3 (7–24 GHz, 6G critical band)
    - 6G_ISAC (Integrated Sensing and Communication, 24–300 GHz)
    - WIFI7   (2.4 / 5 / 6 GHz 802.11be)

References:
  3GPP TR 38.821: NTN.
  3GPP TR 38.901: Channel models 0.5–100 GHz.
  ITU-R P.618-13: Propagation for Earth-space paths.
  3GPP Release 19: ISAC study.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch


class ProviderType(Enum):
    """All connectivity provider types in the unified graph."""
    # Non-Terrestrial Network (NTN)
    LEO   = auto()   # Low Earth Orbit satellite
    MEO   = auto()   # Medium Earth Orbit satellite
    GEO   = auto()   # Geostationary satellite
    HAPS  = auto()   # High-Altitude Platform Station

    # Terrestrial (TR)
    FR1   = auto()   # 5G sub-6 GHz
    FR3   = auto()   # 5G/6G 7–24 GHz (the "critical band")
    ISAC  = auto()   # 6G Integrated Sensing and Communication
    WIFI7 = auto()   # 802.11be tri-band


# Categorization helpers
NTN_TYPES = {ProviderType.LEO, ProviderType.MEO, ProviderType.GEO, ProviderType.HAPS}
TR_TYPES  = {ProviderType.FR1, ProviderType.FR3, ProviderType.ISAC, ProviderType.WIFI7}


@dataclass
class ProviderPhysics:
    """
    Physical and operational parameters for a connectivity provider type.

    All frequency values are in GHz, distances in km, latencies in ms,
    bandwidths in MHz.
    """
    provider_type:        ProviderType
    display_name:         str

    # Altitude / coverage
    altitude_km_min:      float         # min operational altitude
    altitude_km_max:      float         # max operational altitude
    coverage_radius_km:   float         # nominal footprint radius

    # Spectrum
    freq_bands_ghz:       List[Tuple[float, float]]  # [(f_lo, f_hi), ...]
    max_bandwidth_mhz:    float
    typical_eirp_dbw:     float         # effective isotropic radiated power

    # Link budget defaults
    typical_pl_db:        float         # typical path loss at coverage edge
    typical_snr_db:       float         # expected SNR mid-coverage
    noise_figure_db:      float

    # Latency and dynamics
    rtt_ms_min:           float         # min round-trip time
    rtt_ms_max:           float         # max round-trip time
    doppler_hz_max:       float         # max Doppler shift at band centre
    handover_period_s:    Optional[float]  # None = fixed (GEO/HAPS/TR)

    # Timescale for LTC multi-scale encoder (seconds)
    # The dominant channel decorrelation timescale
    channel_decorr_s:     float

    # Representative centre frequency for propagation model calls (GHz)
    freq_representative_ghz: float = 3.5

    # Coexistence: frequency overlap with other provider types
    coexists_with:        List[ProviderType] = field(default_factory=list)

    # Node feature dimension in the unified graph
    node_feature_dim:     int = 16


def build_provider_registry() -> Dict[ProviderType, ProviderPhysics]:
    """
    Return the full registry of connectivity providers.

    Numeric values follow 3GPP TR 38.821, ITU-R P.618-13, and published
    operator data sheets where available.
    """
    return {
        # ── Non-Terrestrial ────────────────────────────────────────────────
        ProviderType.LEO: ProviderPhysics(
            provider_type      = ProviderType.LEO,
            display_name       = "LEO Satellite (Starlink-class 550 km)",
            altitude_km_min    = 340.0,
            altitude_km_max    = 1200.0,
            coverage_radius_km = 1000.0,
            freq_bands_ghz     = [(10.7, 12.7), (14.0, 14.5), (17.8, 18.6),
                                  (19.7, 20.2), (27.5, 29.1), (29.5, 30.0)],
            max_bandwidth_mhz  = 500.0,
            typical_eirp_dbw   = 35.0,
            typical_pl_db      = 168.0,
            typical_snr_db     = 12.0,
            noise_figure_db    = 1.5,
            rtt_ms_min         = 20.0,
            rtt_ms_max         = 60.0,
            doppler_hz_max     = 48000.0,   # ~3.5 km/s at Ka-band
            handover_period_s  = 90.0,
            channel_decorr_s   = 1.0,       # scintillation dominant
            freq_representative_ghz = 20.0,  # Ka-band downlink
            coexists_with      = [ProviderType.MEO, ProviderType.GEO,
                                  ProviderType.FR3],
        ),
        ProviderType.MEO: ProviderPhysics(
            provider_type      = ProviderType.MEO,
            display_name       = "MEO Satellite (O3b mPOWER 8000 km)",
            altitude_km_min    = 2000.0,
            altitude_km_max    = 20200.0,
            coverage_radius_km = 4000.0,
            freq_bands_ghz     = [(17.7, 20.2), (27.5, 30.0)],
            max_bandwidth_mhz  = 2000.0,   # O3b uses wide beams
            typical_eirp_dbw   = 50.0,
            typical_pl_db      = 186.0,
            typical_snr_db     = 10.0,
            noise_figure_db    = 2.0,
            rtt_ms_min         = 100.0,
            rtt_ms_max         = 250.0,
            doppler_hz_max     = 6000.0,
            handover_period_s  = 600.0,
            channel_decorr_s   = 30.0,     # rain fade timescale
            freq_representative_ghz = 28.0,  # Ka-band
            coexists_with      = [ProviderType.LEO, ProviderType.GEO],
        ),
        ProviderType.GEO: ProviderPhysics(
            provider_type      = ProviderType.GEO,
            display_name       = "GEO Satellite (35786 km)",
            altitude_km_min    = 35786.0,
            altitude_km_max    = 35786.0,
            coverage_radius_km = 8000.0,
            freq_bands_ghz     = [(3.4, 4.2), (5.9, 6.4),
                                  (10.7, 12.75), (14.0, 14.5),
                                  (17.7, 20.2), (27.5, 30.0)],
            max_bandwidth_mhz  = 3000.0,
            typical_eirp_dbw   = 60.0,
            typical_pl_db      = 207.0,
            typical_snr_db     = 8.0,
            noise_figure_db    = 2.5,
            rtt_ms_min         = 480.0,
            rtt_ms_max         = 600.0,
            doppler_hz_max     = 0.0,      # truly geostationary
            handover_period_s  = None,     # fixed position
            channel_decorr_s   = 300.0,    # very slow fading
            freq_representative_ghz = 12.0,  # Ku-band
            coexists_with      = [ProviderType.LEO, ProviderType.MEO,
                                  ProviderType.FR3],
        ),
        ProviderType.HAPS: ProviderPhysics(
            provider_type      = ProviderType.HAPS,
            display_name       = "HAPS (Zephyr-class ~20 km)",
            altitude_km_min    = 18.0,
            altitude_km_max    = 22.0,
            coverage_radius_km = 300.0,
            freq_bands_ghz     = [(0.7, 0.9), (1.9, 2.1),
                                  (3.4, 3.8), (26.0, 28.0)],
            max_bandwidth_mhz  = 400.0,
            typical_eirp_dbw   = 22.0,
            typical_pl_db      = 115.0,
            typical_snr_db     = 18.0,
            noise_figure_db    = 5.0,
            rtt_ms_min         = 1.0,
            rtt_ms_max         = 5.0,
            doppler_hz_max     = 200.0,    # HAPS drifts slowly
            handover_period_s  = 3600.0,   # station-keeping cycles
            channel_decorr_s   = 10.0,
            freq_representative_ghz = 3.5,   # sub-6 band
            coexists_with      = [ProviderType.FR1, ProviderType.FR3,
                                  ProviderType.LEO],
        ),
        # ── Terrestrial ───────────────────────────────────────────────────
        ProviderType.FR1: ProviderPhysics(
            provider_type      = ProviderType.FR1,
            display_name       = "5G FR1 (sub-6 GHz)",
            altitude_km_min    = 0.0,
            altitude_km_max    = 0.1,      # tower height
            coverage_radius_km = 3.0,
            freq_bands_ghz     = [(0.7, 0.9), (1.8, 2.1), (2.5, 2.7),
                                  (3.3, 3.8), (4.4, 5.0)],
            max_bandwidth_mhz  = 100.0,
            typical_eirp_dbw   = 16.0,    # 43 dBm + 3 dB antenna
            typical_pl_db      = 105.0,
            typical_snr_db     = 20.0,
            noise_figure_db    = 7.0,
            rtt_ms_min         = 1.0,
            rtt_ms_max         = 10.0,
            doppler_hz_max     = 1800.0,   # v=300 km/h at 3.5 GHz
            handover_period_s  = 30.0,
            channel_decorr_s   = 0.01,     # fast urban fading (10 ms)
            freq_representative_ghz = 3.5,   # n78 band
            coexists_with      = [ProviderType.FR3, ProviderType.WIFI7,
                                  ProviderType.HAPS],
        ),
        ProviderType.FR3: ProviderPhysics(
            provider_type      = ProviderType.FR3,
            display_name       = "5G/6G FR3 (7–24 GHz)",
            altitude_km_min    = 0.0,
            altitude_km_max    = 0.05,
            coverage_radius_km = 0.5,
            freq_bands_ghz     = [(7.125, 8.5), (10.0, 10.5),
                                  (12.75, 13.25), (14.8, 15.35),
                                  (17.1, 17.3), (24.25, 24.45)],
            max_bandwidth_mhz  = 400.0,   # 3GPP Release 18 target
            typical_eirp_dbw   = 20.0,
            typical_pl_db      = 120.0,
            typical_snr_db     = 25.0,
            noise_figure_db    = 7.0,
            rtt_ms_min         = 0.5,
            rtt_ms_max         = 5.0,
            doppler_hz_max     = 5600.0,
            handover_period_s  = 10.0,
            channel_decorr_s   = 0.005,   # 5 ms urban micro-cell
            freq_representative_ghz = 15.0,  # mid FR3
            coexists_with      = [ProviderType.FR1, ProviderType.LEO,
                                  ProviderType.GEO, ProviderType.ISAC],
        ),
        ProviderType.ISAC: ProviderPhysics(
            provider_type      = ProviderType.ISAC,
            display_name       = "6G ISAC (24–300 GHz)",
            altitude_km_min    = 0.0,
            altitude_km_max    = 0.05,
            coverage_radius_km = 0.1,     # short range sensing
            freq_bands_ghz     = [(24.25, 29.5), (37.0, 40.0),
                                  (57.0, 71.0), (92.0, 114.25)],
            max_bandwidth_mhz  = 2000.0,
            typical_eirp_dbw   = 10.0,
            typical_pl_db      = 100.0,   # very short range
            typical_snr_db     = 30.0,
            noise_figure_db    = 10.0,
            rtt_ms_min         = 0.1,
            rtt_ms_max         = 1.0,
            doppler_hz_max     = 30000.0,  # radar Doppler at mmWave
            handover_period_s  = 2.0,
            channel_decorr_s   = 0.001,   # 1 ms mmWave coherence
            freq_representative_ghz = 28.0,  # mmWave sensing
            coexists_with      = [ProviderType.FR3, ProviderType.WIFI7],
        ),
        ProviderType.WIFI7: ProviderPhysics(
            provider_type      = ProviderType.WIFI7,
            display_name       = "Wi-Fi 7 (802.11be tri-band)",
            altitude_km_min    = 0.0,
            altitude_km_max    = 0.003,   # indoor AP height
            coverage_radius_km = 0.05,
            freq_bands_ghz     = [(2.4, 2.4835), (5.15, 5.85),
                                  (5.925, 7.125)],
            max_bandwidth_mhz  = 320.0,   # 802.11be max channel
            typical_eirp_dbw   = 6.0,    # 23 dBm + 3 dB
            typical_pl_db      = 75.0,
            typical_snr_db     = 35.0,
            noise_figure_db    = 6.0,
            rtt_ms_min         = 0.1,
            rtt_ms_max         = 2.0,
            doppler_hz_max     = 100.0,   # pedestrian speed
            handover_period_s  = 5.0,
            channel_decorr_s   = 0.02,    # indoor coherence time
            freq_representative_ghz = 5.9,   # 6 GHz band
            coexists_with      = [ProviderType.FR1, ProviderType.ISAC],
        ),
    }


# Singleton registry
PROVIDER_REGISTRY: Dict[ProviderType, ProviderPhysics] = build_provider_registry()


def get_provider(ptype: ProviderType) -> ProviderPhysics:
    """Look up a provider by type."""
    return PROVIDER_REGISTRY[ptype]


def get_all_provider_types() -> List[ProviderType]:
    """Return all provider types in canonical order."""
    return list(ProviderType)


def provider_node_features(
    ptype: ProviderType,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a fixed feature vector encoding the physics of a provider type.

    Returns a 1-D tensor of shape (PROVIDER_FEATURE_DIM,) normalised to [0, 1].
    Used as node-type embeddings in the heterogeneous GNN encoder.
    """
    p = PROVIDER_REGISTRY[ptype]

    # Encode key physics scalars (log-scale where orders-of-magnitude vary)
    import math
    feats = torch.tensor([
        math.log10(max(p.altitude_km_min,   1.0) + 1) / 6.0,   # alt_min (0=ground, 1=GEO)
        math.log10(max(p.altitude_km_max,   1.0) + 1) / 6.0,   # alt_max
        math.log10(p.coverage_radius_km + 1) / 5.0,            # coverage
        math.log10(p.max_bandwidth_mhz)  / 4.0,                # bandwidth
        p.typical_snr_db       / 40.0,                         # SNR
        p.noise_figure_db      / 15.0,                         # NF
        math.log10(p.rtt_ms_min + 0.01) / 4.0,                 # latency_min
        math.log10(p.rtt_ms_max + 0.01) / 4.0,                 # latency_max
        math.log10(p.doppler_hz_max + 1) / 6.0,                # doppler
        math.log10(p.channel_decorr_s  + 1e-4) / 4.0,         # decorr_time
        float(ptype in NTN_TYPES),                              # is_ntn
        float(ptype in TR_TYPES),                               # is_terrestrial
        float(ptype == ProviderType.ISAC),                      # has_sensing
        float(p.handover_period_s is None),                     # is_fixed
        float(len(p.freq_bands_ghz)) / 6.0,                    # num_bands
        float(len(p.coexists_with)) / 7.0,                     # coexistence_degree
    ], dtype=torch.float32, device=device)

    return feats.clamp(0.0, 1.0)


# Fixed dimension for provider node feature vectors
PROVIDER_FEATURE_DIM: int = 16


def coexistence_adjacency(device: torch.device) -> torch.Tensor:
    """
    Build the static coexistence adjacency matrix for all provider types.

    A[i, j] = 1 if provider i and j share spectrum (potential interference).
    Shape: (num_providers, num_providers).
    """
    types = get_all_provider_types()
    n = len(types)
    idx = {t: i for i, t in enumerate(types)}
    adj = torch.zeros(n, n, device=device)
    for ptype, physics in PROVIDER_REGISTRY.items():
        i = idx[ptype]
        for other in physics.coexists_with:
            j = idx[other]
            adj[i, j] = 1.0
            adj[j, i] = 1.0  # symmetric
    return adj
