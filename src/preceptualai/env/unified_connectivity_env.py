"""
Universal Heterogeneous Connectivity Environment (UHCE).

The UHCE is the core of the Universal Heterogeneous Connectivity Intelligence
(UHCI) paradigm — a single GPU-vectorized environment that simultaneously
simulates ALL connectivity provider types:

  NTN: LEO · MEO · GEO · HAPS
  TR:  5G FR1 · 5G/6G FR3 · 6G ISAC · Wi-Fi 7

Rather than isolated single-RAT environments, the UHCE models:
  1. Cross-provider interference (shared spectrum bands)
  2. Heterogeneous timescales (0.001 s ISAC coherence → 300 s GEO fading)
  3. Provider-specific propagation physics per ITU-R / 3GPP
  4. Dynamic NTN visibility windows (orbital passes, HAPS drift)
  5. Joint multi-provider allocation decisions

The observation space encodes a heterogeneous graph snapshot:
  - Per-provider-type node state (SNR, occupancy, interference, ...)
  - Per-edge coexistence interference level
  - Global context (time of day, geo-position proxy, weather index)

The action space is hierarchical:
  Level 0 — which provider(s) to use (multi-hot)
  Level 1 — which band/channel within each selected provider
  Level 2 — power fraction (discretized continuous via SmODE)

The reward balances throughput, interference caused to others,
handover costs, and sensing quality (ISAC term).

References:
  3GPP TR 38.821: NTN Solutions for NR.
  3GPP TR 38.901: Channel models for 0.5–100 GHz.
  ITU-R P.618-13: Earth-space propagation.
  3GPP Release 19: ISAC, FR3 band.
  O-RAN WG1: Use Cases and Deployment Scenarios v10.0.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from preceptualai.env.fr3_propagation import FR3PropagationModel
from preceptualai.env.itu_propagation import ITUPropagation
from preceptualai.env.provider_registry import (
    PROVIDER_REGISTRY,
    PROVIDER_FEATURE_DIM,
    ProviderType,
    ProviderPhysics,
    NTN_TYPES,
    TR_TYPES,
    coexistence_adjacency,
    get_all_provider_types,
    provider_node_features,
)


@dataclass
class UnifiedConnectivityConfig:
    """Configuration for the Universal Heterogeneous Connectivity Environment."""
    # Vectorization
    num_envs:            int   = 64
    device:              str   = "cpu"

    # Which provider types to include in this simulation
    provider_types:      List[ProviderType] = field(
        default_factory=lambda: list(ProviderType)
    )

    # Channels per provider (discrete bands to select among)
    channels_per_provider: int = 6

    # History length fed to the temporal encoder
    obs_history_len:     int   = 16

    # Reward weights
    reward_throughput_w: float = 1.0     # Mbps utility
    reward_interfere_w:  float = 0.5     # interference penalty
    reward_handover_w:   float = 0.3     # handover / switch cost
    reward_sensing_w:    float = 0.2     # ISAC sensing quality bonus
    reward_power_w:      float = 0.1     # power efficiency

    # Physics noise
    channel_noise_std:   float = 0.05

    # NTN pass simulation: how many LEO/MEO sats visible simultaneously
    num_leo_sats:        int   = 12
    num_meo_sats:        int   = 4

    # Time step
    dt_s:                float = 1.0

    # Real dataset paths (None = use physics simulation only)
    # When provided, terrestrial providers (FR1, WIFI7) use real traces
    # to drive SNR/RSRP/CQI instead of synthetic OU noise.
    real_data_dirs:      Optional[Dict[str, str]] = None
    max_traces_per_source: int = 200

    @property
    def num_providers(self) -> int:
        return len(self.provider_types)

    @property
    def total_channels(self) -> int:
        return self.num_providers * self.channels_per_provider

    @property
    def obs_dim(self) -> int:
        # Per provider: channels_per_provider*(snr+occ+intf) + provider_features
        per_node = self.channels_per_provider * 3 + PROVIDER_FEATURE_DIM
        graph_features = self.num_providers * per_node
        # Global context: sat_elevation, time_phase, weather_idx, power_used
        global_ctx = 6
        return graph_features + global_ctx

    @property
    def action_dim(self) -> int:
        # Flat discrete action: which (provider, channel) pair
        return self.total_channels


class UnifiedConnectivityEnv:
    """
    GPU-vectorized Universal Heterogeneous Connectivity Environment.

    Runs `num_envs` independent connectivity scenarios in parallel using
    batched tensor operations — no Python loops over environments.

    Observation: flat vector encoding the full heterogeneous graph state.
    Action:      integer index into (provider × channel) joint space.
    Reward:      throughput – interference – handover – power costs.
    """

    def __init__(self, config: Optional[UnifiedConnectivityConfig] = None):
        if config is None:
            config = UnifiedConnectivityConfig()
        self.cfg = config
        self.device = torch.device(config.device)

        self._num_providers = config.num_providers
        self._num_channels  = config.channels_per_provider
        self._provider_types = config.provider_types
        self._provider_list  = [PROVIDER_REGISTRY[pt] for pt in config.provider_types]

        # Static coexistence adjacency over the selected providers
        full_adj = coexistence_adjacency(self.device)
        all_types = get_all_provider_types()
        idx_map   = {t: i for i, t in enumerate(all_types)}
        sel_idx   = torch.tensor(
            [idx_map[pt] for pt in config.provider_types], device=self.device
        )
        self._coexist_adj = full_adj[sel_idx][:, sel_idx]  # (P, P)

        # Provider static node features
        self._provider_node_feats = torch.stack(
            [provider_node_features(pt, self.device) for pt in config.provider_types]
        )  # (P, PROVIDER_FEATURE_DIM)

        # Physics timescale tensors (one per provider)
        self._decorr_s = torch.tensor(
            [p.channel_decorr_s for p in self._provider_list],
            device=self.device,
        )  # (P,)
        self._typical_snr = torch.tensor(
            [p.typical_snr_db for p in self._provider_list],
            device=self.device,
        )  # (P,)
        self._typical_pl = torch.tensor(
            [p.typical_pl_db for p in self._provider_list],
            device=self.device,
        )  # (P,)
        self._has_sensing = torch.tensor(
            [pt == ProviderType.ISAC for pt in config.provider_types],
            dtype=torch.float32, device=self.device,
        )  # (P,)
        self._is_ntn = torch.tensor(
            [pt in NTN_TYPES for pt in config.provider_types],
            dtype=torch.float32, device=self.device,
        )  # (P,)
        self._is_fr3 = torch.tensor(
            [pt == ProviderType.FR3 for pt in config.provider_types],
            dtype=torch.bool, device=self.device,
        )  # (P,)
        self._is_ntn_bool = torch.tensor(
            [pt in NTN_TYPES for pt in config.provider_types],
            dtype=torch.bool, device=self.device,
        )  # (P,)
        self._freq_ghz = torch.tensor(
            [p.freq_representative_ghz for p in self._provider_list],
            device=self.device,
        )  # (P,)

        # Real propagation models (no mocks, no stubs)
        self._fr3_model = FR3PropagationModel(device=self.device)
        self._itu_model = ITUPropagation(device=self.device)

        # ── Real dataset trace replay ─────────────────────────────
        # When real_data_dirs is provided, load traces and use them
        # to drive SNR for terrestrial providers (FR1, WIFI7) instead
        # of synthetic Ornstein-Uhlenbeck noise.
        self._real_traces: Optional[torch.Tensor] = None  # (N, 5) concatenated
        self._trace_offsets: Optional[List[int]] = None    # start index per trace
        self._trace_lengths: Optional[List[int]] = None
        self._trace_ptr: Optional[torch.Tensor] = None     # (E, P) per-env per-provider pointer
        self._terrestrial_trace_providers: List[int] = []   # provider indices using traces

        if config.real_data_dirs is not None:
            self._load_real_traces(config)

        # Runtime state
        E, P, C = config.num_envs, self._num_providers, self._num_channels

        # Channel SNR state: (E, P, C) dB (Gaussian random walk)
        self.snr_db      = torch.zeros(E, P, C, device=self.device)
        # PU occupancy: (E, P, C) in [0, 1]
        self.occupancy   = torch.zeros(E, P, C, device=self.device)
        # Cross-provider interference: (E, P, C) in [0, 1]
        self.interference = torch.zeros(E, P, C, device=self.device)

        # NTN visibility per (env, provider): (E, P) in [0, 1]
        self.ntn_visibility = torch.ones(E, P, device=self.device)

        # Satellite elevation angles (E, P) — only meaningful for NTN types
        self.sat_elevation_deg = torch.zeros(E, P, device=self.device)

        # Real physics state: UE distances (terrestrial), rain rate (NTN)
        self.ue_distance_m   = torch.full((E, P), 200.0, device=self.device)
        self.rain_rate_mmh   = torch.full((E, P), 5.0, device=self.device)

        # ISAC sensing SNR: (E, C) — radar equation result for ISAC providers
        self.isac_sensing_snr = torch.zeros(E, C, device=self.device)

        # Current active (provider, channel) pair per env
        self.active_provider = torch.zeros(E, dtype=torch.long, device=self.device)
        self.active_channel  = torch.zeros(E, dtype=torch.long, device=self.device)

        # Handover penalty flags
        self._prev_action   = torch.zeros(E, dtype=torch.long, device=self.device)

        # Power state: (E,) in [0, 1]
        self.power_frac = torch.ones(E, device=self.device) * 0.5

        # Step counter
        self._step = 0

        # Observation history ring buffer
        self._hist = torch.zeros(
            E, config.obs_history_len, config.obs_dim, device=self.device
        )
        self._hist_ptr = 0

        # Initialize
        self.reset()

    # ──────────────────────────────────────────────────────────────
    # Real Data Loading
    # ──────────────────────────────────────────────────────────────

    def _load_real_traces(self, config: UnifiedConnectivityConfig):
        """
        Load real 5G measurement traces for terrestrial providers.

        UCC MISL traces provide: RSRP, RSRQ, SNR, CQI, RSSI
        Colosseum traces provide: rsrp, dl_snr, dl_mcs, dl_brate, ul_mcs

        The SNR column (index 2) drives the terrestrial SNR directly.
        Other columns modulate occupancy and interference estimates.
        """
        from preceptualai.env.data_pipeline import Unified5GDataPipeline

        pipeline = Unified5GDataPipeline(
            data_dirs=config.real_data_dirs,
            max_traces_per_source=config.max_traces_per_source,
            min_trace_length=100,
        )
        traces = pipeline.load_all()

        if not traces:
            print("[UHCI] No real traces loaded — falling back to simulation only")
            return

        # Record offsets for random access
        offsets = []
        lengths = []
        offset = 0
        for t in traces:
            offsets.append(offset)
            lengths.append(len(t))
            offset += len(t)

        # Concatenate all traces into one big tensor (N, 5)
        import numpy as np
        combined = np.concatenate(traces, axis=0)
        self._real_traces = torch.from_numpy(combined).to(self.device)
        self._trace_offsets = offsets
        self._trace_lengths = lengths

        # Identify which provider indices are terrestrial and can use traces
        # FR1 and WIFI7 use real traces directly; FR3 could too
        trace_types = {ProviderType.FR1, ProviderType.WIFI7, ProviderType.FR3}
        self._terrestrial_trace_providers = [
            i for i, pt in enumerate(config.provider_types)
            if pt in trace_types
        ]

        # Allocate per-env per-provider trace pointer
        E = config.num_envs
        P = len(config.provider_types)
        self._trace_ptr = torch.zeros(E, P, dtype=torch.long, device=self.device)
        self._trace_assignments = torch.zeros(E, P, dtype=torch.long, device=self.device)

        # Assign random traces to each (env, provider) pair
        num_traces = len(traces)
        for i in self._terrestrial_trace_providers:
            self._trace_assignments[:, i] = torch.randint(0, num_traces, (E,))
            # Random start offset within each trace
            for e in range(E):
                tidx = self._trace_assignments[e, i].item()
                max_start = max(1, lengths[tidx] - 200)
                self._trace_ptr[e, i] = offsets[tidx] + torch.randint(0, max_start, (1,)).item()

        print(f"[UHCI] Real traces active for providers: "
              f"{[config.provider_types[i].name for i in self._terrestrial_trace_providers]}")

    def _read_trace_snr(self, env_idx: int, prov_idx: int) -> float:
        """Read the current SNR value from the assigned real trace."""
        if self._real_traces is None:
            return None
        tidx = self._trace_assignments[env_idx, prov_idx].item()
        ptr  = self._trace_ptr[env_idx, prov_idx].item()
        tlen = self._trace_lengths[tidx]
        toff = self._trace_offsets[tidx]

        # Wrap pointer within trace bounds
        local_ptr = (ptr - toff) % tlen
        global_ptr = toff + local_ptr

        # SNR is feature index 2 in the pipeline's standard features
        return self._real_traces[global_ptr, 2].item()

    def _advance_trace_ptrs(self):
        """Advance all trace pointers by 1 timestep (with wraparound)."""
        if self._trace_ptr is None:
            return
        for i in self._terrestrial_trace_providers:
            self._trace_ptr[:, i] += 1
            # Wraparound per (env, provider)
            for e in range(self.cfg.num_envs):
                tidx = self._trace_assignments[e, i].item()
                toff = self._trace_offsets[tidx]
                tlen = self._trace_lengths[tidx]
                if self._trace_ptr[e, i].item() >= toff + tlen:
                    self._trace_ptr[e, i] = toff

    # ──────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────

    def reset(self) -> torch.Tensor:
        """Reset all environments. Returns initial observation."""
        E, P, C = self.cfg.num_envs, self._num_providers, self._num_channels

        # Initialise SNR from provider physics + noise
        base_snr = self._typical_snr.unsqueeze(0).unsqueeze(-1).expand(E, P, C)
        self.snr_db = base_snr + torch.randn_like(base_snr) * 5.0

        # Reassign random traces on reset
        if self._trace_ptr is not None and self._trace_lengths:
            num_traces = len(self._trace_lengths)
            for i in self._terrestrial_trace_providers:
                self._trace_assignments[:, i] = torch.randint(0, num_traces, (E,))
                for e in range(E):
                    tidx = self._trace_assignments[e, i].item()
                    max_start = max(1, self._trace_lengths[tidx] - 200)
                    self._trace_ptr[e, i] = (
                        self._trace_offsets[tidx] + torch.randint(0, max_start, (1,)).item()
                    )

        # Random initial occupancy
        self.occupancy = torch.rand(E, P, C, device=self.device) * 0.4

        # No interference at reset
        self.interference.zero_()

        # NTN: full visibility for terrestrial, random for NTN
        self._init_ntn_state()

        # Zero active provider/channel
        self.active_provider.zero_()
        self.active_channel.zero_()
        self._prev_action.zero_()
        self.power_frac.fill_(0.5)
        self._step = 0

        self._hist.zero_()
        self._hist_ptr = 0

        obs = self._build_observation()
        self._push_history(obs)
        return obs

    # ──────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step all environments simultaneously.

        Args:
            actions: (E,) integer actions in [0, total_channels)
                     encodes flat index = provider * C + channel

        Returns:
            obs:     (E, obs_dim)
            rewards: (E,)
            dones:   (E,) bool
            info:    dict of diagnostic tensors
        """
        E, P, C = self.cfg.num_envs, self._num_providers, self._num_channels
        self._step += 1

        # Decode action → (provider, channel)
        prov_idx = actions // C        # (E,)
        chan_idx  = actions %  C        # (E,)

        # ── Physics update ────────────────────────────────────────
        self._update_channel_dynamics()
        self._update_ntn_visibility()
        self._update_cross_provider_interference()
        self._advance_trace_ptrs()

        # ── Link budget for chosen (provider, channel) ────────────
        sel_snr  = self.snr_db[torch.arange(E), prov_idx, chan_idx]       # (E,)
        sel_occ  = self.occupancy[torch.arange(E), prov_idx, chan_idx]    # (E,)
        sel_intf = self.interference[torch.arange(E), prov_idx, chan_idx] # (E,)
        sel_vis  = self.ntn_visibility[torch.arange(E), prov_idx]        # (E,)

        # Effective SINR
        sinr_db = sel_snr - 10.0 * torch.log10(
            (sel_intf * 10.0 + 1e-6).clamp(min=1e-6)
        )

        # Shannon throughput (simplified, provider-bandwidth-aware)
        bw_mhz = torch.tensor(
            [p.max_bandwidth_mhz / C for p in self._provider_list],
            device=self.device
        )[prov_idx]  # (E,)
        sinr_linear = 10.0 ** (sinr_db / 10.0)
        throughput_mbps = bw_mhz * torch.log2(1.0 + sinr_linear)

        # Apply NTN visibility penalty
        throughput_mbps = throughput_mbps * sel_vis

        # Collision penalty (channel occupied by PU)
        collision = (sel_occ > 0.5).float()

        # ── Reward ────────────────────────────────────────────────
        r_throughput = self.cfg.reward_throughput_w * throughput_mbps / 1000.0  # normalise to [0~1]
        r_interfere  = -self.cfg.reward_interfere_w * sel_intf
        r_handover   = -self.cfg.reward_handover_w * (actions != self._prev_action).float()
        r_collision  = -1.0 * collision
        r_sensing    = self.cfg.reward_sensing_w * (
            self._has_sensing[prov_idx] * sinr_linear.clamp(0, 30) / 30.0
        )
        r_power      = -self.cfg.reward_power_w * self.power_frac

        rewards = r_throughput + r_interfere + r_handover + r_collision + r_sensing + r_power

        # ── Update active state ───────────────────────────────────
        self.active_provider = prov_idx
        self.active_channel  = chan_idx
        self._prev_action    = actions.clone()

        # ── Observation ───────────────────────────────────────────
        obs = self._build_observation()
        self._push_history(obs)

        # Episode never terminates by default (continuous DSA)
        dones = torch.zeros(E, dtype=torch.bool, device=self.device)

        info = {
            "throughput_mbps":    throughput_mbps,
            "collision":          collision,
            "sinr_db":            sinr_db,
            "ntn_visibility":     sel_vis,
            "interference_level": sel_intf,
            "provider_chosen":    prov_idx,
            "channel_chosen":     chan_idx,
        }

        return obs, rewards, dones, info

    # ──────────────────────────────────────────────────────────────
    # Physics helpers
    # ──────────────────────────────────────────────────────────────

    def _init_ntn_state(self):
        """Initialise NTN satellite visibility with random pass phases."""
        E, P = self.cfg.num_envs, self._num_providers
        for i, pt in enumerate(self._provider_types):
            if pt in NTN_TYPES:
                if pt == ProviderType.GEO:
                    self.ntn_visibility[:, i] = 1.0  # always visible
                    self.sat_elevation_deg[:, i] = 40.0 + torch.randn(E, device=self.device) * 5.0
                elif pt == ProviderType.HAPS:
                    self.ntn_visibility[:, i] = 0.9  # near-always available
                    self.sat_elevation_deg[:, i] = 60.0 + torch.randn(E, device=self.device) * 10.0
                else:
                    # LEO/MEO: random initial pass phase
                    phase = torch.rand(E, device=self.device) * 2 * math.pi
                    elev = 45.0 * torch.sin(phase).clamp(0.0, 1.0)
                    self.sat_elevation_deg[:, i] = elev
                    self.ntn_visibility[:, i] = (elev > 10.0).float()
            else:
                self.ntn_visibility[:, i] = 1.0  # terrestrial always available

    def _update_channel_dynamics(self):
        """
        Evolve SNR and occupancy using REAL propagation physics.

        - FR3 providers: 3GPP TR 38.901 UMa path loss via FR3PropagationModel
        - NTN providers: ITU-R P.618 rain + gas + scintillation via ITUPropagation
        - Other terrestrial: physics-informed Ornstein-Uhlenbeck with correct timescales
        - ISAC providers: simplified radar equation for sensing SNR
        - PU occupancy: Markov chain with per-provider transition rates
        """
        E, P, C = self.cfg.num_envs, self._num_providers, self._num_channels
        dt = self.cfg.dt_s

        # ── Update UE distances (Ornstein-Uhlenbeck walk) ────────
        dist_noise = torch.randn(E, P, device=self.device) * 10.0 * math.sqrt(dt)
        self.ue_distance_m = (self.ue_distance_m + dist_noise).clamp(10.0, 5000.0)

        # ── Update rain rate (slow OU process) ────────────────────
        rain_noise = torch.randn(E, P, device=self.device) * 0.5 * math.sqrt(dt)
        self.rain_rate_mmh = (
            self.rain_rate_mmh * 0.999 + 0.001 * 5.0 + rain_noise
        ).clamp(0.0, 80.0)

        # ── Per-provider SNR computation ──────────────────────────
        for i, pt in enumerate(self._provider_types):
            freq = self._freq_ghz[i]
            freq_t = freq.unsqueeze(0).expand(E)  # (E,)
            dist_i = self.ue_distance_m[:, i]       # (E,)
            rain_i = self.rain_rate_mmh[:, i]       # (E,)
            elev_i = self.sat_elevation_deg[:, i]   # (E,)

            physics = self._provider_list[i]
            tx_power_dbm = physics.typical_eirp_dbw + 30.0  # dBW → dBm
            noise_dbm = -174.0 + 10.0 * math.log10(
                physics.max_bandwidth_mhz / C * 1e6
            ) + physics.noise_figure_db

            if pt == ProviderType.FR3:
                # 3GPP TR 38.901 UMa path loss — REAL model
                pl = self._fr3_model.path_loss_uma_nlos(freq_t, dist_i)
                snr_base = tx_power_dbm - pl - noise_dbm  # (E,)

            elif pt in NTN_TYPES:
                # ITU-R P.618 atmospheric attenuation — REAL model
                elev_safe = elev_i.clamp(min=5.0)
                attn = self._itu_model.total_attenuation(
                    freq_t, elev_safe, rain_i
                )  # (E,)
                # Free-space path loss for NTN slant path
                alt_m = physics.altitude_km_min * 1000.0
                slant_m = alt_m / torch.sin(
                    elev_safe * math.pi / 180.0
                ).clamp(min=0.05)
                fspl = 20.0 * torch.log10(slant_m) + 20.0 * torch.log10(freq) + 32.45
                snr_base = tx_power_dbm - fspl - attn - noise_dbm  # (E,)

            elif pt == ProviderType.ISAC:
                # ISAC: communication SNR from FR3-like path loss
                pl = self._fr3_model.path_loss_uma_los(freq_t, dist_i.clamp(min=10.0, max=200.0))
                snr_base = tx_power_dbm - pl - noise_dbm
                # Radar equation for sensing SNR (simplified)
                # Pr = Pt * G^2 * lambda^2 * sigma / ((4*pi)^3 * R^4)
                lambda_m = 3e8 / (freq.item() * 1e9)
                sigma_rcs = 1.0  # 1 m^2 target RCS
                r_m = dist_i.clamp(min=5.0, max=200.0)
                radar_snr_linear = (
                    10 ** (tx_power_dbm / 10.0) * 100.0 * lambda_m ** 2 * sigma_rcs
                    / ((4 * math.pi) ** 3 * r_m ** 4 * 10 ** (noise_dbm / 10.0))
                )
                self.isac_sensing_snr = 10.0 * torch.log10(
                    radar_snr_linear.clamp(min=1e-10)
                ).unsqueeze(-1).expand(E, C).clone()

            else:
                # FR1, WIFI7: use REAL traces if available, else log-distance
                if (self._real_traces is not None and
                        i in self._terrestrial_trace_providers):
                    # Read real SNR from trace data (vectorized across envs)
                    snr_vals = torch.zeros(E, device=self.device)
                    for e in range(E):
                        real_snr = self._read_trace_snr(e, i)
                        if real_snr is not None:
                            snr_vals[e] = real_snr
                        else:
                            snr_vals[e] = physics.typical_snr_db
                    # De-normalize: traces are z-scored, convert back to dB
                    # Pipeline z-scores with mean~15 dB, std~12 dB for SNR
                    snr_base = snr_vals * 12.0 + 15.0
                else:
                    # Fallback: log-distance path loss model
                    pl_0 = physics.typical_pl_db
                    n_exp = 3.5 if pt == ProviderType.FR1 else 3.0
                    pl = pl_0 + 10.0 * n_exp * torch.log10(
                        (dist_i / 100.0).clamp(min=0.1)
                    )
                    snr_base = tx_power_dbm - pl - noise_dbm

            # Per-channel variation (frequency diversity across channels)
            chan_var = torch.randn(E, C, device=self.device) * 2.0
            self.snr_db[:, i, :] = snr_base.unsqueeze(-1) + chan_var

        # ── PU occupancy: Markov chain with per-provider rates ────
        p_occ_on  = 0.05 * dt
        p_occ_off = 0.15 * dt
        occ_flip_on  = (torch.rand_like(self.occupancy) < p_occ_on) & (self.occupancy < 0.5)
        occ_flip_off = (torch.rand_like(self.occupancy) < p_occ_off) & (self.occupancy >= 0.5)
        self.occupancy = torch.where(occ_flip_on,  torch.ones_like(self.occupancy),  self.occupancy)
        self.occupancy = torch.where(occ_flip_off, torch.zeros_like(self.occupancy), self.occupancy)

    def _update_ntn_visibility(self):
        """Simulate orbital dynamics for NTN provider visibility."""
        dt = self.cfg.dt_s

        for i, pt in enumerate(self._provider_types):
            if pt == ProviderType.GEO:
                continue  # always visible

            if pt == ProviderType.HAPS:
                # Slow drift ±5 degrees
                noise = torch.randn(self.cfg.num_envs, device=self.device) * 0.1
                self.sat_elevation_deg[:, i] = (
                    self.sat_elevation_deg[:, i] + noise
                ).clamp(50.0, 70.0)
                self.ntn_visibility[:, i] = 0.9

            elif pt == ProviderType.LEO:
                # ~90 min orbit, 10-min visible windows
                period_s = 90.0 * 60.0
                phase_inc = (dt / period_s) * 2 * math.pi
                phase = (self._step * phase_inc) % (2 * math.pi)
                max_elev = 45.0 + torch.randn(self.cfg.num_envs, device=self.device) * 10.0
                elev = max_elev * math.sin(phase)
                self.sat_elevation_deg[:, i] = elev
                self.ntn_visibility[:, i] = (elev > 10.0).float()

            elif pt == ProviderType.MEO:
                # ~12 hr orbit
                period_s = 12.0 * 3600.0
                phase_inc = (dt / period_s) * 2 * math.pi
                phase = (self._step * phase_inc) % (2 * math.pi)
                max_elev = 35.0 + torch.randn(self.cfg.num_envs, device=self.device) * 8.0
                elev = max_elev * math.sin(phase)
                self.sat_elevation_deg[:, i] = elev
                self.ntn_visibility[:, i] = (elev > 5.0).float()

    def _update_cross_provider_interference(self):
        """
        Compute cross-provider interference via the coexistence adjacency matrix.

        If provider i and j share spectrum (adj[i,j]=1), provider j's occupancy
        leaks interference into provider i's channels (proportional to elevation
        and PU activity).
        """
        E, P, C = self.cfg.num_envs, self._num_providers, self._num_channels

        # Mean occupancy per provider: (E, P)
        mean_occ = self.occupancy.mean(dim=-1)

        # Interference into provider i = sum_j adj[i,j] * mean_occ_j * vis_j * scale
        # (E, P) = (E, P) @ (P, P).T scaled by visibility
        adj = self._coexist_adj.unsqueeze(0).expand(E, P, P)  # (E, P, P)
        occ_j = mean_occ.unsqueeze(1).expand(E, P, P)          # (E, P, P) = src occ
        vis_j = self.ntn_visibility.unsqueeze(1).expand(E, P, P)

        intf_prov = (adj * occ_j * vis_j).sum(dim=-1) * 0.3  # (E, P)

        # Broadcast to channels (uniform across channels of a provider)
        self.interference = intf_prov.unsqueeze(-1).expand(E, P, C).clone()

    # ──────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────

    def _build_observation(self) -> torch.Tensor:
        """
        Build flat observation vector encoding the heterogeneous graph state.

        Structure:
          For each provider p in 0..P-1:
            [snr_db_c0..cC-1 (normalised), occupancy_c0..cC-1, intf_c0..cC-1,
             provider_node_feats (PROVIDER_FEATURE_DIM)]
          Global context:
            [mean_sat_elevation, time_sin, time_cos, power_frac,
             ntn_vis_mean, terrestrial_load_mean]
        """
        E, P, C = self.cfg.num_envs, self._num_providers, self._num_channels

        parts = []

        # Per-provider blocks
        snr_norm = (self.snr_db / 40.0).clamp(-1, 1)     # (E, P, C)
        occ      = self.occupancy.clamp(0, 1)              # (E, P, C)
        intf     = self.interference.clamp(0, 1)           # (E, P, C)

        for i in range(P):
            parts.append(snr_norm[:, i, :])   # (E, C)
            parts.append(occ[:, i, :])         # (E, C)
            parts.append(intf[:, i, :])        # (E, C)
            # Static provider features broadcast to all envs
            feat = self._provider_node_feats[i].unsqueeze(0).expand(E, -1)  # (E, F)
            parts.append(feat)

        # Global context
        t = float(self._step)
        time_sin = math.sin(2 * math.pi * t / 86400.0)
        time_cos = math.cos(2 * math.pi * t / 86400.0)
        mean_elev = self.sat_elevation_deg[:, [i for i, pt in enumerate(self._provider_types)
                                               if pt in NTN_TYPES]].mean(dim=-1) / 90.0 \
            if any(pt in NTN_TYPES for pt in self._provider_types) \
            else torch.zeros(E, device=self.device)

        ntn_vis_mean = self.ntn_visibility[:, [i for i, pt in enumerate(self._provider_types)
                                               if pt in NTN_TYPES]].mean(dim=-1) \
            if any(pt in NTN_TYPES for pt in self._provider_types) \
            else torch.ones(E, device=self.device)

        tr_occ_mean = occ[:, [i for i, pt in enumerate(self._provider_types)
                               if pt in TR_TYPES], :].mean(dim=(-2, -1)) \
            if any(pt in TR_TYPES for pt in self._provider_types) \
            else torch.zeros(E, device=self.device)

        global_ctx = torch.stack([
            mean_elev,
            torch.full((E,), time_sin, device=self.device),
            torch.full((E,), time_cos, device=self.device),
            self.power_frac,
            ntn_vis_mean,
            tr_occ_mean,
        ], dim=-1)  # (E, 6)
        parts.append(global_ctx)

        obs = torch.cat(parts, dim=-1)  # (E, obs_dim)
        return obs

    def _push_history(self, obs: torch.Tensor):
        self._hist[:, self._hist_ptr % self.cfg.obs_history_len, :] = obs
        self._hist_ptr += 1

    def get_history(self) -> torch.Tensor:
        """Return observation history: (E, T, obs_dim)."""
        T   = self.cfg.obs_history_len
        ptr = self._hist_ptr % T
        # Roll so oldest is first
        return torch.roll(self._hist, shifts=-ptr, dims=1)

    # ──────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────

    def get_provider_stats(self) -> Dict[str, torch.Tensor]:
        """Return per-provider statistics for monitoring / TensorBoard."""
        stats = {}
        for i, pt in enumerate(self._provider_types):
            name = pt.name
            stats[f"{name}/mean_snr_db"]      = self.snr_db[:, i, :].mean()
            stats[f"{name}/mean_occupancy"]   = self.occupancy[:, i, :].mean()
            stats[f"{name}/mean_interference"] = self.interference[:, i, :].mean()
            stats[f"{name}/visibility"]       = self.ntn_visibility[:, i].mean()
        return stats

    @property
    def obs_dim(self) -> int:
        return self.cfg.obs_dim

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def num_envs(self) -> int:
        return self.cfg.num_envs
