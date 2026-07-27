"""Sionna link-level PHY → Horizon telemetry seam.

This is the concrete integration point the project's core comment declares
(``pyproject.toml``: "the Shield validates the *decisions* a neural-PHY
block emits (in production, from NVIDIA Aerial cuBB / O-RAN E2 KPM)"). Here
the neural-PHY block is stood in for by a **real coded MIMO-OFDM link-level
simulation** (NVIDIA Sionna, Apache-2.0) driven over **real ray-traced MIMO
channels** (DeepMIMO). Each receiver location becomes a PHY KPI record
(post-equalisation SINR, coded BLER, throughput, spectral efficiency), which
this module turns into a ``TelemetryEvent`` (modality ``ue_qos``) and drives
through the existing ``DecisionPipeline`` — planner → Decision Safety Shield →
guard chain → A1 policy.

The pure-Python half (``PhyMeasurement`` → telemetry → risk → pipeline) has no
heavy dependencies and is unit-tested directly. The link-level simulation
(``run_link_level_over_channels``) imports ``sionna`` lazily, so importing this
module never pulls the PyTorch stack; callers that need the simulation install
the ``phy`` extra (``sionna`` 2.x, PyTorch-backed) and the ``realdata`` extra
(``deepmimo``) to source the channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from horizon_ric.io.schemas import TelemetryEvent

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = structlog.get_logger(__name__)

# Risk-model identifier stamped into telemetry so a regulator can tell which
# KPI→risk mapping produced a decision. Bump on any change to _risk().
PHY_RISK_MODEL = "phy_risk_v1"

# Nominal 5G NR reference for the spectral-efficiency / throughput scaling and
# the SLA thresholds below. These are documented operating points, not tuning
# knobs pulled from the data.
_TARGET_BLER = 0.10          # 3GPP link-adaptation operating point (10% iBLER)
_MIN_USABLE_SINR_DB = -3.0   # below this an LDPC MCS-0 link does not close
_GOOD_SINR_DB = 20.0         # at/above this the link is comfortably healthy


@dataclass
class PhyMeasurement:
    """One receiver's link-level PHY KPIs over a real ray-traced channel.

    Field names align to 3GPP TS 28.552 where a counterpart exists
    (``DRB.UEThpDl`` ↔ ``throughput_mbps``, ``DRB.RlcSduDelayDl`` has no
    link-level analogue here so latency is left to the RAN).
    """

    receiver_index: int
    position_m: tuple[float, float, float]
    post_eq_sinr_db: float
    coded_bler: float
    throughput_mbps: float
    spectral_eff_bps_hz: float
    prb_util_pct: float
    mcs_bits_per_symbol: float
    n_tx: int
    n_rx: int
    ebn0_db: float
    scenario: str = "asu_campus_3p5"
    extra: dict[str, Any] = field(default_factory=dict)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _risk(m: PhyMeasurement) -> float:
    """Map link-level KPIs → ``sla_risk_30s`` in [0, 1] (``phy_risk_v1``).

    Three physical failure modes, combined by taking the worst (an SLA is
    breached if *any* of them is bad), then lightly averaged so a single
    marginal indicator does not saturate the risk:

    * decode risk — coded BLER relative to the 10% link-adaptation target;
    * SINR risk — how far post-equalisation SINR sits below a healthy link;
    * starvation risk — spectral efficiency shortfall vs a 1 bps/Hz floor.
    """
    bler_risk = _clamp01(m.coded_bler / max(_TARGET_BLER, 1e-6)) if m.coded_bler < _TARGET_BLER \
        else _clamp01(0.5 + 0.5 * (m.coded_bler - _TARGET_BLER) / (1.0 - _TARGET_BLER))

    sinr_span = _GOOD_SINR_DB - _MIN_USABLE_SINR_DB
    sinr_risk = _clamp01((_GOOD_SINR_DB - m.post_eq_sinr_db) / sinr_span)

    starvation_risk = _clamp01(1.0 - m.spectral_eff_bps_hz / 1.0)

    worst = max(bler_risk, sinr_risk, starvation_risk)
    mean = (bler_risk + sinr_risk + starvation_risk) / 3.0
    return round(_clamp01(0.6 * worst + 0.4 * mean), 6)


class SionnaPhyBridge:
    """Convert link-level PHY measurements into Horizon telemetry.

    ``to_telemetry`` is pure and deterministic given a measurement; the risk
    it stamps into ``payload['sla_risk_30s']`` is what the downstream
    ``DecisionPipeline`` planner reads.
    """

    def __init__(self, source_id: str = "sionna-phy") -> None:
        self.source_id = source_id
        self._seq = 0

    def to_telemetry(
        self, m: PhyMeasurement, ts_utc: datetime | None = None
    ) -> TelemetryEvent:
        risk = _risk(m)
        event = TelemetryEvent(
            event_id=f"phy-{m.scenario}-{m.receiver_index}",
            modality="ue_qos",
            source_id=self.source_id,
            ts_utc=ts_utc or datetime.now(timezone.utc),
            sequence=self._seq,
            payload={
                "sla_risk_30s": risk,
                "risk_model": PHY_RISK_MODEL,
                "post_eq_sinr_db": round(m.post_eq_sinr_db, 4),
                "coded_bler": round(m.coded_bler, 6),
                "throughput_mbps": round(m.throughput_mbps, 4),
                "spectral_eff_bps_hz": round(m.spectral_eff_bps_hz, 4),
                "prb_util_pct": round(m.prb_util_pct, 4),
                "mcs_bits_per_symbol": m.mcs_bits_per_symbol,
                "mimo": f"{m.n_tx}x{m.n_rx}",
                "ebn0_db": m.ebn0_db,
                "position_m": list(m.position_m),
                "channel_source": "deepmimo_ray_tracing",
                "scenario": m.scenario,
                **m.extra,
            },
        )
        self._seq += 1
        return event


async def phy_measurements_to_pipeline(
    measurements: list[PhyMeasurement],
    pipeline: Any,
    *,
    bridge: SionnaPhyBridge | None = None,
) -> list[Any]:
    """Drive PHY measurements through a ``DecisionPipeline``.

    ``pipeline`` is a ``horizon_ric.rapp.pipeline.DecisionPipeline`` (kept as
    ``Any`` so importing this module does not pull the rapp/A1 stack). Returns
    the list of ``PipelineResult``s — one guarded A1 decision per receiver.
    """
    bridge = bridge or SionnaPhyBridge()
    results = []
    for m in measurements:
        event = bridge.to_telemetry(m)
        results.append(await pipeline.process_event(event))
    return results


def run_link_level_over_channels(
    channel_freq: "np.ndarray",
    *,
    ebn0_db: "float | list[float] | np.ndarray",
    num_bits_per_symbol: int = 4,
    code_rate: float = 0.5,
    num_codewords: int = 24,
    bandwidth_hz: float = 100e6,
    seed: int = 1,
) -> list[dict[str, float]]:
    """Real coded MIMO-OFDM link-level simulation over ray-traced channels.

    ``channel_freq`` is a complex array ``[num_rx_locations, num_subcarriers,
    n_rx, n_tx]`` — the frequency-domain MIMO channel matrices produced by
    DeepMIMO for each receiver location. For every location this runs a genuine
    5G-NR LDPC + QAM + spatial-multiplexing link with LMMSE detection (NVIDIA
    Sionna) and returns per-location PHY KPIs.

    Per standard link-level methodology, large-scale and small-scale effects
    are separated: each site's channel is normalised to unit average power so
    the matrix carries only the **spatial/frequency structure** (MIMO
    conditioning, selectivity) that determines how well spatial multiplexing
    works, while ``ebn0_db`` sets the **operating point** — pass a per-site
    array to fold the site's real ray-traced path loss into a link budget
    (see ``deploy/sionna-phy/run_phy_pipeline.py``), or a scalar to sweep all
    sites at one SNR.

    Imports ``sionna`` (PyTorch backend) lazily — install the ``phy`` extra.
    """
    import numpy as np
    import torch
    from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
    from sionna.phy.mapping import Constellation, Demapper, Mapper
    from sionna.phy.mimo import lmmse_equalizer
    from sionna.phy.utils import ebnodb2no

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_loc, n_sc, n_rx, n_tx = channel_freq.shape
    coderate = float(code_rate)
    ebn0 = np.broadcast_to(np.asarray(ebn0_db, dtype=float), (n_loc,))
    # One 5G-NR LDPC codeword's worth of coded bits per stream, sized to the
    # subcarrier grid so every subcarrier carries one QAM symbol per stream.
    n_coded = n_sc * num_bits_per_symbol
    k = int(n_coded * coderate)
    encoder = LDPC5GEncoder(k, n_coded)
    decoder = LDPC5GDecoder(encoder, hard_out=True)
    constellation = Constellation("qam", num_bits_per_symbol)
    mapper = Mapper(constellation=constellation)
    demapper = Demapper("app", constellation=constellation)

    results: list[dict[str, float]] = []
    eye = torch.eye(n_rx, dtype=torch.complex64).unsqueeze(0).repeat(n_sc, 1, 1)

    with torch.no_grad():
        for loc in range(n_loc):
            h = torch.as_tensor(channel_freq[loc], dtype=torch.complex64)  # [n_sc, n_rx, n_tx]
            # Unit average power per site: separate small-scale structure from
            # the large-scale gain, which is applied via the per-site Eb/N0.
            avg_pow = float(torch.mean((h.abs() ** 2)))
            if avg_pow > 0:
                h = h / math.sqrt(avg_pow)
            no = float(ebnodb2no(float(ebn0[loc]), num_bits_per_symbol, coderate))
            s = eye * complex(no, 0.0)                              # noise cov [n_sc, n_rx, n_rx]
            block_errors = 0
            sinr_lin_acc = 0.0
            sinr_count = 0
            for _ in range(num_codewords):
                # Transmit: independent LDPC codeword per spatial stream.
                bits = torch.randint(0, 2, (n_tx, k)).float()
                coded = encoder(bits)                              # [n_tx, n_coded]
                x = mapper(coded)                                  # [n_tx, n_sc] complex
                x = x.transpose(0, 1).unsqueeze(-1)                # [n_sc, n_tx, 1]

                # Channel: y = H x + n, per subcarrier.
                y = torch.matmul(h, x)                             # [n_sc, n_rx, 1]
                noise = torch.complex(
                    torch.randn(y.shape) * math.sqrt(no / 2),
                    torch.randn(y.shape) * math.sqrt(no / 2),
                )
                y = (y + noise).squeeze(-1)                        # [n_sc, n_rx]

                # LMMSE spatial equalisation per subcarrier.
                x_hat, no_eff = lmmse_equalizer(y, h, s)           # [n_sc, n_tx], [n_sc, n_tx]
                sinr_lin_acc += float(
                    torch.mean(1.0 / torch.clamp(no_eff.real, min=1e-9))
                )
                sinr_count += 1

                llr = demapper(x_hat.transpose(0, 1), no_eff.transpose(0, 1))  # [n_tx, n_coded]
                bits_hat = decoder(llr)                            # [n_tx, k]
                block_errors += int(
                    torch.sum(torch.any(bits != bits_hat, dim=1).int())
                )

            total_blocks = num_codewords * n_tx
            bler = block_errors / max(total_blocks, 1)
            sinr_lin = sinr_lin_acc / max(sinr_count, 1)
            sinr_db = 10.0 * math.log10(max(sinr_lin, 1e-9))
            goodput_frac = (1.0 - bler)
            spectral_eff = num_bits_per_symbol * coderate * n_tx * goodput_frac
            throughput_mbps = spectral_eff * bandwidth_hz / 1e6
            results.append(
                {
                    "post_eq_sinr_db": sinr_db,
                    "coded_bler": bler,
                    "throughput_mbps": throughput_mbps,
                    "spectral_eff_bps_hz": spectral_eff,
                    "mcs_bits_per_symbol": float(num_bits_per_symbol),
                    "n_tx": int(n_tx),
                    "n_rx": int(n_rx),
                    "ebn0_db": round(float(ebn0[loc]), 3),
                }
            )
    return results


__all__ = [
    "PHY_RISK_MODEL",
    "PhyMeasurement",
    "SionnaPhyBridge",
    "phy_measurements_to_pipeline",
    "run_link_level_over_channels",
]
