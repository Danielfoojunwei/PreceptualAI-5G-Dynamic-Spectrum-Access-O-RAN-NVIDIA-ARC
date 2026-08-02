"""Derive the Shield's frequency envelope from a real OCUDU cell configuration.

The seam this closes
--------------------
``horizon_ric.rapp.pipeline.PipelineConfig`` declares the band the Shield
polices as three hand-typed defaults::

    band_lo_hz: float = 3.40e9
    band_hi_hz: float = 3.50e9
    max_eirp_dbm: float = 33.0

Those numbers came from nowhere. They are not read from the radio, not
derived from a carrier, not checked against a 3GPP band. Meanwhile the gNB
the demonstrator actually talks to — OCUDU, in
``third_party/ocudu/configs/`` — states its carrier precisely, as an
NR-ARFCN, a band number, a channel bandwidth and a subcarrier spacing.

This module computes the envelope from the second thing, so the first thing
stops being a guess. It is deliberately additive: nothing under ``src/`` is
edited. ``PipelineConfig.from_env`` already reads ``HORIZON_SHIELD_BAND_LO_HZ``
/ ``HORIZON_SHIELD_BAND_HI_HZ`` / ``HORIZON_SHIELD_MAX_EIRP_DBM``, so the CLI
here emits exactly those, and the *unmodified* pipeline consumes them::

    eval "$(python -m horizon_ocudu.cell_config CONFIG.yml --export \\
              --antenna-gain-dBi 8)"

What is derivable and what is not
---------------------------------
The band edges ARE derivable, exactly, from the config: NR-ARFCN → F_REF is
an integer formula in TS 38.104 §5.4.2.1, and the channel edges are
F_REF ± BW/2.

The EIRP ceiling is NOT. ``ssb_block_power_dbm`` is the SS/PBCH block EPRE
— power per resource element, at the antenna connector. Turning that into an
EIRP needs the antenna gain and the feeder loss, and an OCUDU config carries
neither, because they are properties of the site, not of the software. So
:func:`derive_eirp_ceiling_dBm` REFUSES to run without an explicit
``antenna_gain_dBi`` from the operator rather than inventing one. A safety
ceiling assembled from a number nobody supplied is worse than no ceiling,
because it looks like it was measured.

References, all normative
-------------------------
* TS 38.104 §5.4.2.1 — NR-ARFCN to F_REF (Table 5.4.2.1-1 global raster).
* TS 38.104 Table 5.3.2-1 — FR1 transmission bandwidth configuration N_RB.
* TS 38.104 Table 5.2-1 — FR1 operating bands.
* TS 38.213 §4.1 — SS/PBCH block EPRE is per-RE, not per-carrier.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "NR_BANDS",
    "TRANSMISSION_BANDWIDTH_PRB",
    "CellConfigError",
    "NrBand",
    "OcuduCell",
    "ShieldEnvelope",
    "derive_eirp_ceiling_dBm",
    "derive_shield_envelope",
    "n_rb",
    "nr_arfcn_to_hz",
    "parse_gnb_config",
]


class CellConfigError(ValueError):
    """The configuration cannot be turned into a Shield envelope."""


# ── TS 38.104 §5.4.2.1, Table 5.4.2.1-1 — the global frequency raster ────────
#
# F_REF = F_REF-Offs + ΔF_Global × (N_REF − N_REF-Offs)
#
# (ΔF_Global Hz, F_REF-Offs Hz, N_REF-Offs, N_REF range inclusive)
_GLOBAL_RASTER = (
    (5_000, 0, 0, (0, 599_999)),
    (15_000, 3_000_000_000, 600_000, (600_000, 2_016_666)),
    (60_000, 24_250_080_000, 2_016_667, (2_016_667, 3_279_165)),
)


def nr_arfcn_to_hz(arfcn: int) -> float:
    """NR-ARFCN → F_REF in Hz, per TS 38.104 §5.4.2.1.

    Exact integer arithmetic on the global raster. Raises for an ARFCN
    outside the defined range rather than extrapolating off the end of the
    table.
    """
    if not isinstance(arfcn, int) or isinstance(arfcn, bool):
        raise CellConfigError(f"NR-ARFCN must be an int, got {type(arfcn).__name__}")
    for delta, f_offs, n_offs, (lo, hi) in _GLOBAL_RASTER:
        if lo <= arfcn <= hi:
            return float(f_offs + delta * (arfcn - n_offs))
    raise CellConfigError(
        f"NR-ARFCN {arfcn} is outside the global raster defined in "
        "TS 38.104 Table 5.4.2.1-1 (0..3279165)"
    )


# ── TS 38.104 Table 5.3.2-1 — FR1 transmission bandwidth configuration ──────
#
# (channel bandwidth MHz, SCS kHz) -> N_RB. "N/A" cells are simply absent, so
# an unsupported combination raises instead of silently picking a neighbour.
TRANSMISSION_BANDWIDTH_PRB: dict[tuple[int, int], int] = {
    (5, 15): 25, (5, 30): 11, (5, 60): 0,
    (10, 15): 52, (10, 30): 24, (10, 60): 11,
    (15, 15): 79, (15, 30): 38, (15, 60): 18,
    (20, 15): 106, (20, 30): 51, (20, 60): 24,
    (25, 15): 133, (25, 30): 65, (25, 60): 31,
    (30, 15): 160, (30, 30): 78, (30, 60): 38,
    (35, 15): 188, (35, 30): 92, (35, 60): 44,
    (40, 15): 216, (40, 30): 106, (40, 60): 51,
    (45, 15): 242, (45, 30): 119, (45, 60): 58,
    (50, 15): 270, (50, 30): 133, (50, 60): 65,
    (60, 30): 162, (60, 60): 79,
    (70, 30): 189, (70, 60): 93,
    (80, 30): 217, (80, 60): 107,
    (90, 30): 245, (90, 60): 121,
    (100, 30): 273, (100, 60): 135,
}


def n_rb(channel_bandwidth_mhz: int, scs_khz: int) -> int:
    """N_RB for an (channel bandwidth, SCS) pair, per TS 38.104 Table 5.3.2-1."""
    key = (int(channel_bandwidth_mhz), int(scs_khz))
    if key not in TRANSMISSION_BANDWIDTH_PRB:
        raise CellConfigError(
            f"{channel_bandwidth_mhz} MHz at {scs_khz} kHz SCS is not a defined "
            "FR1 transmission bandwidth configuration (TS 38.104 Table 5.3.2-1)"
        )
    prb = TRANSMISSION_BANDWIDTH_PRB[key]
    if prb == 0:
        raise CellConfigError(
            f"{channel_bandwidth_mhz} MHz at {scs_khz} kHz SCS is marked N/A in "
            "TS 38.104 Table 5.3.2-1"
        )
    return prb


@dataclass(frozen=True)
class NrBand:
    """One FR1 operating band from TS 38.104 Table 5.2-1."""

    number: int
    ul_lo_hz: float
    ul_hi_hz: float
    dl_lo_hz: float
    dl_hi_hz: float
    duplex: str  # "FDD" | "TDD" | "SDL" | "SUL"


def _band(n: int, ul: tuple[float, float], dl: tuple[float, float], d: str) -> NrBand:
    return NrBand(n, ul[0] * 1e6, ul[1] * 1e6, dl[0] * 1e6, dl[1] * 1e6, d)


# Every band an OCUDU config in third_party/ocudu/configs/ actually selects,
# plus the common neighbours. Kept short on purpose: an entry that is never
# exercised is an entry nothing checks.
NR_BANDS: dict[int, NrBand] = {
    1: _band(1, (1920, 1980), (2110, 2170), "FDD"),
    3: _band(3, (1710, 1785), (1805, 1880), "FDD"),
    7: _band(7, (2500, 2570), (2620, 2690), "FDD"),
    8: _band(8, (880, 915), (925, 960), "FDD"),
    20: _band(20, (832, 862), (791, 821), "FDD"),
    28: _band(28, (703, 748), (758, 803), "FDD"),
    38: _band(38, (2570, 2620), (2570, 2620), "TDD"),
    41: _band(41, (2496, 2690), (2496, 2690), "TDD"),
    77: _band(77, (3300, 4200), (3300, 4200), "TDD"),
    78: _band(78, (3300, 3800), (3300, 3800), "TDD"),
    79: _band(79, (4400, 5000), (4400, 5000), "TDD"),
}


@dataclass(frozen=True)
class OcuduCell:
    """The subset of an OCUDU ``cell_cfg`` that determines the RF envelope."""

    band: int
    dl_arfcn: int
    channel_bandwidth_mhz: int
    common_scs_khz: int
    source_path: str
    pci: int | None = None
    plmn: str | None = None
    tac: int | None = None
    nof_antennas_dl: int | None = None
    ssb_block_power_dbm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "dl_arfcn": self.dl_arfcn,
            "channel_bandwidth_MHz": self.channel_bandwidth_mhz,
            "common_scs": self.common_scs_khz,
            "pci": self.pci,
            "plmn": self.plmn,
            "tac": self.tac,
            "nof_antennas_dl": self.nof_antennas_dl,
            "ssb_block_power_dbm": self.ssb_block_power_dbm,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class ShieldEnvelope:
    """A frequency envelope derived from a cell, ready for the Shield.

    ``band_lo_hz``/``band_hi_hz`` are the CHANNEL edges (F_REF ± BW/2), not
    the operating-band edges: the Shield's job is to keep an agent inside the
    carrier this cell was licensed and configured to use, not merely inside
    the 500 MHz of spectrum band n78 spans.
    """

    centre_hz: float
    band_lo_hz: float
    band_hi_hz: float
    channel_bandwidth_hz: float
    transmission_bandwidth_hz: float
    n_rb: int
    band: NrBand
    cell: OcuduCell
    notes: list[str] = field(default_factory=list)

    @property
    def guard_to_band_edge_hz(self) -> float:
        """Smallest distance from a channel edge to the operating-band edge."""
        return min(
            self.band_lo_hz - self.band.dl_lo_hz, self.band.dl_hi_hz - self.band_hi_hz
        )

    def contains(self, lo_hz: float, hi_hz: float) -> bool:
        """True when ``[lo_hz, hi_hz]`` fits inside this channel."""
        return lo_hz >= self.band_lo_hz and hi_hz <= self.band_hi_hz

    def env(self) -> dict[str, str]:
        """The exact environment variables ``PipelineConfig.from_env`` reads.

        Only the two the cell determines. EIRP is absent by construction —
        see :func:`derive_eirp_ceiling_dBm`.
        """
        return {
            "HORIZON_SHIELD_BAND_LO_HZ": repr(self.band_lo_hz),
            "HORIZON_SHIELD_BAND_HI_HZ": repr(self.band_hi_hz),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "centre_hz": self.centre_hz,
            "band_lo_hz": self.band_lo_hz,
            "band_hi_hz": self.band_hi_hz,
            "channel_bandwidth_hz": self.channel_bandwidth_hz,
            "transmission_bandwidth_hz": self.transmission_bandwidth_hz,
            "n_rb": self.n_rb,
            "guard_to_band_edge_hz": self.guard_to_band_edge_hz,
            "band": {
                "number": self.band.number,
                "duplex": self.band.duplex,
                "dl_lo_hz": self.band.dl_lo_hz,
                "dl_hi_hz": self.band.dl_hi_hz,
            },
            "cell": self.cell.to_dict(),
            "env": self.env(),
            "notes": list(self.notes),
        }


# ── parsing ─────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover — yaml is a hard dependency
        raise CellConfigError(
            "PyYAML is required to parse an OCUDU configuration"
        ) from exc
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise CellConfigError(f"{path} does not contain a YAML mapping")
    return loaded


def parse_gnb_config(path: str | Path) -> OcuduCell:
    """Parse an OCUDU gNB YAML into the RF-relevant subset.

    Reads ``cell_cfg`` only. A config with no ``cell_cfg`` (a standalone
    ``cu_cp.yml``, say) raises rather than returning a partly-empty cell.
    """
    p = Path(path)
    doc = _load_yaml(p)
    cell = doc.get("cell_cfg")
    if not isinstance(cell, dict):
        raise CellConfigError(
            f"{p} has no `cell_cfg` section; it does not describe a cell"
        )

    missing = [k for k in ("band", "dl_arfcn", "channel_bandwidth_MHz", "common_scs")
               if k not in cell]
    if missing:
        raise CellConfigError(
            f"{p} `cell_cfg` is missing {', '.join(missing)} — the RF envelope "
            "cannot be derived from it"
        )

    ssb = cell.get("ssb")
    ssb_power = None
    if isinstance(ssb, dict) and "ssb_block_power_dbm" in ssb:
        ssb_power = float(ssb["ssb_block_power_dbm"])

    plmn = cell.get("plmn")
    return OcuduCell(
        band=int(cell["band"]),
        dl_arfcn=int(cell["dl_arfcn"]),
        channel_bandwidth_mhz=int(cell["channel_bandwidth_MHz"]),
        common_scs_khz=int(cell["common_scs"]),
        source_path=str(p),
        pci=int(cell["pci"]) if "pci" in cell else None,
        plmn=str(plmn) if plmn is not None else None,
        tac=int(cell["tac"]) if "tac" in cell else None,
        nof_antennas_dl=(
            int(cell["nof_antennas_dl"]) if "nof_antennas_dl" in cell else None
        ),
        ssb_block_power_dbm=ssb_power,
    )


# ── derivation ──────────────────────────────────────────────────────────────


def derive_shield_envelope(cell: OcuduCell) -> ShieldEnvelope:
    """Compute the channel envelope for ``cell``, checked against its band.

    Raises when the configured carrier does not fit inside the operating band
    it declares — that is a misconfigured gNB, and deriving a safety envelope
    from it would launder the error into the Shield.
    """
    band = NR_BANDS.get(cell.band)
    if band is None:
        raise CellConfigError(
            f"band n{cell.band} is not in this module's TS 38.104 Table 5.2-1 "
            f"subset (have: {sorted(NR_BANDS)})"
        )

    centre = nr_arfcn_to_hz(cell.dl_arfcn)
    ch_bw = cell.channel_bandwidth_mhz * 1e6
    lo = centre - ch_bw / 2.0
    hi = centre + ch_bw / 2.0

    prb = n_rb(cell.channel_bandwidth_mhz, cell.common_scs_khz)
    tx_bw = prb * 12 * cell.common_scs_khz * 1e3

    notes: list[str] = []
    if not (band.dl_lo_hz <= lo and hi <= band.dl_hi_hz):
        raise CellConfigError(
            f"carrier {lo / 1e6:.2f}-{hi / 1e6:.2f} MHz (ARFCN {cell.dl_arfcn}, "
            f"{cell.channel_bandwidth_mhz} MHz) does not fit inside band "
            f"n{band.number} DL {band.dl_lo_hz / 1e6:.0f}-"
            f"{band.dl_hi_hz / 1e6:.0f} MHz"
        )
    if band.duplex == "TDD" and band.ul_lo_hz != band.dl_lo_hz:  # pragma: no cover
        notes.append(f"band n{band.number} is TDD with asymmetric UL/DL ranges")

    unused = ch_bw - tx_bw
    notes.append(
        f"channel {ch_bw / 1e6:.0f} MHz, transmission bandwidth "
        f"{tx_bw / 1e6:.2f} MHz ({prb} PRB at {cell.common_scs_khz} kHz); "
        f"{unused / 1e6:.2f} MHz is guard, split either side"
    )
    notes.append(
        "envelope is the CHANNEL, not the operating band: an agent may not "
        f"wander the {(band.dl_hi_hz - band.dl_lo_hz) / 1e6:.0f} MHz of "
        f"n{band.number} merely because the cell sits in it"
    )

    return ShieldEnvelope(
        centre_hz=centre,
        band_lo_hz=lo,
        band_hi_hz=hi,
        channel_bandwidth_hz=ch_bw,
        transmission_bandwidth_hz=tx_bw,
        n_rb=prb,
        band=band,
        cell=cell,
        notes=notes,
    )


def derive_eirp_ceiling_dBm(
    envelope: ShieldEnvelope,
    *,
    antenna_gain_dBi: float,
    feeder_loss_dB: float = 0.0,
) -> tuple[float, list[str]]:
    """EIRP ceiling implied by the cell's SSB EPRE and a SUPPLIED antenna gain.

    ``antenna_gain_dBi`` is keyword-only and has no default on purpose. It is
    a property of the site — the antenna bolted to the mast — and no OCUDU
    configuration carries it. Guessing it would produce a safety ceiling that
    looks derived and is not.

    The arithmetic, given SS/PBCH block EPRE ``P_ssb`` in dBm (TS 38.213 §4.1,
    power per resource element at the connector):

        P_carrier = P_ssb + 10·log10(N_RB × 12)      # all REs, flat
        EIRP      = P_carrier − feeder_loss + gain

    A flat EPRE across the carrier is the standard planning assumption and is
    stated in the returned notes; it is an upper bound for any non-flat
    allocation of the same total power.
    """
    cell = envelope.cell
    if cell.ssb_block_power_dbm is None:
        raise CellConfigError(
            f"{cell.source_path} has no cell_cfg.ssb.ssb_block_power_dbm; "
            "the EIRP ceiling cannot be derived from this configuration. "
            "Supply HORIZON_SHIELD_MAX_EIRP_DBM from the site's licence "
            "instead of inferring it."
        )
    if not math.isfinite(antenna_gain_dBi):
        raise CellConfigError("antenna_gain_dBi must be finite")

    n_re = envelope.n_rb * 12
    p_carrier = cell.ssb_block_power_dbm + 10.0 * math.log10(n_re)
    eirp = p_carrier - feeder_loss_dB + antenna_gain_dBi
    notes = [
        f"SS/PBCH EPRE {cell.ssb_block_power_dbm:.1f} dBm/RE over {n_re} RE "
        f"({envelope.n_rb} PRB × 12) → {p_carrier:.2f} dBm conducted, assuming "
        "flat EPRE across the carrier (TS 38.213 §4.1)",
        f"antenna gain {antenna_gain_dBi:.1f} dBi and feeder loss "
        f"{feeder_loss_dB:.1f} dB were SUPPLIED, not read from the OCUDU "
        "configuration — no gNB config carries them",
        f"EIRP ceiling {eirp:.2f} dBm",
    ]
    return eirp, notes


# ── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m horizon_ocudu.cell_config",
        description=(
            "Derive the Shield's frequency envelope from an OCUDU gNB "
            "configuration and emit it as the environment variables "
            "PipelineConfig.from_env already reads."
        ),
    )
    p.add_argument("config", help="path to an OCUDU gNB YAML with a cell_cfg section")
    p.add_argument(
        "--export",
        action="store_true",
        help="emit `export NAME=VALUE` lines for eval, instead of JSON",
    )
    p.add_argument(
        "--antenna-gain-dBi",
        type=float,
        default=None,
        help="site antenna gain; REQUIRED to derive an EIRP ceiling. Without "
        "it the EIRP variable is not emitted and the pipeline default stands.",
    )
    p.add_argument(
        "--feeder-loss-dB",
        type=float,
        default=0.0,
        help="site feeder loss in dB (default 0.0)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        cell = parse_gnb_config(args.config)
        envelope = derive_shield_envelope(cell)
    except CellConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    env = dict(envelope.env())
    report = envelope.to_dict()

    if args.antenna_gain_dBi is not None:
        try:
            eirp, eirp_notes = derive_eirp_ceiling_dBm(
                envelope,
                antenna_gain_dBi=args.antenna_gain_dBi,
                feeder_loss_dB=args.feeder_loss_dB,
            )
        except CellConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        env["HORIZON_SHIELD_MAX_EIRP_DBM"] = repr(eirp)
        report["eirp"] = {"max_eirp_dBm": eirp, "notes": eirp_notes}
        report["env"] = env
    else:
        report["eirp"] = {
            "max_eirp_dBm": None,
            "notes": [
                "no --antenna-gain-dBi supplied, so no EIRP ceiling was "
                "derived; the pipeline's own default stands. This is "
                "deliberate — see derive_eirp_ceiling_dBm."
            ],
        }

    if args.export:
        for name, value in env.items():
            print(f"export {name}={value}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess in tests
    raise SystemExit(main())
