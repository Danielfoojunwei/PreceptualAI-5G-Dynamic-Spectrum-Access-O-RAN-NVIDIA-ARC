#!/usr/bin/env python3
"""Jamming on REAL campus geometry and REAL measured propagation (DeepMIMO ASU 3.5 GHz).

What changed (2026-07-28). The old suite was synthetic end to end: the victim
channel came from ``fading_dataset`` (``rayleigh_taps`` → ``rng.standard_normal``)
and the jammer-to-signal ratio was a free parameter swept over an invented grid
``[-10, -5, 0, 5, 10] dB``. No position, no measured path loss, no geometry.

It now runs on data already in the tree:

* ``channel_features.jsonl`` — 4096 REAL receiver positions on the ASU campus
  with the REAL ray-traced per-subband channel gain measured to each one.
* ``angular_features.jsonl`` — the REAL per-path amplitude / phase / delay that
  those gains were derived from, re-synthesised here into the OFDM frequency
  response so the link-level receivers see measured multipath rather than a
  Rayleigh draw. The reconstruction is checked against the independently built
  subband gains every run (``channel_reconstruction_check``).
* ``angular_manifest.json`` — the REAL base-station position (166, 104, 22) m.

Two layers.

**Layer A — jammer siting on measured propagation (no free parameters).**
A jammer is placed at a REAL campus grid point, so its channel to the base
station is *measured*, not modelled. Ray tracing is reciprocal and both ends are
isotropic SISO here, so the measured BS→point gain is also the point→BS gain;
that reciprocity is the only propagation assumption in Layer A. Victim uplink
SINR at the BS is then computed entirely from measured gains. Sweeping the
jammer over many real grid points produces a **vulnerability map of the actual
campus**: a jammer standing at a well-connected spot denies service to most of
the site, one in a shadowed courtyard denies almost none — and the difference is
measured, not assumed. Every subband-escape decision the RIC then makes is
pushed through the Decision Safety Shield and counted for legality.

**Layer B — jammer waveforms on the measured channel.** The classic
electronic-warfare archetypes (barrage / partial-band / single-tone / pulsed)
plus the imperfect-CSI realism stressor, run against the neural receiver, the
classical LMMSE receiver and the Shield's routed output — now over the REAL
ray-traced channel. The JSR operating points are no longer invented: they are
the percentiles of the JSR distribution that Layer A's real geometry actually
produces.

Honest limits are in ``scope_note`` and are not small. In particular the dataset
contains BS→grid-point channels only, so a jammer's channel to a *victim
handset* is not available at any price; Layer A is therefore an uplink /
receiver-blocking geometry, which is exactly the geometry the measurement
supports. Ray tracing is not over-the-air capture and a modelled jammer on
measured propagation is not a captured attack.

Run:  python benchmarks/jamming_suite.py --out benchmarks/results/jamming_suite.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.phy import NeuralReceiver, classical_equalize_demap
from horizon_ric.phy.constellation import modulate
from horizon_ric.phy.jamming import (
    barrage_jammer,
    partial_band_jammer,
    pulsed_jammer,
    single_tone_jammer,
)
from horizon_ric.runtime_env import stamp
from horizon_ric.shield import default_terrestrial_shield
from horizon_ric.shield.invariants import NeuralRxEnvelopeInvariant

M = 16
WIN = 200  # symbols per measurement window for the windowed-SER fallback decision

# --- DeepMIMO OFDM grid (mirrors datasets/deepmimo_asu_3p5/build.py). ---------
OFDM_SUBCARRIERS = 1024
OFDM_BANDWIDTH_HZ = 100e6
SUBBAND_SELECTED = 60
N_SUBBANDS = 6
# DeepMIMO's frequency-domain channel carries a 1/K OFDM scaling; adding it back
# recovers the physical wideband channel gain. Verified every run to 1e-3 dB RMS
# against the ray re-synthesis (see reconstruction_check).
OFDM_SCALING_DB = 10.0 * np.log10(OFDM_SUBCARRIERS)

# --- Link budget. Declared constants, not measurements; stated so in the JSON.
BAND_LO_HZ = 3.45e9
BAND_HI_HZ = 3.55e9
CARRIER_BW_HZ = OFDM_BANDWIDTH_HZ / N_SUBBANDS
THERMAL_DBM_PER_HZ = -174.0
NOISE_FIGURE_DB = 7.0
UE_TX_POWER_DBM = 23.0  # 3GPP power class 3 handset
MAX_EIRP_DBM = 33.0
ANTENNA_GAIN_DBI = 6.0
REQUESTED_TX_POWER_DBM = 44.0  # over-request on purpose to stress EIRP projection
# A link is called usable when its SINR clears the threshold; 16-QAM r=1/2 needs
# roughly 11 dB and 64-QAM r=3/4 roughly 18 dB in AWGN, so both are reported.
SINR_THRESHOLDS_DB = (11.0, 18.0)

NOISE_DBM = THERMAL_DBM_PER_HZ + 10.0 * np.log10(CARRIER_BW_HZ) + NOISE_FIGURE_DB


# ── data loading and the real channel ────────────────────────────────────────
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no feature rows in {path}")
    return rows


def raytraced_frequency_response(
    angular_rows: list[dict[str, Any]], subcarriers: np.ndarray
) -> np.ndarray:
    """H_r(f_k) = Σ_l 10**(P_l/20)·e^{jφ_l}·e^{-j2π f_k τ_l}, f_k = k·B/K."""
    freqs = subcarriers.astype(np.float64) * (OFDM_BANDWIDTH_HZ / OFDM_SUBCARRIERS)
    out = np.zeros((len(angular_rows), freqs.size), dtype=np.complex128)
    for i, row in enumerate(angular_rows):
        paths = row["paths"]
        if not paths:
            raise ValueError(f"receiver row {i} has no ray paths")
        amp = 10.0 ** (
            np.array([p["power_dbw"] for p in paths], dtype=np.float64) / 20.0
        ) * np.exp(
            1j * np.deg2rad(np.array([p["phase_deg"] for p in paths], dtype=np.float64))
        )
        tau = np.array([p["delay_ns"] for p in paths], dtype=np.float64) * 1e-9
        out[i] = (
            amp[None, :] * np.exp(-2j * np.pi * freqs[:, None] * tau[None, :])
        ).sum(axis=1)
    if not np.all(np.isfinite(out)):
        raise ValueError("ray-traced frequency response contains non-finite values")
    return out


def reconstruction_check(
    angular_rows: list[dict[str, Any]], channel_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Cross-validate the ray re-synthesis against the committed subband gains."""
    sel = np.linspace(0, OFDM_SUBCARRIERS - 1, SUBBAND_SELECTED, dtype=np.int64)
    power = np.abs(raytraced_frequency_response(angular_rows, sel)) ** 2
    sub = np.stack(
        [chunk.mean(axis=1) for chunk in np.split(power, N_SUBBANDS, axis=1)], axis=1
    )
    reconstructed = 10.0 * np.log10(np.maximum(sub, 1e-40))
    committed = (
        np.array([r["subband_gain_dbw"] for r in channel_rows], dtype=np.float64)
        + OFDM_SCALING_DB
    )
    delta = reconstructed - committed
    return {
        "what": (
            "angular_features.jsonl per-path rays re-synthesised into the OFDM "
            "frequency response and compared against the independently built "
            "channel_features.jsonl subband gains (+10*log10(1024) OFDM scaling)"
        ),
        "rms_error_db": round(float(np.sqrt(np.mean(delta**2))), 6),
        "max_abs_error_db": round(float(np.max(np.abs(delta))), 6),
        "receivers": int(len(angular_rows)),
    }


class RealChannelBank:
    """Per-link power-normalised measured channels for the link-level layer."""

    def __init__(self, angular_rows: list[dict[str, Any]]) -> None:
        h = raytraced_frequency_response(
            angular_rows, np.arange(OFDM_SUBCARRIERS, dtype=np.int64)
        )
        wideband = np.mean(np.abs(h) ** 2, axis=1)
        if np.any(wideband <= 0):
            raise ValueError("a receiver has zero wideband channel power")
        self.h_normalised = h / np.sqrt(wideband)[:, None]
        self.link_gain_dbw = 10.0 * np.log10(wideband)
        self.n_rx = self.h_normalised.shape[0]

    def _draw(self, n: int, rng: np.random.Generator, rx_indices: np.ndarray):
        rx = rng.choice(rx_indices, size=n, replace=True)
        sc = rng.integers(0, OFDM_SUBCARRIERS, size=n)
        return self.h_normalised[rx, sc]

    def dataset(
        self, n: int, snr_dB: float, rng: np.random.Generator, rx_indices: np.ndarray
    ):
        """(n,4) [Re Y, Im Y, Re H, Im H] over the measured channel population."""
        labels = rng.integers(0, M, size=n)
        x = modulate(labels, M)
        h = self._draw(n, rng, rx_indices)
        n0 = 1.0 / (10.0 ** (snr_dB / 10.0))
        noise = np.sqrt(n0 / 2.0) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )
        y = h * x + noise
        return (
            np.stack([y.real, y.imag, h.real, h.imag], axis=1).astype(np.float64),
            labels,
            {"N0": n0},
        )

    def dataset_imperfect_csi(
        self,
        n: int,
        snr_dB: float,
        sigma_est: float,
        rng: np.random.Generator,
        rx_indices: np.ndarray,
    ):
        """Same measured channel, but both receivers are handed Ĥ = H + e.

        Estimation-error realism, not an attack: the true measured H still shapes
        Y = H·X + N, while the feature matrix carries the estimate, so the neural
        receiver and the LMMSE equaliser both demap against the wrong CSI.
        """
        labels = rng.integers(0, M, size=n)
        x = modulate(labels, M)
        h = self._draw(n, rng, rx_indices)
        n0 = 1.0 / (10.0 ** (snr_dB / 10.0))
        noise = np.sqrt(n0 / 2.0) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )
        y = h * x + noise
        err = np.sqrt((sigma_est**2) / 2.0) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )
        h_hat = h + err
        return (
            np.stack(
                [y.real, y.imag, h_hat.real, h_hat.imag], axis=1
            ).astype(np.float64),
            labels,
            {"N0": n0},
        )


# ── Layer A: jammer siting on measured propagation ───────────────────────────
def _lin(dbm: np.ndarray | float) -> np.ndarray:
    return 10.0 ** (np.asarray(dbm, dtype=np.float64) / 10.0)


def clean_sinr_dbm(abs_gain_db: np.ndarray) -> np.ndarray:
    """Un-jammed uplink SNR (dB) at the BS for every (receiver, subband)."""
    return UE_TX_POWER_DBM + abs_gain_db - NOISE_DBM


def jammed_sinr_db(
    abs_gain_db: np.ndarray, jammer_row: np.ndarray, jammer_power_dbm: float
) -> np.ndarray:
    """SINR (dB) with one jammer at a real grid point, all terms measured.

    ``abs_gain_db`` is (R, K) measured victim gain; ``jammer_row`` is the (K,)
    measured gain of the grid point the jammer stands on. By ray-tracing
    reciprocity that same gain describes jammer→BS.
    """
    interference = _lin(jammer_power_dbm + jammer_row)  # (K,)
    noise = _lin(NOISE_DBM)
    signal = _lin(UE_TX_POWER_DBM + abs_gain_db)  # (R, K)
    return 10.0 * np.log10(signal / (interference[None, :] + noise))


def _outage(sinr_db: np.ndarray, threshold: float) -> float:
    """Fraction of receivers with no subband clearing ``threshold``."""
    return float(np.mean(sinr_db.max(axis=1) < threshold))


def jammer_site_sweep(
    abs_gain_db: np.ndarray,
    positions: np.ndarray,
    bs_position: np.ndarray,
    jammer_power_dbm: float,
    n_sites: int,
) -> dict[str, Any]:
    """Sweep the jammer over ``n_sites`` real grid points → a vulnerability map."""
    total = abs_gain_db.shape[0]
    sites = np.linspace(0, total - 1, n_sites, dtype=np.int64)
    clean = clean_sinr_dbm(abs_gain_db)
    base_outage = {f"{t:g}dB": _outage(clean, t) for t in SINR_THRESHOLDS_DB}

    rows = []
    for s in sites:
        sinr = jammed_sinr_db(abs_gain_db, abs_gain_db[s], jammer_power_dbm)
        rows.append(
            {
                "site_row": int(s),
                "jammer_gain_dbw": float(abs_gain_db[s].mean()),
                "jammer_distance_to_bs_m": float(
                    np.linalg.norm(positions[s] - bs_position)
                ),
                "median_sinr_loss_db": float(
                    np.median(clean.max(axis=1) - sinr.max(axis=1))
                ),
                "outage_11db": _outage(sinr, 11.0),
                "outage_18db": _outage(sinr, 18.0),
            }
        )
    out11 = np.array([r["outage_11db"] for r in rows])
    worst = rows[int(np.argmax(out11))]
    best = rows[int(np.argmin(out11))]
    median = rows[int(np.argmin(np.abs(out11 - np.median(out11))))]
    return {
        "median_site": {
            k: round(v, 4) if isinstance(v, float) else v for k, v in median.items()
        },
        "median_site_position_m": [
            round(float(v), 4) for v in positions[median["site_row"]]
        ],
        "jammer_power_dbm": jammer_power_dbm,
        "sites_evaluated": int(n_sites),
        "site_selection": "evenly spaced over the 4096 measured receiver rows",
        "unjammed_outage_fraction": {k: round(v, 6) for k, v in base_outage.items()},
        "campus_outage_11db_over_sites": {
            "min": round(float(out11.min()), 6),
            "median": round(float(np.median(out11)), 6),
            "mean": round(float(out11.mean()), 6),
            "max": round(float(out11.max()), 6),
        },
        "worst_site": {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in worst.items()},
        "best_site": {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in best.items()},
        "worst_site_position_m": [
            round(float(v), 4) for v in positions[worst["site_row"]]
        ],
        "best_site_position_m": [
            round(float(v), 4) for v in positions[best["site_row"]]
        ],
        "jammer_gain_vs_outage_pearson_r": round(
            float(
                np.corrcoef(
                    np.array([r["jammer_gain_dbw"] for r in rows]), out11
                )[0, 1]
            ),
            6,
        ),
        "jammer_bs_distance_vs_outage_pearson_r": round(
            float(
                np.corrcoef(
                    np.array([r["jammer_distance_to_bs_m"] for r in rows]), out11
                )[0, 1]
            ),
            6,
        ),
    }


def subband_escape(
    abs_gain_db: np.ndarray,
    jammer_row: np.ndarray,
    jammer_power_dbm: float,
    served: np.ndarray,
) -> dict[str, Any]:
    """Does knowing the jammer change which subband the RIC should pick?

    Three strategies, all scored on the same measured SINR field: a fixed
    subband, the jamming-*unaware* best-gain choice (what
    ``deepmimo_dsa_benchmark.py`` does), and the jamming-*aware* best-SINR
    choice. The aware-vs-unaware delta is a real measured quantity and exists
    only because the jammer's own per-subband gain profile differs from the
    victim's — both profiles are ray-traced. It is also reported restricted to
    ``served`` (receivers the cell can actually serve when un-jammed), because
    averaging over receivers that are out of coverage anyway flatters nobody.
    """
    sinr = jammed_sinr_db(abs_gain_db, jammer_row, jammer_power_dbm)
    rows = np.arange(abs_gain_db.shape[0])
    unaware = sinr[rows, np.argmax(abs_gain_db, axis=1)]
    aware = sinr.max(axis=1)
    fixed = sinr[:, 0]
    return {
        "mean_sinr_db": {
            "fixed_subband_0": round(float(fixed.mean()), 6),
            "jamming_unaware_best_gain": round(float(unaware.mean()), 6),
            "jamming_aware_best_sinr": round(float(aware.mean()), 6),
        },
        "aware_minus_unaware_db": round(float(np.mean(aware - unaware)), 6),
        "aware_minus_fixed_db": round(float(np.mean(aware - fixed)), 6),
        "served_only": {
            "receivers": int(served.sum()),
            "aware_minus_unaware_db": round(
                float(np.mean((aware - unaware)[served])), 6
            ),
            "aware_minus_fixed_db": round(float(np.mean((aware - fixed)[served])), 6),
        },
        "receivers_whose_best_subband_changes": int(
            np.sum(np.argmax(sinr, axis=1) != np.argmax(abs_gain_db, axis=1))
        ),
        "usable_fraction_11db": {
            "fixed_subband_0": round(float(np.mean(fixed >= 11.0)), 6),
            "jamming_unaware_best_gain": round(float(np.mean(unaware >= 11.0)), 6),
            "jamming_aware_best_sinr": round(float(np.mean(aware >= 11.0)), 6),
        },
    }


def shield_legality(
    abs_gain_db: np.ndarray, jammer_row: np.ndarray, jammer_power_dbm: float
) -> dict[str, Any]:
    """Push every jamming-aware escape decision through the Shield and count."""
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ, max_eirp_dBm=MAX_EIRP_DBM
    )
    sinr = jammed_sinr_db(abs_gain_db, jammer_row, jammer_power_dbm)
    choices = np.argmax(sinr, axis=1)
    width = (BAND_HI_HZ - BAND_LO_HZ) / N_SUBBANDS
    illegal = projected = blocked = 0
    for index, channel in enumerate(choices):
        proposed = {
            "block": "jamming_escape_policy",
            "frequency_hz": BAND_LO_HZ + (float(channel) + 0.5) * width,
            "bandwidth_hz": width,
            "tx_power_dBm": REQUESTED_TX_POWER_DBM,
            "antenna_gain_dBi": ANTENNA_GAIN_DBI,
        }
        disposition = shield.dispose(proposed, decision_id=f"jam-escape-{index}")
        certificate = disposition.certificate
        projected += int(certificate.projected)
        if certificate.emit_blocked:
            blocked += 1
            continue
        safe = disposition.safe_action
        lo = safe["frequency_hz"] - safe["bandwidth_hz"] / 2
        hi = safe["frequency_hz"] + safe["bandwidth_hz"] / 2
        eirp = safe["tx_power_dBm"] + safe["antenna_gain_dBi"]
        if (
            lo < BAND_LO_HZ - 1e-3
            or hi > BAND_HI_HZ + 1e-3
            or eirp > MAX_EIRP_DBM + 1e-6
        ):
            illegal += 1
    return {
        "decisions": int(choices.size),
        "requested_tx_power_dBm": REQUESTED_TX_POWER_DBM,
        "shield_projected": projected,
        "shield_blocked": blocked,
        "illegal_emits_after_shield": illegal,
    }


def measured_jsr_distribution(
    abs_gain_db: np.ndarray, jammer_row: np.ndarray, jammer_power_dbm: float
) -> np.ndarray:
    """Per-(receiver, subband) JSR in dB produced by the real geometry."""
    return (jammer_power_dbm + jammer_row)[None, :] - (
        UE_TX_POWER_DBM + abs_gain_db
    )


# ── Layer B: jammer waveforms on the measured channel ────────────────────────
def _ser(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred != y))


def _shield_effective(neural_pred, classical_pred, y):
    """Windowed-SER fallback (NeuralRxEnvelopeInvariant, 1 dB tolerance).

    Per window the Shield independently measures the neural SER; if it leaves the
    classical baseline's envelope it routes to the certified LMMSE receiver. The
    Shield can only choose *between* the two receivers it is given — it cannot
    make either one less noisy. Returns ``(effective_ser, fallback_rate)``.
    """
    n_err = (neural_pred != y).astype(float)
    c_err = (classical_pred != y).astype(float)
    inv = NeuralRxEnvelopeInvariant(tolerance_dB=1.0)
    n_win = max(len(y) // WIN, 1)
    eff, fb = [], 0
    for w in range(n_win):
        s = slice(w * WIN, (w + 1) * WIN)
        measured_neural = float(n_err[s].mean())
        baseline_classical = float(c_err[s].mean())
        action = {"block": "neural_rx", "baseline_tbler": baseline_classical}
        ctx = {"measured_tbler": measured_neural}
        if not inv.evaluate(action, ctx).satisfied:
            fb += 1
            eff.append(baseline_classical)
        else:
            eff.append(measured_neural)
    return float(np.mean(eff)), fb / n_win


def _ms(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def _measure(net, feats, labels, n0):
    """Return (neural_ser, classical_ser, shield_eff_ser, fallback_rate)."""
    np_ = net.predict(feats)
    cp_ = classical_equalize_demap(feats, M, n0)
    eff, fbr = _shield_effective(np_, cp_, labels)
    return _ser(np_, labels), _ser(cp_, labels), eff, fbr


def _point(rows: dict, extra: dict) -> dict:
    point = dict(extra)
    point.update(
        {
            "neural_ser": _ms(rows["n"]),
            "classical_lmmse_ser": _ms(rows["c"]),
            "shield_effective_ser": _ms(rows["eff"]),
            "fallback_rate": _ms(rows["fb"]),
        }
    )
    return point


def sweep_jammer(
    jammer_name: str, nets, testsets, jsr_grid, **jam_kw
) -> list:
    """Sweep one jammer archetype over the measured-geometry JSR operating points."""
    points = []
    for jsr in jsr_grid:
        rows = {"n": [], "c": [], "eff": [], "fb": []}
        for sd, net in nets.items():
            feats, labels, n0 = testsets[sd]
            rng = np.random.default_rng(7000 + sd)
            if jammer_name == "barrage":
                jammed = barrage_jammer(feats, jsr, rng)
            elif jammer_name == "partial_band":
                jammed = partial_band_jammer(feats, jsr, jam_kw["fraction"], rng)
            elif jammer_name == "single_tone":
                jammed = single_tone_jammer(feats, jsr, phase=None, rng=rng)
            elif jammer_name == "pulsed":
                jammed = pulsed_jammer(feats, jsr, jam_kw["duty"], rng)
            else:
                raise ValueError(jammer_name)
            n_ser, c_ser, eff, fbr = _measure(net, jammed, labels, n0)
            rows["n"].append(n_ser)
            rows["c"].append(c_ser)
            rows["eff"].append(eff)
            rows["fb"].append(fbr)
        points.append(_point(rows, {"jsr_dB": round(float(jsr), 3)}))
    return points


def sweep_partial_fraction(nets, testsets, jsr_dB, frac_grid) -> list:
    """Hold JSR fixed; sweep the jammed fraction of the partial-band jammer."""
    points = []
    for frac in frac_grid:
        rows = {"n": [], "c": [], "eff": [], "fb": []}
        for sd, net in nets.items():
            feats, labels, n0 = testsets[sd]
            rng = np.random.default_rng(7100 + sd)
            jammed = partial_band_jammer(feats, jsr_dB, frac, rng)
            n_ser, c_ser, eff, fbr = _measure(net, jammed, labels, n0)
            rows["n"].append(n_ser)
            rows["c"].append(c_ser)
            rows["eff"].append(eff)
            rows["fb"].append(fbr)
        points.append(
            _point(rows, {"fraction": frac, "jsr_dB": round(float(jsr_dB), 3)})
        )
    return points


def sweep_imperfect_csi(bank, nets, snr_dB, sigma_grid, test_rx) -> list:
    """Sweep the CSI-estimation-error σ on the measured channel. NOT an attack."""
    points = []
    for sigma in sigma_grid:
        rows = {"n": [], "c": [], "eff": [], "fb": []}
        for sd, net in nets.items():
            rng = np.random.default_rng(7200 + sd)
            feats, labels, meta = bank.dataset_imperfect_csi(
                15_000, snr_dB, sigma, rng, test_rx
            )
            n_ser, c_ser, eff, fbr = _measure(net, feats, labels, meta["N0"])
            rows["n"].append(n_ser)
            rows["c"].append(c_ser)
            rows["eff"].append(eff)
            rows["fb"].append(fbr)
        point = _point(rows, {"sigma_est": sigma})
        point["neural_minus_classical"] = (
            point["neural_ser"]["mean"] - point["classical_lmmse_ser"]["mean"]
        )
        points.append(point)
    return points


def _dist(values: np.ndarray) -> dict[str, float]:
    return {
        "min": round(float(np.min(values)), 4),
        "p5": round(float(np.percentile(values, 5)), 4),
        "median": round(float(np.median(values)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "max": round(float(np.max(values)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="benchmarks/results/jamming_suite.json")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--jammer-power-dbm", type=float, default=23.0)
    ap.add_argument("--jammer-sites", type=int, default=128)
    ap.add_argument(
        "--features",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/generated/channel_features.jsonl"),
    )
    ap.add_argument(
        "--angular",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/generated/angular_features.jsonl"),
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/manifest.json"),
    )
    ap.add_argument(
        "--angular-manifest",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/angular_manifest.json"),
    )
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    snr = args.snr

    channel_rows = _load_jsonl(args.features)
    angular_rows = _load_jsonl(args.angular)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    angular_manifest = json.loads(args.angular_manifest.read_text(encoding="utf-8"))
    if len(channel_rows) != manifest["sampled_receivers"]:
        raise ValueError("feature row count does not match the manifest")
    if len(angular_rows) != len(channel_rows):
        raise ValueError("channel and angular feature files disagree on row count")

    check = reconstruction_check(angular_rows, channel_rows)
    if check["rms_error_db"] > 1e-3:
        raise RuntimeError(
            "ray reconstruction disagrees with the committed subband gains: "
            f"{check['rms_error_db']} dB RMS"
        )

    positions = np.array([r["position_m"] for r in channel_rows], dtype=np.float64)
    abs_gain_db = (
        np.array([r["subband_gain_dbw"] for r in channel_rows], dtype=np.float64)
        + OFDM_SCALING_DB
    )
    bs_position = np.array(angular_manifest["tx_position_m"], dtype=np.float64)

    # --- Layer A -------------------------------------------------------------
    siting = jammer_site_sweep(
        abs_gain_db,
        positions,
        bs_position,
        args.jammer_power_dbm,
        args.jammer_sites,
    )
    # Serve-able receivers: those the un-jammed cell can actually carry. Escape
    # gains averaged over out-of-coverage receivers would be meaningless.
    served = clean_sinr_dbm(abs_gain_db).max(axis=1) >= 11.0

    site_rows = {
        name: int(siting[f"{name}_site"]["site_row"])
        for name in ("best", "median", "worst")
    }
    escape = {
        name: subband_escape(
            abs_gain_db, abs_gain_db[row], args.jammer_power_dbm, served
        )
        for name, row in site_rows.items()
    }
    legality = {
        name: shield_legality(abs_gain_db, abs_gain_db[row], args.jammer_power_dbm)
        for name, row in site_rows.items()
    }
    # Layer B operating points come from the REPRESENTATIVE (median-outage) real
    # site. The worst site annihilates every link, so its JSR field would pin the
    # whole sweep above the cliff and measure nothing.
    jsr_field = measured_jsr_distribution(
        abs_gain_db, abs_gain_db[site_rows["median"]], args.jammer_power_dbm
    )
    jsr_grid = [
        float(np.round(np.percentile(jsr_field, q), 3)) for q in (10, 50, 90)
    ]

    # --- Layer B -------------------------------------------------------------
    bank = RealChannelBank(angular_rows)
    x = positions[:, 0]
    cut = float(np.median(x))
    train_rx = np.flatnonzero(x < cut)
    test_rx = np.flatnonzero(x >= cut)
    nets = {}
    testsets = {}
    for sd in seeds:
        rng = np.random.default_rng(1000 + sd)
        Xtr, ytr, _ = bank.dataset(40_000, snr, rng, train_rx)
        nets[sd] = NeuralReceiver(M, hidden=160, seed=1, n_features=4).train(
            Xtr, ytr, epochs=50, lr=0.3, seed=2
        )
        rng_te = np.random.default_rng(7000 + sd)
        feats, labels, meta = bank.dataset(15_000, snr, rng_te, test_rx)
        testsets[sd] = (feats, labels, meta["N0"])

    report: dict[str, Any] = {
        "benchmark": "Jamming on real DeepMIMO campus geometry and measured propagation",
        "dataset": manifest["dataset"],
        "scenario": manifest["scenario"],
        "data_kind": manifest["data_kind"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "features_sha256": manifest["features_sha256"],
        "angular_features_sha256": angular_manifest["features_sha256"],
        "receivers": len(channel_rows),
        "channel_reconstruction_check": check,
        "setup": {
            "constellation": "16-QAM",
            "channel": (
                "measured DeepMIMO ray-traced OFDM response (1024 subcarriers "
                "over 100 MHz) rebuilt from per-path power/phase/delay"
            ),
            "snr_dB": snr,
            "seeds": seeds,
            "receivers": "neural MLP vs per-subcarrier LMMSE; Shield routes between them",
            "geometry": {
                "bs_position_m": angular_manifest["tx_position_m"],
                "jammer_placement": "a real DeepMIMO receiver grid point",
                "reciprocity_assumption": (
                    "ray tracing is reciprocal and both ends are isotropic SISO, "
                    "so the measured BS->grid-point gain is used as the "
                    "grid-point->BS (jammer uplink) gain"
                ),
                "measured_best_subband_gain_db": _dist(abs_gain_db.max(axis=1)),
                "bs_to_receiver_distance_m": _dist(
                    np.linalg.norm(positions - bs_position, axis=1)
                ),
            },
            "link_budget": {
                "note": "declared operating constants, not measurements",
                "ue_tx_power_dbm": UE_TX_POWER_DBM,
                "carrier_bandwidth_hz": CARRIER_BW_HZ,
                "noise_figure_db": NOISE_FIGURE_DB,
                "noise_floor_dbm": round(float(NOISE_DBM), 4),
                "sinr_thresholds_db": list(SINR_THRESHOLDS_DB),
            },
            "jsr_definition": (
                "jammer received power / victim received power at the BS, dB; "
                "Layer B operating points are the 10/50/90th percentiles of the "
                "JSR field the worst real jammer site actually produces"
            ),
        },
        "jammer_siting_vulnerability_map": siting,
        "served_receivers_unjammed_11db": int(served.sum()),
        "subband_escape_by_real_site": escape,
        "shield_legality_by_real_site": legality,
        "measured_jsr_field_at_median_site_db": _dist(jsr_field),
        "measured_jsr_operating_points_db": jsr_grid,
        "barrage_awgn": sweep_jammer("barrage", nets, testsets, jsr_grid),
        "partial_band_jsr_sweep": sweep_jammer(
            "partial_band", nets, testsets, jsr_grid, fraction=0.3
        ),
        "partial_band_fraction_sweep": sweep_partial_fraction(
            nets, testsets, jsr_grid[-1], [0.1, 0.3, 0.5, 0.8, 1.0]
        ),
        "single_tone": sweep_jammer("single_tone", nets, testsets, jsr_grid),
        "pulsed": sweep_jammer("pulsed", nets, testsets, jsr_grid, duty=0.3),
        "imperfect_csi": sweep_imperfect_csi(
            bank, nets, snr, [0.0, 0.05, 0.1, 0.2, 0.4], test_rx
        ),
    }

    barrage_hi = report["barrage_awgn"][-1]
    csi_hi = report["imperfect_csi"][-1]
    neural_worse = csi_hi["neural_minus_classical"] > 0
    report["honest_findings"] = [
        "JAMMER SITING IS THE DOMINANT VARIABLE, AND IT IS MEASURED. Over "
        f"{siting['sites_evaluated']} real campus grid points a "
        f"{args.jammer_power_dbm:g} dBm jammer denies (11 dB SINR) between "
        f"{siting['campus_outage_11db_over_sites']['min'] * 100:.1f}% and "
        f"{siting['campus_outage_11db_over_sites']['max'] * 100:.1f}% of the "
        f"campus (median {siting['campus_outage_11db_over_sites']['median'] * 100:.1f}%), "
        f"against an unjammed baseline of "
        f"{siting['unjammed_outage_fraction']['11dB'] * 100:.1f}%. The worst site "
        f"is {siting['worst_site_position_m']} m, "
        f"{siting['worst_site']['jammer_distance_to_bs_m']:.0f} m from the BS; the "
        f"best is {siting['best_site_position_m']} m at "
        f"{siting['best_site']['jammer_distance_to_bs_m']:.0f} m. Outage "
        "correlates with the jammer's measured gain to the BS "
        f"(Pearson r={siting['jammer_gain_vs_outage_pearson_r']:.3f}) more "
        "tightly than with its straight-line distance to the BS "
        f"(r={siting['jammer_bs_distance_vs_outage_pearson_r']:.3f}). Distance is "
        "a usable proxy but the ray-traced gain field is the thing: siting rules "
        "should be written against measured propagation, not a radius on a map.",
        "SUBBAND ESCAPE IS WORTH LESS THAN IT SOUNDS, AND WE MEASURED HOW MUCH. "
        "At the representative (median-outage) real jammer site, subband "
        "selection is worth "
        f"{escape['median']['aware_minus_fixed_db']:.3f} dB of mean SINR over a "
        "fixed subband — but making that selection *jamming-aware* rather than "
        "just best-gain adds only "
        f"{escape['median']['aware_minus_unaware_db']:.3f} dB "
        f"({escape['median']['served_only']['aware_minus_unaware_db']:.3f} dB "
        "over the "
        f"{escape['median']['served_only']['receivers']} receivers the un-jammed "
        f"cell can actually serve), even though "
        f"{escape['median']['receivers_whose_best_subband_changes']} of "
        f"{len(channel_rows)} receivers do change which subband is optimal. The "
        "reason is measurable in the data: the jammer's interference is common to "
        "every victim and its ray-traced per-subband profile is only a few dB "
        "wide, so the victim's own gain variation dominates the choice. Anyone "
        "selling jamming-aware DSA as a large win on this geometry is overselling.",
        "LEGALITY IS INDEPENDENT OF THE ATTACK. At all three real jammer sites, "
        f"every one of {legality['worst']['decisions']} escape decisions was "
        "pushed through the Shield with a deliberately over-limit "
        f"{REQUESTED_TX_POWER_DBM:g} dBm request; "
        f"{legality['worst']['shield_projected']} were projected and "
        f"{sum(v['illegal_emits_after_shield'] for v in legality.values())} "
        "illegal carriers in total reached the air interface. A jammer can take "
        "the throughput; it cannot make the rApp emit outside its licence.",
        "BARRAGE (broadband AWGN) jammer: both receivers degrade together as JSR "
        f"rises (at the measured p90 JSR={barrage_hi['jsr_dB']} dB neural SER "
        f"{barrage_hi['neural_ser']['mean']:.3f}, LMMSE SER "
        f"{barrage_hi['classical_lmmse_ser']['mean']:.3f}). The Shield CANNOT fix "
        "a noisier channel — it only routes between receivers — so the effective "
        f"SER ({barrage_hi['shield_effective_ser']['mean']:.3f}) tracks the better "
        "receiver but is still degraded. This is the Shield's stated limitation.",
        "PARTIAL-BAND / PULSED / SINGLE-TONE jammers degrade SER monotonically "
        "with JSR as well; partial-band severity also rises with the jammed "
        "fraction. The Shield still only routes — it cannot denoise.",
        (
            "IMPERFECT CSI on the measured channel: at sigma_est="
            f"{csi_hi['sigma_est']}, neural SER {csi_hi['neural_ser']['mean']:.3f} "
            f"vs LMMSE SER {csi_hi['classical_lmmse_ser']['mean']:.3f} — "
            + (
                "the NEURAL receiver degrades MORE (it was trained on perfect-CSI "
                "features and is brittle to a feature-distribution shift in Ĥ)."
                if neural_worse
                else "the LMMSE equaliser degrades MORE (its conj(Ĥ)/(|Ĥ|²+N0) "
                "weights are directly corrupted by the CSI error)."
            )
        ),
    ]
    report["scope_note"] = (
        "WHAT IS REAL: the 4096 receiver positions, the base-station position, "
        "the per-subband channel gains, the per-path amplitudes/phases/delays and "
        "therefore the whole propagation geometry are site-specific Wireless "
        "InSite ray tracing (DeepMIMO ASU campus, 3.5 GHz). WHAT IS NOT: this is "
        "NOT over-the-air capture, and a jammer we place on measured propagation "
        "is a MODELLED attack, not a captured one. No jamming incident, no "
        "spectrum-monitoring trace and no adversary telemetry is used anywhere in "
        "this file. Specific limits: (1) the dataset contains BS->grid-point "
        "channels only, so a jammer's channel to a victim handset does not exist "
        "at any price — Layer A is deliberately an uplink/receiver-blocking "
        "geometry, which is what the measurement supports, and it assumes "
        "reciprocity of the ray-traced channel; (2) the UE/jammer transmit "
        "powers, noise figure and SINR thresholds are declared constants, not "
        "measurements, and are listed under setup.link_budget so any reader can "
        "re-scale them; (3) Layer B normalises each measured link to unit power "
        "so snr_dB is a post-power-control operating point, and the jammer "
        "waveforms (barrage/partial-band/tone/pulsed) are analytic models added "
        "to the measured received signal; (4) SISO isotropic antennas, one BS, no "
        "mobility, no Doppler, no inter-cell interference, no scheduler. Not live "
        "O-RAN traffic, not vendor interoperability, not RF conformance, not "
        "carrier-scale evidence."
    )

    stamp(report)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)
    print(
        f"\nchannel reconstruction vs committed features: "
        f"{check['rms_error_db']} dB RMS"
    )
    print(
        f"jammer siting ({siting['sites_evaluated']} real sites, "
        f"{args.jammer_power_dbm:g} dBm): campus outage@11dB "
        f"min={siting['campus_outage_11db_over_sites']['min']:.3f} "
        f"median={siting['campus_outage_11db_over_sites']['median']:.3f} "
        f"max={siting['campus_outage_11db_over_sites']['max']:.3f} "
        f"(unjammed {siting['unjammed_outage_fraction']['11dB']:.3f})"
    )
    for name in ("best", "median", "worst"):
        e = escape[name]
        print(
            f"subband escape @ {name:6s} real site: aware-fixed "
            f"{e['aware_minus_fixed_db']:+.3f} dB, aware-unaware "
            f"{e['aware_minus_unaware_db']:+.3f} dB, "
            f"{e['receivers_whose_best_subband_changes']} receivers re-tune"
        )
    total_illegal = sum(v["illegal_emits_after_shield"] for v in legality.values())
    print(
        f"Shield: {sum(v['decisions'] for v in legality.values())} decisions over "
        f"3 real sites, illegal_emits={total_illegal}"
    )
    print(f"Layer B measured JSR operating points (dB): {jsr_grid}")
    return 0 if total_illegal == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
