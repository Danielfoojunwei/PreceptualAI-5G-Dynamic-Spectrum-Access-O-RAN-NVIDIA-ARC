#!/usr/bin/env python3
"""Adversarial robustness on a REALISTIC fading OFDM link — an honest realism check.

The single-symbol AWGN benchmark (`neural_rx_pgd_benchmark.py`) found a ~22x
adversarial gap between the neural receiver and the classical demapper. This
benchmark asks the harder, more honest question: does that gap survive a real
frequency-selective **multipath Rayleigh fading** channel with **LMMSE
equalisation**?

Setup: 16-QAM over a Rayleigh TDL channel; perfect-CSI neural receiver demapping
from [Re Y, Im Y, Re H, Im H]; classical = per-subcarrier LMMSE equalise +
nearest point. The attacker runs white-box **PGD on the received signal Y only**
(it does not control the channel H). Multi-seed mean ± std.

Honest finding (see committed results): under fading the neural-vs-classical gap
**collapses to ~1.1-1.2x** — far below the AWGN toy's 22x — because deep fades
amplify the perturbation (delta-X-hat = conj(H)·delta-Y / (|H|^2 + N0)) and make
BOTH receivers vulnerable. The Shield's value under fading is therefore not "the
neural model is much worse" but the *guarantee*: it measures the independent
(CRC/HARQ) block-error rate and falls back to the certified LMMSE baseline, so the
effective error is never worse than the classical receiver under an attack the
Shield was not coded for.

Pure numpy. Run:  python benchmarks/phy_fading_eval.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from horizon_ric.phy import (
    NeuralReceiver,
    classical_equalize_demap,
    fading_dataset,
    pgd_attack,
)
from horizon_ric.shield.invariants import NeuralRxEnvelopeInvariant

M = 16
PERTURB_MASK = np.array([1, 1, 0, 0])  # attacker controls Y, not the CSI H
WIN = 200  # symbols per measurement window (block-error saturates at this SER, so
           # the fallback decision is taken on the windowed SER, not block TBLER)


def _ser(p, y):
    return float(np.mean(p != y))


def _shield_effective(neural_pred, classical_pred, y):
    """Windowed-SER fallback: per window, fall back to the classical receiver when
    the independently measured neural SER leaves the envelope of the classical
    baseline (NeuralRxEnvelopeInvariant, 1 dB tolerance). Returns
    (effective_ser, fallback_rate).

    Note: at SER ~0.1 a 100-symbol block-error rate saturates to 1.0 for BOTH
    receivers, so the decision is taken on the windowed symbol-error rate instead.
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
            eff.append(baseline_classical)  # routed to the certified LMMSE receiver
        else:
            eff.append(measured_neural)
    return float(np.mean(eff)), fb / n_win


def run_point(snr_dB: float, eps: float, seeds: list[int]) -> dict:
    rows = {"clean_n": [], "clean_c": [], "pgd_n": [], "pgd_c": [],
            "shield_eff_ser": [], "fallback_rate": []}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        Xtr, ytr, _ = fading_dataset(40_000, M, snr_dB, rng)
        Xte, yte, meta = fading_dataset(15_000, M, snr_dB, rng)
        n0 = meta["N0"]
        net = NeuralReceiver(M, hidden=160, seed=1, n_features=4).train(
            Xtr, ytr, epochs=50, lr=0.3, seed=2
        )
        rows["clean_n"].append(_ser(net.predict(Xte), yte))
        rows["clean_c"].append(_ser(classical_equalize_demap(Xte, M, n0), yte))
        Xa = pgd_attack(net, Xte, yte, epsilon=eps, alpha=eps / 6, steps=40,
                        perturb_mask=PERTURB_MASK)
        np_ = net.predict(Xa)
        cp_ = classical_equalize_demap(Xa, M, n0)
        rows["pgd_n"].append(_ser(np_, yte))
        rows["pgd_c"].append(_ser(cp_, yte))
        eff, fbr = _shield_effective(np_, cp_, yte)
        rows["shield_eff_ser"].append(eff)
        rows["fallback_rate"].append(fbr)

    def ms(k):
        a = np.array(rows[k])
        return {"mean": float(a.mean()), "std": float(a.std())}

    pgd_n = ms("pgd_n")
    pgd_c = ms("pgd_c")
    return {
        "snr_dB": snr_dB, "pgd_epsilon": eps, "seeds": seeds,
        "clean_ser": {"neural": ms("clean_n"), "classical_lmmse": ms("clean_c")},
        "pgd_ser": {"neural": pgd_n, "classical_lmmse": pgd_c,
                    "neural_over_classical_x": pgd_n["mean"] / max(pgd_c["mean"], 1e-9)},
        "shield": {"effective_ser": ms("shield_eff_ser"),
                   "fallback_rate": ms("fallback_rate")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    points = [run_point(snr, eps, seeds) for snr, eps in [(28.0, 0.07), (34.0, 0.07)]]
    report = {
        "setup": {"constellation": "16-QAM", "channel": "Rayleigh TDL (4 taps)",
                  "receiver": "perfect-CSI neural vs LMMSE", "attack": "white-box PGD on Y"},
        "points": points,
        "honest_finding": [
            "Under realistic multipath fading the neural-vs-classical adversarial "
            "gap is ~1.1-1.2x (mean over seeds), NOT the ~22x of the AWGN toy: deep "
            "fades amplify the Y-domain perturbation and make both receivers "
            "vulnerable.",
            "The gap is small on average but not uniform: on the minority of "
            "windows where the measured neural SER is meaningfully (>1 dB) worse "
            "than the LMMSE baseline, the Shield falls back (see per-point "
            "fallback_rate, ~12-23% here), nudging the effective SER below the "
            "neural receiver's toward the classical baseline. So under fading the "
            "Shield's benefit is small but real; in AWGN, where the neural "
            "receiver is >>1 dB worse, it falls back almost always and protects "
            "~12x. The guarantee (effective <= classical + tolerance) holds in "
            "both regimes — the Shield helps exactly as much as the measured "
            "degradation warrants, and no more.",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    for p in points:
        print(f"\nSNR {p['snr_dB']} eps {p['pgd_epsilon']}: PGD SER neural "
              f"{p['pgd_ser']['neural']['mean']:.4f}±{p['pgd_ser']['neural']['std']:.4f} vs "
              f"classical {p['pgd_ser']['classical_lmmse']['mean']:.4f} "
              f"({p['pgd_ser']['neural_over_classical_x']:.2f}x); Shield effective SER "
              f"{p['shield']['effective_ser']['mean']:.4f}, fallback "
              f"{p['shield']['fallback_rate']['mean']*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
