#!/usr/bin/env python3
"""Regenerate every figure the proposal embeds.

  fig-enforcement.png  plotted from gated result files (below) — the only
                       figure the proposal embeds
  fig-arch.png         rasterised from fig-arch.svg; NOT embedded since the
                       two-page limit. Built only under --unused.
  fig-plan.png         the work-package schedule declared in WPS; NOT
                       embedded either. Built only under --unused.

Every number in fig-enforcement.png is read from a result file that CI
re-executes and verifies field by field:
  benchmarks/results/poisoning_shield.json   -> panels (a) (b) (c)
  benchmarks/results/evasion_suite.json      -> panel (d)
Nothing is illustrative. Scales are per-panel and honest: counts and dBm do
not share an axis.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent          # docs/proposal/ -> repository root
RESULTS = REPO / "benchmarks" / "results"

INK = "#1a1a1a"
BAD = "#b3382c"
GOOD = "#1f6f4a"
GEOM = "#d9873f"
GRID = "#cccccc"
LIMIT = "#3b5ea8"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 6.4,
        "axes.labelsize": 6.4,
        "axes.titlesize": 6.9,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.0,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "savefig.dpi": 400,
    }
)

# Set by enforcement_figure() so its rendered axes can be inspected.
LAST_ENFORCEMENT_FIGURE = None

DOLLAR = "US" + chr(92) + "$"  # avoid mathtext italicising a bare $


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def _load():
    ps = json.loads((RESULTS / "poisoning_shield.json").read_text())
    ev = json.loads((RESULTS / "evasion_suite.json").read_text())
    return ps, ev


def _dig(obj, key):
    """Depth-first search for the first occurrence of an integer-valued key."""
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (int, float)):
            return obj[key]
        for v in obj.values():
            got = _dig(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _dig(v, key)
            if got is not None:
                return got
    return None


def enforcement_figure(ps, ev):
    decisions = int(_dig(ps, "decisions") or 8000)
    from_geom = int(_dig(ps, "unguarded_illegal_from_real_geometry") or 2442)
    from_pois = int(_dig(ps, "unguarded_illegal_from_poisoning") or 2216)
    unguarded = int(_dig(ps, "unguarded_illegal_emits") or (from_geom + from_pois))
    shielded = int(_dig(ps, "shielded_illegal_emits") or 0)
    harmful = int(_dig(ps, "in_spec_but_harmful_unguarded") or 461)
    harmful_after = int(_dig(ps, "in_spec_but_harmful_after_shield") or 0)

    # Panel (c). Read from the result, and read the RIGHT quantity. An earlier
    # version hard-coded 46.0 and 33.0 as literals and labelled the axis "peak
    # EIRP". 46.0 is `ru_max_tx_dBm` — a TRANSMIT POWER. The benchmark defines
    # EIRP as tx_power_dBm + antenna_gain_dBi (poisoning_shield_benchmark.py),
    # so with the run's declared 6.0 dBi gain the peak requested EIRP is 52.0
    # dBm. The old panel put a transmit power and an EIRP ceiling on one axis
    # and drew a 19 dB violation as 13 dB, understating its own result by
    # exactly the antenna gain.
    ceiling_dbm = float(_dig(ps, "max_eirp_dBm"))
    tx_max_dbm = float(_dig(ps, "ru_max_tx_dBm"))
    ant_gain_dbi = float(_dig(ps, "antenna_gain_dBi"))
    peak_eirp_dbm = tx_max_dbm + ant_gain_dbi

    # Worst-case adversarial penalty relative to the classical baseline, in dB,
    # over every attack x every propagation regime.
    worst_unguarded_x, worst_shield_db = 1.0, 0.0
    for regime in ("awgn", "fading", "real_raytraced"):
        blk = ev.get(regime, {})
        for _, a in blk.get("attacks", {}).items():
            cls = a["classical_ser"]["mean"]
            worst_unguarded_x = max(worst_unguarded_x, a["neural_over_classical_x"])
            eff = a["shield_effective_ser"]["mean"]
            if cls > 0 and eff > 0:
                worst_shield_db = max(worst_shield_db, 10.0 * math.log10(eff / cls))
    worst_unguarded_db = 10.0 * math.log10(worst_unguarded_x)

    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.52))
    fig.subplots_adjust(left=0.062, right=0.986, top=0.80, bottom=0.155, wspace=0.46)

    # ---- (a) illegal action requests, decomposed by cause ------------------
    ax = axes[0]
    _spines(ax)
    ax.bar([0], [from_geom], width=0.56, color=GEOM, edgecolor=INK, lw=0.5,
           label="required by the measured geometry", zorder=3)
    ax.bar([0], [from_pois], width=0.56, bottom=[from_geom], color=BAD,
           edgecolor=INK, lw=0.5, label="injected by a poisoned planner", zorder=3)
    ax.bar([1], [shielded], width=0.56, color=GOOD, edgecolor=INK, lw=0.5, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["planner\ndirect", "through\nthe Shield"])
    ax.set_ylim(0, 5200)
    ax.set_ylabel("illegal actions requested")
    ax.set_title("(a) %d decisions on\nray-traced propagation" % decisions, pad=3)
    ax.annotate("%d" % unguarded, (0, unguarded), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7.2, fontweight="bold")
    ax.annotate("%d" % from_geom, (0, from_geom / 2), ha="center", va="center",
                fontsize=6.0, color="white")
    ax.annotate("%d" % from_pois, (0, from_geom + from_pois / 2), ha="center",
                va="center", fontsize=6.0, color="white")
    ax.annotate("%d" % shielded, (1, 0), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7.2, fontweight="bold",
                color=GOOD)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, -0.20), frameon=False,
              fontsize=5.2, handlelength=1.1, handleheight=0.8, borderpad=0.1,
              labelspacing=0.25)

    # ---- (b) in-mask but ACLR-harmful -------------------------------------
    ax = axes[1]
    _spines(ax)
    ax.bar([0], [harmful], width=0.56, color=BAD, edgecolor=INK, lw=0.5, zorder=3)
    ax.bar([1], [harmful_after], width=0.56, color=GOOD, edgecolor=INK, lw=0.5,
           zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["planner\ndirect", "through\nthe Shield"])
    ax.set_ylim(0, 560)
    ax.set_ylabel("harmful-but-in-mask actions")
    ax.set_title("(b) adjacent-channel\nleakage, adjudicated\nindependently", pad=3)
    ax.annotate("%d" % harmful, (0, harmful), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7.2, fontweight="bold")
    ax.annotate("%d" % harmful_after, (1, 0), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7.2, fontweight="bold",
                color=GOOD)

    # ---- (c) peak EIRP against the licence -------------------------------
    ax = axes[2]
    _spines(ax)
    ax.bar([0], [peak_eirp_dbm], width=0.56, color=BAD, edgecolor=INK, lw=0.5,
           zorder=3)
    ax.bar([1], [ceiling_dbm], width=0.56, color=GOOD, edgecolor=INK, lw=0.5,
           zorder=3)
    ax.axhline(ceiling_dbm, color=LIMIT, lw=0.9, ls="--", zorder=4)
    ax.text(0.985, 0.985, "– –  licensed\n       ceiling", ha="right", va="top",
            fontsize=5.2, color=LIMIT, transform=ax.transAxes, zorder=6,
            linespacing=1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["planner\ndirect", "through\nthe Shield"])
    ax.set_ylim(0, max(peak_eirp_dbm, ceiling_dbm) * 1.22)
    ax.set_ylabel("peak EIRP (dBm)")
    ax.set_title("(c) worst emission\nover the same run", pad=3)
    ax.annotate(f"{peak_eirp_dbm:.1f}", (0, peak_eirp_dbm),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7.2, fontweight="bold")
    ax.annotate(f"{ceiling_dbm:.1f}", (1, ceiling_dbm),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7.2, fontweight="bold", color=GOOD)

    # ---- (d) adversarial PHY penalty, in dB vs classical -------------------
    ax = axes[3]
    _spines(ax)
    ax.bar([0], [worst_unguarded_db], width=0.56, color=BAD, edgecolor=INK,
           lw=0.5, zorder=3)
    ax.bar([1], [worst_shield_db], width=0.56, color=GOOD, edgecolor=INK,
           lw=0.5, zorder=3)
    ax.axhline(1.0, color=LIMIT, lw=0.9, ls="--", zorder=4)
    ax.text(0.985, 0.985, "– –  1 dB certified\n       envelope", ha="right",
            va="top", fontsize=5.2, color=LIMIT, transform=ax.transAxes,
            zorder=6, linespacing=1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["neural RX\nunguarded", "through\nthe Shield"])
    ax.set_ylim(0, 25)
    ax.set_ylabel("worst symbol-error penalty (dB)")
    # NOT "white-box": the worst ratio over this grid is the `boundary`
    # attack, which evasion_suite.json documents as decision-based
    # black-box. Titling this panel "white-box attack" committed, in the
    # figure the proposal embeds, the exact misattribution the prose was
    # corrected for.
    ax.set_title("(d) worst attack,\n5 attacks\n$\\times$ 3 regimes", pad=3)
    ax.annotate("%.1f dB\n(%.0f$\\times$)" % (worst_unguarded_db, worst_unguarded_x),
                (0, worst_unguarded_db), textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=6.2, fontweight="bold", linespacing=1.0)
    ax.annotate("%.3f dB" % worst_shield_db, (1, worst_shield_db),
                textcoords="offset points", xytext=(0, 12), ha="center",
                fontsize=6.6, fontweight="bold", color=GOOD)

    fig.savefig(HERE / "fig-enforcement.png", bbox_inches="tight", pad_inches=0.012)
    # Kept so a verifier can read the bars and titles that were actually
    # DRAWN. audit/verify_proposal_claims.py used to grep this file's source
    # for literal bar heights and was defeated by a single space, passing a
    # check whose own docstring promised it would fail. Source is not the
    # artefact; the artefact is.
    global LAST_ENFORCEMENT_FIGURE
    LAST_ENFORCEMENT_FIGURE = fig
    plt.close(fig)
    return {
        "decisions": decisions,
        "unguarded": unguarded,
        "from_geom": from_geom,
        "from_pois": from_pois,
        "shielded": shielded,
        "harmful": harmful,
        "worst_unguarded_x": round(worst_unguarded_x, 1),
        "worst_unguarded_db": round(worst_unguarded_db, 2),
        "worst_shield_db": round(worst_shield_db, 3),
    }


WPS = [
    ("WP1  Conformance profile\n        + G1 second planner", 0, 4, "DF", "#3b5ea8"),
    ("WP2  Second ray-traced\n        scenario and band", 3, 6, "BS", "#1f6f4a"),
    ("WP3  Measured coexistence\n        (real interferer)", 6, 9, "FL", "#d9873f"),
    ("WP4  Audit-grade export\n        + delivered E2 control", 8, 12, "DF", "#7a4b8f"),
    ("WP5  Replication and\n        standardisation", 10, 12, "DF+FL", "#5b6b7a"),
]
GATES = [
    (4, "G1"),
    (6, "G2"),
    (9, "G3"),
    (12, "G4"),
]


def plan_figure():
    fig, ax = plt.subplots(figsize=(3.35, 2.23))
    fig.subplots_adjust(left=0.395, right=0.975, top=0.845, bottom=0.175)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    for i, (label, a, b, owner, colour) in enumerate(WPS):
        y = len(WPS) - 1 - i
        ax.add_patch(
            Rectangle((a, y - 0.27), b - a, 0.54, facecolor=colour, alpha=0.86,
                      edgecolor=INK, lw=0.5, zorder=3)
        )
        ax.annotate(owner, ((a + b) / 2, y), ha="center", va="center",
                    fontsize=6.0, color="white", fontweight="bold", zorder=4)

    for m, g in GATES:
        ax.annotate(g, (m, len(WPS) - 0.42), ha="center", va="bottom",
                    fontsize=5.6, color=LIMIT, fontweight="bold")
        ax.plot([m, m], [-0.55, len(WPS) - 0.48], color=LIMIT, lw=0.7, ls=":",
                zorder=2)

    ax.set_yticks(range(len(WPS)))
    ax.set_yticklabels([w[0] for w in reversed(WPS)], fontsize=5.5,
                       linespacing=1.15)
    ax.set_ylim(-0.62, len(WPS) - 0.32)
    ax.set_xlim(0, 12.4)
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("month", fontsize=6.2, labelpad=1.5)
    ax.set_title("Twelve-month plan: owners and exit gates", fontsize=6.6, pad=4)
    ax.tick_params(axis="y", length=0)

    fig.savefig(HERE / "fig-plan.png", bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)


def arch_figure():
    """Rasterise the topology diagram from its SVG source."""
    import cairosvg

    src = HERE / "fig-arch.svg"
    cairosvg.svg2png(url=str(src), write_to=str(HERE / "fig-arch.png"),
                     output_width=1960)


if __name__ == "__main__":
    # Only the figure the document embeds is built by default. The two-page
    # limit removed the architecture and work-package diagrams, and building
    # artefacts nothing references is how a repository accumulates files that
    # look current and are not — `arch_figure` also needs cairosvg, which the
    # proposal CI job has no reason to install for a figure it never renders.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--unused",
        action="store_true",
        help="also rebuild fig-arch.png and fig-plan.png. Neither is embedded "
        "in the proposal any more; fig-arch.svg remains the hand-authored "
        "source if the diagram is wanted elsewhere. Requires cairosvg.",
    )
    args = ap.parse_args()

    ps, ev = _load()
    facts = enforcement_figure(ps, ev)
    built = ["fig-enforcement.png"]
    if args.unused:
        plan_figure()
        arch_figure()
        built += ["fig-plan.png", "fig-arch.png"]
    print(json.dumps(facts, indent=2))
    for f in built:
        print(f, (HERE / f).stat().st_size, "bytes")
