#!/usr/bin/env python3
"""FL model-poisoning battery on **REAL** federated clients (DeepMIMO ASU 3.5 GHz).

What changed (2026-07-28)
-------------------------
This suite used to invent its entire federation. ``honest_population()`` drew the
"honest client gradients" from ``rng.normal(0, 1)`` with a random mean and a
random anisotropic scale — 12 clients x 80 coordinates of pure noise — and the
only thing measured was the L2/cosine displacement of an aggregate of noise.
Nothing in it touched a radio, a position, or a measurement.

It now runs on the **real ray-traced channel data already in the tree**
(``datasets/deepmimo_asu_3p5/generated/channel_features.jsonl``: 4096 real
receiver positions on the ASU campus and their measured per-subband channel
gains from Wireless InSite). The federation is real in every respect an
attacker-free federation can be:

* **Real clients.** The 4096 receivers are partitioned by their REAL 2D
  positions into ``N_SITES`` geographic cohorts (Lloyd/k-means++ style seeding),
  exactly the non-IID split ``benchmarks/federated_coverage_loop.py`` uses. Each
  client is a campus region holding its own slice of the measured propagation
  field — genuinely non-IID, because ray tracing is site-specific.
* **Real learning task.** Each client fits the O-RAN CCO coverage map: predict
  all six MEASURED subband gains (dBW) from the receiver's real position, over
  whitened quadratic position features. 6 features x 6 subbands = 36 real
  parameters.
* **Real benign updates.** A client's uploaded vector is its FedProx proximal
  ridge step computed on its own measured receivers. These are the "honest
  gradients" now — measured-physics updates, not Gaussian draws.
* **Real success metric.** Attack damage is reported as the coverage model's
  RMSE **in dB on the real measured gains** after a full multi-round federation,
  not only as an abstract L2 displacement. A number a radio engineer can read.

WHAT IS STILL SYNTHETIC, AND WHY IT HAS TO BE
---------------------------------------------
**The malicious updates are algorithmic, not captured.** No public corpus of
real malicious federated-learning client updates exists; every published
poisoning result (PoisonedFL, Fang, Shejwalkar & Houmansadr, ALIE, ...)
*synthesises* the malicious updates, because capturing them would require a real
adversary to attack a real production federation and the operator to publish the
raw gradient tensors. Nobody has done that. So the honest construction — and the
one implemented here — is:

    REAL clients + REAL data + REAL benign updates
        + a PUBLISHED, CITED attack algorithm applied to those real updates.

Every attack below is cited to its paper and is computed *from* the real benign
updates it is attacking (that is exactly the full-knowledge threat model those
papers assume). This is NOT a corpus of captured attacks and the result JSON
says so in ``data_provenance``.

Attacks (all real attack math, numpy-only):
  - sign_flip   : Bernstein et al. ICLR'19 / Blanchard et al. NeurIPS'17
  - scaling     : Bagdasaryan et al. AISTATS'20 (model replacement)
  - gaussian    : Blanchard et al. NeurIPS'17 (random Byzantine baseline)
  - min_max     : Shejwalkar & Houmansadr NDSS'21 (Min-Max)
  - min_sum     : Shejwalkar & Houmansadr NDSS'21 (Min-Sum)
  - alie        : Baruch et al. NeurIPS'19 (A Little Is Enough)
  - fang_krum   : Fang et al. USENIX-Sec'20 (Krum-targeted)
  - fang_median : Fang et al. USENIX-Sec'20 (median-targeted)

Defences: fedavg (none), krum, coordinate median, trimmed mean, and FLTrust
(Cao et al., NDSS'21) whose server root set is a deterministic stride sample of
receivers never given to any client.

The real feature file is licence-gated (not redistributed in-repo); build it
with ``datasets/deepmimo_asu_3p5/build.py`` or point ``--features`` at a cached
copy. Pure numpy — no torch.

Run:  python benchmarks/fl_poisoning_suite.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.federated import poison_attacks as pa
from horizon_ric.federated import robust
from horizon_ric.runtime_env import stamp

AGGREGATORS = ("fedavg", "krum", "median", "trimmed_mean", "fltrust")
ATTACKS = (
    "sign_flip",
    "scaling",
    "gaussian",
    "min_max",
    "min_sum",
    "alie",
    "fang_krum",
    "fang_median",
)

ATTACK_CITATIONS = {
    "sign_flip": "Bernstein et al., signSGD with Majority Vote, ICLR 2019; "
    "Blanchard et al., NeurIPS 2017 (inverted-gradient Byzantine)",
    "scaling": "Bagdasaryan et al., How To Backdoor Federated Learning, AISTATS 2020 "
    "(model replacement / boosting)",
    "gaussian": "Blanchard et al., Machine Learning with Adversaries, NeurIPS 2017 "
    "(random Byzantine baseline)",
    "min_max": "Shejwalkar & Houmansadr, Manipulating the Byzantine, NDSS 2021 (Min-Max)",
    "min_sum": "Shejwalkar & Houmansadr, Manipulating the Byzantine, NDSS 2021 (Min-Sum)",
    "alie": "Baruch, Baruch & Goldberg, A Little Is Enough, NeurIPS 2019",
    "fang_krum": "Fang, Cao, Jia & Gong, Local Model Poisoning Attacks to "
    "Byzantine-Robust Federated Learning, USENIX Security 2020 (Krum-targeted)",
    "fang_median": "Fang, Cao, Jia & Gong, USENIX Security 2020 (median-targeted)",
}
DEFENCE_CITATIONS = {
    "krum": "Blanchard et al., NeurIPS 2017",
    "median": "Yin et al., ICML 2018 (coordinate-wise median)",
    "trimmed_mean": "Yin et al., ICML 2018 (coordinate-wise trimmed mean)",
    "fltrust": "Cao, Fang, Liu & Gong, FLTrust, NDSS 2021",
}

# --- Real federation geometry (mirrors benchmarks/federated_coverage_loop.py) ---
N_SITES = 16          # geographic client cohorts carved out of the real positions
ROUNDS = 12           # federated rounds per configuration
PROX_LAMBDA = 2.0     # FedProx proximal strength (keeps local solves stable)
PROX_ETA = 0.6        # local step size on the proximal solution
ROOT_STRIDE = 8       # every 8th receiver is server-held (FLTrust root set)
N_SUBBANDS = 6

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = _REPO / "datasets" / "deepmimo_asu_3p5" / "generated" / "channel_features.jsonl"
DEFAULT_MANIFEST = _REPO / "datasets" / "deepmimo_asu_3p5" / "manifest.json"


# ---------------------------------------------------------------------------
# Real data -> real federated learning problem
# ---------------------------------------------------------------------------
def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"missing real feature file {path}\n"
            "This benchmark runs on real DeepMIMO ray-traced channels. Build them "
            "with: python datasets/deepmimo_asu_3p5/build.py  (licence-gated, not "
            "redistributed in-repo), or pass --features <cached copy>."
        )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no feature rows in {path}")
    return rows


def _whiten_design(pos: np.ndarray) -> np.ndarray:
    """Whitened quadratic position design ``[whitened(x,y,x²,y²,xy), 1]``.

    Centering + SVD whitening gives the *global* feature covariance identity, so
    the local proximal solves are well conditioned even on geographically
    concentrated, non-IID clients. Fixed, deterministic, data-published
    preprocessing (identical to ``federated_coverage_loop.py``).
    """
    x, y = pos[:, 0], pos[:, 1]
    feats = np.column_stack([x, y, x * x, y * y, x * y])
    feats = feats - feats.mean(axis=0)
    u, _s, _vt = np.linalg.svd(feats, full_matrices=False)
    whitened = u * np.sqrt(len(feats))
    return np.column_stack([whitened, np.ones(len(feats))])


def _partition_geographic(pos: np.ndarray, n_sites: int, *, seed: int) -> list[np.ndarray]:
    """Partition receivers into ``n_sites`` clients by their REAL 2D positions."""
    rng = np.random.default_rng(seed)
    centers = [pos[int(rng.integers(len(pos)))]]
    for _ in range(1, n_sites):
        d = np.min([np.linalg.norm(pos - c, axis=1) for c in centers], axis=0)
        centers.append(pos[int(np.argmax(d))])
    centers = np.asarray(centers)
    assign = np.zeros(len(pos), dtype=np.int64)
    for _ in range(50):
        assign = np.argmin(
            np.stack([np.linalg.norm(pos - c, axis=1) for c in centers], axis=1), axis=1
        )
        new = np.asarray(
            [
                pos[assign == k].mean(axis=0) if np.any(assign == k) else centers[k]
                for k in range(n_sites)
            ]
        )
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    return [np.flatnonzero(assign == k) for k in range(n_sites)]


class CoverageFederation:
    """The real multi-output coverage-map problem, split into real clients.

    Target ``B`` is the measured per-subband gain matrix (n_receivers, 6) in dBW,
    standardised per subband; design ``A`` is the whitened quadratic position
    basis. The model ``W`` is (6 features, 6 subbands) = 36 real parameters, and
    a client's uploaded update is its FedProx proximal step on its own measured
    receivers, flattened.
    """

    def __init__(self, rows: list[dict[str, Any]], *, n_sites: int, seed: int) -> None:
        gains = np.asarray([r["subband_gain_dbw"] for r in rows], dtype=np.float64)
        pos = np.asarray([r["position_m"][:2] for r in rows], dtype=np.float64)
        if gains.shape[1] != N_SUBBANDS:
            raise ValueError(f"expected {N_SUBBANDS} measured subbands, got {gains.shape}")
        if not (np.all(np.isfinite(gains)) and np.all(np.isfinite(pos))):
            raise ValueError("feature rows carry non-finite gains or positions")

        self.gains = gains
        self.pos = pos
        self.design = _whiten_design(pos)
        self.mean = gains.mean(axis=0)
        self.std = gains.std(axis=0)
        self.target = (gains - self.mean) / self.std

        root = np.arange(0, len(rows), ROOT_STRIDE)
        mask = np.ones(len(rows), dtype=bool)
        mask[root] = False
        self.root_idx = root
        self.private_idx = np.flatnonzero(mask)

        sub = _partition_geographic(pos[self.private_idx], n_sites, seed=seed)
        self.sites = [self.private_idx[s] for s in sub]
        if min(len(s) for s in self.sites) < 6:
            raise ValueError("a geographic client holds fewer receivers than parameters")

        self.n_params = self.design.shape[1] * N_SUBBANDS

    # -- metrics -------------------------------------------------------------
    @property
    def baseline_rmse_db(self) -> float:
        """Predict-the-per-subband-mean everywhere: the do-nothing coverage model."""
        return float(np.sqrt(np.mean((self.gains - self.mean) ** 2)))

    def rmse_db(self, w: np.ndarray) -> float:
        pred = (self.design @ w.reshape(-1, N_SUBBANDS)) * self.std + self.mean
        return float(np.sqrt(np.mean((pred - self.gains) ** 2)))

    # -- federated primitives ------------------------------------------------
    def _prox(self, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """FedProx proximal ridge update on receiver subset ``idx`` (flattened)."""
        a = self.design[idx]
        b = self.target[idx]
        wm = w.reshape(-1, N_SUBBANDS)
        n = a.shape[0]
        hess = a.T @ a / n + PROX_LAMBDA * np.eye(a.shape[1])
        grad = a.T @ (a @ wm - b) / n
        return (-PROX_ETA * np.linalg.solve(hess, grad)).ravel()

    def client_updates(self, w: np.ndarray) -> list[np.ndarray]:
        """Every honest client's REAL update from its own measured receivers."""
        return [self._prox(w, idx) for idx in self.sites]

    def root_update(self, w: np.ndarray) -> np.ndarray:
        """The server's FLTrust reference update on its held-out root receivers."""
        return self._prox(w, self.root_idx)

    def optimum_rmse_db(self) -> float:
        """Centralised least-squares fit — the best this model class can do."""
        w, *_ = np.linalg.lstsq(self.design, self.target, rcond=None)
        return self.rmse_db(w.ravel())


# ---------------------------------------------------------------------------
# Attacks + aggregation
# ---------------------------------------------------------------------------
def make_malicious(attack: str, benign: list[np.ndarray], n_byz: int, seed: int) -> list[np.ndarray]:
    """Apply a PUBLISHED attack algorithm to the REAL benign client updates.

    Full-knowledge threat model (the one all the cited papers assume): the
    adversary controls ``n_byz`` clients and observes the honest clients'
    updates for the round. The malicious vectors are *computed from* the real
    updates — they are algorithmic, not captured traffic.
    """
    if n_byz <= 0:
        return []
    if attack == "gaussian":
        return list(pa.gaussian_attack(benign, n_byz, seed=seed))
    fn = pa.ATTACK_BATTERY[attack][0]
    return list(fn(benign, n_byz))


def aggregate(
    method: str, updates: list[np.ndarray], n_byz: int, root_update: np.ndarray
) -> tuple[np.ndarray | None, bool | None]:
    """Aggregate with one defence. Returns (aggregate, krum_selected_byzantine).

    ``None`` means the aggregator's structural precondition is not met at this
    Byzantine count, and we refuse to run it rather than report a meaningless
    number.
    """
    n = len(updates)
    if method == "fedavg":
        return robust.fedavg(updates), None
    if method == "median":
        return robust.coordinate_median(updates), None
    if method == "krum":
        if n <= 2 * n_byz + 2:
            return None, None
        kr = robust.krum(updates, f=n_byz)
        return kr.aggregate, bool(kr.selected_index >= n - n_byz)
    if method == "trimmed_mean":
        if n <= 2 * n_byz:
            return None, None
        return robust.trimmed_mean(updates, beta=n_byz), None
    if method == "fltrust":
        return robust.fltrust(updates, root_update).aggregate, None
    raise ValueError(f"unknown aggregator {method!r}")


def federate(
    fed: CoverageFederation,
    *,
    method: str,
    attack: str | None,
    n_byz: int,
    rounds: int,
    seed: int,
) -> dict[str, Any]:
    """Run a full federation on the real data, poisoning ``n_byz`` clients.

    The compromised clients are the LAST ``n_byz`` sites, so the honest cohort is
    the first ``K - n_byz`` real geographic clients; their real updates are the
    input the attack algorithm optimises against.
    """
    k = len(fed.sites)
    w = np.zeros(fed.n_params, dtype=np.float64)
    first_round: dict[str, float] = {}
    byz_selected = 0
    rounds_run = 0
    for r in range(rounds):
        honest = fed.client_updates(w)
        if attack is None or n_byz == 0:
            updates = honest
            honest_mean = np.mean(honest, axis=0)
        else:
            honest = honest[: k - n_byz]
            malicious = make_malicious(attack, honest, n_byz, seed=seed * 1000 + r)
            updates = honest + malicious
            honest_mean = np.mean(honest, axis=0)
        agg, sel = aggregate(method, updates, n_byz, fed.root_update(w))
        if agg is None:
            return {"skipped": f"{method} precondition not met at f={n_byz}"}
        if sel:
            byz_selected += 1
        if r == 0:
            hn = float(np.linalg.norm(honest_mean))
            an = float(np.linalg.norm(agg))
            cos = float(agg @ honest_mean / (an * hn)) if an > 0 and hn > 0 else 0.0
            first_round = {
                "l2_from_honest_mean": round(float(np.linalg.norm(agg - honest_mean)), 6),
                "cos_err": round(1.0 - cos, 6),
                "honest_mean_norm": round(hn, 6),
            }
        w = w + agg
        rounds_run += 1
        if not np.all(np.isfinite(w)) or float(np.linalg.norm(w)) > 1e12:
            break
    diverged = not np.all(np.isfinite(w)) or float(np.linalg.norm(w)) > 1e12
    rmse = float("inf") if diverged else fed.rmse_db(w)
    out: dict[str, Any] = {
        "round_1": first_round,
        "rounds_run": rounds_run,
        "diverged": diverged,
        "final_rmse_db": None if diverged else round(rmse, 4),
        "w_norm": None if diverged else round(float(np.linalg.norm(w)), 4),
    }
    if method == "krum":
        out["krum_byzantine_selection_rate"] = round(byz_selected / max(rounds_run, 1), 4)
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def _mean_std(v: list[float]) -> dict[str, float]:
    a = np.asarray(v, dtype=np.float64)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4)}


def run_sweep(
    rows: list[dict[str, Any]],
    *,
    n_sites: int,
    rounds: int,
    seeds: int,
    byz_grid: list[int],
) -> dict[str, Any]:
    feds = [CoverageFederation(rows, n_sites=n_sites, seed=1234 + s) for s in range(seeds)]
    fed0 = feds[0]

    clean: dict[str, dict] = {}
    for method in AGGREGATORS:
        per = [
            federate(f, method=method, attack=None, n_byz=0, rounds=rounds, seed=s)
            for s, f in enumerate(feds)
        ]
        vals = [p["final_rmse_db"] for p in per if p.get("final_rmse_db") is not None]
        clean[method] = _mean_std(vals) if vals else {"skipped": True}

    results: dict[str, dict] = {}
    for attack in ATTACKS:
        results[attack] = {}
        for f_byz in byz_grid:
            bucket: dict[str, Any] = {}
            for method in AGGREGATORS:
                rmses, l2s, coss, sels = [], [], [], []
                skipped = None
                for s, fed in enumerate(feds):
                    r = federate(
                        fed, method=method, attack=attack, n_byz=f_byz,
                        rounds=rounds, seed=s,
                    )
                    if "skipped" in r:
                        skipped = r["skipped"]
                        break
                    if r["final_rmse_db"] is not None:
                        rmses.append(r["final_rmse_db"])
                    l2s.append(r["round_1"]["l2_from_honest_mean"])
                    coss.append(r["round_1"]["cos_err"])
                    if "krum_byzantine_selection_rate" in r:
                        sels.append(r["krum_byzantine_selection_rate"])
                if skipped:
                    bucket[method] = {"skipped": skipped}
                    continue
                entry: dict[str, Any] = {
                    "final_rmse_db": _mean_std(rmses) if rmses else None,
                    "diverged_seeds": len(feds) - len(rmses),
                    "round1_l2_from_honest_mean": _mean_std(l2s),
                    "round1_cos_err": _mean_std(coss),
                }
                if sels:
                    entry["krum_byzantine_selection_rate"] = _mean_std(sels)
                bucket[method] = entry
            results[attack][f"f={f_byz}"] = bucket
    return {
        "clean_federation_rmse_db": clean,
        "results": results,
        "problem": {
            "clients": n_sites,
            "client_receivers_min": int(min(len(s) for s in fed0.sites)),
            "client_receivers_max": int(max(len(s) for s in fed0.sites)),
            "server_root_receivers": int(fed0.root_idx.size),
            "client_held_receivers": int(fed0.private_idx.size),
            "model_parameters": int(fed0.n_params),
            "predict_the_mean_baseline_rmse_db": round(fed0.baseline_rmse_db, 4),
            "centralised_least_squares_rmse_db": round(fed0.optimum_rmse_db(), 4),
            "measured_gain_span_db": round(
                float(fed0.gains.max() - fed0.gains.min()), 4
            ),
        },
    }


def build_efficacy_matrix(sweep: dict, byz_grid: list[int], degrade_db: float = 1.0) -> dict:
    """Per (attack, defence): did the attack materially damage the REAL model?

    "DEFEATED" means, at the largest swept Byzantine count, the coverage RMSE on
    the measured gains degraded by more than ``degrade_db`` dB versus the same
    aggregator's clean-federation RMSE, or the federation diverged, or (Krum)
    the attack forced Krum to select a malicious update in >50% of rounds.
    """
    worst = f"f={byz_grid[-1]}"
    clean = sweep["clean_federation_rmse_db"]
    matrix: dict[str, dict[str, str]] = {}
    for attack in ATTACKS:
        row: dict[str, str] = {}
        for method in AGGREGATORS:
            slot = sweep["results"][attack][worst].get(method, {})
            if "skipped" in slot:
                row[method] = "N/A (precondition not met)"
                continue
            base = clean[method].get("mean")
            rm = slot.get("final_rmse_db")
            if rm is None:
                row[method] = "DIVERGED (unbounded)"
                continue
            delta = rm["mean"] - base
            sel = slot.get("krum_byzantine_selection_rate", {}).get("mean")
            beaten = delta > degrade_db or (sel is not None and sel > 0.5)
            tag = "DEFEATED" if beaten else "bounded"
            extra = f", krum_byz_sel={sel:.0%}" if sel is not None else ""
            row[method] = (
                f"{tag} rmse {base:.2f}->{rm['mean']:.2f} dB (+{delta:.2f} dB{extra})"
            )
        matrix[attack] = row
    return matrix


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--sites", type=int, default=N_SITES)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--seeds", type=int, default=4, help="geographic partition seeds")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "fl_poisoning_suite.json",
    )
    args = ap.parse_args()

    rows = _load_rows(args.features)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    # Byzantine counts: 1 client, Krum's structural ceiling (f < (K-2)/2), and
    # one point beyond it where Krum is undefined and median/trimmed-mean are at
    # their breakdown edge.
    k = args.sites
    krum_max = (k - 3) // 2
    byz_grid = sorted({1, max(k // 8, 1), max(krum_max, 1), min(k // 2, k - 2)})

    sweep = run_sweep(
        rows, n_sites=args.sites, rounds=args.rounds, seeds=args.seeds, byz_grid=byz_grid
    )
    matrix = build_efficacy_matrix(sweep, byz_grid)

    report: dict[str, Any] = {
        "benchmark": "Federated model-poisoning battery on real DeepMIMO federated clients",
        "dataset": "DeepMIMO ASU Campus 3.5 GHz",
        "scenario": manifest.get("scenario"),
        "data_kind": manifest.get("data_kind"),
        "features_sha256": manifest.get("features_sha256"),
        "source_archive_sha256": manifest.get("source_archive_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "receivers": len(rows),
        "data_provenance": {
            "clients": "real — 4096 measured receiver positions partitioned into "
            f"{args.sites} geographic cohorts by their real 2D coordinates",
            "client_data": "real — Wireless InSite ray-traced per-subband channel "
            "gains (dBW) at those positions",
            "benign_updates": "real — FedProx proximal ridge steps computed on each "
            "client's own measured receivers",
            "malicious_updates": "ALGORITHMIC, CITED, NOT CAPTURED — published attack "
            "algorithms applied to the real benign updates under the full-knowledge "
            "threat model those papers assume",
            "why": "No public corpus of captured malicious federated-learning client "
            "updates exists; every published poisoning result synthesises them. "
            "Claiming 'real attack data' here would be false.",
            "attack_citations": ATTACK_CITATIONS,
            "defence_citations": DEFENCE_CITATIONS,
        },
        "config": {
            "clients": args.sites,
            "rounds": args.rounds,
            "partition_seeds": args.seeds,
            "byzantine_counts_swept": byz_grid,
            "byzantine_fractions_swept": [round(f / args.sites, 3) for f in byz_grid],
            "prox_lambda": PROX_LAMBDA,
            "prox_eta": PROX_ETA,
            "root_stride": ROOT_STRIDE,
            "krum_structural_ceiling_f": krum_max,
        },
        "breakdown_points": pa.BREAKDOWN_POINTS,
        "sweep": sweep,
        "efficacy_matrix_at_worst_f": matrix,
    }

    clean = sweep["clean_federation_rmse_db"]
    prob = sweep["problem"]
    report["honest_findings"] = [
        "The federation is real: {} geographic client cohorts carved out of {} "
        "measured ASU-campus receivers, each fitting the measured per-subband "
        "coverage field. Clean FedAvg reaches {:.2f} dB RMSE against a "
        "{:.2f} dB predict-the-mean baseline and a {:.2f} dB centralised "
        "least-squares optimum, so the task has real, large, learnable signal — "
        "which is what makes the poisoning result legible.".format(
            args.sites, len(rows), clean["fedavg"]["mean"],
            prob["predict_the_mean_baseline_rmse_db"],
            prob["centralised_least_squares_rmse_db"],
        ),
        "The malicious updates are NOT captured attacks. They are published "
        "algorithms (ALIE NeurIPS'19, Fang USENIX-Sec'20, Min-Max/Min-Sum "
        "NDSS'21, model replacement AISTATS'20) applied to the real benign "
        "updates. No corpus of real malicious FL updates exists to use instead.",
        "Damage is now reported in dB of coverage-map error on the measured "
        "gains, not only as L2 displacement of an aggregate of noise.",
    ]
    stamp(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"FL poisoning suite (REAL DeepMIMO federation) -> {args.out}")
    print(
        f"clients={args.sites} rounds={args.rounds} seeds={args.seeds} "
        f"byz={byz_grid} params={prob['model_parameters']}"
    )
    print(
        "clean RMSE dB: "
        + ", ".join(
            f"{m}={clean[m]['mean']:.2f}" for m in AGGREGATORS if "mean" in clean[m]
        )
        + f"  | baseline {prob['predict_the_mean_baseline_rmse_db']:.2f}"
    )
    print(f"\nATTACK x DEFENCE at f={byz_grid[-1]} of {args.sites} clients "
          f"(real coverage RMSE, dB):")
    header = f"{'attack':12s} | " + " | ".join(f"{a:>13s}" for a in AGGREGATORS)
    print(header)
    print("-" * len(header))
    for attack in ATTACKS:
        cells = []
        for method in AGGREGATORS:
            slot = sweep["results"][attack][f"f={byz_grid[-1]}"].get(method, {})
            if "skipped" in slot:
                cells.append(f"{'N/A':>13s}")
            elif slot.get("final_rmse_db") is None:
                cells.append(f"{'DIVERGED':>13s}")
            else:
                cells.append(f"{slot['final_rmse_db']['mean']:>13.2f}")
        print(f"{attack:12s} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
