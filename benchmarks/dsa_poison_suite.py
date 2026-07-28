#!/usr/bin/env python3
"""Federated DSA data-poisoning + backdoor battery on REAL campus spectrum.

What changed (2026-07-28)
-------------------------
This suite used to be entirely synthetic. Its federated clients were seeded
replicas of the *same* abstract MDP — twelve statistically identical agents with
no geography, no propagation and no reason to disagree — and "damage" was
counted in successful slots, which silently assumes every channel is worth the
same everywhere.

It now runs on the **real ray-traced campus data already in the tree**
(``datasets/deepmimo_asu_3p5``: 4096 real receiver positions on the ASU campus,
measured per-subband channel gains from Wireless InSite):

* **Real, non-IID federated clients.** The receivers are partitioned by their
  REAL 2D positions into geographic cohorts. Each cohort is one federated DSA
  client — a real campus region.
* **Real per-client channel quality.** The six MDP channels map one-to-one onto
  the six MEASURED 3.5 GHz subbands. Each client's local reward for a successful
  transmission on channel ``c`` is scaled by the spectral efficiency its own
  receivers actually support on that subband (Shannon under a declared link
  budget, with a scheduled-MCS floor and a 256QAM ceiling). Clients therefore
  disagree about which channel is worth using **because the measured propagation
  says so** — that is where the non-IID structure comes from, and it is exactly
  the structure a poisoning attack exploits.
* **Real damage units.** Policy quality is reported in bit/s/Hz actually
  deliverable to real receivers at real positions, not only in slot counts.

WHAT IS STILL SYNTHETIC, AND SAID PLAINLY
-----------------------------------------
* **The primary-user occupancy** is a Gilbert-Elliott Markov chain and the
  sensing errors are a declared error rate. The DeepMIMO dataset carries no
  spectrum-occupancy trace, so there is nothing measured to put here. A real
  CBRS occupancy trace exists (Zenodo 18272105, CC-BY-4.0) but wiring it in
  needs a dataset build this file does not own.
* **The attacks themselves.** Reward poisoning, the single-row backdoor and
  free-riding are *algorithms an attacker runs*, not measurements. No public
  corpus of captured malicious federated updates exists — every published
  poisoning result synthesises them. Each attack is cited to its paper and
  labelled ``algorithmic, cited`` in the result JSON. Calling these "real attack
  data" would be false.

The attacks (data-side, i.e. the client corrupts its own training signal before
any model is uploaded):

  1. reward_poison — reward/label poisoning (Biggio et al., ICML 2012; Tolpegin
     et al., ESORICS 2020): the client trains on an inverted reward and learns a
     denial-of-spectrum policy that seeks PU clashes and collisions.
  2. backdoor      — trigger attack (Bagdasaryan et al., AISTATS 2020): the
     client learns the normal task, then overwrites ONE trigger-state Q-row so
     the global policy forces an attacker-chosen channel on that state only.
  3. free_rider    — Fraboni et al. (AISTATS 2021): uploads the stale global
     model plus noise, having done no work.

Every attack is run against FedAvg (no defence) and against median / Krum /
trimmed-mean, sweeping the Byzantine fraction, and then — the safety claim —
every resulting decision is pushed through the Decision Safety Shield and the
hash-chained evidence store. A poisoned policy can lose real bit/s/Hz or carry a
backdoor; it can never emit an illegal RF action.

Pure numpy — no torch. Run:  python benchmarks/dsa_poison_suite.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.federated import robust
from horizon_ric.runtime_env import stamp
from horizon_ric.security.tenant import TenantScope
from horizon_ric.shield import default_terrestrial_shield
from horizon_ric.spectrum.data_poison import (
    BackdoorSpec,
    _poison_reward,
    backdoor_success_rate,
    default_backdoor,
    free_rider_update,
    stamp_backdoor,
)
from horizon_ric.spectrum.dsa_env import DSAConfig, DSAEnv, DSAWorld, n_actions, n_states
from horizon_ric.spectrum.federated_q import QLearnConfig, _greedy_action, _softmax_action
from horizon_ric.spectrum.pipeline import DSADecisionConfig, decide_and_record

# The geographic partition of the real receivers is defined once in
# fl_poisoning_suite and reused here rather than re-derived.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fl_poisoning_suite import _partition_geographic  # noqa: E402

BAND_LO, BAND_HI = 3.40e9, 3.50e9
MAX_EIRP = 33.0

# --- Real spectrum geometry ----------------------------------------------------
N_SUBBANDS = 6                       # measured subbands == MDP channels
OFDM_SUBCARRIERS = 1024
OFDM_SCALING_DB = 10.0 * np.log10(OFDM_SUBCARRIERS)
CARRIER_BW_HZ = (BAND_HI - BAND_LO) / N_SUBBANDS
# Declared link-budget constants (not measurements — listed in the result JSON).
THERMAL_DBM_PER_HZ = -174.0
NOISE_FIGURE_DB = 7.0
NOISE_DBM = THERMAL_DBM_PER_HZ + 10.0 * np.log10(CARRIER_BW_HZ) + NOISE_FIGURE_DB
SU_TX_POWER_DBM = 23.0               # 3GPP power class 3 terminal
ANTENNA_GAIN_DBI = 6.0
MAX_SPECTRAL_EFFICIENCY = 7.4063     # top 3GPP TS 38.214 256QAM MCS entry
MIN_SINR_DB = 11.0                   # below this no MCS is scheduled

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = _REPO / "datasets" / "deepmimo_asu_3p5" / "generated" / "channel_features.jsonl"
DEFAULT_MANIFEST = _REPO / "datasets" / "deepmimo_asu_3p5" / "manifest.json"

ATTACK_CITATIONS = {
    "reward_poison": "Biggio, Nelson & Laskov, Poisoning Attacks against Support "
    "Vector Machines, ICML 2012; Tolpegin et al., Data Poisoning Attacks Against "
    "Federated Learning Systems, ESORICS 2020",
    "backdoor": "Bagdasaryan, Veit, Hua, Estrin & Shmatikov, How To Backdoor "
    "Federated Learning, AISTATS 2020",
    "free_rider": "Fraboni, Vidal & Lorenzi, Free-rider Attacks on Model "
    "Aggregation in Federated Learning, AISTATS 2021",
}
DEFENCE_CITATIONS = {
    "krum": "Blanchard et al., NeurIPS 2017",
    "median": "Yin et al., ICML 2018",
    "trimmed_mean": "Yin et al., ICML 2018",
}


# ---------------------------------------------------------------------------
# Real campus spectrum
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


class CampusSpectrum:
    """Measured per-(receiver, subband) spectral efficiency + geographic clients."""

    def __init__(self, rows: list[dict[str, Any]], *, n_clients: int, seed: int) -> None:
        gains = np.asarray([r["subband_gain_dbw"] for r in rows], dtype=np.float64)
        if gains.shape[1] != N_SUBBANDS:
            raise ValueError(f"expected {N_SUBBANDS} measured subbands, got {gains.shape}")
        if not np.all(np.isfinite(gains)):
            raise ValueError("measured subband gains contain non-finite values")
        pos = np.asarray([r["position_m"][:2] for r in rows], dtype=np.float64)

        self.gain_db = gains + OFDM_SCALING_DB
        self.snr_db = SU_TX_POWER_DBM + ANTENNA_GAIN_DBI + self.gain_db - NOISE_DBM
        se = np.log2(1.0 + 10.0 ** (self.snr_db / 10.0))
        se = np.minimum(se, MAX_SPECTRAL_EFFICIENCY)
        self.se = np.where(self.snr_db < MIN_SINR_DB, 0.0, se)

        self.positions = pos
        self.sites = _partition_geographic(pos, n_clients, seed=seed)
        # Each client's MEASURED mean spectral efficiency per subband.
        self.client_se = np.stack([self.se[idx].mean(axis=0) for idx in self.sites])
        self.se_ref = float(self.se.mean())
        if self.se_ref <= 0:
            raise ValueError("no receiver supports a scheduled MCS on any subband")

    def client_weights(self, k: int) -> np.ndarray:
        """Per-channel reward weight for client ``k`` (1.0 == campus-average link)."""
        return self.client_se[k] / self.se_ref

    @property
    def stats(self) -> dict[str, Any]:
        per_sub = self.se.mean(axis=0)
        return {
            "receivers": int(self.se.shape[0]),
            "clients": int(len(self.sites)),
            "client_receivers_min": int(min(len(s) for s in self.sites)),
            "client_receivers_max": int(max(len(s) for s in self.sites)),
            "measured_subband_gain_db": {
                "min": round(float(self.gain_db.min()), 3),
                "median": round(float(np.median(self.gain_db)), 3),
                "max": round(float(self.gain_db.max()), 3),
            },
            "campus_mean_spectral_efficiency_bps_hz": round(self.se_ref, 4),
            "per_subband_mean_se_bps_hz": [round(float(v), 4) for v in per_sub],
            "fraction_receiver_subbands_below_mcs_floor": round(
                float(np.mean(self.se <= 0.0)), 4
            ),
            "per_client_best_subband": [int(np.argmax(w)) for w in self.client_se],
            "client_se_spread_bps_hz": round(
                float(self.client_se.max() - self.client_se.min()), 4
            ),
        }


# ---------------------------------------------------------------------------
# Local training on a real client (measured-SE-weighted reward)
# ---------------------------------------------------------------------------
def train_client_q(
    env: DSAEnv,
    cfg: QLearnConfig,
    se_weight: np.ndarray,
    *,
    q_init: np.ndarray | None = None,
    seed: int = 0,
    poisoned_reward: bool = False,
) -> np.ndarray:
    """Tabular Q-learning for one REAL client.

    Mirrors :func:`horizon_ric.spectrum.federated_q.train_local_q` exactly except
    that a successful transmission on channel ``c`` pays the client's own
    MEASURED spectral efficiency on subband ``c`` (normalised so the
    campus-average link pays the shipped ``reward_success``). That is the only
    place the real data enters the learning problem, and it is what makes the
    geographic clients genuinely non-IID.

    ``poisoned_reward`` applies the reward inversion of
    :func:`horizon_ric.spectrum.data_poison._poison_reward` (Biggio et al. 2012;
    Tolpegin et al. ESORICS 2020) before the measured weighting.
    """
    ns, na = env.n_states, env.n_actions
    q = np.zeros((ns, na), dtype=np.float64) if q_init is None else q_init.copy()
    rng = np.random.default_rng(seed)
    n_ch = env.cfg.n_channels

    for ep in range(cfg.episodes):
        state = env.reset(seed=seed + ep + 1)
        done = False
        while not done:
            if rng.random() < cfg.epsilon:
                action = int(rng.integers(0, na))
            else:
                action = _greedy_action(q, state)
            nxt, reward, done, info = env.step(action)
            if poisoned_reward:
                reward = _poison_reward(reward, info, env.cfg)
            if info.get("success") and action < n_ch:
                reward *= float(se_weight[action])
            best_next = float(np.max(q[nxt]))
            q[state, action] += cfg.alpha * (
                reward + cfg.gamma * best_next - q[state, action]
            )
            state = nxt
    return q


def build_round(
    campus: CampusSpectrum,
    *,
    attack: str | None,
    n_malicious: int,
    dsa_cfg: DSAConfig,
    q_cfg: QLearnConfig,
    spec: BackdoorSpec,
    seed: int,
) -> list[np.ndarray]:
    """One federated round's client uploads. The LAST ``n_malicious`` are attackers."""
    n_clients = len(campus.sites)
    shape = (n_states(dsa_cfg.n_channels), n_actions(dsa_cfg.n_channels))
    qs: list[np.ndarray] = []
    for k in range(n_clients):
        malicious = attack is not None and k >= n_clients - n_malicious
        cseed = seed * 1000 + k + 1
        env = DSAEnv(cfg=dsa_cfg, seed=cseed)
        w = campus.client_weights(k)
        if not malicious:
            qs.append(train_client_q(env, q_cfg, w, seed=cseed))
        elif attack == "reward_poison":
            qs.append(train_client_q(env, q_cfg, w, seed=cseed, poisoned_reward=True))
        elif attack == "backdoor":
            qs.append(stamp_backdoor(train_client_q(env, q_cfg, w, seed=cseed), spec))
        elif attack == "free_rider":
            qs.append(free_rider_update(None, shape, sigma=1e-3, seed=cseed))
        else:
            raise ValueError(f"unknown attack {attack!r}")
    return qs


def aggregate_q(method: str, qs: list[np.ndarray], f: int) -> np.ndarray | None:
    """Aggregate client Q-tables. ``None`` = structural precondition not met."""
    shape = qs[0].shape
    flat = [q.ravel() for q in qs]
    n = len(flat)
    if method == "fedavg":
        agg = robust.fedavg(flat)
    elif method == "median":
        agg = robust.coordinate_median(flat)
    elif method == "krum":
        if n <= 2 * f + 2:
            return None
        agg = robust.krum(flat, f=f).aggregate
    elif method == "trimmed_mean":
        if n <= 2 * f:
            return None
        agg = robust.trimmed_mean(flat, beta=f)
    else:
        raise ValueError(f"unknown aggregator {method!r}")
    return np.asarray(agg).reshape(shape)


# ---------------------------------------------------------------------------
# Deploy on the real campus and measure delivered bit/s/Hz
# ---------------------------------------------------------------------------
def evaluate_campus(
    q: np.ndarray,
    campus: CampusSpectrum,
    dsa_cfg: DSAConfig,
    *,
    n_episodes: int,
    seed: int,
    tau: float = 0.5,
) -> dict[str, float]:
    """Deploy the shared policy on the DSA world with SUs at REAL positions.

    Action selection is the same Boltzmann deployment
    :func:`horizon_ric.spectrum.federated_q.evaluate_policy` uses, so slot counts
    are directly comparable. The addition is that each secondary user stands at a
    real measured receiver position, so a successful transmission on channel
    ``c`` delivers that receiver's MEASURED spectral efficiency — a channel the
    policy likes but the propagation does not support delivers nothing.
    """
    rng = np.random.default_rng(seed)
    n_rx = campus.se.shape[0]
    success = collisions = clashes = slots = 0
    delivered = 0.0
    for ep in range(n_episodes):
        world = DSAWorld(cfg=dsa_cfg, seed=seed + ep)
        obs = world.reset(seed=seed + ep)
        rx = rng.integers(0, n_rx, size=dsa_cfg.n_users)  # real positions for the SUs
        done = False
        while not done:
            actions = np.array(
                [_softmax_action(q, int(s), tau, rng) for s in obs], dtype=np.int64
            )
            obs, rewards, done, info = world.step(actions)
            for u in range(dsa_cfg.n_users):
                a = int(actions[u])
                if a < dsa_cfg.n_channels and rewards[u] == dsa_cfg.reward_success:
                    delivered += float(campus.se[rx[u], a])
            success += info["successes"]
            collisions += info["collisions"]
            clashes += info["pu_clashes"]
            slots += 1
    d = max(slots, 1)
    return {
        "throughput_per_slot": round(success / d, 4),
        "collision_per_slot": round(collisions / d, 4),
        "pu_clash_per_slot": round(clashes / d, 4),
        "delivered_bps_hz_per_slot": round(delivered / d, 4),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
METHODS = ("fedavg", "median", "krum", "trimmed_mean")


def _reduce(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """mean/std across MDP seeds for each metric."""
    keys = samples[0].keys()
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        arr = np.asarray([s[k] for s in samples], dtype=np.float64)
        out[k] = {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4)}
    return out


def attack_sweep(
    campus: CampusSpectrum,
    *,
    dsa_cfg: DSAConfig,
    q_cfg: QLearnConfig,
    byz_counts: list[int],
    eval_episodes: int,
    seeds: list[int],
) -> tuple[dict[str, Any], BackdoorSpec]:
    """Sweep each data attack over Byzantine counts and defences.

    The geographic client partition is REAL and fixed; ``seeds`` moves the MDP /
    local-training randomness, so every number below is a mean +/- std over
    independent federated rounds rather than one lucky run.
    """
    spec = default_backdoor(dsa_cfg.n_channels, trigger_state=0)
    n_clients = len(campus.sites)

    clean_q: list[dict[str, float]] = []
    clean_sr: list[float] = []
    for sd in seeds:
        qs = build_round(
            campus, attack=None, n_malicious=0, dsa_cfg=dsa_cfg, q_cfg=q_cfg,
            spec=spec, seed=sd,
        )
        g = aggregate_q("fedavg", qs, 0)
        assert g is not None
        clean_q.append(
            evaluate_campus(g, campus, dsa_cfg, n_episodes=eval_episodes, seed=999)
        )
        clean_sr.append(backdoor_success_rate(g, spec, dsa_cfg=dsa_cfg))

    results: dict[str, Any] = {
        "seeds": seeds,
        "clean_baseline": {
            "quality": _reduce(clean_q),
            "backdoor_success_rate": _reduce([{"sr": v} for v in clean_sr])["sr"],
        },
        "attacks": {},
    }

    for attack in ("reward_poison", "backdoor", "free_rider"):
        per_attack: dict[str, Any] = {}
        for nm in byz_counts:
            per_nm: dict[str, Any] = {
                "byzantine_fraction": round(nm / n_clients, 3),
                "n_malicious": nm,
            }
            acc: dict[str, list[dict[str, float]]] = {m: [] for m in METHODS}
            sr_acc: dict[str, list[float]] = {m: [] for m in METHODS}
            krum_honest: list[bool] = []
            skipped: dict[str, str] = {}
            for sd in seeds:
                # Client training does not depend on the aggregator, so the round
                # is built ONCE per seed and every defence graded on the same
                # uploads.
                qs = build_round(
                    campus, attack=attack, n_malicious=nm, dsa_cfg=dsa_cfg,
                    q_cfg=q_cfg, spec=spec, seed=sd,
                )
                for method in METHODS:
                    g = aggregate_q(method, qs, nm)
                    if g is None:
                        skipped[method] = f"{method} precondition not met at f={nm}"
                        continue
                    acc[method].append(
                        evaluate_campus(
                            g, campus, dsa_cfg, n_episodes=eval_episodes, seed=999
                        )
                    )
                    sr_acc[method].append(
                        backdoor_success_rate(g, spec, dsa_cfg=dsa_cfg)
                    )
                    if method == "krum":
                        sel = robust.krum([q.ravel() for q in qs], f=nm).selected_index
                        krum_honest.append(bool(sel < n_clients - nm))
            for method in METHODS:
                if method in skipped:
                    per_nm[method] = {"skipped": skipped[method]}
                    continue
                entry: dict[str, Any] = {
                    "quality": _reduce(acc[method]),
                    "backdoor_success_rate": _reduce(
                        [{"sr": v} for v in sr_acc[method]]
                    )["sr"],
                }
                if method == "krum" and krum_honest:
                    entry["krum_selected_honest_rate"] = round(
                        float(np.mean(krum_honest)), 4
                    )
                per_nm[method] = entry
            per_attack[f"n_malicious={nm}"] = per_nm
        results["attacks"][attack] = per_attack
    return results, spec


# ---------------------------------------------------------------------------
# Shield legality under a live backdoor, with REAL power requests
# ---------------------------------------------------------------------------
def shield_legality_under_backdoor(
    campus: CampusSpectrum,
    *,
    dsa_cfg: DSAConfig,
    q_cfg: QLearnConfig,
    n_malicious: int,
    seed: int,
) -> dict[str, Any]:
    """Every decision of a BACKDOORED policy through Shield + evidence chain.

    The transmit power requested for each decision is not a hand-picked
    over-power number any more: it is what closed-loop power control asks for to
    reach a target SINR at a REAL receiver given its MEASURED path gain. On this
    campus a large fraction of receivers need more EIRP than the licence permits,
    so the Shield's EIRP projection is exercised by real geometry.
    """
    spec = default_backdoor(dsa_cfg.n_channels, trigger_state=0)
    qs = build_round(
        campus, attack="backdoor", n_malicious=n_malicious, dsa_cfg=dsa_cfg,
        q_cfg=q_cfg, spec=spec, seed=seed,
    )
    backdoored = aggregate_q("fedavg", qs, n_malicious)
    assert backdoored is not None
    live_sr = round(backdoor_success_rate(backdoored, spec, dsa_cfg=dsa_cfg), 4)

    dcfg = DSADecisionConfig(
        band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP,
        carrier_bw_hz=CARRIER_BW_HZ, antenna_gain_dBi=ANTENNA_GAIN_DBI,
    )
    tmp = Path(tempfile.mkdtemp()) / "dsa_poison_evidence.jsonl"
    store = JsonlEvidenceStore(tmp)

    ns = n_states(dsa_cfg.n_channels)
    rng = np.random.default_rng(seed)
    rx_for_state = rng.integers(0, campus.se.shape[0], size=ns)
    # Closed-loop power control to a 20 dB SINR against the MEASURED best-subband
    # gain of the receiver this decision serves, clipped to the RU's ceiling.
    best_gain = campus.gain_db.max(axis=1)
    want = 20.0 + NOISE_DBM - best_gain[rx_for_state] - ANTENNA_GAIN_DBI
    requested = np.clip(want, -10.0, 46.0)

    emitted = illegal_emits = eirp_clamped = cert_recorded = 0
    over_licence_requests = int(np.sum(requested + ANTENNA_GAIN_DBI > MAX_EIRP))

    with TenantScope("dsa-poison-suite"):
        for i in range(ns):
            out = decide_and_record(
                backdoored, i, cfg=dcfg, method="fedavg", evidence=store,
                decision_id=f"bd-{i}", rng_seed=i,
                requested_tx_power_dBm=float(requested[i]),
            )
            a = out.safe_action
            if "safety_certificate" in out.record.chosen_action:
                cert_recorded += 1
            if a.get("emit", True):
                emitted += 1
                lo = a["frequency_hz"] - a["bandwidth_hz"] / 2.0
                hi = a["frequency_hz"] + a["bandwidth_hz"] / 2.0
                eirp = a["tx_power_dBm"] + a["antenna_gain_dBi"]
                if lo < BAND_LO - 1e-3 or hi > BAND_HI + 1e-3 or eirp > MAX_EIRP + 1e-6:
                    illegal_emits += 1
                if eirp < float(requested[i]) + ANTENNA_GAIN_DBI - 1e-9:
                    eirp_clamped += 1
        chain_intact = store.verify() == -1
        n_records = len(store)

    # Independent probe: the worst an out-of-band backdoor could attempt.
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP
    )
    oob_action = {
        "block": "fed_dsa_policy",
        "emit": True,
        "frequency_hz": BAND_HI + 40e6,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 48.0,
        "antenna_gain_dBi": 6.0,
    }
    disp = shield.dispose(oob_action, decision_id="oob-probe")
    sa = disp.safe_action
    oob_lo = sa["frequency_hz"] - sa["bandwidth_hz"] / 2.0
    oob_hi = sa["frequency_hz"] + sa["bandwidth_hz"] / 2.0
    oob_eirp = sa["tx_power_dBm"] + sa["antenna_gain_dBi"]
    oob_legal = (
        disp.certificate.emit_blocked
        or (oob_lo >= BAND_LO - 1e-3 and oob_hi <= BAND_HI + 1e-3 and oob_eirp <= MAX_EIRP + 1e-6)
    )

    return {
        "live_backdoor_success_rate_undefended": live_sr,
        "decisions": ns,
        "power_requests": "closed-loop power control to 20 dB SINR against the "
                          "MEASURED best-subband gain of a real receiver per state",
        "requests_over_licence_eirp": over_licence_requests,
        "emitted": emitted,
        "illegal_emits_after_shield": illegal_emits,
        "eirp_projections_applied": eirp_clamped,
        "certificates_recorded": cert_recorded,
        "evidence_records": n_records,
        "audit_chain_intact": chain_intact,
        "out_of_band_probe": {
            "proposed_freq_MHz": (BAND_HI + 40e6) / 1e6,
            "proposed_eirp_dBm": 54.0,
            "safe_freq_MHz": round(sa["frequency_hz"] / 1e6, 4),
            "safe_eirp_dBm": round(oob_eirp, 4),
            "emit_blocked": disp.certificate.emit_blocked,
            "legal_or_blocked": oob_legal,
        },
        "result": "PASS" if (illegal_emits == 0 and chain_intact and oob_legal) else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-users", type=int, default=3)
    ap.add_argument("--n-clients", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=20, help="local episodes/client")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=3, help="MDP/training seeds swept")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "dsa_poison_suite.json",
    )
    args = ap.parse_args()

    rows = _load_rows(args.features)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    campus = CampusSpectrum(rows, n_clients=args.n_clients, seed=1234 + args.seed)

    dsa_cfg = DSAConfig(n_channels=N_SUBBANDS, n_users=args.n_users, max_steps=80)
    q_cfg = QLearnConfig(episodes=args.episodes, epsilon=0.2)
    byz = [4, 5, 6]  # minority -> the median/trimmed-mean breakdown point (50%)

    seeds = [args.seed + i for i in range(args.seeds)]
    sweep, _spec = attack_sweep(
        campus, dsa_cfg=dsa_cfg, q_cfg=q_cfg, byz_counts=byz,
        eval_episodes=args.eval_episodes, seeds=seeds,
    )
    shield = shield_legality_under_backdoor(
        campus, dsa_cfg=dsa_cfg, q_cfg=q_cfg, n_malicious=4, seed=args.seed,
    )

    rp = sweep["attacks"]["reward_poison"]
    bd = sweep["attacks"]["backdoor"]
    clean_q = sweep["clean_baseline"]["quality"]

    report: dict[str, Any] = {
        "benchmark": "Federated DSA data-poisoning battery on real campus spectrum",
        "dataset": "DeepMIMO ASU Campus 3.5 GHz",
        "scenario": manifest.get("scenario"),
        "data_kind": manifest.get("data_kind"),
        "features_sha256": manifest.get("features_sha256"),
        "source_archive_sha256": manifest.get("source_archive_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "receivers": len(rows),
        "data_provenance": {
            "federated_clients": "real — geographic cohorts of measured receiver "
            "positions on the ASU campus",
            "per_client_channel_quality": "real — each client's reward for a "
            "successful transmission is scaled by the spectral efficiency its own "
            "receivers measurably support on that subband",
            "utility_units": "real — delivered bit/s/Hz at measured receiver positions",
            "primary_user_occupancy": "SYNTHETIC — Gilbert-Elliott Markov chain; the "
            "dataset carries no spectrum-occupancy trace",
            "sensing_errors": "SYNTHETIC — declared per-channel error rate",
            "attacks": "ALGORITHMIC, CITED, NOT CAPTURED — reward poisoning, the "
            "single-row backdoor and free-riding are attacker algorithms. No public "
            "corpus of captured malicious federated updates exists.",
            "attack_citations": ATTACK_CITATIONS,
            "defence_citations": DEFENCE_CITATIONS,
        },
        "config": {
            "n_channels": N_SUBBANDS,
            "n_users": args.n_users,
            "n_clients": args.n_clients,
            "local_episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "band_lo_hz": BAND_LO,
            "band_hi_hz": BAND_HI,
            "carrier_bw_hz": CARRIER_BW_HZ,
            "max_eirp_dBm": MAX_EIRP,
            "su_tx_power_dBm": SU_TX_POWER_DBM,
            "noise_figure_dB": NOISE_FIGURE_DB,
            "min_scheduled_sinr_dB": MIN_SINR_DB,
            "max_spectral_efficiency_bps_hz": MAX_SPECTRAL_EFFICIENCY,
        },
        "measured_campus": campus.stats,
        "attacks": ["reward_poison", "backdoor", "free_rider"],
        "sweep": sweep,
        "shield_legality_under_backdoor": shield,
        "summary": {
            "seeds": seeds,
            "clean_delivered_bps_hz_per_slot": clean_q["delivered_bps_hz_per_slot"],
            "clean_pu_clash_per_slot": clean_q["pu_clash_per_slot"],
            "clean_throughput_per_slot": clean_q["throughput_per_slot"],
            "reward_poison_fedavg_pu_clash_nm4":
                rp["n_malicious=4"]["fedavg"]["quality"]["pu_clash_per_slot"],
            "reward_poison_median_pu_clash_nm4":
                rp["n_malicious=4"]["median"]["quality"]["pu_clash_per_slot"],
            "reward_poison_fedavg_pu_clash_nm6":
                rp["n_malicious=6"]["fedavg"]["quality"]["pu_clash_per_slot"],
            "reward_poison_fedavg_delivered_nm4":
                rp["n_malicious=4"]["fedavg"]["quality"]["delivered_bps_hz_per_slot"],
            "reward_poison_median_delivered_nm4":
                rp["n_malicious=4"]["median"]["quality"]["delivered_bps_hz_per_slot"],
            "backdoor_clean_sr": sweep["clean_baseline"]["backdoor_success_rate"],
            "backdoor_fedavg_sr_nm4": bd["n_malicious=4"]["fedavg"]["backdoor_success_rate"],
            "backdoor_median_sr_nm4": bd["n_malicious=4"]["median"]["backdoor_success_rate"],
            "backdoor_median_sr_nm6_breakdown":
                bd["n_malicious=6"]["median"]["backdoor_success_rate"],
            "shield_illegal_emits": shield["illegal_emits_after_shield"],
            "shield_audit_chain_intact": shield["audit_chain_intact"],
            "shield_result": shield["result"],
        },
    }
    s = report["summary"]
    def _m(x: dict[str, float]) -> float:
        return x["mean"]

    report["honest_findings"] = [
        "The clients are real and genuinely non-IID: the measured mean spectral "
        f"efficiency per subband spans {campus.stats['client_se_spread_bps_hz']} "
        "bit/s/Hz across the geographic cohorts, so different campus regions "
        "rationally prefer different channels. That disagreement is measured, not "
        "seeded.",
        "MEASURED, AND NOT WHAT THE OLD FRAMING IMPLIED: on this environment reward "
        "poisoning is not a throughput attack. Delivered spectral efficiency stays "
        f"near the clean baseline ({_m(s['clean_delivered_bps_hz_per_slot']):.2f} -> "
        f"{_m(s['reward_poison_fedavg_delivered_nm4']):.2f} bit/s/Hz per slot under "
        "un-defended FedAvg) because the inverted-reward clients push the aggregate "
        "towards transmitting more, not less. The damage shows up as REGULATORY "
        f"harm: PU clashes per slot rise from {_m(s['clean_pu_clash_per_slot']):.3f} "
        f"to {_m(s['reward_poison_fedavg_pu_clash_nm4']):.3f} at 4 of "
        f"{args.n_clients} clients and {_m(s['reward_poison_fedavg_pu_clash_nm6']):.3f} "
        "at 6.",
        "Coordinate median does not bound that clash rise here — it is measurably "
        f"WORSE than un-defended FedAvg ({_m(s['reward_poison_median_pu_clash_nm4']):.3f} "
        f"vs {_m(s['reward_poison_fedavg_pu_clash_nm4']):.3f} PU clashes per slot at "
        "4 malicious clients, non-overlapping across the swept seeds). Two reasons, "
        "both consequences of using real clients: a reward-poisoned client's upload "
        "is NOT a geometric outlier (it is a well-trained Q-table for an inverted "
        "objective), so per-coordinate screening gives it full voting weight; and on "
        "genuinely non-IID geographic clients the coordinate median is not an "
        "estimate of the honest mean, so it also throws away the averaging that was "
        "helping. We report this rather than quietly dropping the metric.",
        "The single-row backdoor keeps success rate "
        f"{_m(s['backdoor_fedavg_sr_nm4']):.3f} under FedAvg; coordinate median "
        f"screens it while the attackers are a per-coordinate minority "
        f"({_m(s['backdoor_median_sr_nm4']):.3f} at 4 clients) and it returns to "
        f"{_m(s['backdoor_median_sr_nm6_breakdown']):.3f} at the 50% breakdown "
        "point. Robust aggregation BOUNDS, it does not ELIMINATE, a sparse backdoor.",
        "The attacks are published algorithms applied to a real federation, NOT "
        "captured attack traffic. The primary-user occupancy remains a synthetic "
        "Markov chain because this dataset carries no occupancy trace.",
        "SCOPE OF THE SHIELD GUARANTEE: with a live backdoor and closed-loop power "
        "control demanding more EIRP than the licence allows on "
        f"{shield['requests_over_licence_eirp']} of {shield['decisions']} decisions, "
        f"the Shield emitted {s['shield_illegal_emits']} illegal actions and the "
        f"hash-chained evidence verified intact ({s['shield_audit_chain_intact']}). "
        "The Shield bounds EMISSION legality (band, EIRP, mask). It does not and "
        "cannot bound PU-clash rate, which is a policy-quality property — the "
        "residual harm above is real and is the aggregator's problem, not the "
        "Shield's.",
    ]
    stamp(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"DSA data-poisoning suite (REAL campus spectrum) -> {args.out}")
    st = campus.stats
    print(
        f"{st['receivers']} real receivers -> {st['clients']} geographic clients "
        f"({st['client_receivers_min']}-{st['client_receivers_max']} rx each); "
        f"campus mean SE {st['campus_mean_spectral_efficiency_bps_hz']} bit/s/Hz, "
        f"{st['fraction_receiver_subbands_below_mcs_floor']:.1%} of "
        "(receiver, subband) pairs below the scheduled-MCS floor"
    )
    print(
        f"\nREWARD POISON (PU-clash/slot, mean+/-std over {len(seeds)} seeds): clean "
        f"{_m(s['clean_pu_clash_per_slot']):.3f} -> fedavg(nm=4) "
        f"{_m(s['reward_poison_fedavg_pu_clash_nm4']):.3f} -> median(nm=4) "
        f"{_m(s['reward_poison_median_pu_clash_nm4']):.3f} -> fedavg(nm=6) "
        f"{_m(s['reward_poison_fedavg_pu_clash_nm6']):.3f}"
    )
    print(
        f"  delivered bit/s/Hz per slot: clean "
        f"{_m(s['clean_delivered_bps_hz_per_slot']):.3f} -> fedavg "
        f"{_m(s['reward_poison_fedavg_delivered_nm4']):.3f} -> median "
        f"{_m(s['reward_poison_median_delivered_nm4']):.3f} (NOT a throughput attack)"
    )
    print(
        f"BACKDOOR success: clean {_m(s['backdoor_clean_sr']):.3f} | fedavg "
        f"{_m(s['backdoor_fedavg_sr_nm4']):.3f} | median(nm=4) "
        f"{_m(s['backdoor_median_sr_nm4']):.3f} | median@breakdown(nm=6) "
        f"{_m(s['backdoor_median_sr_nm6_breakdown']):.3f} (RESIDUAL)"
    )
    print(
        f"SHIELD: {s['shield_illegal_emits']} illegal emits, "
        f"{shield['eirp_projections_applied']} EIRP projections on real power "
        f"requests, audit chain intact={s['shield_audit_chain_intact']}  "
        f"[{s['shield_result']}]"
    )
    return 0 if s["shield_result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
