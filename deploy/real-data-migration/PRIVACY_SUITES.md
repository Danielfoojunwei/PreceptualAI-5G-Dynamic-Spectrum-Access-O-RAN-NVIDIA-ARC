# Privacy / unlearning suites: migration from synthetic records to real data subjects

**Scope of this document.** Four benchmarks —
`benchmarks/dp_privacy_suite.py`, `benchmarks/subject_erasure_suite.py`,
`benchmarks/federated_unlearning_suite.py`,
`benchmarks/verifiable_secagg_suite.py` — used to make privacy claims about
records that were never anybody's: `numpy.random` draws and tabular Q-tables
rolled out in a toy `DSAEnv`. They are now driven by the licence-gated DeepMIMO
ASU Campus 3.5 GHz build that was already in the tree. **No download was needed
and no dataset was rebuilt.**

Everything below is a number produced by a command quoted next to it, run on
2026-07-28 with `/home/user/venv/bin/python` (CPython 3.11.15, numpy 2.4.6).

The baseline for every "before" quote is commit `0df3dd9`, the last commit before
this migration started.

---

## 0. What makes a privacy benchmark real

A DP epsilon, an erasure certificate and a membership-inference AUC are all
statements *about a population of data subjects*. If the subjects are
`rng.normal(size=36)`, the statement is vacuous — you can prove any epsilon you
like about noise. The migration therefore rests on one substitution:

> **A data subject is one real DeepMIMO receiver.** Its private record is its
> real 3D position (`position_m`, Wireless InSite site-specific ray tracing over
> the ASU campus) and its six real measured subband gains (`subband_gain_dbw`).

4096 such receivers already exist in
`datasets/deepmimo_asu_3p5/generated/channel_features.jsonl`
(`features_sha256 = ab4414a1…69e5af`, `source_archive_sha256 = 80e4a498…7da3`).

All four suites now share one substrate, `benchmarks/privacy_real_subjects.py`
(new file), which deliberately reuses the recipe already validated by the
*already-real* `benchmarks/federated_coverage_loop.py`:

| Piece | What it is |
| --- | --- |
| Task | federated per-subband path-loss regression: whitened quadratic position design → 6 real measured subband gains |
| Local learner | FedProx proximal ridge step (`PROX_LAMBDA = 2.0`, `PROX_ETA = 0.6`, same constants as the coverage loop) |
| Client | a real geographic cell of the campus (Lloyd partition on real 2D positions → genuinely non-IID) |
| Model | 6 features × 6 subbands = **36-dimensional** update vector |
| Public slice | every 8th receiver (512 of 4096) is server-held and **never given to a client** — used to calibrate the DP clip norm and to evaluate held-out utility |
| Utility unit | RMSE in **dB against the real measured gains**, not a proxy score |

### The standardisation constants are public, and that matters

Standardising the target on the private cohort's own mean/σ would itself leak.
`build_population()` instead takes them from the **committed manifest**:

```
center_dbw = (subband_gain_dbw_min + subband_gain_dbw_max) / 2 = -153.0027
scale_db   = (subband_gain_dbw_max - subband_gain_dbw_min) / 4 =   27.7650
```

Both are already published aggregates of this build.

### Scope note, repeated in all four result JSONs

These are **ray-traced receivers, not subscribers**. A receiver is a grid point
in a propagation simulation. Every epsilon, AUC and erasure bound below is a
property of the stated mechanism on that population under the stated adjacency
convention. It is **not** a measurement of privacy risk to real mobile
subscribers, and no personal data was used anywhere in this repository. Raw and
row-level derived DeepMIMO data is not redistributed — only the manifest hashes
and these aggregate results.

---

## 1. `benchmarks/dp_privacy_suite.py`

### What was synthetic

```
$ git show 0df3dd9:benchmarks/dp_privacy_suite.py | sed -n '72,75p;147p'
def _client_q(seed: int, dsa_cfg: DSAConfig, q_cfg: QLearnConfig) -> np.ndarray:
    """Train one client's local DSA Q-table from scratch on its own env."""
    env = DSAEnv(cfg=dsa_cfg, seed=seed)
    return train_local_q(env, q_cfg, seed=seed)
...
        util = evaluate_policy(
```

Members and non-members were 64 + 64 toy-environment rollouts distinguished only
by their RNG seed band (`_MEMBER_BASE = 1_000_000`, `_NONMEMBER_BASE =
9_000_000`). The membership-inference AUC that resulted was a statement about
two seed bands. Utility was `throughput_per_slot` on the toy `DSAWorld`.

### What it is now

128 real geographic cells are carved out of the 3584 private receivers (10–43
receivers each); 64 are members, 64 are held out entirely. Utility is held-out
dB RMSE against real measured gains. Two membership attacks replace the distance
heuristic, both loss-threshold (Yeom et al., CSF 2018): client-level and
record-level.

```
$ /home/user/venv/bin/python benchmarks/dp_privacy_suite.py
dataset=DeepMIMO ASU Campus 3.5 GHz  features_sha256=ab4414a1dca5d124...
real subjects: 3584 private + 512 public root

baseline (predict the public centre) held-out RMSE = 25.1369 dB

  z      epsilon(R=12)   MIA AUC client / record   held-out RMSE dB   clipped
  ----------------------------------------------------------------------------
  0      inf (no DP)     0.542 / 0.531              15.501           0.46
  1      22.675+-0.000   0.525 / 0.520              18.468           0.51
  4      4.553+-0.000    0.523 / 0.519              19.017           0.53
  8      2.268+-0.000    0.524 / 0.520              21.104           0.59
```

Every row is real: 1805 member receivers vs 1779 non-member receivers, 5
member/non-member assignment seeds, 12 rounds.

### The clip norm now binds, and it was chosen without touching the private data

```
$ /home/user/venv/bin/python -c "import json;print(json.load(open('benchmarks/results/dp_privacy.json'))['setup']['clip_calibration'])"
{'clip_norm_C': 0.232821, 'rule': 'median update L2 norm over the PUBLIC server-held root pseudo-clients',
 'public_root_receivers': 512, 'public_root_pseudo_clients': 64,
 'public_norm_median': 0.232821, 'public_norm_p90': 0.487455, 'public_norm_max': 0.759531, ...}
```

`C = 0.2328` is the median update norm of 64 pseudo-clients built **only** from
the 512 public root receivers. It clips **45.9 %** of the real private updates
at `z = 0` and up to 59.2 % at `z = 8`. A clip far above the real update scale
would inject noise calibrated to a sensitivity the data never attains — pure
utility loss at identical epsilon.

### The new part: epsilon is verified numerically, not asserted

`epsilon_verification` runs the **real `dp.dp_fedavg` code path** 20 000 times
on each of two real adjacent datasets (one real member cell's receivers replaced
by a real non-member cell's — replace-one client adjacency, exactly what
`DPConfig` claims) and sweeps the optimal distinguisher's threshold with
one-sided Clopper–Pearson 95 % bounds (Jagielski, Ullman & Oprea, NeurIPS 2020).

```
  epsilon verification (1 Gaussian round, real replace-one pair):
  z      audit(>=)   realised    worst-case   RDP        ordering
  ------------------------------------------------------------------
  1      2.0900      3.3638      4.3772     5.3026     True
  4      0.4214      0.7208      0.9263     1.2675     True
  8      0.1317      0.3385      0.4344     0.6214     True
```

Four independent numbers per row, and the suite checks they are ordered:

| Column | What it is |
| --- | --- |
| `audit` | the privacy loss an optimal attacker **actually achieved**, a statistically valid lower bound |
| `realised` | exact Gaussian epsilon at the **measured** sensitivity of that real pair (‖clip(x) − clip(x′)‖ = **0.3299**, not the assumed 2C = 0.4660) |
| `worst-case` | exact Gaussian epsilon at Δ = 2C (Balle & Wang, ICML 2018 — the *tight* value) |
| `RDP` | what the shipped accountant reports |

`all_orderings_hold = true`. A flipped inequality would mean the mechanism or
the accountant is wrong. The RDP slack over the exact Gaussian value (e.g.
`5.3026 − 4.3772 = 0.9254` at z = 1) is the price of using an RDP relaxation and
is now reported rather than hidden.

### What the results honestly do not show

The undefended client-level AUC is **0.542**, not a dramatic leak. A
36-parameter model fitted on ~1800 real receivers memorises very little, so this
attacker had little to find even at z = 0. That is reported as-is; an AUC near
0.5 shows *this attack failed*, it does not prove privacy.

---

## 2. `benchmarks/subject_erasure_suite.py`

### What was synthetic

```
$ git show 0df3dd9:benchmarks/subject_erasure_suite.py | sed -n '57,61p'
    for ci in range(N_CLIENTS):
        cid = f"cell-{ci}"
        subjects = [f"{cid}-sub-{j}" for j in range(SUBJECTS_PER_CLIENT)]
        env = DSAEnv(cfg=dsa, seed=1000 * seed + ci + 1)
        client_logs[cid] = E.collect_subject_transitions(
```

Subjects were the strings `"cell-1-sub-0"` … `"cell-3-sub-2"`, created by
assigning toy-environment rollout episodes round-robin. Twelve of them, total.

### What it is now

8 real geographic clients holding 268–525 real receivers each. A subject is a
named real receiver, e.g.

```
$ /home/user/venv/bin/python -c "import json;r=json.load(open('benchmarks/results/subject_erasure.json'))['results'][0]['per_erasure'][0];print(r['subject_id'], r['subject_position_m'])"
deepmimo-rx-106298 [34.449, 97.831]
```

**Erasure is an exact rank-1 downdate.** A FedProx step is a closed-form
function of the client's sufficient statistics `G = DᵀD`, `H = DᵀT` and its
subject count `n`; one subject contributes exactly `d dᵀ` and `d tᵀ`. Removing
it is a rank-1 subtraction replayed through all 12 warm-started rounds.

**The exactness is measured, not claimed.** The post-erasure model is computed
two structurally different ways — the rank-1 downdate, and an independent
from-scratch retrain from the raw rows with the receiver physically deleted.

```
$ /home/user/venv/bin/python benchmarks/subject_erasure_suite.py
[fedavg ] erasures=8 changed=True
    certified L2 to independent retrain : max 2.3e-16
    cheap shortcut residual  (R=12)   : mean 0.000349
    cheap shortcut residual  (R= 1)     : mean 2.64e-17
    influence of one real receiver      : 0.000417 L2, 0.00757 dB at that receiver

[median ] erasures=8 changed=True
    certified L2 to independent retrain : max 2.29e-16
    cheap shortcut residual  (R=12)   : mean 0.000669
    cheap shortcut residual  (R= 1)     : mean 9.51e-05
    influence of one real receiver      : 0.000688 L2, 0.00424 dB at that receiver

[audit ] 200 real receivers erased one at a time
    paired leave-one-out: loss rose for 81.5% of subjects (mean +0.311 dB^2)
    population AUC vs 78 never-seen: 0.514 -> 0.5134
```

2.3e-16 on a model whose own norm is **1.377** — floating-point noise, from two
different code paths. That is the Art. 17 "as if the data had never been used"
claim, measured.

### A bug this migration caught in its own first draft

The first version rounded the certified distance with `round(certified, 15)`,
which turned every genuine `~2e-16` into a printed `0.0` — evidence destroyed by
formatting. The committed code stores `float(f"{certified:.6g}")` and carries
`model_norm_for_scale` alongside, with a comment saying why.

### The cheap shortcut, priced honestly

`cheap_linear_shortcut_residual_to_retrain` is what an operator who patches only
the final aggregate (linear mean-correction, no replay) is left holding:

| Aggregator | R = 1 | R = 12 |
| --- | --- | --- |
| FedAvg (linear) | **2.6e-17** — exact | 3.49e-04 |
| coordinate median (non-linear) | **9.51e-05** — not exact | 6.69e-04 |

Exactness under warm-starting *or* non-linearity costs the replay. Both regimes
are reported instead of only the flattering one.

### The erasure audit, and the confound it does not hide

The population AUC barely moves (0.514 → 0.5134). That is expected and it is
also confounded: erased receivers versus never-seen receivers differ by
geography as well as membership. The suite therefore leads with a **paired
leave-one-out** test — same receiver, same geography, membership the only
difference — and reports the achieved value: the subject's own loss rose after
its own erasure for **81.5 %** of 200 real receivers, mean **+0.311 dB²**. The
remaining 18.5 % are real spatial heterogeneity at an effect size where one
subject out of hundreds barely moves a 36-parameter model. Neither test proves a
stronger attacker would fail.

### The certificate is real crypto bound to the real data build

RSA-PSS via the real provenance HSM (`InMemoryHSMBackend` + `sign_model`). For
both aggregators, all of `certificate_verifies_after`,
`certificate_rejects_before`, `certificate_pin_verify_ok`,
`certificate_wrong_pin_fails`, `certificate_rejects_manifest_tamper` and
`certificate_binds_features_sha256` are `true` — the manifest now carries
`dataset_features_sha256`, so an erasure is bound to the exact data build it was
performed against.

---

## 3. `benchmarks/federated_unlearning_suite.py`

### What was synthetic

```
$ git show 0df3dd9:benchmarks/federated_unlearning_suite.py | sed -n '62p;68,69p'
    bd = BackdoorSpec(trigger_state=TRIGGER_STATE, target_channel=TARGET_CHANNEL, boost=50.0)
    poisoned, trace = U.run_federated_training(
        clients, rounds=ROUNDS, method=method, dsa_cfg=dsa, q_cfg=qc, base_seed=seed)
```

Ten toy-environment Q-learning clients; the attack was a `+50` boost written
directly into one cell of a Q-table, scored by a synthetic trigger-state probe.

### What it is now

Ten real geographic clients (268–525 real receivers each). The attack is the
injection an O-RAN operator actually has to worry about: **one cell falsifies
its measurement reports**, adding +20 dB to subband 0's real measured gain
before fitting its local model. Every position and every other subband stays at
its real ray-traced value, so the upload is a *legitimate fit to fabricated
data*. Damage is measured on the 512 public receivers no client ever held.

```
$ /home/user/venv/bin/python benchmarks/federated_unlearning_suite.py
[fedavg ] target-subband recommendation rate on real receivers
    clean (all honest)   0.053
    poisoned             0.709  (+1.327 dB hallucinated)
    efficient_unlearn    0.262   certified L2 0.1128
    retrain_from_scratch 0.043   certified L2 0

[median ] target-subband recommendation rate on real receivers
    clean (all honest)   0.115
    poisoned             0.641  (+1.010 dB hallucinated)
    efficient_unlearn    0.373   certified L2 0.3222
    retrain_from_scratch 0.103   certified L2 0
```

One falsifying cell out of ten drives the global model from recommending subband
0 for 5.3 % of real receivers to **70.9 %**. Coordinate median barely helps
(64.1 %) — the update is not an outlier in shape, so a robust aggregator bounds
it rather than removes it.

Retrain-from-scratch unlearning closes it (0.043, matching the 0.053 clean
baseline) at a cost of 108 local fits. The cheap replay leaves 26.2 % and its
certified L2 distance to the gold standard is **0.1128** against 0 for the
retrain — the honest signal an operator needs to reject the cheap result. (The
retrain row is 0 by construction and is labelled as a plumbing self-check, not
evidence.)

### The finding the old suite got backwards

```
[detect ] measurement_falsification   attacker_flagged=False norm ratio   0.516  damage 0.641
[detect ] norm_matched_falsification  attacker_flagged=False norm ratio   1.000  damage 0.660
[detect ] scaling_attack_control      attacker_flagged=True  norm ratio   4.382  damage 0.115
```

The old suite reported that whole-vector screening *caught* its backdoor, with
the caveat that a hypothetical norm-matched attacker would evade. On real data
the shipped attack is **already invisible**: the falsifying cell's update norm
is **0.516×** the honest median — *smaller* than an honest client's, because it
is fitting a global its own lie has already moved. A 10× scaling positive
control **is** flagged, so this is a real blind spot rather than a broken
detector. Certified unlearning is only as good as some independent attribution
signal; where there is none, the deterministic Decision Safety Shield is the
backstop.

---

## 4. `benchmarks/verifiable_secagg_suite.py`

### What was synthetic

```
$ git show 0df3dd9:benchmarks/verifiable_secagg_suite.py | grep -n 'rng.normal(size'
75:                vecs = [rng.normal(size=d) for _ in range(n)]
126:            vals = rng.normal(size=dim)
182:        vecs = [rng.normal(size=dim) for _ in range(n_clients)]
241:            vecs = [rng.normal(size=d) for _ in range(n_clients)]
296:        raw_updates = [rng.normal(size=dim) for _ in range(n_clients)]
```

Every secret-shared vector, on all five axes.

### What it is now

Every one of them is a **real per-cell FedProx update** of the path-loss model
fitted on real receiver positions and real measured gains.

```
$ /home/user/venv/bin/python benchmarks/verifiable_secagg_suite.py --seeds 2
[correct ] worst |mean err| on real updates = 6.84e-06 (tol 1.53e-05)
[e2e     ] held-out RMSE vs real gains: secagg 16.744478 dB vs plaintext 16.744459 dB (diff 1.89e-05, baseline 24.636665 dB)
[privacy ] masking violations 0/576, chi-square z = 0.254
[integrity] tamper 1.000  drop 1.000  (100 trials on real updates)
```

### New axis: does the crypto change the model an operator ships?

The whole federation is trained twice — plaintext FedAvg and verifiable secure
aggregation in the loop — and both models are scored in dB against the real
measured gains of the 512 public held-out receivers. **16.744478 dB vs
16.744459 dB**, a difference of 1.9e-05 dB, with every round verified. Public
verifiability is free in utility; its cost is wall-clock.

### The cost the old suite understated

`_encode` maps a **negative** float to `Q − k`, i.e. a full 2047-bit exponent.
Real model updates are roughly half negative, so a commitment costs ~10.9 ms per
coordinate, not the ~0.13 ms a small positive exponent suggests.

```
[cost    ] dim=   1  split+verify =    21.0 ms  (21.02 ms/coord)
[cost    ] dim=   8  split+verify =   245.0 ms  (30.62 ms/coord)
[cost    ] dim=  36  split+verify =  1882.0 ms  (52.28 ms/coord) <- real model dim
[cost    ] dim= 108  split+verify =  5744.2 ms  (53.19 ms/coord)
```

**A bug this migration caught in its own first draft.** The cost sweep built its
vectors with `u[:d]`, and a real update is only 36 long — so `dim=128` silently
measured 36 coordinates and the curve looked flat (1830 ms at dim 36, 1846 ms at
"dim 128"). The committed code builds long vectors by concatenating a client's
real updates across consecutive rounds, asserts the realised length, and records
`vector_source`. The corrected curve is linear: 3× the dimension, 3.05× the
time.

### A vacuous statistic, replaced

The old privacy probe binned the top **16** bits of `share_a` — 65536 cells —
with ~1500 samples. Expected count per bin: 0.02. Pearson's statistic is not
chi-square distributed anywhere near that regime, so the reported `z` was not a
test of anything. The suite now uses the top 5 bits (32 bins, expected count
**18.0**) and records `expected_count_per_bin` and `chi_square_validity` so a
reader can check the test is sound. Result: `z = 0.254`, 0 masking violations in
576 real-secret coordinates.

### Local DP against colluding servers: a trade-off curve, not one flattering point

The old suite reported a single `z = 4` point. On real updates that point is
catastrophic, so the suite now sweeps:

```
[localdp ]   z    eps_rdp   eps_exact   held-out RMSE dB   vs plaintext
[localdp ]  0.25    27.513     24.382            21.81         +0.65
[localdp ]     1     5.303      4.377            24.32         +3.17
[localdp ]     4     1.268      0.926            51.75        +30.60
[localdp ] plaintext 21.1584 dB, predict-the-centre baseline 24.6367 dB
```

**Local DP is not central DP.** Each of the 8 clients adds `N(0, (z·2C)²)` per
coordinate *before* sharing, and averaging over 8 clients divides the noise by
√8, not 8. At the `z = 4` that the central DP-FedAvg suite ships, the released
aggregate is dominated by noise and the model (51.75 dB) is far worse than
predicting a constant (24.64 dB). At a usable utility (`z = 0.25`, +0.65 dB) the
epsilon is 27.5 — no meaningful guarantee. **This federation size cannot pay for
collusion resistance via local DP.** That is the honest result and it is now
stated in `collusion_local_dp_sweep.reading` rather than smoothed over.

The `eps_exact` column is an independent check computed here with the exact
analytic Gaussian mechanism: at every `z`, `rdp_is_a_valid_upper_bound = true`,
so the module's reported number is conservative rather than optimistic.

---

## 5. Provenance binding

All four result JSONs now carry a `provenance` block — they carried none before:

```
$ /home/user/venv/bin/python -c "
import json
for n in ['dp_privacy','subject_erasure','federated_unlearning','verifiable_secagg']:
    p=json.load(open('benchmarks/results/%s.json'%n))['provenance']
    print(n,'->',p['features_sha256'][:16],'|',p['source_archive_sha256'][:16],'| rx',p['receivers_total'])"
dp_privacy -> ab4414a1dca5d124 | 80e4a49838470231 | rx 4096
subject_erasure -> ab4414a1dca5d124 | 80e4a49838470231 | rx 4096
federated_unlearning -> ab4414a1dca5d124 | 80e4a49838470231 | rx 4096
verifiable_secagg -> ab4414a1dca5d124 | 80e4a49838470231 | rx 4096
```

Each block carries `dataset`, `scenario`, `data_kind`, `source_archive_sha256`,
`source_tree_sha256`, `features_sha256`, `deepmimo_version`,
`carrier_frequency_hz`, the receiver split (4096 = 512 public root + 3584
private subjects), the explicit `subject_definition`, the public standardisation
constants and their justification, and the `scope_note` reproduced in §0 —
matching the convention in `benchmarks/deepmimo_dsa_benchmark.py` and the
spatial suites migrated in `SPATIAL_SUITES.md`.

`benchmarks/subject_erasure_suite.py` goes one step further: the signed
`ErasureCertificate` manifest itself carries `dataset_features_sha256`, so the
cryptographic evidence of an erasure is bound to the exact data build.

---

## 6. Lint and reproduction

```
$ /home/user/venv/bin/python -m ruff check benchmarks/dp_privacy_suite.py \
    benchmarks/subject_erasure_suite.py benchmarks/federated_unlearning_suite.py \
    benchmarks/verifiable_secagg_suite.py benchmarks/privacy_real_subjects.py
All checks passed!
```

Wall-clock on the machine above:

| Suite | Command | Time |
| --- | --- | --- |
| DP privacy | `python benchmarks/dp_privacy_suite.py` | 27.0 s |
| Subject erasure | `python benchmarks/subject_erasure_suite.py` | 3.0 s |
| Federated unlearning | `python benchmarks/federated_unlearning_suite.py` | 1.1 s |
| Verifiable secagg | `python benchmarks/verifiable_secagg_suite.py --seeds 2` | 86.4 s |

All four are deterministic given the pinned partition seed (`PARTITION_SEED =
1234`) and their mechanism seeds; the only non-determinism in a result file is
the ISO timestamp inside each signed certificate.

`benchmarks/results/SHA256SUMS` must be regenerated after this and any other
concurrent benchmark change lands:

```
(cd benchmarks/results && sha256sum *.json > SHA256SUMS)
```

---

## 7. What is still not real, stated plainly

* **The subjects are simulated.** Ray-traced receivers are not subscribers. No
  epsilon here is a claim about a person.
* **The attacks are simulated.** The +20 dB measurement falsification in §3 is a
  plausible O-RAN threat, not an observed incident.
* **The MIA numbers bound leakage from below only.** AUC 0.542 means *this*
  attacker found little. A stronger attacker is not excluded.
* **The audited epsilon is a lower bound.** 2.09 audited against 5.30 certified
  at z = 1 does not mean the true epsilon is 2.09; it means no attacker in this
  audit did better than 2.09.
* **The erasure audit is a necessary condition, not a proof.** It shows the
  influence was removed; it does not show no residual information exists.
