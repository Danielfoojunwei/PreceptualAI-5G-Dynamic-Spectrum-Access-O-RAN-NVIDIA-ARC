# Injection / integrity / evasion suites: migration from synthetic to captured attack data

**Scope of this document.** Three benchmarks —
`benchmarks/integrity_attack_suite.py`, `benchmarks/control_plane_load_suite.py`,
`benchmarks/evasion_suite.py` — carried no external data provenance. Two of them
are now driven by **real captured attack traffic**; the third could not be, and
this document says exactly why.

Every number below came from a command quoted next to it, run on 2026-07-28 with
`/home/user/venv/bin/python` (CPython 3.11.15, numpy 2.4.6) on this host.

---

## 0. The dataset that made this possible

**5GAD-2022**, Idaho National Laboratory.

| Property | Value |
| --- | --- |
| Repository | <https://github.com/IdahoLabResearch/5GAD> |
| Pinned commit | `d6c3643dafb0683ecac2557e1a5ca29c3b5d7ecb` |
| **Licence** | **MIT**, `Copyright (c) 2022 Battelle Energy Alliance, LLC` |
| Dataset DOI | `10.11578/dc.20220811.1` |
| Paper DOI | `10.1109/GCWkshps56602.2022.10008647` |
| What it is | Over-the-wire packet captures taken **while ten real attacks were executed against a real free5GC 5G standalone core** (UERANSIM RAN) on a physical test bench |
| Full size | 35.46 GB across 273 Git-LFS objects |
| **This build fetches** | **12 captures, 24.1 MB** |

This is the best-licensed and most CI-tractable option the scouting turned up.
MIT is strictly more permissive than everything else considered — it permits
redistribution of derived rows outright, so unlike the DeepMIMO build there is no
licence question hanging over the derived features at all. (The build still does
not commit them; see §5.)

### Licence position, stated plainly

MIT permits use, modification and redistribution, including of derived works,
subject only to preserving the copyright notice. There is **no non-commercial
clause, no share-alike clause and no click-through**. The attribution obligation
is discharged in `datasets/5gad_inl/manifest.json` under
`licensing.attribution_required`, and is propagated into every results JSON this
migration touches. Verified verbatim:

```
$ GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/IdahoLabResearch/5GAD.git
$ head -3 5GAD/LICENSE
MIT License

Copyright (c) 2022 Battelle Energy Alliance, LLC
```

### CI reproducibility, stated plainly

**Fully non-interactive. No login, no token, no click-through, no registration.**
Git-LFS objects are served as plain HTTPS GETs from the media host at a pinned
commit, and the repository's own LFS pointer publishes the SHA-256, so the pin is
*upstream-published* rather than something we computed ourselves:

```
$ curl -sSL -o /tmp/fakeamfinsert.pcapng \
  "https://media.githubusercontent.com/media/IdahoLabResearch/5GAD/d6c3643dafb0683ecac2557e1a5ca29c3b5d7ecb/Attacks/FakeAMFInsert/Attacks_FakeAMFInsert.pcapng"
real  0m0.710s
$ ls -l /tmp/fakeamfinsert.pcapng
-rw-r--r-- 1 root root 991376 ...
$ sha256sum /tmp/fakeamfinsert.pcapng
41cca32c909ffaaa80f4b6e78825c94a5c09530f9f21cb78f35cda3a8e8ceaa8
```

and the LFS pointer in the pinned commit reads

```
oid sha256:41cca32c909ffaaa80f4b6e78825c94a5c09530f9f21cb78f35cda3a8e8ceaa8
size 991376
```

— an exact match. The largest object fetched (101 MB, evaluated and then
rejected, see §4) came down in 1.7 s. **No opt-in gate is needed and none was
added**; unlike the licence-gated DeepMIMO build, this one runs unattended.

The pin is enforced *before* any parsing, and hard-fails:

```
$ sed 's/"41cca32c9.../"deadbeef0.../' build.py > /tmp/build_badpin.py && python /tmp/build_badpin.py
RuntimeError: 5GAD object Attacks/FakeAMFInsert/Attacks_FakeAMFInsert.pcapng failed its
pinned checksum: expected sha256=deadbeef09ffaaa80f4b6e78825c94a5c09530f9f21cb78f35cda3a8e8ceaa8
size=991376, got sha256=41cca32c909ffaaa80f4b6e78825c94a5c09530f9f21cb78f35cda3a8e8ceaa8 size=991376
```

Determinism, verified by building twice and diffing the manifest hashes:

```
$ python datasets/5gad_inl/build.py   # twice
DETERMINISM: identical hashes across two builds
source_archive_sha256 f568a8a0a291f5814c2a6fd6dc9c568c8cda2d96f88706776efd7c301cf7428e
events_sha256         d167e8ecb711d445934ed234b05805a5ee816b765a0417e067e374680d67da57
features_sha256       5f75d84aac8a7a94cf900a4694ad6281484adf7e8ecb0ee6633c72d6bdfac8fe
```

Timestamps are integers and no floating point enters the derivation, so unlike
the DeepMIMO builds this `features_sha256` is exactly reproducible across hosts,
not merely tolerance-reproducible.

### No new dependencies

The pcapng / Ethernet / IPv4 / TCP / UDP / HTTP-1.1 / PFCP reader in `build.py`
is written against the stdlib `struct` module. No scapy, pyshark or tshark, and
therefore **no change to `pyproject.toml` or the lockfile is required** — which
matters, because this agent does not own those files.

### What the build actually extracted

```
$ python datasets/5gad_inl/build.py
FakeAMFInsert                packets=794     events=397    nf_writes=397   accepted_2xx=0
FakeAMFInsert_full           packets=18824   events=398    nf_writes=398   accepted_2xx=398
randomAMFInsert              packets=1198    events=799    nf_writes=399   accepted_2xx=0
FakeAMFDelete                packets=35029   events=809    nf_writes=399   accepted_2xx=800
CrashNRF                     packets=398     events=398    nf_writes=0     accepted_2xx=0
GetAllNFs                    packets=398     events=398    nf_writes=0     accepted_2xx=0
GetUserData                  packets=398     events=398    nf_writes=0     accepted_2xx=0
AMFLookingForUDM             packets=398     events=398    nf_writes=0     accepted_2xx=0
randomDataDump               packets=397     events=397    nf_writes=0     accepted_2xx=0
automatedDropWithTimer       packets=360     events=360    nf_writes=0     accepted_2xx=0
automatedRedirectWithTimer   packets=360     events=360    nf_writes=0     accepted_2xx=0
Normal-1UE                   packets=374     events=0      nf_writes=0     accepted_2xx=0
{"control_plane_events": 5112, "http_requests": 4392, "nf_profile_writes": 1593,
 "pfcp_messages": 720, "captures": 12,
 "attack_classes": ["benign","denial_of_service","network_reconfiguration","reconnaissance"]}
```

Real content recovered, verified by parsing rather than by reading the abstract:

| Capture | What is actually on the wire |
| --- | --- |
| `FakeAMFInsert` | 397/398 × `PUT /nnrf-nfm/v1/nf-instances/b01dface-bead-cafe-bade-cabledfabled`, each carrying a 2081-byte forged AMF NF profile (`"nfType":"AMF"`, `"nfStatus":"REGISTERED"`, `"ipv4Addresses":["127.0.0.18"]`) |
| `randomAMFInsert` | 399 `PUT` + 399 `DELETE` against 399 *distinct random* instance UUIDs |
| `FakeAMFDelete` | 399 `PUT` (→ `201 Created`) + 399 `DELETE` (→ `204 No Content`) |
| `CrashNRF` | 398 × `GET /nnrf-disc/v1/nf-instances?requester-nf-type=&target-nf-type=` — **both required parameters empty**, which is what takes the real NRF down |
| `GetUserData` | 398 × `GET /nudm-sdm/v1/imsi-208930000000003/am-data` — real subscriber-data exfiltration by IMSI |
| `randomDataDump` | 397 discovery requests with attacker-generated random `requester-nf-type` values |
| `automatedDropWithTimer`, `automatedRedirectWithTimer` | 720 **PFCP Session Modification Requests** on N4 (UDP 8805) — real forged session modification to drop / redirect user traffic |

---

## 1. `integrity_attack_suite.py` — the strongest result of this migration

**Before:** 8 probes, all self-contained. `total_attacks: 8`, no provenance
fields anywhere in the results JSON (`git show HEAD:benchmarks/results/integrity_attack_suite.json`
→ keys `['all_defended','attacks','bypassed','bypasses','detected_or_blocked','generated_at','suite','total_attacks']`).

**After:** 11 probes. Probes 1–8 are unchanged and remain constructed-adversary
probes — **correctly so**: a signature check or a fail-closed gate is a property
of the code, not of a dataset, and there is nothing external to measure. Probes
9–11 are new and driven by the captures.

### Probe 9 — the headline number

The FakeAMFInsert attack PUTs a forged AMF NF profile to the NRF. The full
capture records **what the real free5GC core did about it**. It accepted every
single one:

```
$ python benchmarks/integrity_attack_suite.py
9_captured_forged_nf_injection -> captured_forged_registrations=398
   real_free5gc_nrf_accepted_2xx=398/398
   horizon_pinned_reject=398/398
   attacker_signature_self_consistent=398/398
   replayed_bytes_match_recorded_digest=398/398
   vault_IntegrityError=8/8
```

Read that carefully, because the whole value of the probe is in the comparison:

* **398/398** — a real, production-grade 5G core admitted every forged NF
  registration, answering `HTTP/1.1 200 OK` with a `Location` header pointing at
  the attacker's forged instance. Nothing in free5GC's NF-management path pins a
  signer.
* **398/398** — the *identical bytes*, presented to Horizon's artefact-admission
  gate, were rejected against a pinned trusted public key.
* **398/398** — those attacker signatures were nonetheless internally
  self-consistent, i.e. the rejection came from the pin, not from a malformed
  signature. This is the distinction that matters.
* **398/398** — the replayed bytes hash to the digest recorded at build time, so
  the corpus and the probe have not drifted apart.

**What this is NOT.** It is not a claim that Horizon-RIC defends free5GC, or any
5G core network function. Horizon is an rApp; the NRF is not in its trust
boundary. It is a measured comparison on identical inputs: what a real core
accepted, versus what Horizon's own gate does with the same bytes. Stated that
way, it is defensible; stated any more broadly, it would not be.

### Probes 10 and 11

```
10_captured_malformed_input_failclosed -> captured_malformed_requests=1594
   armA_hostile_text_handled=1594/1594
   armB_empty_required_fields_contained=795/795
   unhandled_exceptions=0 illegal_emits=0

11_evidence_chain_at_captured_volume -> chain_length=5112 (= captured control-plane events)
   clean_verify=-1 tamper_index=796 first_broken_index=796
```

Probe 10 arm B is the one worth explaining: CrashNRF works by sending a request
whose *required* parameters are present-but-empty, which the real NRF then
dereferences and dies on. Arm B reproduces that **shape** against Horizon — the
action's required fields present but empty — for the 795 captured requests that
actually had that shape on the wire. All 795 were blocked or projected; none
emitted, none raised.

Probe 11 builds a 5112-record hash chain, one record per captured control-plane
event in captured order, tampers the record at the index of the first captured
NF-profile write (796 — the first point at which an attacker actually mutated
core state), and the chain break is detected at exactly 796.

**Honest scoping for 10 and 11.** 5GAD contains no Horizon artefacts, records or
radio actions. Probe 9 is the strongest of the three because it feeds captured
bytes through **unchanged**. Probes 10 and 11 take the real attacker-supplied
strings and the real event sequence and use them to drive Horizon's own objects;
the mapping is documented in each probe's docstring and in the results JSON.

### Failure mode is honest

If the corpus has not been built, probes 9–11 report `status:
"skipped_dataset_absent"`, **not** as passing, and `all_defended` goes to
`false`:

```
$ python -c "... _DATASET_DIR = Path('/tmp/nonexistent'); run_battery() ..."
total_probes 11 run 8 blocked 8 skipped ['9_captured_forged_nf_injection',
  '10_captured_malformed_input_failclosed', '11_evidence_chain_at_captured_volume']
all_defended False
```

Final state: **11/11 probes detected/blocked, 0 bypassed, 0 skipped.**

---

## 2. `control_plane_load_suite.py` — real arrival profile, derived radio parameters

**Before** (`git show HEAD:benchmarks/results/control_plane_load.json`):

```
benchmark:    single_process_shield_decision_path
traffic_mix:  "deterministic valid and projectable terrestrial policy proposals"
              (i.e. index % 2 and index % 3 — entirely invented)
throughput:   38932 decisions/s
p99 latency:  40.4 us
```

**After:**

```
$ python benchmarks/control_plane_load_suite.py
capture                      class                     events  peak/s  offered/s   shield/s  headroom
AMFLookingForUDM             reconnaissance               398       4       3.47      39329     9832x
CrashNRF                     denial_of_service            398       4       3.70      44760    11190x
FakeAMFDelete                network_reconfiguration      809       5       1.57      30519     6104x
FakeAMFInsert                network_reconfiguration      397       4       3.63      43790    10948x
FakeAMFInsert_full           network_reconfiguration      398       4       3.64      41674    10418x
GetAllNFs                    reconnaissance               398       4       3.45      42023    10506x
GetUserData                  reconnaissance               398       4       3.47      44647    11162x
automatedDropWithTimer       denial_of_service            360       2       0.20      30435    15217x
automatedRedirectWithTimer   denial_of_service            360       2       0.20      33002    16501x
randomAMFInsert              network_reconfiguration      799       1       0.87      29914    29914x
randomDataDump               reconnaissance               397       4       3.60      38994     9749x
  aggregate: 5112 real events, peak 35/s offered; Shield serviced 36193/s (p99 65.8 us)
  worst-case headroom over any capture: 6104x
  synthetic control (pre-migration mix): 30123/s
```

### What is real here, precisely

**Real (measured, from the capture):** which events occurred, in what order, at
what nanosecond timestamps; the SBI method and resource path of every request;
the PFCP message type of every N4 datagram; the offered arrival rate and
burstiness.

**Real (measured, this host):** the Shield's latency and throughput.

**Derived, not measured — and labelled as such in the JSON under
`data_provenance.derived_not_measured`:** the *radio parameters* of the action
each captured event maps to. 5GAD is a core-network capture and contains no RIC
policy actions, so there is nothing to measure. Each event maps to one proposal
by a deterministic function of that event's own bytes. The mapping preserves the
real read/write distinction and nothing more.

### The honest limits, which are real limits

1. **The replay runs at maximum speed, not in real time.** The longest capture
   spans 1791 s; sleeping out the real gaps would measure the clock, not the
   Shield. Real timing is reported separately as the measured arrival profile
   and compared against the measured service rate — that ratio is the `headroom`
   column.
2. **The headroom numbers are large because the real attacks are slow.** The
   fastest captured attack offered 5 events/s peak. This is a genuine finding
   about attack tooling against a 5G core, not a stress test: a real hping3-class
   flood would be orders of magnitude faster. The result honestly says *the
   Shield is not the bottleneck for this class of attack*, and nothing stronger.
3. **This benchmark measures load, not detection.** The Shield is an
   action-legality projector, not an intrusion detector. It does not detect the
   captured attacks and the JSON says so under `scope.does_not_prove`.
4. **There is no benign SBI baseline, and 5GAD does not have one.** See §4.

The synthetic control arm is retained verbatim under `synthetic_control` and
labelled `"data_provenance": "synthetic"` so the pre-migration number stays
visible rather than being quietly replaced.

**Schema contract preserved.** `tests/test_control_plane_load_evidence.py` (not
owned by this agent) asserts on `parameters.iterations` and a top-level
`results` block. Rather than break it, the new schema keeps both, with `results`
now pointing at the **real** aggregate (`"source":
"real_capture_replay.aggregate"`) and `parameters.iterations` = 5112 captured
events × 10 replay repeats = 51120. `python -m pytest
tests/test_control_plane_load_evidence.py tests/test_integrity_attacks.py -q` →
**17 passed**.

---

## 3. `evasion_suite.py` — the attacks CANNOT be made real, and no dataset exists

This is a negative finding, and it is a real one. It is reported here rather than
papered over.

**Every attack in this suite (FGSM, BIM, MIM, transfer, boundary) is a gradient
function of this repository's own neural receiver's weights, computed at attack
time.** No downloadable artefact can substitute for it:

* An "adversarial RF dataset" would carry perturbations crafted against somebody
  else's classifier. Replayed here they would be a weak black-box transfer
  attack — not the white-box attack the benchmark measures.
* The two canonical over-the-air adversarial-perturbation papers
  (arXiv `2002.02400`, arXiv `2202.11197`) demonstrate the effect over SDRs and
  **release no dataset**.
* RadioML / DeepSig — the obvious candidate — contains **no adversarial
  perturbation at all** (it is a modulation-classification corpus), is
  **CC BY-NC-SA 4.0** (a non-commercial + share-alike licence, incompatible with
  this Apache-2.0 repository), and its download host's TLS certificate
  **expired on 2023-06-12**. Three independent disqualifications.

So the attacks stay synthetic **by necessity, not by laziness**, and the results
JSON says exactly that under `data_provenance.attacks`:

```json
{"kind": "synthetic_by_necessity",
 "not_claimed": "these are NOT captured or replayed real attacks"}
```

### What WAS made real: the channel

The suite gained a third arm that runs the identical attack battery over **real
ray-traced channels** built from the DeepMIMO ASU Campus 3.5 GHz per-path data
already in the tree — 4096 real receiver positions, 1–10 real propagation paths
each, `H(f) = Σ_p √(10^(P_p/10))·e^{j(φ_p − 2πf·τ_p)}` over 64 subcarriers across
100 MHz, normalised to unit mean power per receiver. Measured selectivity:
**19.7 dB median peak-to-null** across the band.

Note the wording: DeepMIMO ASU is **site-specific Wireless InSite ray tracing,
not an over-the-air measurement** — the upstream manifest says so and so does
this arm's provenance block (`honest_scope`). Calling it "measured" would be
wrong.

The synthetic Rayleigh arm is **kept as a labelled control**, and the comparison
is the point:

| Attack | AWGN (synthetic) | Fading (**synthetic** Rayleigh) | **Real ray-traced** |
| --- | --- | --- | --- |
| fgsm | 28.6× | 1.14× | **1.16×** |
| bim | 28.6× | 1.14× | **1.16×** |
| mim | 28.6× | 1.14× | **1.16×** |
| transfer | 28.5× | 1.11× | **1.13×** |
| boundary | 84.5× | 1.59× | **1.61×** |

(neural-over-classical SER ratio, 3 seeds, 16-QAM @ 30 dB, ε=0.08)

The interesting result is in the absolute SERs, not the ratios. Under MIM:

| | synthetic Rayleigh | real ray-traced |
| --- | --- | --- |
| neural SER | 0.1426 | **0.0761** |
| classical SER | 0.1246 | **0.0656** |
| Shield effective SER | 0.1357 | **0.0695** |
| fallback rate | 18.3% | **30.0%** |
| clean neural SER | 0.00636 | **0.00400** |

**The synthetic Rayleigh model was roughly 1.9× pessimistic on SER.** Real campus
geometry has fewer catastrophic deep fades than i.i.d. Rayleigh taps, so both
receivers do materially better — and the Shield falls back more often (30% vs
18%) because the neural receiver's degradation is more often measurably outside
its envelope. That difference is only visible because both arms are reported.
The Shield's guarantee (effective ≤ classical + tolerance) holds on the real
channel too.

If the ray-traced features are absent the arm reports `{"skipped": true,
"reason": ...}` rather than silently vanishing.

---

## 4. Things checked and rejected — negative findings

* **5GAD has no usable benign SBI control-plane baseline, and this could not be
  fixed by fetching more data.** The 101 MB `Normal-1UE/allcap_00006` capture was
  fetched (1.7 s, sha256 `e2e76c22…c48f0c9a65a`, matching its LFS pointer) and
  parsed: **165 499 packets yielding 10 HTTP requests, all `GET /`, and zero SBI
  requests.** free5GC's steady state after registration is GTP-U user plane; the
  service-based interface is quiet. Rather than pay 101 MB of CI budget for 10
  events, the 65 KB `Normal-1UE` reference is pinned and its zero-event yield is
  reported honestly in `scope.honest_limits`.
* **5G-NIDD (University of Oulu)** — CC BY 4.0, 1.2 M labelled flows, excellent
  data. Not used here because its CSV `StartTime`/`LastTime` are Excel-mangled to
  `MM:SS.s` and `SIntPkt` is empty for every row, so **per-packet timing fidelity
  is not recoverable from the CSVs**. 5GAD's pcapng gives true nanosecond
  timestamps, which is exactly what a load benchmark needs. 5G-NIDD remains the
  right choice for the FL-poisoning suites (a different agent's files).
* **DLTeamTUC/5GDatasets** — content is an excellent match, but the repository
  has **no LICENSE file and no licence statement**, i.e. all rights reserved.
  Not used, and should not be used without a written grant.
* **RadioML / DeepSig** — rejected on three independent grounds, see §3.

---

## 5. Redistribution discipline

MIT would permit committing the derived rows. This build still does not, keeping
the repository-wide convention: `datasets/5gad_inl/.gitignore` excludes
`generated/`, `manifest.json` carries `"features_committed": false`, and only the
manifest plus the aggregate results JSONs are tracked.

One deliberate exception *inside the gitignored output*:
`nf_profile_writes.jsonl` retains the verbatim captured NF-profile body as
base64. This is necessary, not incidental — a signature check is over **bytes**,
so probe 9 must feed the actual forged bytes through the provenance gate.
Anything less would make it a simulation of the probe rather than the probe.
`control_plane_events.jsonl` stores only digests and lengths.

---

## 6. Verification summary

```
$ python -m ruff check datasets/5gad_inl/build.py benchmarks/integrity_attack_suite.py \
    benchmarks/control_plane_load_suite.py benchmarks/evasion_suite.py
All checks passed!

$ python -m pytest tests/test_control_plane_load_evidence.py tests/test_integrity_attacks.py -q
17 passed

$ python datasets/5gad_inl/build.py          # 12 captures, 24.1 MB, ~5 s, deterministic
$ python benchmarks/integrity_attack_suite.py    # 11/11 blocked, 0 bypassed, 0 skipped
$ python benchmarks/control_plane_load_suite.py  # 5112 real events, 36193 dec/s, p99 65.8 us
$ python benchmarks/evasion_suite.py --seeds 3   # 3 channel arms, real arm present
```

## 7. Scoreboard

| Benchmark | Before | After | Honesty of the claim |
| --- | --- | --- | --- |
| `integrity_attack_suite` | 8 constructed probes, no provenance | 11 probes; 3 driven by captured attacks; probe 9 replays forged bytes unchanged | **Strong.** Real core accepted 398/398; Horizon rejected 398/398 of the same bytes. |
| `control_plane_load_suite` | invented `index % 2` traffic mix | replay of 5112 real captured control-plane events with real timing and message mix | **Medium-strong.** Arrival profile and message mix are real; radio parameters are derived and labelled. |
| `evasion_suite` | synthetic attacks on synthetic channels | synthetic attacks (**by necessity**) on **real ray-traced channels**, synthetic Rayleigh kept as control | **Honest partial.** Channel real; attacks provably cannot be. Said so explicitly. |

## 8. Handoffs (not actioned — files not owned by this agent)

1. **`.gitignore`** — a `datasets/5gad_inl/.gitignore` was added inside the owned
   directory to exclude `generated/`. If the project prefers all dataset
   exclusions centralised in the root `.gitignore` (which currently lists
   `datasets/deepmimo_asu_3p5/generated/`), move the line there and delete the
   local file.
2. **`pyproject.toml` / lockfile** — **no change needed.** The pcapng reader is
   stdlib-only, deliberately, to avoid requiring one.
3. **`docs/CLAIMS_EVIDENCE.json` and `docs/THREAT_MODEL.md`** reference these
   three benchmarks and should be updated to reflect the new provenance —
   particularly the probe-9 comparison, which is the strongest single claim this
   migration produced.
4. **CI** — `datasets/5gad_inl/build.py` needs no opt-in gate and can run
   unattended in CI (24.1 MB, ~5 s). Whoever owns the workflow files may want to
   add it, since probes 9–11 report as *skipped* (and `all_defended: false`)
   without it.
