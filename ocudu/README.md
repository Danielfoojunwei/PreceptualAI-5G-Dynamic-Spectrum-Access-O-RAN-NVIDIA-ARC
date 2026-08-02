# OCUDU integration — controls a real gNB would parse, telemetry a real gNB emits

**Additive and self-contained.** Nothing under `src/horizon_ric/` or
`docs/proposal/` is modified. This subtree imports the main package read-only
(`rc_control.rc_spec`, the vendored E2SM-RC ASN.1) and the agentic package
(`telemetry_trust`), and composes on top.

[OCUDU](https://gitlab.com/ocudu/ocudu) is the Linux Foundation 5G CU/DU stack,
pinned as a submodule at `f46f5804`. It was already used here as a **real gNB**
that joins FlexRIC over E2 ([`../deploy/e2-companion/OCUDU_E2_JOIN_PROOF.md`](../deploy/e2-companion/OCUDU_E2_JOIN_PROOF.md)).
What was left on the table was everything on the *receiving* side: OCUDU does
not merely accept an E2 Setup, it implements control-action executors that
parse RIC control messages and metrics generators that emit real RAN state.

## The finding that changed a claim

Horizon's E2SM-RC encoder carried **placeholder RAN Parameter IDs**, flagged
amber in `docs/conformance/CONFORMANCE.md` with the correct note that E2SM-RC
assigns identifiers per node via `RANFunctionDefinition-Control-Action-Item`.

OCUDU declares them, in code. And the action it declares them for is the one
Horizon cares about most:

> **E2SM-RC Style 2, Action 6 — "Slice-level PRB quota"**
> *"To control the radio resource management policy for slice-specific PRB
> quota allocation"*

That is the same quantity `ProtectedSliceFloorInvariant` guards. The invariant
reserves a fraction of a cell's PRBs for a safety-critical slice and projects
any planner proposal back above the floor; action 2.6 is how a RIC tells a DU
to enforce it. Horizon's floor and the DU's **Min PRB Policy Ratio** are the
same number in different units.

| id | name | role |
|---|---|---|
| 1 | RRM Policy Ratio List | structural |
| 2 | RRM Policy Ratio Group | opens a group |
| 3 | RRM Policy | structural |
| 5 | RRM Policy Member List | structural |
| 6 | RRM Policy Member | opens a member |
| 7 | PLMN Identity | 3 octets, exactly |
| 8 | S-NSSAI | structural |
| 9 | SST | octets |
| 10 | SD | octets |
| 11 | **Min PRB Policy Ratio** | integer percent |
| 12 | Max PRB Policy Ratio | integer percent |
| 13 | Dedicated PRB Policy Ratio | integer percent |

(4 is absent; the standard skips it, and OCUDU's source says so.)

## Extracted, not transcribed

`extract_catalogue.py` parses OCUDU's executors and CCC packer and writes
`catalogue/ocudu-e2sm-catalogue.json`, recording the OCUDU commit and the
SHA-256 of every file it read. Gate G8 **re-runs the extraction and compares**.

That is the difference between an integration and a copy. Hand-writing thirteen
identifiers into a Python module would be correct today and silently wrong the
next time the submodule moves — and the failure would surface as a gNB
rejecting a control message in a lab, weeks later, with no obvious cause.

The catalogue also records which parameters *carry values* versus which only
open a nesting level. OCUDU spells the latter as one `if` arm whose body is
`// No need to parse`; the extractor detects that arm specifically rather than
assuming.

## The ordering contract, which is the easy thing to get wrong

OCUDU's parser walks the RAN-parameter tree in document order and mutates the
**most recently created** group and member:

```
id 2       → rrm_policy_ratio_list.emplace_back()          new group
id 6       → .back().policy_members_list.emplace_back()    new member
id 7/9/10  → fill .back().policy_members_list.back()       that member
id 11/12/13→ set on .back()                                that group
```

So a tree can decode perfectly and still be wrong: a member appended to a group
that does not exist yet, or ratios applied to the previous group. `visit_order()`
exposes the sequence, and G8 asserts the precedences directly rather than
inferring correctness from a successful decode.

A real encode, from the vendored O-RAN E2SM-RC v1.03 ASN.1:

```
visit order : [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
PER         : 79 octets, round-trips identically
0.20 floor  → Min PRB Policy Ratio = 20
```

## What is in here

| Path | What it does |
|---|---|
| `extract_catalogue.py` | Parses OCUDU source into a machine-readable control-action catalogue. |
| `src/horizon_ocudu/rc_slice_quota.py` | Builds and PER-encodes a Slice-level PRB quota control with OCUDU's real ids, in its parser's order. Binds a Horizon PRB fraction to Min PRB Policy Ratio. |
| `src/horizon_ocudu/ccc_rrm_policy.py` | The same intent via E2SM-CCC `O-RRMPolicyRatio` — JSON attributes rather than nested PER. Shares the RC route's validation so the two cannot disagree about what is valid. |
| `src/horizon_ocudu/metrics.py` | Parses OCUDU's JSON scheduler metrics into agentic `TelemetryRecord`s. |
| `verify_g8_ocudu_conformance.py` | The gate. |

Two deliberate choices in the metrics bridge:

**It does not sign on OCUDU's behalf.** OCUDU emits unauthenticated JSON. There
is no key, and minting one here would produce records that *look* authenticated.
`records_from_metrics` returns unsigned records; a deployment either puts them
behind a collector holding a real key or accepts that the trust gate refuses
them. Refusing is what happens if nobody decides, which is correct.

**UE quality metrics take the worst value, not the mean.** A cell whose average
CQI is comfortable can still contain a UE at the edge of coverage, and an
enforcement layer reasoning about the average would approve an action that
harms exactly the user it should protect.

## Running it

```sh
python ocudu/extract_catalogue.py --out ocudu/catalogue/ocudu-e2sm-catalogue.json
python ocudu/verify_g8_ocudu_conformance.py --out ocudu/results/g8-ocudu-conformance.json
PYTHONPATH=src:agentic/src:ocudu/src python -m pytest ocudu/tests -q
```

Needs the `third_party/ocudu` submodule initialised, and the `oran` extra —
`asn1tools` is what compiles the vendored E2SM-RC ASN.1, and
`horizon_ric.e2.__init__` imports it eagerly, so a missing extra fails at
collection rather than at first use. No *new* dependencies are added.

The `ocudu` CI job first went red on exactly this: it installed `.[dev]` while
the main pytest job installs `.[dev,oran,otel]`, and the whole subtree had gone
green locally because the development venv already carried `asn1tools` from an
earlier task. `test_ci_installs_every_extra_this_subtree_needs` now asserts the
workflow's install line against the extra that declares `asn1tools`, so the two
facts — what the code imports and what CI installs — are checked against each
other rather than assumed to agree.

## Gate G8, and that it can fail

Seven checks. `catalogue_is_reproducible` is the load-bearing one — every other
check asserts agreement with the catalogue, and agreement with a stale
catalogue proves nothing.

| Falsification | Result |
|---|---|
| One parameter id renumbered, as an upstream change would | `parameter_ids_match_ocudu` and `visit_order_matches_parser_contract` fail |
| Committed catalogue edited without re-extracting | `catalogue_is_reproducible` fails |
| Member emitted before its group (breaks `.back()`) | `visit_order_matches_parser_contract` fails |
| Minimum ratios allowed to exceed the cell | `impossible_quotas_refused` fails |

## An open item, narrowed — and a wrong answer withdrawn

`OCUDU_E2_JOIN_PROOF.md` records that only the CU-CP E2 agent attached, with
`enable_du_e2: true` in the effective configuration and the DU agent never
starting.

**A first pass at this was wrong and is withdrawn.** It claimed the CU-CP and
DU-high units each create their own `e2:` section, so `enable_du_e2` at top
level bound to the wrong one. The source refutes that: OCUDU's `add_subcommand`
helper (`include/ocudu/support/cli11_utils.h:38`) calls
`get_subcommand_no_throw` and **returns the existing subcommand** when one is
already registered. Both units therefore populate a single shared top-level
`e2:` block, and our configuration was correct.

What the source does establish, tracing the whole path:

| step | source | state |
|---|---|---|
| `enable_du_e2` reaches the DU-high unit config | `o_du_high_unit_config_cli11_schema.h:20` | wired |
| the DU E2 agent is constructed when it is set | `o_du_high_unit_factory.cpp:225` | wired |
| the gNB app supplies the DU an E2 client | `gnb.cpp:543` — `e2_gw_du.get()` | supplied |

So the cause is **not configuration** — which is what the withdrawn answer got
wrong — and the remaining candidates are runtime. The one to check first is the
SCTP shim: two E2 agents mean two concurrent SCTP associations, the join proof
already documents extending the shim to satisfy OCUDU's one-to-one client
style, and nothing in it suggests a second simultaneous association is handled.

> **That paragraph is a candidate, not a finding.** It says where to look, not
> what is true. There is no OCUDU build tree in this sandbox, so nothing here
> has been re-run, and the item stays open.

## What is deliberately not claimed

- **No control has been delivered to a running gNB.** These bytes have not been
  carried over E2AP to OCUDU and executed. What is proven is that they use the
  identifiers OCUDU declares, nested in the order its parser requires, and that
  they round-trip through the standard ASN.1 — not that a DU applied them.
- **The CCC route produces JSON, not a complete CCC control message.** E2SM-CCC
  carries its payload as JSON inside an ASN.1 envelope; only the JSON half is
  built here, and the envelope is not.
- **No UE, no traffic.** OCUDU's metrics parser is exercised against reports
  matching its own generators' field names and structure; it has not consumed a
  report from a gNB carrying live UEs, because the E2 join ran without any.
- **The DU E2 agent is still unjoined** — see above.
- **Handover control is catalogued, not implemented.** CU style 3 action 1 is
  in the catalogue because OCUDU declares it; nothing here encodes one, and
  Horizon has no invariant that governs handover.
