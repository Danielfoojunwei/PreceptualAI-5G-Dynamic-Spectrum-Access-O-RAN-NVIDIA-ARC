# NVIDIA Cloud Functions × Horizon-RIC

_Additive subtree. Nothing under `src/`, `docs/proposal/`, `agentic/` or the
existing `ocudu/` modules is modified by this work._

NVCF went open source (Apache-2.0, [`NVIDIA/nvcf`](https://github.com/NVIDIA/nvcf))
and is deployable self-managed on an operator's own GPU cluster. That changes
what it is, for us: not a hosted API to call, but **the substrate the telco
agents we govern will actually run on**.

This subtree connects the two in both directions, and every path, header and
status code in it is extracted from NVIDIA's own published OpenAPI document
(`api.nvcf.nvidia.com/v3/openapi`, API **2.248.1**, vendored under
[`spec/`](spec/) and pinned by SHA-256) rather than transcribed from
documentation. Gate **G10** re-extracts and compares, so a protocol change
upstream fails CI instead of producing a client that quietly speaks last
year's protocol.

---

## 1. Their architecture, and what it means for a safety plane

NVCF is three planes ([`NVIDIA/nvcf`](https://github.com/NVIDIA/nvcf)):

| NVCF plane | What it does | The Horizon analogue |
|---|---|---|
| **Control plane** | function/deployment lifecycle, secrets | the operator's invariant profile: `ShieldConfig`, `AuthorityPolicy`, the keyring, `profile_digest` on every emitted policy |
| **Invocation plane** | HTTP/gRPC routing, rate limiting | `EnvelopeIngress` → `TransactionCycle` — the serialisation point |
| **Compute plane** | NVCA cluster agent, GPU placement, telemetry | where the agent's model runs. **Untrusted by construction.** |

The split is the useful part. NVCF already separates *who may invoke what*
(control) from *where the model runs* (compute). Horizon's separation is the
same line drawn one level down: *what an action is permitted to be* (the
operator's invariants) from *what a model proposed* (the agent). Deploying
Horizon into NVCF puts both separations on the same axis instead of at odds.

### The invocation protocol is the right shape, for a reason NVIDIA does not need

NVCF's pass-through invocation is asynchronous by default. `POST` holds the
connection for `NVCF-POLL-SECONDS`; if the work is unfinished it answers
**202** with an `NVCF-REQID`, and the caller polls
`GET /v2/nvcf/pexec/status/{requestId}` until **200**. That is normally
explained as a concession to slow GPUs.

For a safety plane it is the *required* shape, for a different reason.
`TransactionCycle.submit()` **cannot** answer when a request arrives, because
answering would decide that agent's request in isolation — and deciding
requests in isolation is the exact failure the whole transaction machinery
exists to remove. It already returns a `SubmissionReceipt`, not a decision.
NVCF's 202-and-poll is the wire form of that receipt:

```
NVCF                              Horizon
────────────────────────────      ───────────────────────────────────
POST /pexec/functions/{id}    →   TransactionCycle.submit(envelope)
202 + NVCF-REQID              →   SubmissionReceipt(epoch, position)
GET  /pexec/status/{reqId}    →   look up that epoch's result
202 + NVCF-PERCENT-COMPLETE   →   epoch still open
200 + body                    →   TransactionCertificate
429                           →   epoch full (max_members)
403                           →   wrong functionId/version, unknown reqId
```

So [`src/horizon_nvcf/service.py`](src/horizon_nvcf/service.py) is not a
wrapper around a mismatch. It is the same idea expressed twice, and G10
drives it over a real socket.

### Where we deliberately depart from their defaults

**`maxInstances: 1`.** NVCF's headline capability is autoscaling from zero to
maximum capacity. For the decision plane that is a correctness bug, not a
tuning knob. The plane is a *serialisation point*: two replicas would each
open their own epoch, and two agents whose actions conflict could land in
different epochs and both be admitted — precisely the failure the transaction
exists to prevent. Scale the **agents**, never this.
[`deploy/horizon-shield-deployment.json`](deploy/horizon-shield-deployment.json)
pins it, and G10's `decision_plane_is_not_horizontally_scaled` gates it.

**`minInstances: 1`.** Scale-to-zero is right for inference and wrong here: a
cold start is a window in which agents can be invoked and the Shield cannot.
The invariant chain is CPU work, so a held replica costs almost nothing.

**`maxRequestConcurrency: 16`**, bounded under `TransactionCycle`'s
`max_members` (32), so the epoch buffer is what refuses excess load — and it
refuses with a 429 that means what NVCF says it means: the request did not
run, retry later.

---

## 2. NVCF as the place agents live

[`src/horizon_nvcf/agent.py`](src/horizon_nvcf/agent.py) presents an
NVCF-hosted model as a governed principal. Three properties, all gated:

**Identity is the version, not the function.** A `functionId` survives a
redeploy; only `versionId` is immutable. So the agent id is
`nvcf:<functionId>/<versionId>`, the client always invokes the *versioned*
path, and a response echoing a different version produces no envelope. A
function rolled forward to a new model is a **different principal** and must
be granted authority again — G10 drives a proposal from an ungranted version
and shows the transaction refuses it outright and that
`TransactionResult.members` **raises** rather than handing back the part that
passed.

**No failure mode produces an action.** G10 drives fourteen of them —
402/403/429/302, an undeclared 500, a 202 with no request id, a 202 that
never resolves, a non-JSON 200, a 200 with no action — and every one yields
no proposal.

Five of those fourteen carry a **valid action body alongside the
non-fulfilled status**, and they exist because the first version of that
check passed for the wrong reason. Real 402/403/429 responses tend to have
empty bodies, so the refusal was coming from *"that is not JSON"* rather than
from the status classification. A misbehaving inference service will happily
return 429 with a complete answer, and then only the classification stands
between that answer and a radio. With the added modes, deliberately
misclassifying 429 as fulfilled now flips the check — before, it did not.

**The response is parsed, never trusted.** Out-of-*scope* keys are dropped
(passing them would make `authorize()` refuse the whole envelope because a
model added an annotation); out-of-*range* values are **kept**, because that
is the model asking for something and the Shield is what must answer.
Non-finite numbers and booleans refuse the proposal outright. The envelope is
signed by *us*, with a key the operator registered — the function has no key
and cannot sign anything. NVCF's transport auth proves we reached NVIDIA, not
that the model is honest.

---

## 3. The composite path, joined to OCUDU

G10's end-to-end leg is the only thing that runs all three subsystems
together:

```
NVCF function response
  → AgentActionEnvelope           (identity = immutable function version)
  → SafetyTransaction             (Shield envelope DERIVED from the OCUDU cell)
  → E2SM-RC Style 2 Action 6      (aligned PER, OCUDU's own parameter IDs)
```

The middle step is what [`ocudu/src/horizon_ocudu/cell_config.py`](../ocudu/src/horizon_ocudu/cell_config.py)
and gate **G9** add, and it closed a real hole. `PipelineConfig` hand-typed
the Shield's band as **3.40–3.50 GHz**. The OCUDU cell in
`gnb_ru_ran550_tdd_n78_100mhz_4x2.yml` transmits **3508.18–3608.18 MHz**
(NR-ARFCN 637212 → F_REF by TS 38.104 §5.4.2.1; 273 PRB at 30 kHz by
Table 5.3.2-1). **Those intervals are disjoint.** The hand-typed band was
never this radio's band.

G10 makes the consequence concrete: an NVCF function proposing **3450.00 MHz**
— comfortably inside the old hand-typed default, 58 MHz below the real
carrier — is projected by the derived envelope to **3518.18 MHz**. The old
envelope would have waved through the one emission this cell is not licensed
to make.

G9 is wired so it cannot rot: it parses the `HORIZON_SHIELD_*` names out of
`pipeline.py`'s **AST**, so a rename in the product source fails the gate
rather than silently unwiring the derivation. And it derives all ten
committed OCUDU cell configurations, not just the one it was aimed at.

EIRP is deliberately **not** derived. `ssb_block_power_dbm` is SS/PBCH block
EPRE (per-RE, at the connector); turning it into an EIRP needs antenna gain
and feeder loss, which are properties of the site and which no gNB config
carries. `derive_eirp_ceiling_dBm` takes `antenna_gain_dBi` keyword-only with
no default and refuses without it. A safety ceiling assembled from a number
nobody supplied is worse than no ceiling, because it looks measured.

Nothing under `src/` changed to achieve any of this: `PipelineConfig.from_env`
already reads those variables, so the derivation is delivered as

```sh
eval "$(python -m horizon_ocudu.cell_config \
          third_party/ocudu/configs/gnb_ru_ran550_tdd_n78_100mhz_4x2.yml \
          --export --antenna-gain-dBi 8)"
```

---

## 4. Reproduce

```sh
python nvcf/extract_contract.py --check          # contract vs vendored spec
PYTHONPATH=src:agentic/src:ocudu/src:nvcf/src \
  python -m pytest nvcf/tests ocudu/tests -q
python ocudu/verify_g9_cell_envelope.py          # 12 checks
python nvcf/verify_g10_nvcf_conformance.py       # 18 checks
```

CI: [`.github/workflows/nvcf.yml`](../.github/workflows/nvcf.yml). Both gates
are re-run and diffed against their committed results, source digests
included, so editing a module to make the numbers agree fails there.

## 5. What this does NOT prove

- **No NVCF function was deployed.** There is no NGC account, no GPU and no
  credential in this environment. The client is driven by a recorded
  transport over the spec's declared responses; `urllib_transport` is the
  code path a real endpoint would take, and it is not exercised against one.
- **The service is a protocol surface, not a production deployment.** TLS,
  the NVCA cluster agent and real autoscaling belong to NVCF itself. What is
  proven is that Horizon answers the contract on a real socket.
- **The composite path stops at PER bytes.** G10 encodes a real E2SM-RC
  Style 2 Action 6 message; delivering it to a live E2 node over a RIC
  Control Request is WP4 work and is gated by G4, not here. See
  [`deploy/e2-companion/OCUDU_E2_JOIN_PROOF.md`](../deploy/e2-companion/OCUDU_E2_JOIN_PROOF.md)
  for exactly how far the real E2 path currently reaches.
- **The OCUDU-derived envelope is not yet the pipeline's default.** It is
  delivered as an export; making it the default would mean editing `src/`,
  which this work deliberately does not do.
