# xApp end-to-end harness — real near-RT RIC A1 mediator + real xApp

This directory builds and runs the **official O-RAN-SC near-RT RIC A1
mediator** and the **official `hw-python` reference xApp** from pinned
upstream source, so the Horizon-RIC rApp can be exercised against a real
policy consumer over the real RMR message bus — no mocks, no emulators
written by this repo.

```
Horizon-RIC rApp ──HTTP A1AP (/A1-P/v2, dialect "legacy")──► a1mediator (Go, :10000)
                                                                 │  RMR A1_POLICY_REQ (20010)
                                                                 ▼
                                                          hw-python xApp (:4560)
                                                                 │  RMR A1_POLICY_RESP (20011)
                                                                 ▼
                                            a1mediator marks enforceStatus=ENFORCED (SDL/redis)
```

| Component | Source | Pin |
| --- | --- | --- |
| RMR message router (C) | `gerrit.o-ran-sc.org/r/ric-plt/lib/rmr` | `8b9a214906a40b338def981d5c16b4ea247176fb` (4.9.4) |
| A1 mediator (Go) | `gerrit.o-ran-sc.org/r/ric-plt/a1` | `09a757b4fd63198d8690d50b52bfd04552d47f1f` |
| hw-python xApp | `gerrit.o-ran-sc.org/r/ric-app/hw-python` | `a6d00525aa2f62f2457d84731da2b7d89e0b1013` |
| ricxappframe (PyPI) | pinned transitively by hw-python `setup.py` | 2.2.0 |
| SDL backend | `redis-server` (the dbaas image is plain redis) | distro package |

## Usage

```bash
deploy/xapp-e2e/run_stack.sh          # build + start redis, mediator, xApp
HORIZON_A1_DIALECT=legacy \
HORIZON_NEAR_RT_RIC_URL=http://127.0.0.1:10000 \
HORIZON_ONCE_REQUIRE_ACCEPTED=1 \
.venv/bin/python scripts/run_horizon_rapp.py \
    --source-config deploy/xapp-e2e/source-replay.yaml \
    --once --once-max-events 12 \
    --report-json /tmp/pipeline-report.json
.venv/bin/python scripts/xapp_e2e_proof.py \
    --report /tmp/pipeline-report.json \
    --mediator-log ~/oran-deps/a1mediator.log \
    --xapp-log ~/oran-deps/hwxapp.log \
    --out /tmp/xapp-e2e-proof.json
```

The proof script cross-checks three independent witnesses: the rApp's
pipeline report, the mediator's per-policy `enforceStatus`, and the
mediator's receive log of the xApp's `A1_POLICY_RESP` payloads — each
log line's embedded JSON payload must carry BOTH the policy instance id
and `handler_id: "hw-python"` — corroborated by the xApp's RMR
send-stats line showing the matching number of successful sends to the
mediator endpoint. (The xApp's own "processed request" lines sit below
its pinned log level, so the receiver-side record is the third witness,
not the xApp's own log.)

## Local deviations from stock upstream (full disclosure)

1. **Mediator patch** `patches/a1mediator-policy-resp-type.patch` — the
   stock mediator serialises `policy_type_id` as a JSON *string* in the
   `A1_POLICY_REQ` it sends (`pkg/rmr/messages.go` builds a
   `map[string]string`), but `Consume()` type-asserts `float64` on the
   echoed `A1_POLICY_RESP` and panics. The stock binary therefore
   crashes on the ACK to its own request when paired with the stock
   hw-python xApp (which echoes request fields per its
   `A1PolicyHandler.buildPolicyResp`). The patch accepts both encodings
   on receive; nothing else is touched. Found live in this harness —
   candidate for an upstream bug report.
2. **hw-python config addition** — `A1PolicyHandler` reads
   `config["xapp_name"]`, which the in-repo `init/config-file.json`
   does not define (cluster deployments inject it via the RIC Helm
   configmap). `run_stack.sh` adds `"xapp_name": "hw-python"` to the
   local config copy. No xApp code is modified.
3. The RMR routing table `a1.rt` is a static two-endpoint table
   (mediator `:4562`, xApp `:4560`) replacing the platform Route
   Manager, which is standard practice for standalone RMR deployments.
