# E2SM-RC control-payload proof — Shield enforcement reaching the near-RT plane

_Recorded 2026-07-30 on the same sandbox host as
[`E2_KPM_PROOF.md`](E2_KPM_PROOF.md) and
[`OCUDU_E2_JOIN_PROOF.md`](OCUDU_E2_JOIN_PROOF.md)._

## What was proven

Horizon's E2 package was receive-only. It decoded E2SM-KPM indications
([`E2_KPM_PROOF.md`](E2_KPM_PROOF.md)) and had no encoder for the near-real-time
control direction at all, so the last bullet of
[`OCUDU_E2_JOIN_PROOF.md`](OCUDU_E2_JOIN_PROOF.md) read *"Near-real-time
enforcement over E2SM-RC remains roadmap work."*

That gap is now closed on the construction side.
[`../../src/horizon_ric/e2/rc_control.py`](../../src/horizon_ric/e2/rc_control.py)
builds the two OCTET STRING payloads a RIC Control Request carries —
`E2SM-RC-ControlHeader` and `E2SM-RC-ControlMessage` — in aligned PER against
the O-RAN **E2SM-RC v1.03 standard** ASN.1 text vendored at
[`../../src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn`](../../src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn),
and it **cannot** build them for an action the Decision Safety Shield refused.

The load-bearing function is `control_from_disposition`. Given a
`ShieldDisposition` it raises `RcControlError` — naming the violated invariant
ids — whenever the certificate reports `emit_blocked`, `not safe`, or a
non-empty `violated_ids`. There is no override flag and no second code path.
When it does encode, it encodes strictly from `disposition.safe_action`, the
projected action, never from `certificate.action_proposed`. E2 is therefore
subject to the same enforcement as A1 instead of being a parallel unguarded
route to the RAN.

The committed evidence is
[`results/e2_rc_control_proof.json`](results/e2_rc_control_proof.json)
(sha256 `1ba8e4c1adea7e55efc9461c877269f6e30bc8c2474c3ac62c5ff74aeb029bfd`,
11 508 B). It is deterministic: the same inputs regenerate the same file
except for the `environment` block, which records the host's Python and
`asn1tools` versions.

## The spec text and its provenance

| Item | Value |
| --- | --- |
| Spec | O-RAN WG3 E2SM-RC v1.03, ASN.1 modules `E2SM-COMMON-IEs` and `E2SM-RC-IEs` |
| Vendored at | `src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn` |
| sha256 | `09f295cb59d4145efd64fe510f602c994d6df65e9beba33aab2e98cdf5bc642b` (50 985 B, 1 575 lines) |
| Copied from | `third_party/flexric/src/sm/rc_sm/ie/asn/e2sm_rc_v1_03_standard.asn` |
| FlexRIC pin | `gitlab.eurecom.fr/mosaic5g/flexric.git` branch `dev`, commit `ef6d722f22191eea74089966983da1f5ec1fedd4` (`v2.0.0-224-gef6d722f`) — the same submodule pin the KPM proof uses |
| Codec | `asn1tools.compile_files([...], codec="per")` — aligned PER, as in [`../../src/horizon_ric/e2/kpm_bridge.py`](../../src/horizon_ric/e2/kpm_bridge.py) |

`diff` between the vendored copy and the upstream path is empty and the two
sha256 sums are equal:

```
09f295cb59d4145efd64fe510f602c994d6df65e9beba33aab2e98cdf5bc642b  third_party/flexric/src/sm/rc_sm/ie/asn/e2sm_rc_v1_03_standard.asn
09f295cb59d4145efd64fe510f602c994d6df65e9beba33aab2e98cdf5bc642b  src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn
```

The `standard` variant was taken deliberately, not FlexRIC's sibling
`e2sm_rc_v1_03_modified.asn`. All 484 asn1c-generated C sources in that
directory carry the banner `From ASN.1 module "E2SM-RC-IEs" found in
"e2sm_rc_v1_03_standard.asn"`, so the text this repo encodes with is the text
FlexRIC's own RC wire codec was generated from. Full provenance:
[`../../src/horizon_ric/e2/asn1/PROVENANCE_RC.md`](../../src/horizon_ric/e2/asn1/PROVENANCE_RC.md).

This is the service model the joined real gNB advertises. From the RIC's own
log in [`OCUDU_E2_JOIN_PROOF.md`](OCUDU_E2_JOIN_PROOF.md):
`[NEAR-RIC]: Accepting RAN function ID 3 with def = ORAN-E2SM-RC`.

## The ASN.1 actually encoded

Field names below are quoted from the vendored text; nothing is invented.

```asn1
E2SM-RC-ControlHeader ::= SEQUENCE {
    ric-controlHeader-formats  CHOICE { controlHeader-Format1 E2SM-RC-ControlHeader-Format1, ... }, ... }

E2SM-RC-ControlHeader-Format1 ::= SEQUENCE {
    ueID                  UEID,
    ric-Style-Type        RIC-Style-Type,
    ric-ControlAction-ID  RIC-ControlAction-ID,
    ric-ControlDecision   ENUMERATED {accept, reject, ...}  OPTIONAL, ... }

E2SM-RC-ControlMessage-Format1 ::= SEQUENCE {
    ranP-List  SEQUENCE (SIZE(0..maxnoofAssociatedRANParameters)) OF E2SM-RC-ControlMessage-Format1-Item, ... }

E2SM-RC-ControlMessage-Format1-Item ::= SEQUENCE {
    ranParameter-ID         RANParameter-ID,
    ranParameter-valueType  RANParameter-ValueType, ... }
```

CHOICE arms selected, and why:

| CHOICE | Arm used | Reason |
| --- | --- | --- |
| `E2SM-RC-ControlHeader.ric-controlHeader-formats` | `controlHeader-Format1` | Format 1 is the per-UE control header; Format 2 makes `ueID` optional and carries no style/action. |
| `E2SM-RC-ControlHeader-Format1.ueID` (`UEID`) | `gNB-UEID` → `UEID-GNB` | `ueID` is mandatory and `UEID` has no scalar arm — every arm is a SEQUENCE. `UEID-GNB` is the 5G-SA gNB arm and the narrowest: only `amf-UE-NGAP-ID` (INTEGER 0..1099511627775) and `guami` are mandatory. The spec's own inline comments say the `gNB-CU-UE-F1AP-ID-List` and `gNB-CU-CP-UE-E1AP-ID-List` IEs "may not be included" in NearRT-RIC → E2 Node messages, so they are omitted. |
| `E2SM-RC-ControlMessage.ric-controlMessage-formats` | `controlMessage-Format1` | A flat list of RAN parameter values; Format 2 wraps the same list per control style/action and is not needed for a single-action request. |
| `...Format1-Item.ranParameter-valueType` (`RANParameter-ValueType`) | `ranP-Choice-ElementTrue` | Its single field `ranParameter-value` is mandatory. The sibling `ranP-Choice-ElementFalse` makes the value OPTIONAL and the spec annotates it `C-ifControl: This IE shall be present if it is part of a RIC Control Request message` — a control request must carry the value, so the arm that mandates it is correct. |
| `RANParameter-Value` | `valueInt` / `valueReal` | Selected by Python type: `int` → `valueInt`, `float` → `valueReal`. Any other type, including `bool`, raises `RcControlError` rather than being coerced. |

`ric-ControlDecision` is OPTIONAL and is omitted: emitting `accept` or
`reject` would assert a semantic the caller never requested.

The GUAMI is a pinned test-network value (PLMN 001/01, TBCD `00 f1 10`, plus
fixed AMF Region/Set/Pointer bit strings) so the encoding is deterministic. It
identifies the serving AMF, not the UE, and must be replaced with the real
GUAMI against a real core. This is documented at the constants in
[`../../src/horizon_ric/e2/rc_control.py`](../../src/horizon_ric/e2/rc_control.py).

`RANParameter-ID` values are **not** spec constants. An E2 node advertises them
in its RAN Function Definition
(`RANFunctionDefinition-Control-Action-Item.ran-ControlActionParameters-List`, a
list of `ControlAction-RANParameter-Item` pairing a `ranParameter-ID` with its
`ranParameter-name`). `DEFAULT_PARAMETER_IDS` maps the four Shield-governed
action keys to local placeholder ids 1–4; against a real node the caller passes
`parameter_ids=` with the ids that node advertises.

## The committed bytes

Shield: `default_terrestrial_shield(band_lo_hz=3.40e9, band_hi_hz=3.50e9,
max_eirp_dBm=33.0)` — invariant chain `numeric_domain_sanity`,
`spectral_mask_ts38104`, `max_eirp`, `neural_rx_envelope`,
`constellation_legality`. UE identity `amf-UE-NGAP-ID = 112358132134`, the same
value FlexRIC's gNB emulator reports in its KPM UE reports. RIC Style Type 1,
RIC Control Action ID 1.

### Case 1 — approved, clean action (`dec-clean`)

Action `frequency_hz=3.45e9, bandwidth_hz=20e6, tx_power_dBm=20.0,
antenna_gain_dBi=10.0`. Certificate: `safe=true`, `emit_blocked=false`,
`projected=false`, `violated_ids=[]`.

```
header  (20 B) 0002001a291109a60000f1108000400101000000
        sha256 c1057b3f3e80f9baa48ab773e9697ce25c3a66aa10799c57b007f375a167c9cd
message (36 B) 000004000002068007019b45a500010205800801312d0002020380020500030203800105
        sha256 02e2b7c4c0b9797db50fb1a0e3948fbd5aa07aa5f698ccf1ae0bada126ab47ce
```

Decoding those bytes back through the same spec yields
`controlHeader-Format1` → `gNB-UEID` → `amf-UE-NGAP-ID = 112358132134`,
`ric-Style-Type = 1`, `ric-ControlAction-ID = 1`, and
`controlMessage-Format1` → `{1: 3450000000.0, 2: 20000000.0, 3: 20.0, 4: 10.0}`
— the four projected values, unchanged. The full decoded structures are in the
proof JSON.

### Case 2 — approved after projection (`dec-projected`)

Same action but `tx_power_dBm=40.0`: in band, 17 dB over the 33 dBm EIRP
ceiling. The Shield projects rather than refuses —
`"Tx power reduced by 17.00 dB to meet EIRP ceiling 33.00 dBm."` — and
`safe_action["tx_power_dBm"] = 23.0`. The encoded message carries **23.0**:

```
message (36 B) 000004000002068007019b45a500010205800801312d0002020380001700030203800105
        sha256 e54b81fe9fa05b346f1d4e87c028bef3aace8593c09470ffe5a71312a3e572e3
```

The header bytes are identical to Case 1 (same UE, style and action); only the
`tx_power_dBm` RAN parameter differs, decoding to
`{1: 3450000000.0, 2: 20000000.0, 3: 23.0, 4: 10.0}`.

The 40 dBm the planner proposed appears nowhere in the bytes. This is the
difference between a Shield that logs and a Shield that enforces.

### Case 3 — refused, fail closed (`dec-blocked`)

Action `frequency_hz=3.90e9, bandwidth_hz=400e6, tx_power_dBm=46.0,
antenna_gain_dBi=12.0` — out of band, over ceiling, and 400 MHz wide against a
100 MHz licensed channel, so the spectral-mask projection cannot place the
carrier at all. Certificate: `emit_blocked=true`, `safe=false`,
`violated_ids=["spectral_mask_ts38104", "max_eirp"]`.

`control_from_disposition` produced **no bytes**:

```
RcControlError: Shield refused this action; it cannot be encoded as an E2SM-RC
control request (decision_id=dec-blocked): emit_blocked; not safe; violated
invariants spectral_mask_ts38104, max_eirp
```

`header_hex` and `message_hex` for this case are `null` in the proof JSON.

## Tests

[`../../tests/test_e2_rc_control.py`](../../tests/test_e2_rc_control.py) — 14
tests, all passing:

```
$ python -m pytest tests/test_e2_rc_control.py -q -p no:warnings
..............                                                           [100%]
```

They pin the vendored spec's sha256 and the presence of the control types (so a
bad vendor copy fails here, not on someone's wire), the header and message
round-trips including both value arms, byte determinism across repeated
encodes, refusal of unsupported value types instead of coercion, and the paired
property: the refused action raises naming the violated invariants, the approved
one encodes the projected values.
[`../../tests/test_e2_kpm_bridge.py`](../../tests/test_e2_kpm_bridge.py) still
passes unchanged.

## HONEST SCOPE

**What this proves.** E2SM-RC RIC Control *payload construction* against the
vendored O-RAN E2SM-RC v1.03 standard ASN.1 — sha256-verified byte-identical to
the FlexRIC tree its own asn1c wire codec was generated from — in aligned PER,
with every encoded payload decoded back through the same spec to the same
fields. And that a Shield-refused action **cannot be encoded**: the only
function that turns a Horizon decision into E2SM-RC bytes raises instead,
naming the violated invariant ids, and when it succeeds it encodes the
projected action rather than the proposed one.

**What this does NOT prove: live delivery.** No RIC Control Request was sent,
accepted, or acted upon. Three independent reasons, all of them real:

- **There is no E2AP/SCTP transport in this package, by design.**
  [`../../src/horizon_ric/e2/__init__.py`](../../src/horizon_ric/e2/__init__.py)
  states the boundary: the near-RT RIC owns the E2 association; Horizon
  produces and consumes E2SM payloads. This module adds payload
  *construction* only. Wrapping the two OCTET STRINGs in an E2AP
  `RICcontrolRequest` and putting it on an SCTP association is the near-RT
  RIC's job and is not implemented here.
- **The sandbox kernel has no SCTP.** Re-verified in
  [`OCUDU_E2_JOIN_PROOF.md`](OCUDU_E2_JOIN_PROOF.md):
  `socket(AF_INET, SOCK_STREAM, IPPROTO_SCTP)` → `OSError [Errno 93] Protocol
  not supported`, `/proc/net/sctp` absent, no `modprobe` in an unprivileged
  container. Existing E2 work here runs under the disclosed
  [`sctp_udp_shim.c`](sctp_udp_shim.c) substitution.
- **No UE is attached to the OCUDU gNB, and only its CU-CP E2 agent
  attaches.** The gNB runs on the ZMQ RF driver with no UE
  ([`OCUDU_E2_JOIN_PROOF.md`](OCUDU_E2_JOIN_PROOF.md) §Scope). A RIC Control
  targeting a `ueID` therefore has no UE context to resolve against, so it
  could not be shown to be accepted and acted upon even with transport in
  place. The DU-side agent never starts either.

So the honest claim is: **the bytes are right and the gate is real; the
delivery is not demonstrated.** Closing that requires an E2 termination that
sends `RICcontrolRequest`, a kernel with SCTP, and a UE attached to the gNB —
none of which this document claims.

Two smaller limits, stated rather than buried: the pinned GUAMI is a
test-network value, and the RAN Parameter IDs are local placeholders because
E2SM-RC leaves them to the E2 node's RAN Function Definition. Both must be
supplied from the target node before these payloads mean anything to it.

## Reproduce

```bash
# the byte-for-byte vendor check
git submodule update --init third_party/flexric
diff third_party/flexric/src/sm/rc_sm/ie/asn/e2sm_rc_v1_03_standard.asn \
     src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn
sha256sum src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn

# the tests
python -m pytest tests/test_e2_rc_control.py -q
```

The committed bytes and the blocked-case exception message above are regenerated
by these calls alone — no FlexRIC process, no network:

```python
from horizon_ric.e2.rc_control import RcControlError, control_from_disposition
from horizon_ric.shield import default_terrestrial_shield

shield = default_terrestrial_shield(band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0)
clean = {"block": "risk_band_planner", "frequency_hz": 3.45e9,
         "bandwidth_hz": 20e6, "tx_power_dBm": 20.0, "antenna_gain_dBi": 10.0}

req = control_from_disposition(shield.dispose(clean, decision_id="dec-clean"), ue_id=112358132134)
print(req.header_bytes.hex(), req.header_sha256)
print(req.message_bytes.hex(), req.message_sha256)

bad = dict(clean, block="poisoned_planner", frequency_hz=3.90e9,
           bandwidth_hz=400e6, tx_power_dBm=46.0, antenna_gain_dBi=12.0)
try:
    control_from_disposition(shield.dispose(bad, decision_id="dec-blocked"), ue_id=112358132134)
except RcControlError as exc:
    print("blocked:", exc)
```

`RcControlRequest.to_dict()` gives the same hex/sha256 fields the proof JSON
records, for wiring into the evidence chain.
