# Known `pyang --strict --canonical` Violations

This file enumerates every vendored YANG module under `deploy/yang/` that does
**not** pass `pyang --strict --canonical -p deploy/yang/ <module>` cleanly,
along with the exact pyang diagnostic. The CI gate (`tests/test_yang_strict.py`
and `.github/workflows/lint.yml`) **skips** these modules so they cannot
silently mask new regressions in the strict-clean modules.

The vendored copies are byte-for-byte identical to the upstream (see
`manifest.json` for SHA-256 + URL); these are upstream issues, not local edits.

---

## Summary

| Module | Source | Class | Reason CI tolerates |
|--------|--------|-------|---------------------|
| `ietf-inet-types@2013-07-15.yang` | RFC 6991 | upstream RFC defect | Published-RFC text; rewriting it would diverge from the RFC. |
| `ietf-netconf-notifications@2012-02-06.yang` | RFC 6470 | upstream RFC warning | Augments cross a module that pyang's --strict resolves with a warning. |
| `_3gpp-common-yang-types.yang` | 3GPP TS 28.623 (Rel-18) | upstream 3GPP defect | Imports `time-with-zone-offset` from a newer ietf-yang-types revision than RFC 6991. |
| `_3gpp-common-yang-extensions.yang` | 3GPP TS 28.623 (Rel-18) | upstream 3GPP canonical-order | Extension definition orders `argument` after siblings — pyang flags as non-canonical. |
| `_3gpp-common-managed-element.yang` | 3GPP TS 28.541 (Rel-18) | unvendored transitive imports | Pulls 5GC NRM modules (`_3gpp-5gc-nrm-configurable5qiset` …) we deliberately did not vendor (out of scope for AI-RAN rApp). |
| `_3gpp-common-managed-function.yang` | 3GPP TS 28.541 (Rel-18) | upstream canonical-order | `type` substatement order. |
| `_3gpp-common-subnetwork.yang` | 3GPP TS 28.541 (Rel-18) | unvendored transitive imports | Imports 5GC NRM modules. |
| `_3gpp-common-mecontext.yang` | 3GPP TS 28.541 (Rel-18) | upstream canonical-order | `key` substatement order. |
| `_3gpp-common-measurements.yang` | 3GPP TS 28.541 (Rel-18) | upstream canonical-order | `key`, `default` order. |
| `_3gpp-common-subscription-control.yang` | 3GPP TS 28.541 (Rel-18) | upstream canonical-order | `key`, `max-elements` order. |
| `_3gpp-common-fm.yang` | 3GPP TS 28.541 (Rel-18) | upstream canonical-order | `max-elements`, `key` order. |
| `_3gpp-common-trace.yang` | 3GPP TS 28.541 (Rel-18) | upstream canonical-order | `key`, `min-elements` order. |
| `_3gpp-common-files.yang` | 3GPP TS 28.541 (Rel-18) | upstream canonical-order | `units`, `config` order. |
| `_3gpp-common-filemanagement.yang` | 3GPP TS 28.541 (Rel-18) | upstream canonical-order | `must` substatement order. |

These canonical-order findings are recurring complaints filed against the 3GPP
SA5 MnS YANG bundle (see issues on `forge.3gpp.org/rep/sa5/MnS`). pyang refuses
to rewrite them automatically without the `--canonical` *output* mode, but our
gate uses `--canonical` as a *check*. The mitigations:

1. We vendor the upstream artifacts as-is (auditable: `manifest.json` records
   the SHA-256 of every file).
2. The CI gate enforces strict-clean on the 14 modules that pass cleanly today.
3. Any new IETF/O-RAN module added to the chart **must** pass strict-clean or
   be added to this allowlist with an upstream-issue link in the next column.

---

## Exact `pyang --strict --canonical -p deploy/yang/` output

```
=== _3gpp-common-filemanagement.yang ===
_3gpp-common-filemanagement.yang:42: error: keyword "must" not in canonical order (see RFC 7950, Section 14)
_3gpp-common-filemanagement.yang:46: error: keyword "must" not in canonical order (see RFC 7950, Section 14)

=== _3gpp-common-files.yang ===
_3gpp-common-files.yang:82: error: keyword "units" not in canonical order (see RFC 7950, Section 14)
_3gpp-common-files.yang:83: error: keyword "config" not in canonical order (see RFC 7950, Section 14)

=== _3gpp-common-fm.yang ===
_3gpp-common-fm.yang:397: error: keyword "max-elements" not in canonical order (see RFC 7950, Section 14)
_3gpp-common-fm.yang:398: error: keyword "key" not in canonical order (see RFC 7950, Section 14)

=== _3gpp-common-managed-element.yang ===
_3gpp-common-managed-element.yang:6: warning: imported module "_3gpp-common-yang-types" not used
_3gpp-common-managed-element.yang:14: error: module "_3gpp-5gc-nrm-configurable5qiset" not found in search path

=== _3gpp-common-managed-function.yang ===
_3gpp-common-managed-function.yang:95: error: keyword "config" not in canonical order, expected "type" (see RFC 7950, Section 14)
_3gpp-common-managed-function.yang:97: error: keyword "type" not in canonical order (see RFC 7950, Section 14)

=== _3gpp-common-measurements.yang ===
_3gpp-common-measurements.yang:184: error: keyword "key" not in canonical order (see RFC 7950, Section 14)
_3gpp-common-measurements.yang:246: error: keyword "default" not in canonical order, expected "type" (see RFC 7950, Section 14)

=== _3gpp-common-mecontext.yang ===
_3gpp-common-mecontext.yang:74: error: keyword "key" not in canonical order (see RFC 7950, Section 14)

=== _3gpp-common-subnetwork.yang ===
_3gpp-common-subnetwork.yang:15: error: module "_3gpp-5gc-nrm-configurable5qiset" not found in search path
_3gpp-common-subnetwork.yang:16: error: module "_3gpp-5gc-nrm-ecmconnectioninfo" not found in search path

=== _3gpp-common-subscription-control.yang ===
_3gpp-common-subscription-control.yang:134: error: keyword "key" not in canonical order (see RFC 7950, Section 14)
_3gpp-common-subscription-control.yang:135: error: keyword "max-elements" not in canonical order (see RFC 7950, Section 14)

=== _3gpp-common-trace.yang ===
_3gpp-common-trace.yang:150: error: keyword "key" not in canonical order (see RFC 7950, Section 14)
_3gpp-common-trace.yang:151: error: keyword "min-elements" not in canonical order (see RFC 7950, Section 14)

=== _3gpp-common-yang-extensions.yang ===
_3gpp-common-yang-extensions.yang:152: error: keyword "argument" not in canonical order (see RFC 7950, Section 14)

=== _3gpp-common-yang-types.yang ===
_3gpp-common-yang-types.yang:98: error: type "time-with-zone-offset" not found in module "ietf-yang-types"
_3gpp-common-yang-types.yang:118: error: keyword "description" not in canonical order, expected "type" (see RFC 7950, Section 14)

=== ietf-inet-types@2013-07-15.yang ===
ietf-inet-types@2013-07-15.yang:361: error: keyword "length" not in canonical order (see RFC 6020, Section 12)

=== ietf-netconf-notifications@2012-02-06.yang ===
ietf-netconf-notifications@2012-02-06.yang:286: warning: node "ietf-netconf-notifications::confirm-event" is not found in module "ietf-netconf-notifications"
```

## Strict-clean modules (as of vendor time)

| Module | Source |
|--------|--------|
| `iana-crypt-hash@2014-08-06.yang` | RFC 7317 |
| `ietf-datastores@2018-02-14.yang` | RFC 8342 |
| `ietf-interfaces@2018-02-20.yang` | RFC 8343 |
| `ietf-netconf@2011-06-01.yang` | RFC 6241 |
| `ietf-netconf-acm@2018-02-14.yang` | RFC 8341 |
| `ietf-system@2014-08-06.yang` | RFC 7317 |
| `ietf-x509-cert-to-name@2014-12-10.yang` | RFC 7407 |
| `ietf-yang-library@2019-01-04.yang` | RFC 8525 |
| `ietf-yang-types@2013-07-15.yang` | RFC 6991 |
| `o-ran-common-identity-refs.yang` | O-RAN WG4 (mirror) |
| `o-ran-common-yang-types.yang` | O-RAN WG4 (mirror) |
| `o-ran-hardware.yang` | O-RAN WG4 (mirror) |
| `o-ran-wg4-features.yang` | O-RAN WG4 (mirror) |
| `_3gpp-common-top.yang` | 3GPP TS 28.541 (Rel-18) |

(14 strict-clean / 28 vendored.)
