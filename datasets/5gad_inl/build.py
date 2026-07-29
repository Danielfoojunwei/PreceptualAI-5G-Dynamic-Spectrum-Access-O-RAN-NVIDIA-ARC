#!/usr/bin/env python3
"""Build a deterministic control-plane attack feature set from 5GAD-2022.

5GAD-2022 is a corpus of packet captures taken from a REAL 5G standalone test
bench (free5GC core + UERANSIM) at Idaho National Laboratory while ten distinct
attacks were actually executed against it. The captures are over-the-wire
recordings of the attacks, not simulations of them.

    repository  https://github.com/IdahoLabResearch/5GAD
    licence     MIT (Copyright (c) 2022 Battelle Energy Alliance, LLC)
    DOI         10.11578/dc.20220811.1
    paper       C. Coldwell, D. Conger, E. Goodell, B. Jacobson, B. Petersen,
                D. Spencer, M. Anderson, M. Sgambati, "Machine Learning 5G
                Attack Detection in Programmable Logic", 2022 IEEE Globecom
                Workshops, pp. 1365-1370, doi 10.1109/GCWkshps56602.2022.10008647

The whole repository is 35.46 GB across 273 Git-LFS objects. This build touches
only the twelve captures pinned in ``PINNED_FILES`` (24.1 MB total), which is a
CI-affordable budget. Each object is fetched from the GitHub LFS media host at a
pinned commit and verified against the SHA-256 recorded in the repository's LFS
pointer *before* it is parsed, exactly as ``deepmimo_asu_3p5/build.py`` verifies
its source archive. The LFS pointer's ``oid sha256:`` field IS the pin: it is
published by the upstream repository, not computed by us.

Reproducibility: fully non-interactive. No login, no token, no click-through.
Verified end to end:

    curl -sSL -o /tmp/x.pcapng \\
      https://media.githubusercontent.com/media/IdahoLabResearch/5GAD/\\
d6c3643dafb0683ecac2557e1a5ca29c3b5d7ecb/Attacks/FakeAMFInsert/\\
Attacks_FakeAMFInsert.pcapng
    # HTTP 200, 991376 bytes,
    # sha256 41cca32c909ffaaa80f4b6e78825c94a5c09530f9f21cb78f35cda3a8e8ceaa8
    # == the LFS pointer oid.

The pcapng reader below is deliberately dependency-free (stdlib ``struct``): the
captures are little-endian pcapng with Ethernet/IPv4/TCP frames carrying
plain-text HTTP/1.1 on the free5GC service-based interface (port 8000), so no
scapy/pyshark/tshark is required and no new dependency is added to the project.

Redistribution: MIT permits redistributing derived rows, but this build keeps
the repository-wide discipline of committing only the manifest and never the
per-request rows. ``generated/`` is gitignored (see ``.gitignore`` in this
directory).

Run with Python 3.11+; no extra install needed beyond the base project.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from horizon_ric.data.lineage import compute_dataset_sha256

DATASET_NAME = "5GAD-2022 (Idaho National Laboratory)"
REPO_URL = "https://github.com/IdahoLabResearch/5GAD"
# Pinned commit. The repository was archived on 2026-06-18, so `main` is frozen,
# but an archived repository can still be deleted or force-pushed by its owner;
# pinning the commit means a moved `main` fails loudly instead of silently
# changing the inputs.
REPO_COMMIT = "d6c3643dafb0683ecac2557e1a5ca29c3b5d7ecb"
REPO_LICENCE = "MIT (Copyright (c) 2022 Battelle Energy Alliance, LLC)"
REPO_DOI = "10.11578/dc.20220811.1"
PAPER_DOI = "10.1109/GCWkshps56602.2022.10008647"
MEDIA_BASE = "https://media.githubusercontent.com/media/IdahoLabResearch/5GAD"

_DOWNLOAD_ATTEMPTS = 4
_SBI_PORT = 8000
_PFCP_PORT = 8805

# PFCP message types carried on N4 (3GPP TS 29.244 Table 7.3-1). Only the types
# actually observed in the pinned captures need names; anything else is emitted
# as "pfcp_type_<n>" rather than being silently dropped.
_PFCP_MESSAGE_TYPES = {
    1: "heartbeat_request",
    2: "heartbeat_response",
    5: "association_setup_request",
    6: "association_setup_response",
    7: "association_update_request",
    8: "association_update_response",
    9: "association_release_request",
    10: "association_release_response",
    12: "node_report_request",
    13: "node_report_response",
    50: "session_establishment_request",
    51: "session_establishment_response",
    52: "session_modification_request",
    53: "session_modification_response",
    54: "session_deletion_request",
    55: "session_deletion_response",
    56: "session_report_request",
    57: "session_report_response",
}

# (label, repo_path, lfs_oid_sha256, lfs_size_bytes, role)
#
# ``lfs_oid_sha256`` and ``lfs_size_bytes`` are copied verbatim from the Git-LFS
# pointer files in the pinned commit (obtained with
# ``GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 <repo>``, which fetches only the
# 1.7 MB pointer tree). The build hard-fails on any mismatch.
#
# ``role``:
#   "attack_only"  - the upstream ``Attacks_*.pcapng``, containing ONLY the
#                    attack packets isolated from the full capture.
#   "attack_full"  - a full capture containing the attack plus the core's
#                    responses; used to recover what the real core actually did.
#   "benign"       - a capture of normal traffic with no attack running.
PINNED_FILES: tuple[tuple[str, str, str, int, str], ...] = (
    (
        "FakeAMFInsert",
        "Attacks/FakeAMFInsert/Attacks_FakeAMFInsert.pcapng",
        "41cca32c909ffaaa80f4b6e78825c94a5c09530f9f21cb78f35cda3a8e8ceaa8",
        991376,
        "attack_only",
    ),
    (
        "FakeAMFInsert_full",
        "Attacks/FakeAMFInsert/allcap_FakeAMFInsert_00001_20220609152541.pcapng",
        "64b7e54fd49916887c1d72c0dc53867befb2c3db8eac83b322e5991bd3964053",
        9287404,
        "attack_full",
    ),
    (
        "randomAMFInsert",
        "Attacks/randomAMFInsert/Attacks_randomAMFInsert.pcapng",
        "25b173646c9a6846325a47b4559164be4fbe2cf1c0c8979b42806507be97828e",
        1098756,
        "attack_only",
    ),
    (
        "FakeAMFDelete",
        "Attacks/FakeAMFDelete/Attacks_FakeAMFDelete.pcapng",
        "1192a62a4f55e0aadaacd447a2b424d269a6b14bb34ff8ff5fb981ecbadabf65",
        12171792,
        # Upstream ships an identical oid for Attacks_ and allcap_ here, i.e.
        # this "attack-only" file is in fact the full capture and does contain
        # the core's responses. Labelled honestly.
        "attack_full",
    ),
    (
        "CrashNRF",
        "Attacks/CrashNRF/Attacks_CrashNRF.pcapng",
        "a81ff70dd5719c81f461304954ed4deb3be0ca54eb9db1a83bc20b4e3a6ff380",
        95984,
        "attack_only",
    ),
    (
        "GetAllNFs",
        "Attacks/GetAllNFs/Attacks_GetAllNFs.pcapng",
        "ea5d5e2d3ed251ef2e09cf7c358c0177657db44c3128c02dbb6516beb1ae75c2",
        95984,
        "attack_only",
    ),
    (
        "GetUserData",
        "Attacks/GetUserData/Attacks_GetUserData.pcapng",
        "7e3ca5ad34c4cba0b6337acab6561d34f397f71e9d5660a68cea33867098f63f",
        110312,
        "attack_only",
    ),
    (
        "AMFLookingForUDM",
        "Attacks/AMFLookingForUDM/Attacks_AMFLookingForUDM.pcapng",
        "e1585016a6604e201c88049e011da7ffc34e12c16f01db3967f5b2863342296a",
        97576,
        "attack_only",
    ),
    (
        "randomDataDump",
        "Attacks/randomDataDump/Attacks_randomDataDump.pcapng",
        "7d57d48d6275ef7668d214c2e8f2100f8b66a8237865fe9c6704967b2d943be2",
        97664,
        "attack_only",
    ),
    (
        "automatedDropWithTimer",
        "Attacks/automatedDropWithTimer/Attacks_automatedDropWithTimer.pcapng",
        "4ee7418e5b9c8d4c35d8f72bce01c105bf25d8ed4123141ecb39ca902c990255",
        45104,
        "attack_only",
    ),
    (
        "automatedRedirectWithTimer",
        "Attacks/automatedRedirectWithTimer/Attacks_automatedRedirectWithTimer.pcapng",
        "b8e6d1054ccd5bf5227cdc96035eac03a35626429affb8fba58b672a87a2ea27",
        58064,
        "attack_only",
    ),
    (
        "Normal-1UE",
        "Normal-1UE/allcap_00001_20220606102554.pcapng",
        "42ff002639d0140d3728254048755bd9561742924a70c0cf2dd758c810ec3cef",
        65564,
        "benign",
    ),
)

# Upstream's own three-way taxonomy (Attacks/README.md).
ATTACK_CLASS = {
    "FakeAMFInsert": "network_reconfiguration",
    "FakeAMFInsert_full": "network_reconfiguration",
    "randomAMFInsert": "network_reconfiguration",
    "FakeAMFDelete": "network_reconfiguration",
    "CrashNRF": "denial_of_service",
    "automatedDropWithTimer": "denial_of_service",
    "automatedRedirectWithTimer": "denial_of_service",
    "GetAllNFs": "reconnaissance",
    "GetUserData": "reconnaissance",
    "AMFLookingForUDM": "reconnaissance",
    "randomDataDump": "reconnaissance",
    "Normal-1UE": "benign",
}

_REQUEST_LINE = re.compile(
    rb"^(GET|PUT|POST|DELETE|PATCH|HEAD|OPTIONS) (\S+) HTTP/1\.[01]\r\n"
)
_STATUS_LINE = re.compile(rb"^HTTP/1\.[01] (\d{3})")
_LOCATION = re.compile(rb"\r\nLocation: ([^\r\n]+)\r\n", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Fetch + pin
# ---------------------------------------------------------------------------
def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _fetch(repo_path: str) -> bytes:
    url = f"{MEDIA_BASE}/{REPO_COMMIT}/{repo_path}"
    last: Exception | None = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        if attempt:
            time.sleep(2**attempt)
            print(f"retrying {repo_path} (attempt {attempt + 1}/{_DOWNLOAD_ATTEMPTS})")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "horizon-ric-dataset-build/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
    raise RuntimeError(
        f"could not fetch {url} after {_DOWNLOAD_ATTEMPTS} attempts: {last!r}. "
        "This is an upstream availability failure, not a reproduction "
        "regression; re-run once the host recovers."
    )


def _resolve(repo_path: str, expected_sha: str, expected_size: int, cache_dir: Path) -> bytes:
    cached = cache_dir / repo_path.replace("/", "__")
    if cached.is_file():
        blob = cached.read_bytes()
    else:
        blob = _fetch(repo_path)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(blob)

    if blob.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{repo_path} came back as a Git-LFS pointer, not the object. The "
            "media host did not resolve the LFS blob."
        )
    actual_sha = _sha256_bytes(blob)
    if actual_sha != expected_sha or len(blob) != expected_size:
        raise RuntimeError(
            f"5GAD object {repo_path} failed its pinned checksum: expected "
            f"sha256={expected_sha} size={expected_size}, got sha256={actual_sha} "
            f"size={len(blob)}"
        )
    return blob


# ---------------------------------------------------------------------------
# Minimal pcapng / Ethernet / IPv4 / TCP / HTTP-1.1 reader (stdlib only)
# ---------------------------------------------------------------------------
def _iter_packets(blob: bytes) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(interface_id, timestamp_raw, frame_bytes)`` for each packet block.

    Handles the little-endian pcapng that 5GAD ships (Section Header Block,
    Interface Description Blocks, Enhanced Packet Blocks). Timestamp resolution
    is read from each IDB's ``if_tsresol`` option (code 9); 5GAD uses 1 ns.
    """
    offset = 0
    total = len(blob)
    while offset + 12 <= total:
        block_type, block_len = struct.unpack_from("<II", blob, offset)
        if block_len < 12 or offset + block_len > total:
            break
        body = blob[offset + 8 : offset + block_len - 4]
        if block_type == 0x00000006:  # Enhanced Packet Block
            iface, ts_hi, ts_lo, captured, _original = struct.unpack_from("<IIIII", body, 0)
            yield iface, (ts_hi << 32) | ts_lo, body[20 : 20 + captured]
        offset += block_len


def _timestamp_resolutions(blob: bytes) -> list[int]:
    """Return the ``if_tsresol`` exponent for each Interface Description Block."""
    resolutions: list[int] = []
    offset = 0
    total = len(blob)
    while offset + 12 <= total:
        block_type, block_len = struct.unpack_from("<II", blob, offset)
        if block_len < 12 or offset + block_len > total:
            break
        if block_type == 0x00000001:  # Interface Description Block
            body = blob[offset + 8 : offset + block_len - 4]
            resolution = 6  # pcapng default: microseconds
            cursor = 8
            while cursor + 4 <= len(body):
                code, length = struct.unpack_from("<HH", body, cursor)
                if code == 0:
                    break
                if code == 9 and length >= 1:
                    resolution = body[cursor + 4]
                cursor += 4 + ((length + 3) // 4) * 4
            resolutions.append(resolution)
        offset += block_len
    return resolutions


def _l4_segment(frame: bytes) -> tuple[int, str, str, int, int, bytes] | None:
    """Return ``(ip_proto, src_ip, dst_ip, src_port, dst_port, payload)``.

    Decodes IPv4 TCP (proto 6) and UDP (proto 17); everything else is ignored.
    """
    if len(frame) < 14:
        return None
    ethertype = struct.unpack_from("!H", frame, 12)[0]
    offset = 14
    if ethertype == 0x8100:  # 802.1Q
        if len(frame) < 18:
            return None
        ethertype = struct.unpack_from("!H", frame, 16)[0]
        offset = 18
    if ethertype != 0x0800:  # IPv4 only
        return None
    if len(frame) < offset + 20:
        return None
    ihl = (frame[offset] & 0x0F) * 4
    proto = frame[offset + 9]
    if proto not in (6, 17):
        return None
    total_length = struct.unpack_from("!H", frame, offset + 2)[0]
    src = ".".join(str(b) for b in frame[offset + 12 : offset + 16])
    dst = ".".join(str(b) for b in frame[offset + 16 : offset + 20])
    l4_at = offset + ihl
    end = min(offset + total_length, len(frame))
    if proto == 6:
        if len(frame) < l4_at + 20:
            return None
        src_port, dst_port = struct.unpack_from("!HH", frame, l4_at)
        data_offset = (frame[l4_at + 12] >> 4) * 4
        return 6, src, dst, src_port, dst_port, frame[l4_at + data_offset : end]
    if len(frame) < l4_at + 8:
        return None
    src_port, dst_port = struct.unpack_from("!HH", frame, l4_at)
    return 17, src, dst, src_port, dst_port, frame[l4_at + 8 : end]


def _pfcp_message(payload: bytes) -> dict[str, Any] | None:
    """Decode a PFCP (N4) header per 3GPP TS 29.244 sec. 7.2.

    Header: flags(1) | message type(1) | length(2) | [SEID(8) if S] | seq(3) | spare(1)
    """
    if len(payload) < 8:
        return None
    flags = payload[0]
    if (flags >> 5) != 1:  # version field must be 1
        return None
    has_seid = bool(flags & 0x01)
    message_type = payload[1]
    length = struct.unpack_from("!H", payload, 2)[0]
    seid = 0
    cursor = 4
    if has_seid:
        if len(payload) < 12:
            return None
        seid = struct.unpack_from("!Q", payload, 4)[0]
        cursor = 12
    if len(payload) < cursor + 3:
        return None
    sequence = int.from_bytes(payload[cursor : cursor + 3], "big")
    return {
        "message_type": message_type,
        "message_name": _PFCP_MESSAGE_TYPES.get(message_type, f"pfcp_type_{message_type}"),
        "seid": seid,
        "sequence": sequence,
        "declared_length": length,
    }


def _http_message(payload: bytes) -> dict[str, Any] | None:
    """Classify a TCP payload as an HTTP/1.1 request or response, if it is one."""
    request = _REQUEST_LINE.match(payload)
    if request is not None:
        head, _, body = payload.partition(b"\r\n\r\n")
        return {
            "kind": "request",
            "method": request.group(1).decode(),
            "path": request.group(2).decode("utf-8", "replace"),
            "header_bytes": len(head) + 4,
            "body": body,
        }
    status = _STATUS_LINE.match(payload)
    if status is not None:
        head, _, body = payload.partition(b"\r\n\r\n")
        location = _LOCATION.search(head + b"\r\n")
        return {
            "kind": "response",
            "status": int(status.group(1)),
            "location": location.group(1).decode("utf-8", "replace") if location else "",
            "header_bytes": len(head) + 4,
            "body": body,
        }
    return None


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
def _nf_summary(body: bytes) -> dict[str, Any]:
    """Structural summary of an NF-profile JSON body, without echoing the body."""
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(obj, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("nfInstanceId", "nfType", "nfStatus"):
        if isinstance(obj.get(key), str):
            summary[key] = obj[key]
    if isinstance(obj.get("ipv4Addresses"), list):
        summary["ipv4Addresses"] = [v for v in obj["ipv4Addresses"] if isinstance(v, str)]
    if isinstance(obj.get("plmnList"), list):
        summary["plmn_count"] = len(obj["plmnList"])
    summary["json_top_level_keys"] = sorted(k for k in obj if isinstance(k, str))
    return summary


def _derive_capture(
    label: str,
    role: str,
    blob: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Derive per-request event rows and forged-NF-profile rows from one capture."""
    resolutions = _timestamp_resolutions(blob)
    events: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    packets = 0
    tcp_packets = 0
    pfcp_messages = 0
    non_http_tcp_payloads = 0
    packet_times: list[int] = []
    responses_by_status: dict[str, int] = {}
    # Correlate a response back to the request on the same 4-tuple (these
    # captures are strictly sequential request/response on loopback, one
    # in-flight request per connection, so ordering is unambiguous).
    pending: dict[tuple[str, str, int, int], int] = {}
    # free5GC's SBI clients send "Expect: 100-continue", so a request's JSON
    # body arrives in a LATER, unframed TCP segment on the same connection
    # (headers -> 100 Continue -> body -> final response). Accumulate those
    # continuation bytes against the request they belong to.
    bodies: dict[int, bytearray] = {}
    first_ts_ns: int | None = None

    def _finalise(index: int) -> None:
        body = bytes(bodies.pop(index, b""))
        if not body:
            return
        row = events[index]
        row["body_bytes"] = len(body)
        row["body_sha256"] = _sha256_bytes(body)
        nf = _nf_summary(body)
        if nf.get("nfType"):
            profiles.append(
                {
                    "capture": row["capture"],
                    "t_ns": row["t_ns"],
                    "method": row["method"],
                    "path": row["path"],
                    "body_bytes": len(body),
                    "body_sha256": row["body_sha256"],
                    # The verbatim captured request body. Needed because the
                    # integrity probe must feed the ACTUAL forged bytes through
                    # the provenance gate, not a reconstruction of them: a
                    # signature check is over bytes, so anything less would make
                    # the probe a simulation. 5GAD is MIT so retaining these is
                    # permitted; generated/ is gitignored regardless.
                    "body_b64": base64.b64encode(body).decode("ascii"),
                    "observed_response_status": row["observed_response_status"],
                    "event_index": index,
                    **nf,
                }
            )

    for iface, ts_raw, frame in _iter_packets(blob):
        packets += 1
        exponent = resolutions[iface] if iface < len(resolutions) else 6
        # pcapng: a tsresol byte with the high bit clear is a power of ten.
        ts_ns = ts_raw * (10 ** (9 - exponent)) if exponent <= 9 else ts_raw
        if first_ts_ns is None or ts_ns < first_ts_ns:
            first_ts_ns = ts_ns
        packet_times.append(ts_ns)
        segment = _l4_segment(frame)
        if segment is None:
            continue
        proto, src, dst, src_port, dst_port, payload = segment
        if not payload:
            continue

        if proto == 17:
            if _PFCP_PORT not in (src_port, dst_port):
                continue
            pfcp = _pfcp_message(payload)
            if pfcp is None:
                non_http_tcp_payloads += 1
                continue
            events.append(
                {
                    "capture": label,
                    "attack_class": ATTACK_CLASS[label],
                    "role": role,
                    "t_ns": ts_ns,
                    "src_ip": src,
                    "dst_ip": dst,
                    "dst_port": dst_port,
                    "protocol": "pfcp",
                    "method": "PFCP",
                    "path": f"pfcp/{pfcp['message_name']}",
                    "sbi_service": "n4-pfcp",
                    "header_bytes": len(payload) - max(0, pfcp["declared_length"] - 4),
                    "body_bytes": len(payload),
                    "body_sha256": _sha256_bytes(payload),
                    "observed_response_status": 0,
                    "pfcp_message_type": pfcp["message_type"],
                    "pfcp_sequence": pfcp["sequence"],
                }
            )
            pfcp_messages += 1
            continue

        tcp_packets += 1
        message = _http_message(payload)
        if message is None:
            # Unframed continuation: the body of a request already seen on this
            # connection, or non-HTTP traffic (GTP-U, NAS, DNS) we do not decode.
            index = pending.get((src, dst, src_port, dst_port))
            if index is None:
                non_http_tcp_payloads += 1
            else:
                bodies.setdefault(index, bytearray()).extend(payload)
            continue

        if message["kind"] == "request":
            key = (src, dst, src_port, dst_port)
            previous = pending.get(key)
            if previous is not None:
                _finalise(previous)
            body = message["body"]
            row = {
                "capture": label,
                "attack_class": ATTACK_CLASS[label],
                "role": role,
                "t_ns": ts_ns,
                "src_ip": src,
                "dst_ip": dst,
                "dst_port": dst_port,
                "protocol": "http",
                "method": message["method"],
                "path": message["path"],
                "sbi_service": message["path"].split("/")[1] if "/" in message["path"] else "",
                "header_bytes": message["header_bytes"],
                "body_bytes": 0,
                "body_sha256": "",
                "observed_response_status": 0,
            }
            events.append(row)
            index = len(events) - 1
            pending[key] = index
            if body:
                bodies.setdefault(index, bytearray()).extend(body)
        else:
            status = message["status"]
            responses_by_status[str(status)] = responses_by_status.get(str(status), 0) + 1
            if status == 100:
                # Interim response: the request is still in flight, its body has
                # not even been sent yet. Do NOT close the correlation here.
                continue
            # Responses travel in the reverse direction of their request.
            index = pending.pop((dst, src, dst_port, src_port), None)
            if index is not None:
                events[index]["observed_response_status"] = status
                _finalise(index)

    for index in list(bodies):
        _finalise(index)

    # Re-base timestamps to the start of the capture: the absolute epoch is
    # retained separately, per-row offsets are what the load replay consumes.
    base = first_ts_ns or 0
    for row in events:
        row["t_offset_ns"] = row.pop("t_ns") - base
    for profile in profiles:
        profile["t_offset_ns"] = profile.pop("t_ns") - base

    events.sort(key=lambda r: (r["t_offset_ns"], r["path"], r["method"]))
    profiles.sort(key=lambda r: (r["t_offset_ns"], r["path"]))

    packet_times.sort()
    packet_gaps = [b - a for a, b in zip(packet_times, packet_times[1:])]
    packet_span_ns = (packet_times[-1] - packet_times[0]) if len(packet_times) > 1 else 0
    request_times = sorted(r["t_offset_ns"] for r in events)
    request_gaps = [b - a for a, b in zip(request_times, request_times[1:])]

    def _gap_stats(gaps: list[int]) -> dict[str, int]:
        if not gaps:
            return {"count": 0, "min_ns": 0, "median_ns": 0, "p95_ns": 0, "max_ns": 0}
        ordered = sorted(gaps)
        return {
            "count": len(ordered),
            "min_ns": ordered[0],
            "median_ns": ordered[len(ordered) // 2],
            "p95_ns": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "max_ns": ordered[-1],
        }

    stats = {
        "capture": label,
        "role": role,
        "attack_class": ATTACK_CLASS[label],
        "packets": packets,
        "tcp_packets": tcp_packets,
        "pfcp_messages": pfcp_messages,
        "http_requests": sum(1 for r in events if r.get("protocol") == "http"),
        "control_plane_events": len(events),
        "sbi_requests": sum(1 for r in events if r["dst_port"] == _SBI_PORT),
        "nf_profile_writes": len(profiles),
        "capture_epoch_ns": base,
        "span_ns": max(request_times, default=0),
        "packet_span_ns": packet_span_ns,
        "packet_interarrival_ns": _gap_stats(packet_gaps),
        "event_interarrival_ns": _gap_stats(request_gaps),
        "responses_by_status": dict(sorted(responses_by_status.items())),
        "non_decoded_payloads": non_http_tcp_payloads,
        "requests_with_observed_response": sum(
            1 for r in events if r["observed_response_status"]
        ),
        "accepted_2xx": sum(1 for r in events if 200 <= r["observed_response_status"] < 300),
    }
    return events, profiles, stats


def build(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_events: list[dict[str, Any]] = []
    all_profiles: list[dict[str, Any]] = []
    per_capture: list[dict[str, Any]] = []
    source_bytes = 0

    for label, repo_path, sha, size, role in PINNED_FILES:
        blob = _resolve(repo_path, sha, size, cache_dir)
        source_bytes += len(blob)
        events, profiles, stats = _derive_capture(label, role, blob)
        stats["repo_path"] = repo_path
        stats["source_sha256"] = sha
        stats["source_bytes"] = size
        per_capture.append(stats)
        all_events.extend(events)
        all_profiles.extend(profiles)
        print(
            f"{label:<28} packets={stats['packets']:<7} "
            f"events={stats['control_plane_events']:<6} "
            f"nf_writes={stats['nf_profile_writes']:<5} "
            f"accepted_2xx={stats['accepted_2xx']}"
        )

    if not all_events:
        raise RuntimeError("no HTTP requests derived — the parser or the pins are wrong")

    # A single SHA-256 over the exact source bytes of every pinned object, in
    # pinned order. This is the archive-level pin the manifest carries.
    digest = hashlib.sha256()
    for _label, _path, sha, size, _role in PINNED_FILES:
        digest.update(f"{sha}:{size}\n".encode())
    source_archive_sha256 = digest.hexdigest()

    def _write(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    _write(args.events, all_events)
    _write(args.profiles, all_profiles)

    manifest = {
        "dataset": DATASET_NAME,
        "data_kind": (
            "over-the-wire packet capture of attacks actually executed against a "
            "real free5GC 5G standalone core (UERANSIM RAN) on a physical test "
            "bench; not simulated, not synthesised"
        ),
        "repo_url": REPO_URL,
        "repo_commit": REPO_COMMIT,
        "repo_doi": REPO_DOI,
        "paper_doi": PAPER_DOI,
        "media_base_url": MEDIA_BASE,
        "licensing": {
            "source_repository": REPO_LICENCE,
            "redistribution_permitted": True,
            "attribution_required": (
                "C. Coldwell et al., 'Machine Learning 5G Attack Detection in "
                "Programmable Logic', 2022 IEEE Globecom Workshops, "
                "doi 10.1109/GCWkshps56602.2022.10008647; dataset "
                "doi 10.11578/dc.20220811.1; MIT, Copyright (c) 2022 Battelle "
                "Energy Alliance, LLC."
            ),
            "note": (
                "MIT permits redistributing the derived rows. This build still "
                "does not commit them, keeping the repository-wide "
                "derived-features-only discipline."
            ),
        },
        "ci_reproducible": True,
        "ci_reproduction_note": (
            "Non-interactive. Objects are plain HTTPS GETs against "
            "media.githubusercontent.com at a pinned commit; no login, token or "
            "click-through. Total fetch is 24.1 MB out of the repository's "
            "35.46 GB."
        ),
        "source_archive_sha256": source_archive_sha256,
        "source_archive_sha256_scope": (
            "sha256 over '<lfs_oid>:<size>\\n' for every pinned object in "
            "PINNED_FILES order. Each object is additionally verified against "
            "its own upstream-published Git-LFS oid before parsing."
        ),
        "source_bytes_total": source_bytes,
        "events_sha256": compute_dataset_sha256(args.events),
        "features_sha256": compute_dataset_sha256(args.profiles),
        "features_sha256_scope": (
            "features_sha256 covers the derived NF-profile write rows "
            "(nf_profile_writes.jsonl); events_sha256 covers the per-request "
            "control-plane event rows (control_plane_events.jsonl). Both are "
            "exact JSONL bytes and are deterministic given the pinned inputs "
            "(integer timestamps, no floating point)."
        ),
        "features_committed": False,
        "transform": {
            "reader": "stdlib pcapng/Ethernet/IPv4/TCP/HTTP-1.1, no external deps",
            "protocol_observed": "plain-text HTTP/1.1 over TCP on the free5GC SBI",
            "sbi_port": _SBI_PORT,
            "timestamps": "integer nanoseconds, re-based to each capture's first packet",
            "bodies": (
                "control_plane_events.jsonl stores only sha256 and length. "
                "nf_profile_writes.jsonl additionally stores the verbatim "
                "captured NF-profile body (base64) because a signature check is "
                "over bytes: the integrity probe must feed the actual forged "
                "bytes through the provenance gate. Neither file is committed."
            ),
            "response_correlation": (
                "response matched to the last unanswered request on the reversed "
                "4-tuple; 100-Continue ignored"
            ),
        },
        "captures": per_capture,
        "totals": {
            "captures": len(PINNED_FILES),
            "control_plane_events": len(all_events),
            "http_requests": sum(c["http_requests"] for c in per_capture),
            "pfcp_messages": sum(c["pfcp_messages"] for c in per_capture),
            "nf_profile_writes": len(all_profiles),
            "attack_classes": sorted({c["attack_class"] for c in per_capture}),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/horizon-5gad"))
    parser.add_argument(
        "--events", type=Path, default=here / "generated" / "control_plane_events.jsonl"
    )
    parser.add_argument(
        "--profiles", type=Path, default=here / "generated" / "nf_profile_writes.jsonl"
    )
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))
    print(f"manifest written to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
