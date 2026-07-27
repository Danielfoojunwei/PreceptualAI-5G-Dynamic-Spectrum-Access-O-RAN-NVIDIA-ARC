#!/usr/bin/env python3
"""AF_PACKET sniffer for the shimmed E2AP-over-UDP loopback traffic.

Captures every UDP datagram to/from ports 36421 (E2AP) and 36422 (E42)
on lo and appends one JSON line per datagram:
{"ts": ..., "src_port": ..., "dst_port": ..., "payload_hex": ...}.
The payloads ARE the E2AP PDUs FlexRIC put on the wire (the shim only
substitutes SCTP with UDP; it never touches the bytes).
"""
import json
import socket
import struct
import sys
import time

PORTS = {36421, 36422}
out_path = sys.argv[1] if len(sys.argv) > 1 else "e2ap_capture.jsonl"

s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
s.bind(("lo", 0))
n = 0
with open(out_path, "a") as out:
    while True:
        frame = s.recv(65535)
        if len(frame) < 34:
            continue
        eth_proto = struct.unpack("!H", frame[12:14])[0]
        if eth_proto != 0x0800:
            continue
        ihl = (frame[14] & 0x0F) * 4
        proto = frame[14 + 9]
        if proto != 17:  # UDP
            continue
        udp = frame[14 + ihl:]
        if len(udp) < 8:
            continue
        sport, dport, ulen = struct.unpack("!HHH", udp[:6])
        if sport not in PORTS and dport not in PORTS:
            continue
        payload = udp[8:ulen]
        out.write(
            json.dumps(
                {
                    "ts": time.time(),
                    "src_port": sport,
                    "dst_port": dport,
                    "payload_hex": payload.hex(),
                }
            )
            + "\n"
        )
        out.flush()
        n += 1
        print(f"captured {n}: {sport}->{dport} {len(payload)} B", file=sys.stderr)
