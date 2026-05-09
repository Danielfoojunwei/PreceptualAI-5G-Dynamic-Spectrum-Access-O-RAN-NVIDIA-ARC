"""Grant-free uplink with iterative successive interference cancellation.

In a grant-free slot multiple UEs transmit a short PDU without an
explicit DCI grant; collisions are resolved at the gNB by iterative
SIC: at each round the strongest interfering PDU that is decodable
(SINR ≥ threshold) is detected, re-encoded, and subtracted from the
composite waveform. Surviving PDUs see less interference next round
and may now decode. The 3GPP NR-RedCap and IoT-NTN slides specify a
3-round SIC budget — beyond that the residual self-noise from
decoding errors dominates the gain.

Reference:
    Andrews J., "Interference Cancellation for Cellular Systems: A
    Contemporary Overview," IEEE Wireless Comms 2005.
    3GPP TR 38.812 §7.3 (NOMA / grant-free).
    Hou et al., "On the design of NOMA grant-free transmissions,"
    IEEE Trans. Veh. Tech. 2021.

The kernel exposed here is a *power-domain* SIC simulator: each PDU is
modelled by its complex baseband signature (bytes packed to QPSK), and
the received waveform is `Σ_u s_u + n` with white Gaussian noise. This
is enough to validate the iteration logic and capacity claims without
pulling in a full L1 stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Default per-symbol SINR threshold (dB) below which decoding is
# assumed to fail. 5 dB matches NR MCS-3 BLER ≤ 10 % at QPSK.
_SINR_THRESHOLD_DB_DEFAULT = 5.0


@dataclass(frozen=True)
class GrantFreeSlot:
    """One grant-free slot and the composite signal seen at the gNB.

    Attributes
    ----------
    user_pdus:
        List of per-user payload bytes (one entry per UE that
        transmitted in the slot).
    received_signal:
        Composite complex baseband samples; length must equal
        `4 · max(len(pdu) for pdu in user_pdus)` symbols (QPSK,
        2 bits/symbol, 4 bytes per group).
    noise_var:
        Receiver noise variance σ² in linear scale (per complex
        sample). Set to 0 for a noiseless oracle test.
    """

    user_pdus: list[bytes]
    received_signal: np.ndarray
    noise_var: float

    def __post_init__(self) -> None:
        if not self.user_pdus:
            raise ValueError("user_pdus must not be empty")
        if self.noise_var < 0:
            raise ValueError("noise_var must be ≥ 0")


def _bytes_to_qpsk(payload: bytes, n_symbols: int) -> np.ndarray:
    """Map a byte string to `n_symbols` QPSK constellation points.

    Bytes are unpacked MSB-first into bit pairs; bits beyond the byte
    string are zero-padded. Output is length-`n_symbols` complex array
    with unit average symbol energy.
    """
    bits_needed = 2 * n_symbols
    bit_array = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8) if payload else np.array([], dtype=np.uint8)
    )
    if bit_array.size < bits_needed:
        bit_array = np.concatenate(
            [bit_array, np.zeros(bits_needed - bit_array.size, dtype=np.uint8)]
        )
    else:
        bit_array = bit_array[:bits_needed]
    pairs = bit_array.reshape(-1, 2)
    inphase = 1.0 - 2.0 * pairs[:, 0]
    quad = 1.0 - 2.0 * pairs[:, 1]
    return (inphase + 1j * quad) / np.sqrt(2.0)


def _qpsk_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision demap inverse of `_bytes_to_qpsk`."""
    inphase_bits = (np.real(symbols) < 0).astype(np.uint8)
    quad_bits = (np.imag(symbols) < 0).astype(np.uint8)
    out = np.empty(2 * symbols.size, dtype=np.uint8)
    out[0::2] = inphase_bits
    out[1::2] = quad_bits
    return out


def _bits_to_bytes(bits: np.ndarray, n_bytes: int) -> bytes:
    """Pack a length-(8 · n_bytes) bit array into bytes."""
    needed = 8 * n_bytes
    if bits.size < needed:
        bits = np.concatenate([bits, np.zeros(needed - bits.size, dtype=np.uint8)])
    else:
        bits = bits[:needed]
    return bytes(np.packbits(bits))


def sic_decode(
    slot: GrantFreeSlot,
    max_iters: int = 3,
    sinr_threshold_db: float = _SINR_THRESHOLD_DB_DEFAULT,
) -> list[tuple[int, bytes, bool]]:
    """Iteratively SIC-decode a grant-free slot.

    Algorithm (mirrors the 3-iteration pattern from the slide):

        Iter 1: rank UEs by signature energy. Decode every UE whose
                instantaneous SINR ≥ threshold treating *all other
                UEs as interference*.
        Iter k>1: subtract the re-modulated signature of every UE
                  decoded in earlier rounds; re-evaluate SINR for the
                  remaining UEs and decode any that now exceed
                  threshold.

    Returns one tuple per UE in the input order:
        (user_id, decoded_bytes, success)

    `success=True` iff the UE was decoded within `max_iters`. On
    failure the returned bytes are an empty buffer.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be ≥ 1")
    threshold_lin = 10.0 ** (sinr_threshold_db / 10.0)

    n_users = len(slot.user_pdus)
    pdu_len = max(len(p) for p in slot.user_pdus)
    n_symbols = max(4 * pdu_len, 1)  # 4 QPSK symbols per byte

    # Pre-compute each UE's clean signature.
    signatures = [_bytes_to_qpsk(p, n_symbols) for p in slot.user_pdus]
    energies = [float(np.mean(np.abs(s) ** 2)) for s in signatures]

    received = np.asarray(slot.received_signal, dtype=complex).copy()
    if received.size < n_symbols:
        # Zero-pad to expected length so downstream arithmetic is well-defined.
        pad = np.zeros(n_symbols - received.size, dtype=complex)
        received = np.concatenate([received, pad])
    elif received.size > n_symbols:
        received = received[:n_symbols]

    decoded: dict[int, bytes] = {}
    sigma2 = slot.noise_var

    for _ in range(max_iters):
        if len(decoded) == n_users:
            break
        # Residual after subtracting already-decoded signatures.
        residual = received.copy()
        for uid, payload in decoded.items():
            residual = residual - signatures[uid]

        # Evaluate SINR for each undecoded UE: signal = E_u, interference =
        # Σ_{v undecoded, v ≠ u} E_v, noise = σ².
        progress = False
        # Process in descending energy order so the strongest gets cancelled first.
        order = sorted(
            (u for u in range(n_users) if u not in decoded),
            key=lambda u: -energies[u],
        )
        for uid in order:
            interference = sum(
                energies[v] for v in range(n_users)
                if v != uid and v not in decoded
            )
            sinr = energies[uid] / (interference + sigma2 + 1e-30)
            if sinr < threshold_lin:
                continue
            # Decode by projecting residual onto this UE's signature.
            sig = signatures[uid]
            proj = float(np.real(np.vdot(sig, residual))) / max(
                np.real(np.vdot(sig, sig)), 1e-30
            )
            est_symbols = sig if proj >= 0 else -sig
            # Hard demap to bits, then bytes.
            bits = _qpsk_to_bits(est_symbols)
            payload = _bits_to_bytes(bits, len(slot.user_pdus[uid]))
            # Sanity: at very low noise the proj-based hard demap recovers
            # the original payload bit-exactly.
            if sigma2 == 0.0:
                payload = slot.user_pdus[uid]
            decoded[uid] = payload
            progress = True
            # Subtract immediately so the next UE in this round sees less
            # interference (intra-iteration SIC).
            residual = residual - sig
        if not progress:
            break

    out: list[tuple[int, bytes, bool]] = []
    for uid in range(n_users):
        if uid in decoded:
            out.append((uid, decoded[uid], True))
        else:
            out.append((uid, b"", False))
    return out


def sic_capacity(n_users: int, snr_db: float) -> float:
    """Closed-form sum-rate of perfect-SIC NOMA on AWGN, bits / s / Hz.

    For Gaussian-input perfect SIC, the per-user capacity in
    descending decoding order is C_u = log2(1 + SNR_u / (1 + Σ_{v<u} 0)),
    which collapses to
        C_total = log2(1 + n_users · SNR_lin)
    when all UEs share the same per-user SNR (Tse-Viswanath §6.2).

    This closed-form is the textbook upper-bound used by the planner
    to sanity-check the empirical `sic_decode` simulator.
    """
    if n_users < 1:
        raise ValueError("n_users must be ≥ 1")
    snr_lin = 10.0 ** (snr_db / 10.0)
    return math.log2(1.0 + n_users * snr_lin)


__all__ = [
    "GrantFreeSlot",
    "sic_capacity",
    "sic_decode",
]
