"""Horizon-RIC ↔ NVIDIA Cloud Functions.

Two directions, one contract:

* :mod:`horizon_nvcf.agent` — an NVCF-hosted inference function presented as
  a governed proposer, whose identity is its *immutable version*, and whose
  every output is graded by the Shield before it can reach a radio.
* :mod:`horizon_nvcf.service` — Horizon's own decision plane presented over
  NVCF's invocation contract, so the Shield deploys on the same self-managed
  GPU cluster as the agents it governs.

Every path, header and status code comes from
:mod:`horizon_nvcf.protocol`, which reads them out of NVIDIA's published
OpenAPI document rather than from anyone's memory.
"""

from horizon_nvcf.protocol import (
    Disposition,
    InvocationOutcome,
    NvcfProtocolError,
    classify,
    contract,
    header,
    invoke_path,
    poll_path,
)

__all__ = [
    "Disposition",
    "InvocationOutcome",
    "NvcfProtocolError",
    "classify",
    "contract",
    "header",
    "invoke_path",
    "poll_path",
]
