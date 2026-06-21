"""Model provenance — cryptographic authenticity for promoted model artefacts.

Complements the content-addressed artefact vault (integrity) with a signature
over (weights ‖ training-manifest) (authenticity), closing the
"byte-consistent but untrusted/backdoored model" gap.
"""

from horizon_ric.provenance.signing import (
    ModelProvenance,
    ProvenanceError,
    manifest_digest,
    sha256_hex,
    sign_model,
    verify_model,
    verify_or_raise,
)

__all__ = [
    "ModelProvenance",
    "ProvenanceError",
    "manifest_digest",
    "sha256_hex",
    "sign_model",
    "verify_model",
    "verify_or_raise",
]
