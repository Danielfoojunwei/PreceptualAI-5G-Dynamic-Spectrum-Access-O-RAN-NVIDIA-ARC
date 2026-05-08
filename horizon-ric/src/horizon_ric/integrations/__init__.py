"""Vendor integration shims for PreceptualAI.

This package exposes vendor-specific clients that prove PreceptualAI's
payloads conform to the published interfaces of major SMO / RAN
platforms — Ericsson EIAP, Nokia MantaRay, NVIDIA Aerial. The A1
dialects themselves live in horizon_ric.rapp.a1_adapter.A1AdapterConfig
because the policy surface is shared across vendors with only
URL-shape and field-name variation; this package houses the
non-A1 management plane (model registry, PM KPIs, billing meter,
multi-tenant export) that each vendor exposes separately.
"""

from horizon_ric.integrations.nvidia_arc import ARCClient
from horizon_ric.integrations.raas import RaaSEndpoint

__all__ = ["ARCClient", "RaaSEndpoint"]
