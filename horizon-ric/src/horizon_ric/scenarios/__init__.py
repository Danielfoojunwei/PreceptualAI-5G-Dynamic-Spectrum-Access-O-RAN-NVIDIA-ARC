"""Scenario playbooks: a generic NTN + AI-RAN framework plus thin
vertical-specific examples (maritime, rural, disaster, enterprise).

The generic framework is the default; the vertical playbooks are
~30-LOC subclasses that adjust defaults and metadata.
"""

from horizon_ric.scenarios.disaster import DisasterRecoveryGenerator
from horizon_ric.scenarios.enterprise import EnterpriseAIRANGenerator
from horizon_ric.scenarios.maritime import (
    MaritimeScenarioConfig,
    MaritimeSyntheticGenerator,
    MaritimeWindow,
)
from horizon_ric.scenarios.ntn_air_ran import (
    NTNAIRANConfig,
    NTNAIRANSyntheticGenerator,
    NTNAIRANWindow,
)
from horizon_ric.scenarios.rural import RuralCoverageGenerator

__all__ = [
    "NTNAIRANConfig",
    "NTNAIRANSyntheticGenerator",
    "NTNAIRANWindow",
    "MaritimeScenarioConfig",
    "MaritimeSyntheticGenerator",
    "MaritimeWindow",
    "RuralCoverageGenerator",
    "DisasterRecoveryGenerator",
    "EnterpriseAIRANGenerator",
]
