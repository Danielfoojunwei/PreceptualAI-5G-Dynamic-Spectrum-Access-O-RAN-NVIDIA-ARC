"""Data layer — training-data lineage.

Only the security-relevant adapter remains: :mod:`horizon_ric.data.lineage`
records the provenance and GDPR lawful basis of the data a model was trained on,
which the provenance + threat-model layers reference (data-poisoning / AI
supply-chain). Dataset/connector adapters live in :mod:`horizon_ric.io`.
"""

from horizon_ric.data.lineage import *  # noqa: F401,F403
