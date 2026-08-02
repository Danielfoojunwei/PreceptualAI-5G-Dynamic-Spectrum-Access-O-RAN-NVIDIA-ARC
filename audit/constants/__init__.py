"""Gate-bearing constant inventory and swap harness — the mechanism of gate G2.

Gate **G2** is the reproducibility claim: *every published gate reproduces
without retuning any constant*. That claim is only meaningful if two things are
true and checkable:

1. there exists an explicit, source-derived list of the numeric/threshold
   constants whose values decide whether a published gate passes — otherwise
   "no constant was retuned" is unfalsifiable because nothing enumerates the
   constants; and
2. each such constant is one a gate is genuinely *sensitive* to — otherwise "not
   retuned" is vacuous for constants no gate can see.

This package supplies both. :mod:`audit.constants.inventory` regenerates
``inventory.json`` from source by AST (same drift-guard pattern as OCUDU's
``extract_catalogue``); :mod:`audit.constants.swap_harness` perturbs a constant
and shows the dependent gate/check flips.
"""

from __future__ import annotations

from audit.constants.inventory import (
    CONSTANT_SPECS,
    ConstantRecord,
    generate_inventory,
    inventory_json_text,
    live_value,
    load_committed_inventory,
    resolve_record,
)

__all__ = [
    "CONSTANT_SPECS",
    "ConstantRecord",
    "generate_inventory",
    "inventory_json_text",
    "live_value",
    "load_committed_inventory",
    "resolve_record",
]
