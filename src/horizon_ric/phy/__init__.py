"""Real (torch-free) neural-PHY layer for adversarial evaluation of the Shield.

A numpy neural receiver (hand-derived backprop), a genuine PGD adversarial
attack, and QAM constellations + the classical ML demapper used as the Shield's
certified fallback. This exists so the Decision Safety Shield can be evaluated
against an attack it was *not* hand-coded against (closing the "the benchmark
only tests violations the Shield was written to catch" critique).
"""

from horizon_ric.phy.channel import (
    classical_equalize_demap,
    fading_dataset,
    lmmse_equalize,
    zf_equalize,
)
from horizon_ric.phy.constellation import (
    awgn,
    classical_ml_demap,
    make_dataset,
    modulate,
    qam_constellation,
    symbols_to_features,
)
from horizon_ric.phy.neural_rx import NeuralReceiver
from horizon_ric.phy.pgd import pgd_attack

__all__ = [
    "NeuralReceiver",
    "pgd_attack",
    "qam_constellation",
    "modulate",
    "symbols_to_features",
    "classical_ml_demap",
    "awgn",
    "make_dataset",
    "fading_dataset",
    "classical_equalize_demap",
    "lmmse_equalize",
    "zf_equalize",
]
