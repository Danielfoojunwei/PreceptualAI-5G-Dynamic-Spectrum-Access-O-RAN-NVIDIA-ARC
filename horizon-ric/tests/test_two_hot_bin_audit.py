"""Two-hot symlog bin spacing audit (Devil-C Finding 10/34).

DreamerV3 (Hafner et al. 2024 §3.2) prescribes:
    * Bins placed at *symlog* targets, equispaced in symlog-space.
    * num_bins = 255, symlog range covering raw values up to ~exp(20)−1.
    * Reconstruction error of any value `t ∈ [bin_low, bin_high]`
      via two-hot encode → softmax-decode is bounded by half the
      adjacent-bin width in bin-space.

This file checks our `make_bins`, `two_hot`, `two_hot_decode` against
those three properties. It closes Devil-C Finding 34 with empirical
evidence the audit is structurally sound.
"""

from __future__ import annotations

import torch

from horizon_ric.heads._two_hot import (
    make_bins,
    symexp,
    symlog,
    two_hot,
    two_hot_decode,
)


def test_bin_spacing_uniform_in_bin_space() -> None:
    """DreamerV3 contract: bin centres equispaced in *bin-space* (the
    space the caller chose — symlog or raw — they are linspace).
    """
    bins = make_bins(num_bins=255, low=-20.0, high=20.0)
    diffs = (bins[1:] - bins[:-1])
    # Uniform spacing in bin-space: every gap is ≈ (40 / 254).
    expected = (20.0 - (-20.0)) / (255 - 1)
    assert torch.allclose(
        diffs, torch.full_like(diffs, expected), atol=1e-6
    )
    # And the spacing matches the DreamerV3 paper's prescription that the
    # bin grid spans [-20, 20] symlog-units — i.e. raw values up to
    # ±(exp(20)-1). Verify that the symexp-mapped extremes are huge.
    assert symexp(bins[-1]).item() > 1e6
    assert symexp(bins[0]).item() < -1e6


def test_reconstruction_error_bounded_by_bin_width() -> None:
    """For any target `t` inside the bin range, encoding + softmax(log p)
    decoding must reconstruct `t` to within HALF the local bin width
    (the worst-case linear-interpolation residual of the two-hot scheme).
    """
    bins = make_bins(num_bins=255, low=-20.0, high=20.0)
    # Pick raw targets covering many decades.
    targets_raw = torch.tensor(
        [-1000.0, -100.0, -1.0, -0.1, 0.0, 0.1, 1.0, 100.0, 1000.0]
    )
    dist = two_hot(targets_raw, bins, apply_symlog=True)
    # Use log of the (already-near-one-hot) distribution as logits so the
    # softmax inside `two_hot_decode` reproduces the same distribution.
    logits = torch.log(dist + 1e-12)
    decoded = two_hot_decode(logits, bins, apply_symlog=True)

    # Bin width in symlog-space — the max bin spacing (constant for our
    # linspace bins). Reconstruction error in bin-space ≤ half the width.
    bin_width = float(bins[1] - bins[0])
    # In symlog-space the decode is exact within bin_width/2; map the
    # bound back to raw space using the local Lipschitz constant of
    # symexp at the target. symexp'(s) = exp(|s|), bounded by symexp at
    # the larger endpoint.
    err_symlog = (symlog(decoded) - symlog(targets_raw)).abs()
    assert (err_symlog <= bin_width / 2 + 1e-5).all(), (
        f"symlog-space reconstruction error {err_symlog} exceeds half-width"
        f" {bin_width / 2}"
    )


def test_logit_space_decoder_unbiased_for_unimodal() -> None:
    """For a SHARP unimodal target distribution (delta-like), the
    expectation-of-bin decoder is unbiased: `Σ p·b ≈ true_centre`.
    This audits the use case in `sla_risk.py` where bins live in logit
    space and `apply_symlog=False`.
    """
    bins = make_bins(num_bins=51, low=-7.0, high=7.0)
    centres = torch.tensor([-4.0, -1.0, 0.0, 1.5, 5.0])
    dist = two_hot(centres, bins, apply_symlog=False)
    logits = torch.log(dist + 1e-12)
    decoded = two_hot_decode(logits, bins, apply_symlog=False)
    # Bound: half a bin width = 14.0/50 / 2 = 0.14.
    bin_width = float(bins[1] - bins[0])
    err = (decoded - centres).abs()
    assert (err <= bin_width / 2 + 1e-5).all(), (
        f"linear-bin reconstruction err {err} > half-width {bin_width / 2}"
    )
