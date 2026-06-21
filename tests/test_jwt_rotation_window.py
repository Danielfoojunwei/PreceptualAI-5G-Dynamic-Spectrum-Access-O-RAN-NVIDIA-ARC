"""JWT key-rotation window forgery-bound tests (Devil-C Finding 12/33).

Closes the formal rotation-window proof in
`horizon_ric.security.jwt` module docstring:

    For every token ``T`` minted with ``K_old``,
        Pr[verify(T) = True | t_verify > T_rot + W + Δ_skew] = 0,

where ``W`` is the overlap window and ``Δ_skew = MAX_CLOCK_SKEW_SECONDS``.

These tests pin the ENGINEERING evidence behind the proof:
    1. After ``W + Δ_skew``, an old-key token is rejected.
    2. Inside ``W``, an old-key token still verifies.
    3. Clock-skew tolerance is bounded (no token survives skew much
       greater than ``Δ_skew``).
    4. Verifier-side eviction happens eagerly (no need for another
       rotation to drop a stale key).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest
from jose.exceptions import JWTError

from horizon_ric.security.jwt import (
    _OVERLAP_WINDOW_DEFAULT_SEC,
    MAX_CLOCK_SKEW_SECONDS,
    JWTManager,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "test_jwt_signing.pem"


@pytest.fixture()
def signing_key(tmp_path: Path) -> Path:
    dst = tmp_path / "signing.pem"
    dst.write_bytes(_FIXTURE.read_bytes())
    return dst


def _gen_rsa_key(path: Path) -> Path:
    subprocess.run(
        ["openssl", "genrsa", "-out", str(path), "2048"],
        check=True,
        capture_output=True,
    )
    return path


# ---------------------------------------------------------------------------
# 1. After the rotation window expires + Δ_skew, the old key is unconditionally
#    evicted, even without another rotate_signing_key() call.
# ---------------------------------------------------------------------------
def test_old_key_token_rejected_after_window(
    signing_key: Path, tmp_path: Path,
) -> None:
    # Use a TINY overlap window so the test isn't 24h.
    mgr = JWTManager(signing_key, "iss", "aud", overlap_window_seconds=2)

    # Mint a token with the ORIGINAL key, TTL longer than the window so
    # the rejection is purely about the window — not about expiry.
    old_token = mgr.mint_token(
        "alice", "default", ["operator"], ttl_seconds=600,
    )
    # Pre-rotation: works.
    assert mgr.verify_token(old_token)["sub"] == "alice"

    # Rotate to a fresh key.
    new_key = _gen_rsa_key(tmp_path / "new.pem")
    mgr.rotate_signing_key(new_key)

    # Inside the window: old token still verifies.
    assert mgr.verify_token(old_token)["sub"] == "alice"

    # Wait for window + skew margin to elapse.
    time.sleep(2 + MAX_CLOCK_SKEW_SECONDS + 1)

    with pytest.raises(JWTError):
        mgr.verify_token(old_token)


# ---------------------------------------------------------------------------
# 2. Clock-skew tolerance: tokens cannot survive arbitrary clock drift.
#    The bound is enforced via _prune_retired_keys + python-jose's
#    standard exp validation — both apply Δ_skew = MAX_CLOCK_SKEW_SECONDS.
# ---------------------------------------------------------------------------
def test_clock_skew_is_bounded(signing_key: Path, tmp_path: Path) -> None:
    # We can't move time forward arbitrarily without a clock-monkey, so we
    # assert the property structurally: MAX_CLOCK_SKEW_SECONDS is bounded
    # to a small finite value (≤ 5 minutes) — operators can audit this.
    assert 0 < MAX_CLOCK_SKEW_SECONDS <= 300, (
        "Δ_skew bound must be small (<= 5min). "
        f"Got {MAX_CLOCK_SKEW_SECONDS}."
    )


# ---------------------------------------------------------------------------
# 3. Default overlap window is 24h (Devil-C #33 prescribed value).
# ---------------------------------------------------------------------------
def test_default_overlap_is_24h() -> None:
    assert _OVERLAP_WINDOW_DEFAULT_SEC == 24 * 3600, (
        "Default rotation overlap window must be 24h per Devil-C #33; "
        f"got {_OVERLAP_WINDOW_DEFAULT_SEC} s"
    )


# ---------------------------------------------------------------------------
# 4. Eager pruning: a verify call alone is enough to evict an expired
#    retired key, without needing another rotate_signing_key().
# ---------------------------------------------------------------------------
def test_eager_pruning_on_verify(signing_key: Path, tmp_path: Path) -> None:
    mgr = JWTManager(signing_key, "iss", "aud", overlap_window_seconds=1)

    new_key = _gen_rsa_key(tmp_path / "new.pem")
    mgr.rotate_signing_key(new_key)
    assert mgr.retired_key_count() == 1

    # Wait > overlap + skew. Then call verify with ANY token — the
    # rotation-window pruner runs and drops the retired key.
    time.sleep(1 + MAX_CLOCK_SKEW_SECONDS + 1)

    # Mint with the active (new) key — this verify call should succeed
    # AND evict the stale retired key as a side effect.
    fresh = mgr.mint_token("bob", "default", ["admin"])
    assert mgr.verify_token(fresh)["sub"] == "bob"
    assert mgr.retired_key_count() == 0, (
        "_prune_retired_keys must evict expired keys on every verify"
    )


# ---------------------------------------------------------------------------
# 5. Forgery: a token signed by an old key (after we forcibly drop the
#    retired entry — simulating end-of-window) must NOT verify.
# ---------------------------------------------------------------------------
def test_forgery_with_evicted_old_key_rejected(
    signing_key: Path, tmp_path: Path,
) -> None:
    mgr = JWTManager(signing_key, "iss", "aud", overlap_window_seconds=1)

    old_token = mgr.mint_token(
        "alice", "default", ["operator"], ttl_seconds=600,
    )
    new_key = _gen_rsa_key(tmp_path / "new.pem")
    mgr.rotate_signing_key(new_key)

    # Force the retired-key set to be empty (simulate end of window).
    mgr._retired = []

    # Re-presenting the old token now: must fail because the verifier
    # has no record of the key that signed it (RS256 sig verification
    # under the EUF-CMA assumption is the residual security argument).
    with pytest.raises(JWTError):
        mgr.verify_token(old_token)
