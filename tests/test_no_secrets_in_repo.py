"""CI gate: fail the build if secrets are committed to the repo.

Walks the working tree (skipping `.venv`, `.git`, build caches) and
fails on:

  * any `*.key`, `*.pem`, `*.p12`, `*.pfx` file not in the allowlist;
  * any text file whose first 4 KiB matches a private-key PEM banner
    (BEGIN ... PRIVATE KEY) regardless of extension;
  * any text file containing a high-entropy token that matches one of
    the well-known cloud-credential patterns (AWS access key id,
    GitHub PAT, GCP service-account marker).

The allowlist is intentionally small. Every entry has a comment
justifying why the file is safe to commit.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that the walk should never enter.
SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv-edge",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "checkpoints",
    "site-packages",
}

# Files that are allowed even though they have a key-like extension.
# Each entry MUST be a concrete reason this file is not a real secret.
ALLOWLISTED_KEY_FILES: dict[str, str] = {
    # JWT signing key used ONLY by tests/test_jwt.py and tests/test_auth_a1ei_o1.py
    # against in-process FastAPI test clients. Never wired into any deployed
    # service; never present in the production container image (the
    # `tests/` directory is excluded by `deploy/docker/Dockerfile`).
    "tests/fixtures/test_jwt_signing.pem": (
        "Test fixture — signs JWTs against the in-process test client only. "
        "Excluded from the production image."
    ),
}

# Suffixes that trigger the strict-allowlist check.
SECRET_SUFFIXES = (".key", ".pem", ".p12", ".pfx")

# PEM banners that indicate private key material regardless of file name.
PRIVATE_KEY_BANNER_RE = re.compile(
    rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP |)PRIVATE KEY-----"
)

# High-confidence cloud-credential patterns.
HIGH_CONFIDENCE_TOKEN_PATTERNS = [
    # AWS access key id (AKIA... 16 chars after).
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    # AWS secret access key shape (40 base64ish chars after the literal "aws_secret_access_key").
    re.compile(rb"aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}"),
    # GitHub personal access token (classic + fine-grained).
    re.compile(rb"ghp_[A-Za-z0-9]{36}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{82}"),
    # GCP service account JSON marker.
    re.compile(rb'"type"\s*:\s*"service_account"'),
    # Slack tokens.
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"),
]

# Suffixes we never scan as text (binary, large).
BINARY_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".whl",
    ".pt",
    ".pth",
    ".bin",
    ".onnx",
    ".npy",
    ".npz",
    ".parquet",
    ".so",
    ".dylib",
    ".dll",
    ".ico",
    ".woff",
    ".woff2",
)

# Cap on bytes scanned per file (4 KiB is enough for PEM/JSON banners).
SCAN_BYTES = 4096


def _walk_repo() -> list[Path]:
    paths: list[Path] = []
    for root, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            paths.append(Path(root) / f)
    return paths


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def test_no_committed_private_key_files() -> None:
    """No `*.key|*.pem|*.p12|*.pfx` in the repo unless explicitly allowlisted."""
    offenders: list[str] = []
    for p in _walk_repo():
        if p.suffix.lower() not in SECRET_SUFFIXES:
            continue
        rel = _rel(p)
        if rel in ALLOWLISTED_KEY_FILES:
            continue
        offenders.append(rel)
    assert not offenders, (
        "Private key material found in repo: "
        + ", ".join(offenders)
        + ". Move to KMS / SealedSecret and add to .gitignore. "
        "If a file is genuinely a public test fixture, allowlist it in "
        "tests/test_no_secrets_in_repo.py::ALLOWLISTED_KEY_FILES with a reason."
    )


def test_no_pem_banner_in_any_file() -> None:
    """Even files with non-key extensions must not contain a PRIVATE KEY banner."""
    offenders: list[str] = []
    for p in _walk_repo():
        if p.suffix.lower() in BINARY_SUFFIXES:
            continue
        # Skip this file itself - it contains the regex pattern as a string.
        if p.resolve() == Path(__file__).resolve():
            continue
        rel = _rel(p)
        if rel in ALLOWLISTED_KEY_FILES:
            continue
        try:
            with p.open("rb") as fh:
                head = fh.read(SCAN_BYTES)
        except (OSError, PermissionError):
            continue
        if PRIVATE_KEY_BANNER_RE.search(head):
            offenders.append(rel)
    assert not offenders, (
        "Private key PEM banner found in committed file(s): "
        + ", ".join(offenders)
    )


def test_no_high_confidence_cloud_tokens() -> None:
    """No AWS / GitHub / GCP credential markers in committed text."""
    offenders: list[tuple[str, str]] = []
    for p in _walk_repo():
        if p.suffix.lower() in BINARY_SUFFIXES:
            continue
        if p.resolve() == Path(__file__).resolve():
            continue
        try:
            with p.open("rb") as fh:
                head = fh.read(SCAN_BYTES)
        except (OSError, PermissionError):
            continue
        for pat in HIGH_CONFIDENCE_TOKEN_PATTERNS:
            if pat.search(head):
                offenders.append((_rel(p), pat.pattern.decode("utf-8", "replace")))
                break
    assert not offenders, (
        "High-confidence credential pattern committed: "
        + "; ".join(f"{f} -> {pat}" for f, pat in offenders)
    )


def test_old_revoked_cosign_key_is_gone() -> None:
    """The specific file `deploy/cosign/cosign.key` must not exist in repo."""
    revoked = REPO_ROOT / "deploy" / "cosign" / "cosign.key"
    assert not revoked.exists(), (
        "deploy/cosign/cosign.key still present. The 2026-05-06 SRE rotation "
        "moved the revoked key to /tmp/cosign_old_REVOKED.key — do not "
        "restore it."
    )


def test_gitignore_blocks_keys() -> None:
    """`.gitignore` patterns must catch the four common key extensions."""
    gi = (REPO_ROOT / ".gitignore").read_text()
    for needle in ("*.key", "*.pem", "deploy/cosign/cosign.key"):
        assert needle in gi, f".gitignore missing pattern: {needle}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
