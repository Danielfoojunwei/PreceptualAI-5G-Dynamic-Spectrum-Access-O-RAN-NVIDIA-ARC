"""Where agent verification keys come from, and how they rotate.

`AgentRegistry` takes a mapping of agent id to public key and says, correctly,
that it must be operator-held — an agent that registers its own key
authenticates as itself under any grant, which is no authentication. It did not
say where the operator's mapping comes from, and "construct it in Python"
is not a key-distribution story.

This loads it from a directory the operator controls. Two properties are worth
stating because both were tempting to skip.

**Rotation needs an overlap, and the overlap must be bounded.** An agent cannot
switch keys atomically with the operator — there is a window where either the
old or the new key may sign. So a registry has to accept both, which means a
compromised old key stays valid for the length of the window. Making that
window an explicit, expiring property of the *file layout* keeps it from
becoming permanent by default: a superseded key lives in `retired/` and a
deployment that never sweeps that directory can see it has not.

**A file that is not an Ed25519 public key is a refusal, not a skip.** Skipping
means an operator who mistypes a filename gets a registry that silently omits
an agent, and the first symptom is a legitimate agent being refused in
production with "no registered key".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from horizon_agentic.identity import AgentRegistry, IdentityVerdict
from horizon_ric.shield.signing import key_fingerprint

__all__ = [
    "KeyringError",
    "MultiKeyRegistry",
    "load_registry",
    "write_public_key",
]

SUFFIX = ".pub.pem"
RETIRED = "retired"


class KeyringError(RuntimeError):
    """Raised when a key directory cannot be turned into a usable registry."""


def _load_public(path: Path) -> Ed25519PublicKey:
    try:
        raw = path.read_bytes()
    except OSError as exc:  # pragma: no cover - filesystem failure
        raise KeyringError(f"cannot read {path}: {exc}") from exc
    try:
        key = serialization.load_pem_public_key(raw)
    except Exception as exc:
        raise KeyringError(
            f"{path} is not a readable PEM public key ({exc}); refusing rather "
            "than skipping, because a skipped agent surfaces later as a "
            "legitimate agent being refused"
        ) from exc
    if not isinstance(key, Ed25519PublicKey):
        raise KeyringError(
            f"{path} holds a {type(key).__name__}, not an Ed25519 public key"
        )
    return key


@dataclass(frozen=True)
class MultiKeyRegistry:
    """A registry that accepts more than one key per agent, for rotation.

    `AgentRegistry` maps one id to one key, which cannot express an overlap.
    This holds the current key and any still-accepted retired ones, and
    :meth:`authenticate_with` tries each — reporting *which* key verified, so
    an operator can see whether an agent has actually moved off the old one
    before sweeping it.
    """

    current: Mapping[str, Ed25519PublicKey]
    retired: Mapping[str, tuple[Ed25519PublicKey, ...]]

    def as_registry(self) -> AgentRegistry:
        """The current keys only — what a deployment past its overlap uses."""
        return AgentRegistry(dict(self.current))

    def keys_for(self, agent_id: str) -> tuple[Ed25519PublicKey, ...]:
        keys: list[Ed25519PublicKey] = []
        if agent_id in self.current:
            keys.append(self.current[agent_id])
        keys.extend(self.retired.get(agent_id, ()))
        return tuple(keys)

    def fingerprints(self, agent_id: str) -> tuple[str, ...]:
        return tuple(key_fingerprint(k) for k in self.keys_for(agent_id))

    def verify_during_rotation(
        self, agent_id: str, verify
    ) -> tuple[bool, str]:
        """Try every accepted key; return whether one worked and its fingerprint.

        ``verify`` is a callable taking a public key and returning a bool. The
        fingerprint of the key that succeeded is returned so a deployment can
        tell a rotation that has completed from one that has merely been
        configured.
        """
        for key in self.keys_for(agent_id):
            if verify(key):
                return True, key_fingerprint(key)
        return False, ""

    @property
    def agents_still_on_retired_keys(self) -> tuple[str, ...]:
        """Agents for which a retired key is still accepted.

        The list an operator sweeps. Non-empty means the overlap window is
        still open and a compromised old key would still authenticate.
        """
        return tuple(sorted(a for a, keys in self.retired.items() if keys))


def load_registry(directory: str | Path) -> MultiKeyRegistry:
    """Build a registry from ``<agent_id>.pub.pem`` files in ``directory``.

    Retired keys live in ``directory/retired/<agent_id>.<n>.pub.pem`` and are
    accepted alongside the current one. An agent present *only* under
    ``retired/`` is refused: that is a half-finished removal, and treating it
    as valid would keep a decommissioned agent alive.
    """
    root = Path(directory)
    if not root.is_dir():
        raise KeyringError(f"{root} is not a directory")

    current: dict[str, Ed25519PublicKey] = {}
    for path in sorted(root.glob(f"*{SUFFIX}")):
        current[path.name[: -len(SUFFIX)]] = _load_public(path)

    retired: dict[str, list[Ed25519PublicKey]] = {}
    retired_dir = root / RETIRED
    if retired_dir.is_dir():
        for path in sorted(retired_dir.glob(f"*{SUFFIX}")):
            stem = path.name[: -len(SUFFIX)]
            agent = stem.rsplit(".", 1)[0] if "." in stem else stem
            retired.setdefault(agent, []).append(_load_public(path))

    orphaned = sorted(set(retired) - set(current))
    if orphaned:
        raise KeyringError(
            f"agents with only retired keys: {orphaned}. A retired key without "
            "a current one is a half-finished removal; delete the retired key "
            "or restore the current one"
        )

    if not current:
        raise KeyringError(
            f"{root} contains no *{SUFFIX} files; an empty registry refuses "
            "every agent, which is safe but almost certainly not intended"
        )

    return MultiKeyRegistry(
        current=current, retired={a: tuple(k) for a, k in retired.items()}
    )


def write_public_key(directory: str | Path, agent_id: str, key: Ed25519PrivateKey) -> Path:
    """Write an agent's public key into a registry directory.

    Public half only — the private key never leaves the agent, and a helper
    that wrote both would make it easy to build a registry directory that is
    also a key-compromise waiting to happen.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{agent_id}{SUFFIX}"
    path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return path


def unusable_verdict(agent_id: str, reason: str) -> IdentityVerdict:
    """A refusal shaped like the authenticator's, for keyring-level failures."""
    return IdentityVerdict(False, agent_id, (reason,))
