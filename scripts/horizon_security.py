#!/usr/bin/env python
"""PreceptualAI security admin CLI.

Subcommands
-----------
    user add       Bind a user to a role within a tenant.
    user remove    Unbind a user from a role.
    user list      List users for a role+tenant.
    token mint     Issue a JWT for a user (RS256).
    token verify   Verify a JWT and print its claims.
    policy show    Print the loaded Casbin policy + role bindings.
    audit-rotate   Rotate the JWT signing key with a graceful overlap.

The CLI talks directly to the Casbin policy CSV and the JWTManager —
no daemon, no extra service. Changes to the policy CSV are picked up
by any running process via `Casbin.reload()` (or a process restart).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

# Allow running directly from a checkout without `pip install -e .`
_THIS_DIR = Path(__file__).resolve().parent
_REPO_SRC = _THIS_DIR.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from horizon_ric.security.jwt import JWTManager  # noqa: E402
from horizon_ric.security.rbac import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_POLICY_PATH,
    Casbin,
)

app = typer.Typer(help="PreceptualAI security admin CLI", no_args_is_help=True)
user_app = typer.Typer(help="User ↔ role bindings", no_args_is_help=True)
token_app = typer.Typer(help="JWT mint / verify", no_args_is_help=True)
policy_app = typer.Typer(help="Casbin policy inspection", no_args_is_help=True)
app.add_typer(user_app, name="user")
app.add_typer(token_app, name="token")
app.add_typer(policy_app, name="policy")


# ---------------------------------------------------------------------------
# Defaults — overridable via env vars so deployments can keep production
# keys outside the repo.
# ---------------------------------------------------------------------------
DEFAULT_SIGNING_KEY = os.environ.get(
    "HORIZON_JWT_SIGNING_KEY",
    str(Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "test_jwt_signing.pem"),
)
DEFAULT_ISSUER = os.environ.get("HORIZON_JWT_ISSUER", "horizon-ric")
DEFAULT_AUDIENCE = os.environ.get("HORIZON_JWT_AUDIENCE", "horizon-ric-api")


def _make_rbac() -> Casbin:
    return Casbin(model_path=DEFAULT_MODEL_PATH, policy_path=DEFAULT_POLICY_PATH)


def _make_jwt(signing_key: Path | None = None) -> JWTManager:
    return JWTManager(
        signing_key_path=signing_key or DEFAULT_SIGNING_KEY,
        issuer=DEFAULT_ISSUER,
        audience=DEFAULT_AUDIENCE,
    )


# ---------------------------------------------------------------------------
# user
# ---------------------------------------------------------------------------
@user_app.command("add")
def user_add(
    user: str = typer.Option(..., help="User identity (subject)"),
    role: str = typer.Option(..., help="Role name (admin, operator, ...)"),
    tenant: str = typer.Option("default", help="Tenant (Casbin domain)"),
) -> None:
    rbac = _make_rbac()
    added = rbac.add_role(user, role, tenant)
    typer.echo(
        json.dumps(
            {"user": user, "role": role, "tenant": tenant, "added": added},
            indent=2,
        )
    )


@user_app.command("remove")
def user_remove(
    user: str = typer.Option(..., help="User identity (subject)"),
    role: str = typer.Option(..., help="Role name"),
    tenant: str = typer.Option("default", help="Tenant"),
) -> None:
    rbac = _make_rbac()
    removed = rbac.remove_role(user, role, tenant)
    typer.echo(
        json.dumps(
            {"user": user, "role": role, "tenant": tenant, "removed": removed},
            indent=2,
        )
    )


@user_app.command("list")
def user_list(
    role: str = typer.Option(None, help="If set, list users for this role only"),
    tenant: str = typer.Option("default", help="Tenant"),
) -> None:
    rbac = _make_rbac()
    if role:
        users = rbac.list_users_for_role(role, tenant)
        typer.echo(json.dumps({"role": role, "tenant": tenant, "users": users}, indent=2))
    else:
        bindings = [
            b for b in rbac.list_grouping_policies()
            if len(b) >= 3 and b[2] == tenant
        ]
        typer.echo(json.dumps({"tenant": tenant, "bindings": bindings}, indent=2))


# ---------------------------------------------------------------------------
# token
# ---------------------------------------------------------------------------
@token_app.command("mint")
def token_mint(
    user: str = typer.Option(..., help="Subject of the token"),
    tenant: str = typer.Option("default", help="Tenant claim"),
    ttl: int = typer.Option(3600, help="Time-to-live in seconds"),
    signing_key: Path = typer.Option(None, help="Override signing key path"),
) -> None:
    rbac = _make_rbac()
    roles = rbac.list_roles_for_user(user, tenant)
    jwt_manager = _make_jwt(signing_key)
    token = jwt_manager.mint_token(
        subject=user, tenant=tenant, roles=roles, ttl_seconds=ttl
    )
    typer.echo(token)


@token_app.command("verify")
def token_verify(
    token: str = typer.Argument(..., help="JWT to verify"),
    signing_key: Path = typer.Option(None, help="Override signing key path"),
) -> None:
    jwt_manager = _make_jwt(signing_key)
    try:
        claims = jwt_manager.verify_token(token)
    except Exception as exc:  # noqa: BLE001 — CLI surface
        typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        raise typer.Exit(code=1)
    typer.echo(json.dumps({"ok": True, "claims": claims}, indent=2))


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
@policy_app.command("show")
def policy_show() -> None:
    rbac = _make_rbac()
    typer.echo(
        json.dumps(
            {
                "model": str(DEFAULT_MODEL_PATH),
                "policy_file": str(DEFAULT_POLICY_PATH),
                "policies": rbac.list_policies(),
                "bindings": rbac.list_grouping_policies(),
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# audit-rotate (signing key rotation)
# ---------------------------------------------------------------------------
@app.command("audit-rotate")
def audit_rotate(
    new_key: Path = typer.Option(..., "--new-key", help="Path to the new RSA private key"),
    old_key: Path = typer.Option(None, "--old-key", help="Override current signing key path"),
    overlap_seconds: int = typer.Option(
        3600, help="Seconds during which the old key still verifies"
    ),
) -> None:
    """Rotate the JWT signing key with a graceful overlap window.

    This command performs an in-process rotation as a smoke test and
    prints the public key fingerprint of both the new active key and
    each retired key so operators can verify the rotation completed.
    """
    jwt_manager = JWTManager(
        signing_key_path=old_key or DEFAULT_SIGNING_KEY,
        issuer=DEFAULT_ISSUER,
        audience=DEFAULT_AUDIENCE,
        overlap_window_seconds=overlap_seconds,
    )
    jwt_manager.rotate_signing_key(new_key)
    typer.echo(
        json.dumps(
            {
                "ok": True,
                "active_public_key_chars": len(jwt_manager.public_pem),
                "retired_keys": jwt_manager.retired_key_count(),
                "overlap_seconds": overlap_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
