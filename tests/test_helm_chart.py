"""Verify the Helm chart renders the full-pipeline + A1 dialect/auth contract.

We can't run a live cluster on the test runner, so these tests shell out to
`helm lint` / `helm template` and assert on the rendered manifests:

  * defaults          — pipeline args, mounted source.yaml, synthetic source,
                        probes on the metrics port, Recreate strategy (RWO
                        PVC), single replica, no HPA, ghcr image, osc dialect
                        env, shield/budget/status-poll env, empty knobs
                        omitted.
  * pipeline disabled — no args, no source ConfigMap, probes/scrape fall
                        back to the health port.
  * ricId + token set — conditional env appears, Secret carries the token.
  * auth false values — verifyTls=false / productionMode=false must render
                        (regression: `with` used to drop falsy strings).
  * secret wiring     — create=false without existingSecret must FAIL the
                        render (dangling secretRef), dashboard JWT secret
                        key renders only when set.
  * evidence DSN      — HORIZON_EVIDENCE_DSN renders only with
                        evidence.useTimescaleDb, password via secretKeyRef.

Skipped when no helm binary is available; point HELM_BIN at one otherwise
(CI downloads helm v3.16.x into the runner and exports HELM_BIN).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = REPO_ROOT / "deploy" / "helm" / "horizon-ric"
HELM = os.environ.get("HELM_BIN", "helm")

pytestmark = pytest.mark.skipif(
    shutil.which(HELM) is None,
    reason="helm binary not found (install helm or set HELM_BIN)",
)


def _helm_raw(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [HELM, *args], capture_output=True, text=True, check=False, timeout=120
    )


def _helm(*args: str) -> str:
    proc = _helm_raw(*args)
    assert proc.returncode == 0, (
        f"helm {' '.join(args)} failed (rc={proc.returncode}):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    return proc.stdout


def _template(*set_args: str) -> str:
    args = ["template", "test-rel", str(CHART_DIR)]
    for s in set_args:
        args += ["--set", s]
    return _helm(*args)


def _template_expect_failure(*set_args: str) -> str:
    args = ["template", "test-rel", str(CHART_DIR)]
    for s in set_args:
        args += ["--set", s]
    proc = _helm_raw(*args)
    assert proc.returncode != 0, (
        f"helm {' '.join(args)} unexpectedly succeeded:\n{proc.stdout}"
    )
    return proc.stderr


# ---------- lint ----------

def test_chart_lints_clean() -> None:
    out = _helm("lint", str(CHART_DIR))
    assert "0 chart(s) failed" in out


# ---------- defaults: full-pipeline mode on ----------

def test_default_render_wires_pipeline_mode() -> None:
    out = _template()
    # Container args switch the entrypoint into full-pipeline mode.
    assert 'args: ["--source-config", "/etc/horizon/source.yaml"]' in out
    # Source ConfigMap is rendered and mounted read-only at /etc/horizon.
    assert "name: test-rel-horizon-ric-source" in out
    assert "source.yaml: |" in out
    assert "mountPath: /etc/horizon" in out
    # Rollouts must pick up edits to the source config.
    assert "checksum/config-source:" in out


def test_default_source_is_synthetic() -> None:
    """A stock install must not point at a telemetry file that never exists.

    `file` sources with a missing path HARD-ERROR at startup in daemon mode,
    so the default is an unbounded synthetic stream; operators switch to
    file/kafka with real telemetry (documented in values.yaml).
    """
    out = _template()
    assert "type: synthetic" in out
    assert "count: 0" in out
    assert "interval_seconds: 10" in out
    # The old crash-looping default must be gone.
    assert "path: /var/lib/horizon/replay.jsonl" not in out


def test_default_probes_target_metrics_port() -> None:
    """Pipeline mode serves /healthz + /readyz ONLY on the metrics port.

    Probing the health port in pipeline mode crash-loops the default
    install — the daemon binds nothing on 8081 in that mode.
    """
    out = _template()
    assert "path: /healthz\n              port: metrics" in out
    assert "path: /readyz\n              port: metrics" in out
    assert "port: health\n            initialDelaySeconds" not in out
    # Scrape targets follow suit.
    assert 'prometheus.io/port: "8082"' in out


def test_default_image_matches_ci_registry() -> None:
    """values.yaml default must be the ghcr image CI actually publishes."""
    out = _template()
    assert 'image: "ghcr.io/preceptualai/horizon-ric:0.2.0"' in out
    assert "horizonric/rapp" not in out


def test_default_single_writer_topology() -> None:
    """RWO state PVC: one replica, Recreate strategy, no HPA by default."""
    out = _template()
    assert "replicas: 1" in out
    assert "type: Recreate" in out
    assert "kind: HorizontalPodAutoscaler" not in out
    assert "maxSurge" not in out


def test_no_persistence_restores_rolling_update() -> None:
    out = _template("persistence.enabled=false")
    assert "type: RollingUpdate" in out
    assert "maxSurge: 1" in out
    assert "type: Recreate" not in out


def test_default_render_a1_and_shield_env() -> None:
    out = _template()
    assert 'HORIZON_A1_DIALECT: "osc"' in out
    assert 'HORIZON_A1_TIMEOUT_S: "10"' in out
    # Env vars are strings — numbers must render quoted.
    assert 'HORIZON_SHIELD_BAND_LO_HZ: "3.4e+09"' in out
    assert 'HORIZON_SHIELD_BAND_HI_HZ: "3.5e+09"' in out
    assert 'HORIZON_SHIELD_MAX_EIRP_DBM: "33"' in out
    assert 'HORIZON_DECISION_BUDGET_MS: "1000"' in out
    assert 'HORIZON_A1_STATUS_POLL_ATTEMPTS: "3"' in out
    assert 'HORIZON_A1_STATUS_POLL_INTERVAL_S: "1"' in out


def test_default_render_state_path_env() -> None:
    """The code reads HORIZON_STATE_PATH (full file path), not *_DIR alone."""
    out = _template()
    assert 'HORIZON_STATE_PATH: "/var/lib/horizon/state.json"' in out
    # STATE_DIR is kept as the documented base-dir convention.
    assert 'HORIZON_STATE_DIR: "/var/lib/horizon"' in out
    # Dead envs: nothing reads HORIZON_METRICS_PORT (pipeline mode takes the
    # port from source.yaml, lifecycle mode from HORIZON_HEALTH_PORT) or
    # HORIZON_LOG_LEVEL (structlog configures its own level).
    assert "HORIZON_METRICS_PORT" not in out
    assert "HORIZON_LOG_LEVEL" not in out


def test_default_render_omits_empty_optional_env() -> None:
    """Empty knobs must be absent so adapter defaults apply (set-vs-empty)."""
    out = _template()
    for var in (
        "HORIZON_A1_RIC_ID",
        "HORIZON_A1_SERVICE_ID",
        "HORIZON_A1_OAUTH_TOKEN_URL",
        "HORIZON_A1_OAUTH_CLIENT_ID",
        "HORIZON_A1_OAUTH_CLIENT_SECRET",
        "HORIZON_A1_OAUTH_SCOPE",
        "HORIZON_A1_CLIENT_CERT_PATH",
        "HORIZON_A1_CLIENT_KEY_PATH",
        "HORIZON_A1_CA_BUNDLE_PATH",
        "HORIZON_A1_VERIFY_TLS",
        "HORIZON_PRODUCTION_MODE",
        "HORIZON_API_JWT_SECRET",
        "HORIZON_EVIDENCE_DSN",
    ):
        assert var not in out, f"{var} must not render when unset"


# ---------- pipeline disabled: bare lifecycle daemon ----------

def test_pipeline_disabled_falls_back_to_lifecycle_daemon() -> None:
    out = _template("pipeline.enabled=false")
    assert "--source-config" not in out
    assert "source.yaml" not in out
    assert "name: test-rel-horizon-ric-source" not in out
    assert "mountPath: /etc/horizon" not in out
    assert "checksum/config-source" not in out
    # Shield env still renders (harmless for the bare daemon).
    assert "HORIZON_SHIELD_BAND_LO_HZ" in out


def test_pipeline_disabled_probes_target_health_port() -> None:
    """Lifecycle mode serves /healthz + /readyz + /metrics on the health port."""
    out = _template("pipeline.enabled=false")
    assert "path: /healthz\n              port: health" in out
    assert "path: /readyz\n              port: health" in out
    assert 'prometheus.io/port: "8081"' in out
    # ServiceMonitor scrapes the health port in this mode.
    assert "- port: health\n      path: /metrics" in out


def test_default_servicemonitor_scrapes_metrics_port() -> None:
    out = _template()
    assert "- port: metrics\n      path: /metrics" in out


# ---------- ricId + token set ----------

def test_ric_id_and_token_render() -> None:
    out = _template(
        "config.a1RicId=ric_osc_1",
        "config.a1ServiceId=horizon-svc",
        "secret.data.a1ClientToken=sekrit",
    )
    assert 'HORIZON_A1_RIC_ID: "ric_osc_1"' in out
    assert 'HORIZON_A1_SERVICE_ID: "horizon-svc"' in out
    # Secret wraps the plaintext with b64enc: b64("sekrit") == "c2Vrcml0".
    assert 'A1_CLIENT_TOKEN: "c2Vrcml0"' in out


# ---------- auth: explicit false values must reach the pod ----------

def test_auth_false_values_render() -> None:
    """Regression: `{{ with }}` silently dropped falsy strings, so an
    operator's explicit verifyTls=false / productionMode=false never
    reached the pod."""
    out = _template(
        "config.auth.verifyTls=false",
        "config.auth.productionMode=false",
    )
    assert 'HORIZON_A1_VERIFY_TLS: "false"' in out
    assert 'HORIZON_PRODUCTION_MODE: "false"' in out


def test_auth_true_values_render() -> None:
    out = _template(
        "config.auth.verifyTls=true",
        "config.auth.productionMode=true",
    )
    assert 'HORIZON_A1_VERIFY_TLS: "true"' in out
    assert 'HORIZON_PRODUCTION_MODE: "true"' in out


# ---------- secret wiring ----------

def test_secret_create_false_without_existing_fails() -> None:
    """secret.create=false with no existingSecret used to render a dangling
    secretRef (CreateContainerConfigError at deploy time) — must fail fast."""
    err = _template_expect_failure("secret.create=false")
    assert "secret.existingSecret" in err


def test_secret_create_false_with_existing_renders() -> None:
    out = _template("secret.create=false", "secret.existingSecret=my-creds")
    assert "name: my-creds" in out
    # The chart-managed Secret itself must not render.
    assert "SPACE_TRACK_IDENTITY" not in out


# ---------- dashboard ----------

def test_dashboard_disabled_by_default() -> None:
    out = _template()
    assert "name: dashboard" not in out
    assert "containerPort: 8083" not in out


def test_dashboard_enabled_exposes_port_and_jwt_secret() -> None:
    out = _template(
        "dashboard.enabled=true",
        "secret.data.apiJwtSecret=jwtsekrit",
    )
    assert "containerPort: 8083" in out
    # Service port + NetworkPolicy ingress open up.
    assert "targetPort: dashboard" in out
    assert 'HORIZON_API_PORT: "8083"' in out
    # b64("jwtsekrit") == "and0c2Vrcml0"
    assert 'HORIZON_API_JWT_SECRET: "and0c2Vrcml0"' in out


# ---------- evidence store DSN (TimescaleDB) ----------

def test_evidence_dsn_disabled_by_default() -> None:
    out = _template()
    assert "HORIZON_EVIDENCE_DSN" not in out


def test_evidence_dsn_renders_with_timescaledb() -> None:
    out = _template("evidence.useTimescaleDb=true")
    assert "HORIZON_EVIDENCE_DSN" in out
    # Password comes from the TimescaleDB Secret via dependent-env expansion —
    # never plaintext in the DSN.
    assert (
        "postgresql://horizon:$(POSTGRES_PASSWORD)"
        "@test-rel-horizon-ric-timescaledb" in out
    )
    assert "/decisions" in out


def test_evidence_dsn_requires_timescaledb_enabled() -> None:
    out = _template("evidence.useTimescaleDb=true", "timescaledb.enabled=false")
    assert "HORIZON_EVIDENCE_DSN" not in out
