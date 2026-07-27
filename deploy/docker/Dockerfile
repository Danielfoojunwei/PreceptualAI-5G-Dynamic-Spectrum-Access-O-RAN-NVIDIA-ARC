# syntax=docker/dockerfile:1.6
# HorizonRIC rApp container — production image.
# Multi-stage build: stage 1 builds the venv, stage 2 ships only what runs.

ARG PYTHON_VERSION=3.12
ARG GIT_SHA=unknown
ARG VERSION=0.2.0

# --------------------------------------------------------------------------
# Stage 1: builder
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/horizon/.venv \
    PATH=/opt/horizon/.venv/bin:$PATH

# Build deps for native wheels (cryptography, ncclient lxml, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/horizon

# Create the venv we will copy to the runtime stage.
# The optional `ca-bundle` secret lets builds behind a TLS-intercepting
# egress proxy trust its CA without baking it into the image:
#   docker build --secret id=ca-bundle,src=/path/to/ca.crt ...
RUN --mount=type=secret,id=ca-bundle,target=/tmp/ca-bundle.crt \
    if [ -s /tmp/ca-bundle.crt ]; then export PIP_CERT=/tmp/ca-bundle.crt; fi \
    && python -m venv /opt/horizon/.venv \
    && /opt/horizon/.venv/bin/pip install --upgrade pip setuptools wheel

# Copy only the build metadata first, so dep installs cache cleanly.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package (no [dev] in the runtime image — keeps it slim, but
# we install dev for the test image variant via a build arg if needed).
# uvicorn is a declared runtime dependency in pyproject.toml (the lifecycle
# daemon and run_horizon_rapp health/metrics servers), so no explicit pin.
# Extras must exist in pyproject [project.optional-dependencies]:
#   oran        — NETCONF/YANG + ASN.1 southbound wire integrations
#   persistence — psycopg for the Postgres/TimescaleDB evidence store
#                 (HORIZON_EVIDENCE_DSN); SQLite needs no extra.
#   kafka       — aiokafka for the Kafka telemetry source/sink
#                 (pipeline source.type: kafka).
#   hsm         — python-pkcs11 for HSM-backed signing (security/hsm.py).
ARG INSTALL_EXTRAS="oran,persistence,kafka,hsm"
RUN --mount=type=secret,id=ca-bundle,target=/tmp/ca-bundle.crt \
    if [ -s /tmp/ca-bundle.crt ]; then export PIP_CERT=/tmp/ca-bundle.crt; fi \
    && /opt/horizon/.venv/bin/pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"

# --------------------------------------------------------------------------
# Stage 2: runtime
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG GIT_SHA
ARG VERSION

LABEL org.opencontainers.image.title="HorizonRIC rApp" \
      org.opencontainers.image.description="NTN-Aware AI-RAN Resource-Orchestration rApp Suite" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.source="https://github.com/preceptualai/horizon-ric" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Preceptual AI"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HORIZON_GIT_SHA=${GIT_SHA} \
    HORIZON_VERSION=${VERSION} \
    VIRTUAL_ENV=/opt/horizon/.venv \
    PATH=/opt/horizon/.venv/bin:$PATH \
    # The daemon now ships inside the package (horizon_ric.rapp.daemon);
    # PYTHONPATH is kept only so the historical `scripts.run_horizon_rapp`
    # shim import path keeps working for tooling that still uses it.
    PYTHONPATH=/opt/horizon \
    HORIZON_HEALTH_HOST=0.0.0.0 \
    HORIZON_HEALTH_PORT=8081 \
    HORIZON_METRICS_PORT=8082 \
    HORIZON_STATE_DIR=/var/lib/horizon

# Runtime libs: curl for healthcheck, libxml/xslt for ncclient, libgomp for torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libxml2 \
        libxslt1.1 \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN groupadd --system --gid 10000 horizon \
    && useradd --system --uid 10000 --gid horizon --home /opt/horizon \
       --shell /usr/sbin/nologin horizon \
    && mkdir -p /opt/horizon /var/lib/horizon /var/log/horizon \
    && chown -R horizon:horizon /opt/horizon /var/lib/horizon /var/log/horizon

WORKDIR /opt/horizon

# Copy the venv from the builder.
COPY --from=builder --chown=horizon:horizon /opt/horizon/.venv /opt/horizon/.venv

# Copy source + scripts (so the daemon entrypoint is on PYTHONPATH).
COPY --chown=horizon:horizon src/ /opt/horizon/src/
COPY --chown=horizon:horizon scripts/ /opt/horizon/scripts/
COPY --chown=horizon:horizon pyproject.toml README.md /opt/horizon/

USER horizon

EXPOSE 8081 8082

# Port story: the bare lifecycle daemon (default CMD, no args) serves
# /healthz on 8081; pipeline mode (`--source-config ...`) serves it on the
# metrics port 8082 instead — probe both so either mode reports healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8081/healthz \
        || curl -fsS http://localhost:8082/healthz \
        || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/horizon/.venv/bin/horizon-rapp"]
CMD []
