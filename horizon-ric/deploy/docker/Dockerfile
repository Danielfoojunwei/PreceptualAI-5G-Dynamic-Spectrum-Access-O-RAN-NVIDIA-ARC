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
RUN python -m venv /opt/horizon/.venv \
    && /opt/horizon/.venv/bin/pip install --upgrade pip setuptools wheel

# Copy only the build metadata first, so dep installs cache cleanly.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package (no [dev] in the runtime image — keeps it slim, but
# we install dev for the test image variant via a build arg if needed).
ARG INSTALL_EXTRAS="oran,crypto"
RUN /opt/horizon/.venv/bin/pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"

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

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8081/healthz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/horizon/.venv/bin/horizon-rapp"]
CMD []
