# SpectrAI Production xApp Container
# ---------------------------------------------------------------------------
# Runs the gRPC inference server with ONNX Runtime (or TensorRT if available).
# Includes the RMR C library required by ricxappframe for O-RAN messaging.
#
# Build:
#   docker build -t spectrai:latest -f docker/Dockerfile .
#
# Run:
#   docker run -p 50051:50051 -p 9090:9090 \
#     -v ./models:/app/models \
#     spectrai:latest --config /app/config.yaml
# ---------------------------------------------------------------------------

FROM python:3.11-slim-bookworm AS base

LABEL maintainer="spectrai@example.com"
LABEL org.opencontainers.image.description="SpectrAI xApp — AI-native spectrum management"

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        libsctp-dev \
        lksctp-tools \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install RMR (RIC Message Router) C library
# ---------------------------------------------------------------------------
ARG RMR_VERSION=4.9.4
RUN curl -fsSL "https://packagecloud.io/o-ran-sc/release/gpgkey" | \
        gpg --dearmor -o /usr/share/keyrings/o-ran-sc.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/o-ran-sc.gpg] https://packagecloud.io/o-ran-sc/release/debian/ bookworm main" \
        > /etc/apt/sources.list.d/o-ran-sc.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        rmr-dev=${RMR_VERSION} \
        rmr=${RMR_VERSION} \
    || true && \
    rm -rf /var/lib/apt/lists/*

# Fallback: build RMR from source if the package repo is unavailable
RUN if ! ldconfig -p | grep -q librmr; then \
        git clone --depth 1 --branch ${RMR_VERSION} \
            https://gerrit.o-ran-sc.org/r/ric-plt/lib/rmr /tmp/rmr && \
        cd /tmp/rmr && \
        mkdir build && cd build && \
        cmake -DDEV_PKG=1 .. && \
        make -j"$(nproc)" install && \
        ldconfig && \
        rm -rf /tmp/rmr; \
    fi

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Install SpectrAI package
# ---------------------------------------------------------------------------
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
COPY proto/ /app/proto/

# Generate protobuf stubs (best-effort — may already be vendored)
RUN mkdir -p /app/src/spectrai/proto && \
    touch /app/src/spectrai/proto/__init__.py && \
    python -m grpc_tools.protoc \
        -Iproto \
        --python_out=src/spectrai/proto \
        --grpc_python_out=src/spectrai/proto \
        proto/spectrai.proto \
    || true

RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
EXPOSE 50051
EXPOSE 9090

# Default RMR routing table (override with a volume mount)
ENV RMR_SEED_RT=/app/routes.rt
RUN echo "newrt | start\nnewrt | end" > /app/routes.rt

# Health check: gRPC reflection
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import grpc; ch=grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(ch).result(timeout=3)" || exit 1

ENTRYPOINT ["python", "-m", "spectrai.xapp.server"]
CMD ["--config", "/app/config.yaml"]
