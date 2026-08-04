# syntax=docker/dockerfile:1

# =============================================================================
# arXiv Paper Recommendation System — Dockerfile
# Phase 7 containerization (docs/architecture.md §9 "Deployment / Docker")
#
# Multi-stage build: builder (uv sync) -> runtime (non-root, lean).
#
# The SAME image serves either service; docker-compose overrides CMD:
#   - app       -> uvicorn src.api.app:app --host 0.0.0.0 --port 8000
#   - dashboard -> streamlit run dashboard/app.py --server.port 8501
#
# PLATFORM GOTCHA (architecture §9): this is an Intel MacBook (macOS 12,
# x86_64, 8GB). Docker Desktop's VM must be capped at ~4GB — the ~1GB
# IVF/SQ8 index + app (~2GB total in the VM) fits; HNSW does NOT fit in any
# VM on this machine. See the docker-compose.yml header comment.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — builder: resolve + install locked deps into a venv (no dev deps)
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/local/bin/python

WORKDIR /app

# Install uv (pinned installer; no network needed at runtime).
RUN pip install --no-cache-dir uv

# Copy dependency manifests first for layer caching.
COPY pyproject.toml uv.lock ./

# --no-environments: pyproject.toml pins required-environments to
#   darwin/x86_64 (the dev laptop). The container builds on linux/amd64, so we
#   must bypass that gate to resolve the locked wheels.
# --no-install-project: source is copied separately in the runtime stage, so we
#   only need the dependency venv here (avoids a build-backend requirement).
RUN uv sync --frozen --no-dev --no-environments --no-install-project

# -----------------------------------------------------------------------------
# Stage 2 — runtime: non-root, lean, serves the API by default
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# OpenMP thread cap — torch + FAISS both use OpenMP; without capping, cores
# thrash (architecture §5 concurrency rules). 2 is safe on the 4GB VM.
ENV OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# curl is required by the compose healthcheck (liveness probe on /health).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (uid/gid 1000).
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

# Copy the dependency venv + application source from the builder.
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY dashboard/ ./dashboard/

# Data + model artifacts are gitignored local files; they are bind-mounted at
# runtime (see docker-compose.yml) so they are NOT baked into the image.
RUN mkdir -p /app/data /app/models && chown -R app:app /app

USER app

# Same image serves both services; compose overrides CMD for the dashboard.
EXPOSE 8000 8501
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]