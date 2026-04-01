# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app/front-end
COPY front-end/package*.json ./
RUN npm ci
COPY front-end/ ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
# Using slim Python instead of the heavy CUDA base image.
# The app runs on CPU by default.
# To enable NVIDIA GPU: set USE_CUDA=1 in docker-compose.yml and switch the
# base image to pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime, then remove
# the --extra-index-url line in requirements.txt and use the standard torch wheel.
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — installed at build time
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source
COPY app.py download_model.py server.py ./

# Copy built React dist from Stage 1
COPY --from=frontend-builder /app/front-end/dist ./front-end/dist

# Models volume — persists weights across container restarts
VOLUME ["/app/models"]
ENV HF_HOME=/app/models/hf_cache

# USE_CUDA=0  → CPU (default, works everywhere)
# USE_CUDA=1  → NVIDIA GPU (requires nvidia-container-toolkit on the host
#               and the CUDA-enabled torch wheel)
ENV USE_CUDA=0

EXPOSE 8000

CMD ["python", "app.py"]
