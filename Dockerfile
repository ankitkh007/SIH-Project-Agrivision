# ==========================
# Stage 1: Builder
# ==========================
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for building scientific libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
      gfortran \
      libopenblas-dev \
      liblapack-dev \
      libatlas-base-dev \
      libffi-dev \
      libssl-dev \
      curl && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for caching
COPY backend/requirements.txt .

# Build wheels in builder stage
RUN pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels


# ==========================
# Stage 2: Final Runtime
# ==========================
FROM python:3.10-slim

WORKDIR /app

# Copy built wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy backend code
COPY backend/ .

# Set environment variables
ENV PORT=8000
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "bend:app", "--host", "0.0.0.0", "--port", "8000"]
