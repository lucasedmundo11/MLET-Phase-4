# ── Build stage: install dependencies ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System libraries needed by TensorFlow and numerical packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Use tensorflow-cpu for inference (smaller image, no GPU needed)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "tensorflow-cpu>=2.15.0" \
        fastapi>=0.111.0 \
        "uvicorn[standard]>=0.29.0" \
        numpy>=1.24.0 \
        pandas>=2.0.0 \
        yfinance>=0.2.37 \
        scikit-learn>=1.3.0 \
        joblib>=1.3.0 \
        "prometheus-client>=0.20.0" \
        "psutil>=5.9.0" \
        "pydantic>=2.0.0" \
        "python-multipart>=0.0.9"


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# System runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Application source
COPY app/ ./app/

# Model artifacts (can be overridden via volume mount in production)
COPY models/ ./models/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check using Python stdlib (no extra deps)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Single worker for inference (TF is not fork-safe; scale horizontally instead)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
