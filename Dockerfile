# ========================================================
# Stage 1: Build & Dependency Resolver
# ========================================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install system utilities required for dependencies compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ========================================================
# Stage 2: Minimal Runtime Image
# ========================================================
FROM python:3.10-slim

WORKDIR /app

# Copy python virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code and config assets
COPY src/ /app/src/

# Set environment defaults
ENV PORT=8000
ENV PYTHONPATH="/app"
ENV COLORS_CONFIG_PATH="/app/src/infrastructure/config/colors.json"

EXPOSE 8000

# Start production server using uvicorn
CMD uvicorn src.presentation.api.main:app --host 0.0.0.0 --port ${PORT}
