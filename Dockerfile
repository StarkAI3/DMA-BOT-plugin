# Backend-Only Dockerfile for DMA Bot API
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000

# Install system dependencies for ML libraries and web scraping
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy dependency file first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python dependencies with optimizations
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only backend source code and necessary data
COPY src/ ./src/
COPY models/ ./models/
COPY optimized_data/ ./optimized_data/
COPY final_data/ ./final_data/

# Create necessary directories
RUN mkdir -p /app/logs

# Expose the API port
EXPOSE 8000

# Health check for API endpoints
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${SERVER_PORT}/health || exit 1

# Start the FastAPI backend server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
