# Base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=5000

# Install system dependencies needed for building some Python packages (e.g., lxml, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy dependency file first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the application port (container port). Host port will be randomized via compose.
EXPOSE 5000

# Healthcheck (container-internal)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -fsS http://127.0.0.1:${SERVER_PORT}/health || exit 1

# Default command: run FastAPI app with uvicorn
CMD ["bash", "-lc", "uvicorn src.server:app --host ${SERVER_HOST} --port ${SERVER_PORT}"]
