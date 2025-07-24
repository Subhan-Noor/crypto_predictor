# Use Python 3.10 slim image for smaller size and better security
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libpq-dev \
        build-essential

# Copy requirements and install Python dependencies
COPY backend/requirements-railway.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools to reduce image size
RUN apt-get purge -y --auto-remove libpq-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY backend/app ./app
COPY backend/ml ./ml
COPY backend/models ./models
COPY backend/config.py .
COPY backend/run.py .
# Removed: COPY backend/startup_debug.py .
# Removed: COPY backend/health_check.py .

# Removed: RUN python startup_debug.py

# If you need any other files, add COPY lines here

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Health check (use FastAPI /health endpoint instead of missing script)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.enhanced_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
