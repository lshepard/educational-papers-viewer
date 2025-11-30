# Backend Dockerfile for Papers Viewer API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
# Using ADD instead of RUN to ensure uv is available
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv pip install --system -e .

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application with multiple workers for concurrent request handling
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
