# Use official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

# Set working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in development mode so stock-tracker command is available
RUN pip install -e .

# Create logs directory and set permissions
RUN mkdir -p logs && \
    chmod -R 755 logs

# Create non-root user for security
RUN useradd -m appuser && \
    chown -R appuser:appuser $APP_HOME

# Switch to non-root user
USER appuser

# Expose port (if needed for web interface in future)
EXPOSE 8000

# Default command - can be overridden
ENTRYPOINT ["stock-tracker"]
CMD ["--help"]
