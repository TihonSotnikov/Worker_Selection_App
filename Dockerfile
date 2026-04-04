# Stage 1: Build
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU version and transformers
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefer-binary \
    'transformers>=4.30,<4.36' \
    'lm-format-enforcer>=0.4.3' \
    pytest

# Install remaining dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project code
COPY . .

# Set Python path to include genai-project
ENV PYTHONPATH=/app/genai-project:$PYTHONPATH

# Default command: run tests
CMD ["pytest", "-v", "genai-project/tests/"]
