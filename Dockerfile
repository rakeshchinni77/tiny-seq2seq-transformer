# Use official slim Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    OUTPUT_DIR=/app/output \
    DEVICE=cpu \
    SEED=42

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (CPU PyTorch)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, and documentation
COPY src /app/src
COPY tests /app/tests
COPY docs /app/docs
COPY submission.json /app/submission.json
COPY .env.example /app/.env.example
COPY README.md /app/README.md

# Ensure data and output directories exist
RUN mkdir -p /app/data /app/output

# Run pipeline: generate -> train -> eval
CMD ["sh", "-c", "python src/data/generate.py && python src/train/train.py && python src/eval/evaluate.py"]
