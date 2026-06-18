FROM python:3.11-slim

WORKDIR /app

# System deps for statsmodels / numpy build (if needed from source)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Parquet data path — override via env var or volume mount
ENV PARQUET_DIR=/data/parquet

EXPOSE 8000

CMD ["uvicorn", "signal_server:app", "--host", "0.0.0.0", "--port", "8000"]
