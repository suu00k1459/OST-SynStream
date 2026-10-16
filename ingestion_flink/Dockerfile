# ingestion_flink/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first for layer caching
COPY ingestion_flink/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy shared config and producer script
COPY common /app/common
COPY ingestion_flink /app/ingestion_flink

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "ingestion_flink/replay_producer.py"]