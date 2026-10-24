FROM python:3.11-slim

# Install OpenJDK 11 (required by PyFlink runtime) and curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download official Flink Kafka SQL Connector JAR
RUN mkdir -p /app/jars && \
    curl -fSL https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.3.0-1.19/flink-sql-connector-kafka-3.3.0-1.19.jar \
    -o /app/jars/flink-sql-connector-kafka.jar

# Install Python dependencies
COPY ingestion_flink/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy shared config and ingestion_flink scripts
COPY common /app/common
COPY ingestion_flink /app/ingestion_flink

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "ingestion_flink/replay_producer.py"]