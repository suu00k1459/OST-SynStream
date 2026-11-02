FROM python:3.11-slim-bookworm

WORKDIR /app

# Install Python dependencies
COPY storage_influx/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy shared config and storage_influx scripts
COPY common /app/common
COPY storage_influx /app/storage_influx

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "storage_influx/kafka_to_influx.py"]