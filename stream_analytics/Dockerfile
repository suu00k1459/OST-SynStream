# Hazem's analytics services. One image, three possible commands:
#   docker run ... python river_anomaly_drift.py --dataset uci --kafka --bootstrap kafka:9092
#   docker run ... python river_anomaly_drift.py --dataset hai --kafka --bootstrap kafka:9092
#   docker run ... python window_comparator.py --kafka --bootstrap kafka:9092
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY results/ ./results/
ENV PYTHONUNBUFFERED=1
CMD ["python", "window_comparator.py", "--kafka", "--bootstrap", "kafka:9092"]
