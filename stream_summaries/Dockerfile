FROM python:3.11-slim-bookworm

WORKDIR /app

COPY stream_summaries/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY stream_summaries /app/stream_summaries

ENV PYTHONUNBUFFERED=1

CMD ["python", "stream_summaries/feature_summaries.py"]
