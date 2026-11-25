FROM python:3.11-slim-bookworm

WORKDIR /app

COPY alert_evaluation/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY alert_evaluation /app/alert_evaluation

ENV PYTHONUNBUFFERED=1

CMD ["python", "alert_evaluation/online_alert_eval.py"]
