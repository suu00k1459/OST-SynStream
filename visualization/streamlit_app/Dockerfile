# Streamlit app for the Batch 2 / Batch 3 results and the PCA / t-SNE tab (Hadil).
# Build context = repository root (see visualization/docker-compose.streamlit.yml).
FROM python:3.11-slim-bookworm

WORKDIR /app

COPY visualization/streamlit_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY visualization/streamlit_app /app/visualization/streamlit_app

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# The result files (stream_analytics/results, synthetic_gen/outputs) are mounted read-only by compose.
CMD ["streamlit", "run", "visualization/streamlit_app/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
