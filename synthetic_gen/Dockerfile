# Suggested Dockerfile for Suyoung's online generator (synthetic_gen/Dockerfile).
# Build context = repo root, because the service imports common/config.py.
FROM python:3.11-slim
WORKDIR /app
# CPU-only PyTorch keeps the image much smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy kafka-python matplotlib   # train_timevae.py imports matplotlib
COPY common /app/common
COPY synthetic_gen /app/synthetic_gen
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
CMD ["python", "synthetic_gen/online_generator_service.py"]
