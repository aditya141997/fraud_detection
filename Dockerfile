FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
COPY deploy ./deploy
ENV FEAST_REPO_PATH=/app/deploy/feast \
    KAFKA_BOOTSTRAP=redpanda.railway.internal:9092 \
    PYTHONUNBUFFERED=1
RUN chmod +x deploy/entrypoint.sh
EXPOSE 8000
CMD ["deploy/entrypoint.sh"]
