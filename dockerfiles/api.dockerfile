FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY configs /app/configs
COPY pyproject.toml /app/pyproject.toml

ENV PYTHONPATH=/app/src

CMD ["sh", "-c", "python -m uvicorn mlops_project.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
