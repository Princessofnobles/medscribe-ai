FROM python:3.11-slim

LABEL maintainer="ParvinAI Agency"
LABEL description="Medical Transcription & SOAP Note Generator"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/main.py .
COPY app/templates/ ./templates/

RUN mkdir -p /app/audio /app/outputs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "300", "main:app"]
