FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY pyproject.toml ./
COPY emotext ./emotext
RUN pip install --no-cache-dir ".[gtts]"

ENV EMOTEXT_DATA_DIR=/data
RUN mkdir -p /data && chown app:app /data
USER app

EXPOSE 8000
CMD ["uvicorn", "emotext.api:app", "--host", "0.0.0.0", "--port", "8000"]
