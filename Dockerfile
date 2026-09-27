FROM python:3.11-slim

# numpy and scipy ship manylinux wheels for the architectures this runs on,
# so no compiler is needed and the image stays a single thin stage.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SPEAKER_SYNC_HOST=0.0.0.0 \
    SPEAKER_SYNC_PORT=8080 \
    SPEAKER_SYNC_STATE_DIR=/data

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[ma]" \
    && useradd --create-home --uid 10001 speaker-sync \
    && mkdir -p /data \
    && chown speaker-sync /data

USER speaker-sync

# Saved listening positions and the probed sync_adjust sign. Mount a volume
# here: without one they are lost on every restart, and re-establishing the
# sign costs two full measurement passes.
VOLUME ["/data"]

# Fixed inside the container; map it to whatever you like from outside. The
# health check below assumes this value.
EXPOSE 8080

# /healthz is deliberately outside the access-token check so this works
# without baking a secret into the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).status == 200 else 1)"

# TLS is the reverse proxy's job — this speaks plain HTTP.
CMD ["speaker-sync", "serve"]
