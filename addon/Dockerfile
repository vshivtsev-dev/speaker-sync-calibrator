# Named here rather than in build.yaml: the Supervisor rejects a bare
# "python:…" there and silently falls back to its own Alpine base, which has no
# Python at all. The official image is multi-arch, so --platform picks amd64 or
# aarch64 by itself.
FROM python:3.12-slim-bookworm

# The Supervisor builds with this directory as the context, so the package is
# not next to the Dockerfile the way it is for the standalone image. It is
# installed from the repository's own archive instead — no git needed, and
# numpy and scipy come as wheels, so no compiler either.
ARG SPEAKER_SYNC_REPO=https://github.com/vshivtsev-dev/speaker-sync-calibrator
ARG SPEAKER_SYNC_REF=main
# Passed by the Supervisor: the add-on version from config.yaml. Named in the
# RUN line below so every new version rebuilds that layer. Without it the line
# reads the same from one version to the next, Docker reuses the cached layer,
# and an "updated" add-on quietly keeps the old code.
ARG BUILD_VERSION=dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN echo "Speaker Sync Calibrator ${BUILD_VERSION} from ${SPEAKER_SYNC_REF}" \
    && pip install --no-cache-dir \
    "speaker-sync[ma] @ ${SPEAKER_SYNC_REPO}/archive/${SPEAKER_SYNC_REF}.tar.gz"

# /data is the add-on's persistent directory: options.json from the
# Supervisor, plus the saved positions, the probed sign and the access token.
CMD ["python", "-m", "speaker_sync.addon"]
