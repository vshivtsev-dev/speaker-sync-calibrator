ARG BUILD_FROM
FROM ${BUILD_FROM}

# The Supervisor builds with this directory as the context, so the package is
# not next to the Dockerfile the way it is for the standalone image. It is
# installed from the repository's own archive instead — no git needed, and
# numpy and scipy come as wheels, so no compiler either.
ARG SPINALIGN_REPO=https://github.com/vshivtsev-dev/audio-sync
ARG SPINALIGN_REF=main

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir \
    "spinalign[ma] @ ${SPINALIGN_REPO}/archive/${SPINALIGN_REF}.tar.gz"

# /data is the add-on's persistent directory: options.json from the
# Supervisor, plus the saved positions, the probed sign and the access token.
CMD ["python", "-m", "spinalign.addon"]
