# -----------------------------------------------------------------------------
# Dockerfile for Larynx (https://github.com/rhasspy/larynx)
# Requires Docker buildx: https://docs.docker.com/buildx/working-with-buildx/
# See scripts/build-docker.sh
# -----------------------------------------------------------------------------

FROM python:3.7-buster as build
ARG TARGETARCH
ARG TARGETVARIANT

ENV LANG C.UTF-8

RUN --mount=type=cache,id=apt-build,target=/var/apt/cache \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential python3-dev

# Create virtual environment
RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \
    python3 -m venv /app/.venv && \
    /app/.venv/bin/pip3 install --upgrade 'pip<=20.2.4' && \
    /app/.venv/bin/pip3 install --upgrade wheel setuptools

# Copy wheel cache
COPY download/ /download/

# Install Larynx
COPY requirements.txt /app/

RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \
    /app/.venv/bin/pip3 install -f /download -r /app/requirements.txt

# Copy and delete extranous gruut data files
COPY gruut/ /gruut/
RUN mkdir -p /gruut && \
    cd /gruut && \
    find . -name lexicon.txt -delete

# -----------------------------------------------------------------------------

FROM python:3.7-buster as run

ENV LANG C.UTF-8

RUN --mount=type=cache,id=apt-run,target=/var/apt/cache \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 python3-pip python3-venv \
        libopenblas-base libatomic1 libgomp1

# Copy virtual environment
COPY --from=build /app/ /app/

# Copy gruut data files
COPY --from=build /gruut/ /app/gruut/

# Copy voices and vocoders
COPY local/ /app/local/

# Copy source code
COPY larynx/ /app/larynx/

WORKDIR /app

EXPOSE 5002

ENTRYPOINT ["/app/.venv/bin/python3", "-m", "larynx.server"]
