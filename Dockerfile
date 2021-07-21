# -----------------------------------------------------------------------------
# Dockerfile for Larynx (https://github.com/rhasspy/larynx)
# Requires Docker buildx: https://docs.docker.com/buildx/working-with-buildx/
# See scripts/build-docker.sh
# -----------------------------------------------------------------------------

FROM debian:buster-slim as build
ARG TARGETARCH
ARG TARGETVARIANT

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "Dir::Cache var/cache/apt/${TARGETARCH}${TARGETVARIANT};" > /etc/apt/apt.conf.d/01cache

RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    mkdir -p /var/cache/apt/${TARGETARCH}${TARGETVARIANT}/archives/partial && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential python3-dev

COPY requirements.txt /app/
COPY scripts/create-venv.sh /app/scripts/

# Copy wheel cache
COPY download/ /download/

# Install Larynx
ENV PIP_INSTALL='install -f /download'
ENV PIP_VERSION='pip<=20.2.4'
ENV PIP_PREINSTALL_PACKAGES='numpy==1.20.1'
RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \
    cd /app && \
    scripts/create-venv.sh

# -----------------------------------------------------------------------------

FROM debian:buster-slim as run
ARG TARGETARCH
ARG TARGETVARIANT

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "Dir::Cache var/cache/apt/${TARGETARCH}${TARGETVARIANT};" > /etc/apt/apt.conf.d/01cache

RUN --mount=type=cache,id=apt-run,target=/var/cache/apt \
    mkdir -p /var/cache/apt/${TARGETARCH}${TARGETVARIANT}/archives/partial && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 python3-pip python3-venv \
        libopenblas-base libatomic1 libgomp1 \
        sox

# Clean up
RUN rm -f /etc/apt/apt.conf.d/01cache

# Copy virtual environment
COPY --from=build /app/ /app/

# Copy source code
COPY larynx/ /app/larynx/

# Copy voices and vocoders
COPY local/ /app/local/

WORKDIR /app

EXPOSE 5002

ENV PYTHONPATH=/app

ENTRYPOINT ["/app/.venv/bin/python3", "-m", "larynx.server"]
