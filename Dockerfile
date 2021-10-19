# -----------------------------------------------------------------------------
# Dockerfile for Larynx (https://github.com/rhasspy/larynx)
#
# Requires Docker buildx: https://docs.docker.com/buildx/working-with-buildx/
# See scripts/build-docker.sh
# -----------------------------------------------------------------------------

FROM debian:bullseye as build
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

# Clean up
RUN rm -f /etc/apt/apt.conf.d/01cache

RUN mkdir -p /home/larynx/app
WORKDIR /home/larynx/app

# Create virtual environment
RUN --mount=type=cache,id=pip-venv,target=/root/.cache/pip \
    python3 -m venv .venv && \
    .venv/bin/pip3 install --upgrade pip && \
    .venv/bin/pip3 install --upgrade wheel setuptools

# Put cached Python wheels here
COPY download/ /download/

COPY requirements.txt requirements.txt

# Install base Python requirements (excluding PyTorch)
RUN --mount=type=cache,id=pip-requirements,target=/root/.cache/pip \
    grep -v '^torch' requirements.txt | \
    xargs .venv/bin/pip3 install \
    -f /download \
    -f 'https://synesthesiam.github.io/prebuilt-apps/'

# -----------------------------------------------------------------------------

FROM debian:bullseye as run
ARG TARGETARCH
ARG TARGETVARIANT

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "Dir::Cache var/cache/apt/${TARGETARCH}${TARGETVARIANT};" > /etc/apt/apt.conf.d/01cache

RUN --mount=type=cache,id=apt-run,target=/var/cache/apt \
    mkdir -p /var/cache/apt/${TARGETARCH}${TARGETVARIANT}/archives/partial && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 sox ca-certificates

RUN --mount=type=cache,id=apt-run,target=/var/cache/apt \
    if [ "${TARGETARCH}${TARGETVARIANT}" = 'armv7' ]; then \
        apt-get install --yes --no-install-recommends libatlas3-base libgfortran5; \
    fi

# Clean up
RUN rm -f /etc/apt/apt.conf.d/01cache

RUN useradd -ms /bin/bash larynx

# Copy virtual environment
COPY --from=build /home/larynx/app/ /home/larynx/app/

# Copy source code
COPY larynx/ /home/larynx/app/larynx/

# Copy voices and vocoders
COPY local/ /home/larynx/app/local/

USER larynx
WORKDIR /home/larynx/app

EXPOSE 5002

ENV PYTHONPATH=/home/larynx/app

ENTRYPOINT ["/home/larynx/app/.venv/bin/python3", "-m", "larynx.server"]
