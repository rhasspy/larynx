#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

version="$(cat "${src_dir}/larynx/VERSION")"

# Directory for cached wheels
mkdir -p "${src_dir}/download"

# -----------------------------------------------------------------------------

: "${PLATFORMS=linux/amd64,linux/arm/v7,linux/arm64}"
: "${DOCKER_REGISTRY=docker.io}"

DOCKERFILE="${src_dir}/Dockerfile"

# Docker tags
tags=()
TAG_PREFIX="${DOCKER_REGISTRY}/rhasspy/larynx"

tags+=('--tag' "${TAG_PREFIX}")
tags+=('--tag' "${TAG_PREFIX}:${version}")

# Use docker buildx (multi-platform)
docker buildx build \
        "${src_dir}" \
        -f "${DOCKERFILE}" \
        "--platform=${PLATFORMS}" \
        "${tags[@]}" \
        --push \
        "$@"
