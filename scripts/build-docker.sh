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

if [[ -n "${PROXY}" ]]; then
    if [[ -z "${PROXY_HOST}" ]]; then
        export PROXY_HOST="$(hostname -I | awk '{print $1}')"
    fi

    : "${APT_PROXY_HOST=${PROXY_HOST}}"
    : "${APT_PROXY_PORT=3142}"
    : "${PYPI_PROXY_HOST=${PROXY_HOST}}"
    : "${PYPI_PROXY_PORT=4000}"

    export APT_PROXY_HOST
    export APT_PROXY_PORT
    export PYPI_PROXY_HOST
    export PYPI_PROXY_PORT

    echo "APT proxy: ${APT_PROXY_HOST}:${APT_PROXY_PORT}"
    echo "PyPI proxy: ${PYPI_PROXY_HOST}:${PYPI_PROXY_PORT}"

    # Use temporary file instead
    temp_dockerfile="$(mktemp -p "${src_dir}")"
    function cleanup {
        rm -f "${temp_dockerfile}"
    }

    trap cleanup EXIT

    # Run through pre-processor to replace variables
    "${src_dir}/docker/preprocess.sh" < "${DOCKERFILE}" > "${temp_dockerfile}"
    DOCKERFILE="${temp_dockerfile}"
fi

# Docker tags
tags=()
TAG_PREFIX="${DOCKER_REGISTRY}/rhasspy/larynx"

# Write .dockerignore file
DOCKERIGNORE="${src_dir}/.dockerignore"
cp -f "${src_dir}/.dockerignore.in" "${DOCKERIGNORE}"

if [[ -n "${LANGUAGE}" ]]; then
    # One language (exclude en-us since it's already included)
    if [[ ! "${LANGUAGE}" == 'en-us' ]]; then
        echo "!gruut/${LANGUAGE}" >> "${DOCKERIGNORE}"
    else
        # Need one file in the directory
        echo '!gruut/.gitkeep' >> "${DOCKERIGNORE}"
    fi

    tags+=('--tag' "${TAG_PREFIX}:${LANGUAGE}")
    tags+=('--tag' "${TAG_PREFIX}:${version}-${LANGUAGE}")
else
    # All languages
    echo '!gruut/' >> "${DOCKERIGNORE}"
    tags+=('--tag' "${TAG_PREFIX}")
    tags+=('--tag' "${TAG_PREFIX}:${version}")
fi

if [[ -z "${VOICES}" ]]; then
    if [[ -n "${LANGUAGE}" ]]; then
        # All voices (one language)
        echo "!local/${LANGUAGE}" >> "${DOCKERIGNORE}"
    else
        # All voices (all languages)
        echo '!local/' >> "${DOCKERIGNORE}"
    fi
else
    # Specific voices in language
    IFS=',' read -ra voices <<< "${VOICES}"

    # One or more voices (comma-separated)
    for voice in "${voices[@]}"; do
        echo "!local/${LANGUAGE}/${voice}" >> "${DOCKERIGNORE}"
    done < <(echo "${VOICES}")
fi

# Exclude Waveglow for now
echo 'local/waveglow' >> "${DOCKERIGNORE}"

if [[ -n "${NOBUILDX}" ]]; then
    # Don't use docker buildx (single platform)

    if [[ -z "${TARGETARCH}" ]]; then
        # Guess architecture
        cpu_arch="$(uname -m)"
        case "${cpu_arch}" in
            armv6l)
                export TARGETARCH=arm
                export TARGETVARIANT=v6
                ;;

            armv7l)
                export TARGETARCH=arm
                export TARGETVARIANT=v7
                ;;

            aarch64|arm64v8)
                export TARGETARCH=arm64
                export TARGETVARIANT=''
                ;;

            *)
                # Assume x86_64
                export TARGETARCH=amd64
                export TARGETVARIANT=''
                ;;
        esac

        echo "Guessed architecture: ${TARGETARCH}${TARGETVARIANT}"
    fi

    docker build \
        "${src_dir}" \
        -f "${DOCKERFILE}" \
        --build-arg "TARGETARCH=${TARGETARCH}" \
        --build-arg "TARGETVARIANT=${TARGETVARIANT}" \
        "${tags[@]}" \
        "$@"
else
    # Use docker buildx (multi-platform)
    docker buildx build \
           "${src_dir}" \
           -f "${DOCKERFILE}" \
           "--platform=${PLATFORMS}" \
           "${tags[@]}" \
           --push \
           "$@"
fi
