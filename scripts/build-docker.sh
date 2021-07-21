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

# Write .dockerignore file
# DOCKERIGNORE="${src_dir}/.dockerignore"
# cp -f "${src_dir}/.dockerignore.in" "${DOCKERIGNORE}"

# if [[ -n "${LARYNX_LANGUAGE}" ]]; then
#     # One language (exclude en-us since it's already included)
#     if [[ ! "${LARYNX_LANGUAGE}" == 'en-us' ]]; then
#         echo "!gruut/${LARYNX_LANGUAGE}" >> "${DOCKERIGNORE}"
#     else
#         # Need one file in the directory
#         echo '!gruut/.gitkeep' >> "${DOCKERIGNORE}"
#     fi

#     tags+=('--tag' "${TAG_PREFIX}:${LARYNX_LANGUAGE}")
#     tags+=('--tag' "${TAG_PREFIX}:${version}-${LARYNX_LANGUAGE}")
# else
#     # All languages
#     echo '!gruut/' >> "${DOCKERIGNORE}"
#     tags+=('--tag' "${TAG_PREFIX}")
#     tags+=('--tag' "${TAG_PREFIX}:${version}")
# fi

# if [[ -z "${VOICES}" ]]; then
#     if [[ -n "${LARYNX_LANGUAGE}" ]]; then
#         # All voices (one language)
#         echo "!local/${LARYNX_LANGUAGE}" >> "${DOCKERIGNORE}"
#     else
#         # All voices (all languages)
#         echo '!local/' >> "${DOCKERIGNORE}"
#     fi
# else
#     # Specific voices in language
#     IFS=',' read -ra voices <<< "${VOICES}"

#     # One or more voices (comma-separated)
#     for voice in "${voices[@]}"; do
#         echo "!local/${LARYNX_LANGUAGE}/${voice}" >> "${DOCKERIGNORE}"
#     done < <(echo "${VOICES}")
# fi

# # Exclude Waveglow for now
# echo 'local/waveglow' >> "${DOCKERIGNORE}"

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
