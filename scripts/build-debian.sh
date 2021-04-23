#!/usr/bin/env bash
set -e

if [[ -z "$1" ]]; then
    echo "Usage: build-debian dist/"
    exit 1
fi

dist_dir="$(realpath "$1")"
mkdir -p "${dist_dir}"
shift

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

version="$(cat "${src_dir}/larynx/VERSION")"
out_version="${version}"

# -----------------------------------------------------------------------------

: "${PLATFORMS=linux/amd64,linux/arm/v7,linux/arm64}"

DOCKERFILE="${src_dir}/Dockerfile.debian"

if [[ -n "${PROXY}" ]]; then
    if [[ -z "${PROXY_HOST}" ]]; then
        export PROXY_HOST="$(hostname -I | awk '{print $1}')"
    fi

    : "${APT_PROXY_HOST=${PROXY_HOST}}"
    : "${APT_PROXY_PORT=3142}"
    : "${PYPI_PROXY_HOST=${PROXY_HOST}}"
    : "${PYPI_PROXY_PORT=4000}"

    if [[ -z "NO_APT_PROXY" ]]; then
        export APT_PROXY='1'
        export APT_PROXY_HOST
        export APT_PROXY_PORT
        echo "APT proxy: ${APT_PROXY_HOST}:${APT_PROXY_PORT}"
    fi

    if [[ -z "NO_PYPI_PROXY" ]]; then
        export PYPI_PROXY='1'
        export PYPI_PROXY_HOST
        export PYPI_PROXY_PORT
        echo "PyPI proxy: ${PYPI_PROXY_HOST}:${PYPI_PROXY_PORT}"
    fi

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

echo 'Building...'

# Write .dockerignore file
DOCKERIGNORE="${src_dir}/.dockerignore"
cp -f "${src_dir}/.dockerignore.in" "${DOCKERIGNORE}"

# if [[ -n "${LANGUAGE}" ]]; then
#     # One language (exclude en-us since it's already included)
#     if [[ ! "${LANGUAGE}" == 'en-us' ]]; then
#         echo "!gruut/${LANGUAGE}" >> "${DOCKERIGNORE}"
#     else
#         # Need one file in the directory
#         echo '!gruut/.gitkeep' >> "${DOCKERIGNORE}"
#     fi

#     tags+=('--tag' "${TAG_PREFIX}:${LANGUAGE}")
#     tags+=('--tag' "${TAG_PREFIX}:${version}-${LANGUAGE}")
# else
#     # All languages
#     echo '!gruut/' >> "${DOCKERIGNORE}"
#     tags+=('--tag' "${TAG_PREFIX}")
#     tags+=('--tag' "${TAG_PREFIX}:${version}")
# fi

# if [[ -z "${VOICES}" ]]; then
#     if [[ -n "${LANGUAGE}" ]]; then
#         # All voices (one language)
#         echo "!local/${LANGUAGE}" >> "${DOCKERIGNORE}"
#     else
#         # All voices (all languages)
#         echo '!local/' >> "${DOCKERIGNORE}"
#     fi
# else
#     # Specific voices in language
#     IFS=',' read -ra voices <<< "${VOICES}"

#     # One or more voices (comma-separated)
#     for voice in "${voices[@]}"; do
#         echo "!local/${LANGUAGE}/${voice}" >> "${DOCKERIGNORE}"
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

    tag="rhasspy/rhasspy:debian-${TARGETARCH}${TARGETVARIANT}"
    docker build \
           "${src_dir}" \
           -f "${DOCKERFILE}" \
           --build-arg "TARGETARCH=${TARGETARCH}" \
           --build-arg "TARGETVARIANT=${TARGETVARIANT}" \
           --tag "${tag}" \
           "$@"

    # Determine directory to copy .deb file into
    if [[ -z "${TARGETVARIANT}" ]]; then
        output_dir="${dist_dir}/linux_${TARGETARCH}"
    else
        output_dir="${dist_dir}/linux_${TARGETARCH}_${TARGETVARIANT}"
    fi

    mkdir -p "${output_dir}"
    deb_path="$(docker run --rm "${tag}" /bin/sh -c 'ls -1 /*.deb' | head -n1)"
    echo "Copying ${deb_path} to ${output_dir}"
    docker cp "$(docker create --rm "${tag}"):${deb_path}" "${output_dir}/"
else
    # Use docker buildx (multi-platform)
    docker buildx build \
           "${src_dir}" \
           -f "${DOCKERFILE}" \
           "--platform=${PLATFORMS}" \
           --output "type=local,dest=${dist_dir}" \
           "$@"
fi

# Manually copy out
in_amd64="${dist_dir}/linux_amd64/rhasspy_${version}_amd64.deb"
out_amd64="${dist_dir}/rhasspy_${out_version}_amd64.deb"
if [[ -f "${in_amd64}" ]]; then
    cp "${in_amd64}" "${out_amd64}"
fi

in_armhf="${dist_dir}/linux_arm_v7/rhasspy_${version}_armhf.deb"
out_armhf="${dist_dir}/rhasspy_${out_version}_armhf.deb"
if [[ -f "${in_armhf}" ]]; then
    cp "${in_armhf}" "${out_armhf}"
fi

in_arm64="${dist_dir}/linux_arm64/rhasspy_${version}_arm64.deb"
out_arm64="${dist_dir}/rhasspy_${out_version}_arm64.deb"
if [[ -f "${in_arm64}" ]]; then
    cp "${in_arm64}" "${out_arm64}"
fi
