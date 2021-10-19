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
: "${DOCKERFILE=Dockerfile.debian}"

echo 'Building...'

# Use docker buildx (multi-platform)
docker buildx build \
        "${src_dir}" \
        -f "${DOCKERFILE}" \
        "--platform=${PLATFORMS}" \
        --output "type=local,dest=${dist_dir}" \
        "$@"

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
