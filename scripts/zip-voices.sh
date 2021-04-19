#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

dist_dir="${src_dir}/dist/voices"
mkdir -p "${dist_dir}"

local_dir="${src_dir}/local"
pushd "${local_dir}" > /dev/null

function zip_voice {
    voice_dir="$1"

    voice="$(basename "${voice_dir}")"
    lang_dir="$(dirname "${voice_dir}")"
    lang="$(basename "${lang_dir}")"

    voice_file="${dist_dir}/${lang}_${voice}.tar.gz"

    rm -f "${voice_file}"
    tar -czf "${voice_file}" "${lang}/${voice}"

    echo "${voice_file}"
}

if [[ -z "$1" ]]; then
    # All voices
    find "${local_dir}" -mindepth 2 -maxdepth 2 -type d | \
        while read -r voice_dir;
        do
            zip_voice "${voice_dir}"
        done
else
    # Specific voices
    while [[ -n "$1" ]]; do
        zip_voice "$1"
        shift 1
    done
fi

popd > /dev/null
