#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

dist_dir="${src_dir}/dist/voices"
mkdir -p "${dist_dir}"

local_dir="${src_dir}/local"
pushd "${local_dir}" > /dev/null

find "${local_dir}" -mindepth 2 -maxdepth 2 -type d | \
    while read -r voice_dir;
    do
        voice="$(basename "${voice_dir}")"
        lang_dir="$(dirname "${voice_dir}")"
        lang="$(basename "${lang_dir}")"

        voice_file="${dist_dir}/${lang}_${voice}.tar.gz"

        rm -f "${voice_file}"
        tar -czf "${voice_file}" "${lang}/${voice}"

        echo "${voice_file}"
    done

popd > /dev/null
