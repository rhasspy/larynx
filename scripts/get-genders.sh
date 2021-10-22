#!/usr/bin/env bash
set -euo pipefail

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

local_dir="${src_dir}/local"

find "${local_dir}" -mindepth 2 -maxdepth 2 -type d | \
    while read -r voice_dir;
    do
        voice_lang="$(dirname "${voice_dir}" | xargs basename)"
        if [ "${voice_lang}" = 'hifi_gan' ] || [ "${voice_lang}" = 'waveglow' ]; then
            continue;
        fi

        voice_name="$(basename "${voice_dir}")"
        voice_gender="$(cat "${voice_dir}/GENDER")"

        full_voice_name="${voice_lang}_${voice_name}"
        echo "${full_voice_name}" "${voice_gender}"
    done
