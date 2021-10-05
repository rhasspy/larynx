#!/usr/bin/env bash
set -e

if [[ -z "$2" ]]; then
    echo "Usage: generate-samples.sh <LOCAL> <OUTPUT>"
    exit 1
fi

local_dir="$1"
output_dir="$2"

mkdir -p "${output_dir}"

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

# -----------------------------------------------------------------------------

declare -A pangrams
while IFS='|' read -r lang text; do
      pangrams["${lang}"]="${text}"
done < "${src_dir}/etc/pangrams.txt"

# -----------------------------------------------------------------------------

while read -r lang_dir; do
    lang="$(basename "${lang_dir}")"
    if [ "${lang}" == 'hifi_gan' ] || [ "${lang}" == 'waveglow' ]; then
        continue
    fi

    text="${pangrams["${lang}"]}"
    if [[ -z "${text}" ]]; then
        echo "No text for ${lang}"
        continue
    fi

    while read -r voice_dir; do
        voice="$(basename "${voice_dir}")"
        sample="${output_dir}/${lang}_${voice}.wav"

        if [[ ! -s "${sample}" ]]; then
            "${src_dir}/bin/larynx" \
                -v "${voice}" \
                "${text}" \
                > "${sample}"

            echo "${sample}"
        fi

    done < <(find "${lang_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

done < <(find "${local_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

# -----------------------------------------------------------------------------

echo 'Done'
