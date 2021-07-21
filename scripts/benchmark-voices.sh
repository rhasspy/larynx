#!/usr/bin/env bash
set -e

if [[ -z "$2" ]]; then
    echo "Usage: benchmark-voices.sh <LOCAL> <OUTPUT>"
    exit 1
fi

local_dir="$1"
output_dir="$2"

mkdir -p "${output_dir}"

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

time_prog="$(which time)"

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

        for quality in high medium low; do
            # Burn one
            "${src_dir}/bin/larynx" \
                -v "${voice}" \
                -q "${quality}" \
                "${text}" \
                > /dev/null \
                2>&1

            for run in 1 2 3; do
                prefix="${output_dir}/${voice}.${quality}.${run}"
                stats_path="${prefix}.txt"

                truncate -s 0 "${stats_path}"

                "${time_prog}" "--output=${prefix}.txt" --append -- \
                     "${src_dir}/bin/larynx" \
                     -v "${voice}" \
                     -q "${quality}" \
                     --debug \
                     "${text}" \
                     2>> "${stats_path}" \
                     > "${prefix}.wav"

                echo "${lang} ${voice} ${quality} ${run}"
            done
        done

    done < <(find "${lang_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

done < <(find "${local_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

# -----------------------------------------------------------------------------

echo 'Done'
