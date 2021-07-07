#!/usr/bin/env bash
set -e

if [[ -z "$1" ]]; then
    echo "Usage: build-debian-lang dist/"
    exit 1
fi

dist_dir="$(realpath "$1")"
mkdir -p "${dist_dir}"
shift

only_langs=("$@")

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

version="$(cat "${src_dir}/larynx/VERSION")"

temp_dir="$(mktemp -d)"
function cleanup {
    rm -rf "${temp_dir}"
}

trap cleanup EXIT

# -----------------------------------------------------------------------------

declare -A lang_names
lang_names['de-de']='German'
lang_names['en-us']='U.S. English'
lang_names['es-es']='Spanish'
lang_names['fr-fr']='French'
lang_names['it-it']='Italian'
lang_names['nl']='Dutch'
lang_names['ru-ru']='Russian'
lang_names['sv-se']='Swedish'

# Dummy package, since en-us is included with Larynx
mkdir -p "${src_dir}/gruut/en-us"

find "${src_dir}/gruut" -mindepth 1 -maxdepth 1 -type d | \
    while read -r lang_dir; do
        lang="$(basename "${lang_dir}")"
        lang_name="${lang_names["${lang}"]}"

        if [[ -z "${lang_name}" ]]; then
            echo "No language name for ${lang}";
            exit 1;
        fi

        if [[ -n "${only_langs[@]}" ]]; then
            skip_lang='yes'
            for check_lang in "${only_langs[@]}"; do
                if [[ "${check_lang}" == "${lang}" ]]; then
                    # Language was found in list
                    skip_lang=''
                    break;
                fi
            done
        fi

        if [[ -n "${skip_lang}" ]]; then
            echo "Skipping ${lang}";
            continue;
        fi

        package_dir="${temp_dir}/larynx-tts-lang-${lang}"
        mkdir -p "${package_dir}/DEBIAN" "${package_dir}/usr/lib/larynx-tts/gruut"
        sed -e "s/@LANGUAGE@/${lang}/" \
            -e "s/@LANGUAGE_NAME@/${lang_name}/" \
            -e "s/@VERSION@/${version}/" \
            < "${src_dir}/debian/control.lang.in" \
            > "${package_dir}/DEBIAN/control"

        cp -R "${lang_dir}" "${package_dir}/usr/lib/larynx-tts/gruut/"

        pushd "${temp_dir}"
        dpkg --build "$(basename "${package_dir}")"
        dpkg-name *.deb
        mv *.deb "${dist_dir}/"
        popd

        rm -rf "${package_dir}"
    done
