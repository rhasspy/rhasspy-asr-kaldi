#!/usr/bin/env bash
kaldi_root="$1"
flat_files="$2"
dir_files="$3"
output="$4"

if [[ -z "${output}" ]]; then
    echo "Usage: install-kaldi.sh KALDIROOT flat-files.txt dir-files.txt output-dir/"
    exit 1
fi

rm -rf "${output}/kaldi"
mkdir -p "${output}/kaldi"

# Flat files (ignore directory structure)
while read -r path_part
do
    input_path="${kaldi_root}/${path_part}"
    output_path="${output}/kaldi"

    if [[ -d "${input_path}" ]]; then \
        # Copy entire directory
        cp --recursive --dereference "${input_path}"/* "${output_path}/"
    else
        # Copy single file
        cp "${input_path}" "${output_path}/"
    fi
done < "${flat_files}"

# Fix rpaths (assume flat files are bin/libs)
find "${output}/kaldi" -type f -exec patchelf --set-rpath '$ORIGIN' {} \;

# Dir files (keep directory structure)
while read -r path_part
do
    input_path="${kaldi_root}/${path_part}"
    output_path="${output}/kaldi/${path_part}"

    if [[ -d "${input_path}" ]]; then \
        # Copy entire directory
        mkdir -p "${output_path}"
        cp --recursive --dereference "${input_path}"/* "${output_path}/"
    else
        # Copy single file
        mkdir -p "$(dirname "${output_path}")"
        cp "${input_path}" "${output_path}"
    fi
done < "${dir_files}"
