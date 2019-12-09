#!/usr/bin/env bash
set -e

if [[ -z "$2" ]]; then
    echo "Usage: fix_rpath.sh KALDI WHEEL"
    exit 1;
fi

this_dir="$( cd "$( dirname "$0" )" && pwd )"
kaldi="$(realpath "$1")"
wheel="$(realpath "$2")"

# Create temporary directory and clean it up when script finishes
temp_dir="$(mktemp -d)"
function finish {
    rm -rf "${temp_dir}"
}

trap finish EXIT

unzip -d "${temp_dir}" "${wheel}"

# Copy Kaldi libs inside
cp "${kaldi}"/tools/openfst/lib/libfst.so* "${temp_dir}/kaldi_speech/"
kaldi_libs=(
    'kaldi-base'
    'kaldi-chain'
    'kaldi-cudamatrix'
    'kaldi-decoder'
    'kaldi-feat'
    'kaldi-fstext'
    'kaldi-gmm'
    'kaldi-hmm'
    'kaldi-ivector'
    'kaldi-kws'
    'kaldi-lat'
    'kaldi-lm'
    'kaldi-matrix'
    'kaldi-nnet'
    'kaldi-nnet2'
    'kaldi-nnet3'
    'kaldi-online2'
    'kaldi-rnnlm'
    'kaldi-sgmm2'
    'kaldi-transform'
    'kaldi-tree'
    'kaldi-util'
)

for lib_name in "${kaldi_libs[@]}"; do
    cp "${kaldi}/src/lib/lib${lib_name}".so* "${temp_dir}/kaldi_speech/"
done

# Patch ORIGIN
for so_file in "${temp_dir}/kaldi_speech"/*.so; do
    patchelf --set-rpath '$ORIGIN' "${so_file}";
done

pushd "${temp_dir}" && zip -r "${wheel}" * && popd

