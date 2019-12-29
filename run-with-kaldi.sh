#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"

kaldi_dir="$(cat "${this_dir}/kaldiroot" | envsubst)"
kaldi_dir="$(realpath "${kaldi_dir}")"

if [[ -z "${kaldi_dir}" ]]; then
    echo "No kaldiroot"
    exit 1
fi

export kaldi_dir

# Required bin/lib directories
lib_dir="${kaldi_dir}/src/lib"
openfst_dir="${kaldi_dir}/tools/openfst"
utils_dir="${kaldi_dir}/egs/wsj/s5/utils"
steps_dir="${kaldi_dir}/egs/wsj/s5/steps"

# Set up paths for Kaldi programs
export PATH="${this_dir}/bin:${kaldi_dir}/src/featbin:${kaldi_dir}/src/latbin:${kaldi_dir}/src/gmmbin:${kaldi_dir}/src/online2bin:$PATH"
export LD_LIBRARY_PATH="${lib_dir}:${openfst_dir}/lib:${LD_LIBRARY_PATH}"

python3 -m rhasspyasr_kaldi "$@"
