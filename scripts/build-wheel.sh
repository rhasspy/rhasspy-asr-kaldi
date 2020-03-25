#!/usr/bin/env bash
set -e

platform="$1"
if [[ -z "${platform}" ]];
then
    echo "Usage: build-wheel.sh platform"
    exit 1
fi

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

venv="${src_dir}/.venv"
if [[ -d "${venv}" ]]; then
    echo "Using virtual environment at ${venv}"
    source "${venv}/bin/activate"
fi

# -----------------------------------------------------------------------------

dist="${src_dir}/dist"

rm -rf "${dist}"
python3 setup.py bdist_wheel --plat-name "${platform}"

# -----------------------------------------------------------------------------

echo "OK"
