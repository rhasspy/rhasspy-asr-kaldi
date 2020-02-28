#!/usr/bin/env bash
if [[ -z "$1" ]]; then
    echo "Usage: download-dep.sh <DEPENDENCY>"
fi

# download/rhasspy-foo-bar-0.X.Y.tar.gz
dest_file="$1"

# rhasspy-foo-bar-0.X.Y.tar.gz
tar_gz="$(basename "${dest_file}")"

# rhasspy-foo-bar
dep_name="$(echo "${tar_gz}" | sed -E 's|-[0-9].+||')"

url="https://github.com/rhasspy/${dep_name}/archive/master.tar.gz"
echo "${url} => ${dest_file}"
wget -O "${dest_file}" "${url}"

