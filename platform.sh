#!/usr/bin/env sh
python3 -c 'import distutils.util; print(distutils.util.get_platform().replace("-", "_").replace(".", "_"))'
