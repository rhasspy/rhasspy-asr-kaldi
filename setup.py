import os
import sys
import platform
from pathlib import Path
import setuptools

import numpy
from Cython.Distutils import build_ext

cmdclass = {}
ext_modules = []

# -----------------------------------------------------------------------------


def find_dependencies():
    include_dirs = []
    library_dirs = []
    libraries = []

    # ATLAS include dir
    atlas_include_found = False
    for atlas_include_dir in [
        "/usr/include/atlas",
        "/usr/include/x86_64-linux-gnu/atlas",
        "/usr/include/arm-linux-gnueabihf/atlas",
        "/usr/include/aarch64-linux-gnu/atlas",
        "/usr/include/i386-linux-gnu/atlas",
    ]:
        atlas_include_dir = Path(atlas_include_dir)
        if atlas_include_dir.is_dir():
            print(f"Found ATLAS include at {atlas_include_dir}")
            atlas_include_found = True
            include_dirs.append(atlas_include_dir)
            break

    if not atlas_include_found:
        raise Exception(
            f"Missing ATLAS include directory ({atlas_include}). Please install libatlas-base-dev."
        )

    # Find libatlas.so.3
    atlas_lib_found = False
    for atlas_lib_dir in [
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/arm-linux-gnueabihf",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/i386-linux-gnu",
    ]:
        atlas_lib_dir = Path(atlas_lib_dir)
        atlas_lib_path = atlas_lib_dir / "libatlas.so.3"
        if atlas_lib_path.is_file():
            print(f"Found ATLAS library at {atlas_lib_dir}")
            atlas_lib_found = True
            libraries.append(atlas_lib_path)

            # Add other libraries
            for other_lib in [
                "libcblas.so.3",
                "libf77blas.so.3",
                "liblapack_atlas.so.3",
            ]:
                other_lib_path = atlas_lib_dir / other_lib
                if not other_lib_path.is_file():
                    raise Exception(
                        f"Missing {other_lib_path}. Please install libatlas3-base."
                    )

                libraries.append(other_lib_path)
            break

    if not atlas_lib_found:
        raise Exception("Failed to find libatlas.so.3. Please install libatlas3-base.")

    # Kaldi
    kaldi_root_file = Path("kaldiroot")
    assert kaldi_root_file.is_file(), f"Missing {kaldi_root}"
    kaldi_root = Path(os.path.expandvars(kaldi_root_file.read_text().strip()))

    if not kaldi_root.is_dir():
        raise Exception(f"Expected Kaldi root at {kaldi_root}")

    include_dirs.append(kaldi_root / "src")
    library_dirs.append(kaldi_root / "src" / "lib")
    include_dirs.append(kaldi_root / "tools" / "openfst" / "include")
    library_dirs.append(kaldi_root / "tools" / "openfst" / "lib")

    kaldi_libs = [
        "fst",
        "kaldi-base",
        "kaldi-chain",
        "kaldi-cudamatrix",
        "kaldi-decoder",
        "kaldi-feat",
        "kaldi-fstext",
        "kaldi-gmm",
        "kaldi-hmm",
        "kaldi-ivector",
        "kaldi-kws",
        "kaldi-lat",
        "kaldi-lm",
        "kaldi-matrix",
        "kaldi-nnet",
        "kaldi-nnet2",
        "kaldi-nnet3",
        "kaldi-online2",
        "kaldi-rnnlm",
        "kaldi-sgmm2",
        "kaldi-transform",
        "kaldi-tree",
        "kaldi-util",
    ]

    libraries.extend(kaldi_libs)

    # Return keyword args
    return {
        "include_dirs": [str(p) for p in include_dirs],
        "library_dirs": [str(p) for p in library_dirs],
        "libraries": [str(p) for p in libraries],
    }


# -----------------------------------------------------------------------------

ext_modules += [
    setuptools.Extension(
        "kaldi_speech.nnet3",
        sources=["kaldi_speech/nnet3.pyx", "kaldi_speech/nnet3_wrappers.cpp"],
        language="c++",
        extra_compile_args=[
            "-Wall",
            "-pthread",
            "-std=c++11",
            "-DKALDI_DOUBLEPRECISION=0",
            "-Wno-sign-compare",
            "-Wno-unused-local-typedefs",
            "-Winit-self",
            "-DHAVE_EXECINFO_H=1",
            "-DHAVE_CXXABI_H",
            "-DHAVE_ATLAS",
            "-g",
        ],
        **find_dependencies(),
    )
]

cmdclass.update({"build_ext": build_ext})

# -----------------------------------------------------------------------------

this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "README.md"), "r") as readme_file:
    long_description = readme_file.read()

with open(os.path.join(this_dir, "requirements.txt"), "r") as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(this_dir, "VERSION"), "r") as version_file:
    version = version_file.read().strip()

setuptools.setup(
    name="rhasspy-asr-kaldi",
    version=version,
    author="Michael Hansen",
    author_email="hansen.mike@gmail.com",
    url="https://github.com/synesthesiam/rhasspy-asr-kaldi",
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Cython",
        "Programming Language :: C++",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)
