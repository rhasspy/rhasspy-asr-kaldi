"""Setup for rhasspyasr_kaldi"""
import os
import typing
from pathlib import Path

import setuptools

# -----------------------------------------------------------------------------


class BinaryDistribution(setuptools.Distribution):
    """Enable packaging of binary artifacts."""

    # pylint: disable=R0201
    def has_ext_modules(self, _):
        """Will have binary artifacts."""
        return True


def get_kaldi_files() -> typing.List[str]:
    """Gets paths to all files installed with scripts/install_kaldi.sh"""
    module_dir = Path("rhasspyasr_kaldi")
    return [str(p.relative_to(module_dir)) for p in (module_dir / "kaldi").glob("**/*")]


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
    url="https://github.com/rhasspy/rhasspy-asr-kaldi",
    dist_class=BinaryDistribution,
    packages=setuptools.find_packages(),
    package_data={
        "rhasspyasr_kaldi": [
            "py.typed",
            "phonetisaurus-apply",
            "phonetisaurus-g2pfst",
            "libfst.so.13",
            "libfstfar.so.13",
            "libfstngram.so.13",
        ]
        + get_kaldi_files()
    },
    install_requires=requirements,
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)
