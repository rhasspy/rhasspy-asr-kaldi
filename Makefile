SHELL := bash
PYTHON_NAME = rhasspyasr_kaldi

PYTHON_FILES = $(PYTHON_NAME)/*.py *.py
SHELL_FILES = bin/* *.sh
PIP_INSTALL ?= install
DOWNLOAD_DIR = download

version := $(shell cat VERSION)
architecture := $(shell bash architecture.sh)

.PHONY: reformat check venv sdist dist downloads debian pyinstaller

all: venv

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	scripts/format-code.sh $(PYTHON_FILES)

check:
	scripts/check-code.sh $(PYTHON_FILES)

venv: downloads
	scripts/create-venv.sh "$(architecture)"

test:
	scripts/run-tests.sh

sdist:
	python3 setup.py sdist

dist: sdist

# -----------------------------------------------------------------------------
# Debian
# -----------------------------------------------------------------------------

pyinstaller:
	scripts/build-pyinstaller.sh "${architecture}" "${version}"

debian:
	scripts/build-debian.sh "${architecture}" "${version}"

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

# Rhasspy development dependencies
RHASSPY_DEPS := $(shell grep '^rhasspy-' requirements.txt | sort | comm -3 - rhasspy_wheels.txt | sed -e 's|^|$(DOWNLOAD_DIR)/|' -e 's/==/-/' -e 's/$$/.tar.gz/')

$(DOWNLOAD_DIR)/%.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	scripts/download-dep.sh "$@"

downloads: $(RHASSPY_DEPS)
