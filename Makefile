SHELL := bash
PYTHON_NAME = rhasspyasr_kaldi

PYTHON_FILES = $(PYTHON_NAME)/*.py *.py
SHELL_FILES = bin/* *.sh
PIP_INSTALL ?= install
DOWNLOAD_DIR = download

version := $(shell cat VERSION)
architecture := $(shell bash architecture.sh)
platform = $(shell sh platform.sh)

.PHONY: reformat check venv sdist dist downloads rhasspy-libs debian pyinstaller

all: venv

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	scripts/format-code.sh $(PYTHON_FILES)

check:
	scripts/check-code.sh $(PYTHON_FILES)

venv: downloads rhasspy-libs etc/kaldi_flat_files.txt etc/kaldi_dir_files.txt
	scripts/create-venv.sh "$(architecture)"

test:
	scripts/run-tests.sh

dist:
	scripts/build-wheel.sh $(platform)

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

downloads: $(DOWNLOAD_DIR)/phonetisaurus-2019-$(architecture).tar.gz $(RHASSPY_DEPS)

# Download pre-built Phonetisaurus binaries.
$(DOWNLOAD_DIR)/phonetisaurus-2019-$(architecture).tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/synesthesiam/docker-phonetisaurus/releases/download/v2019.1/phonetisaurus-2019-$(architecture).tar.gz"

# Download pre-built MITLM binaries.
$(DOWNLOAD_DIR)/mitlm-0.4.2-$(architecture).tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/synesthesiam/docker-mitlm/releases/download/v0.4.2/mitlm-0.4.2-$(architecture).tar.gz"
