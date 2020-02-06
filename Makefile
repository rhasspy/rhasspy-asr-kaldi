SHELL := bash
PYTHON_NAME = rhasspyasr_kaldi

PYTHON_FILES = $(PYTHON_NAME)/*.py *.py
SHELL_FILES = bin/* *.sh
PIP_INSTALL ?= install
DOWNLOAD_DIR = download

architecture := $(shell bash architecture.sh)
platform = $(shell sh platform.sh)

.PHONY: reformat check venv dist downloads rhasspy-libs

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	black .
	isort $(PYTHON_FILES)

check:
	flake8 $(PYTHON_FILES)
	pylint $(PYTHON_FILES)
	mypy $(PYTHON_FILES)
	black --check .
	isort --check-only $(PYTHON_FILES)
	bashate $(SHELL_FILES)
	yamllint .
	pip list --outdated

venv: downloads rhasspy-libs kaldiroot etc/kaldi_flat_files.txt etc/kaldi_dir_files.txt
	scripts/create-venv.sh "$(architecture)"

dist: kaldiroot
	rm -rf dist/
	python3 setup.py bdist_wheel --plat-name $(platform)

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

# Rhasspy development dependencies
rhasspy-libs: $(DOWNLOAD_DIR)/rhasspy-asr-0.1.4.tar.gz $(DOWNLOAD_DIR)/rhasspy-nlu-0.1.6.tar.gz

$(DOWNLOAD_DIR)/rhasspy-asr-0.1.4.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-asr/archive/master.tar.gz"

$(DOWNLOAD_DIR)/rhasspy-nlu-0.1.6.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-nlu/archive/master.tar.gz"

downloads: $(DOWNLOAD_DIR)/mitlm-0.4.2-$(architecture).tar.gz $(DOWNLOAD_DIR)/phonetisaurus-2019-$(architecture).tar.gz

# Download pre-built MITLM binaries.
$(DOWNLOAD_DIR)/mitlm-0.4.2-$(architecture).tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/synesthesiam/docker-mitlm/releases/download/v0.4.2/mitlm-0.4.2-$(architecture).tar.gz"

# Download pre-built Phonetisaurus binaries.
$(DOWNLOAD_DIR)/phonetisaurus-2019-$(architecture).tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/synesthesiam/docker-phonetisaurus/releases/download/v2019.1/phonetisaurus-2019-$(architecture).tar.gz"
