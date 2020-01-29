SHELL := bash
PYTHON_NAME = rhasspyasr_kaldi

PYTHON_FILES = $(PYTHON_NAME)/*.py *.py
SHELL_FILES = bin/* *.sh
PIP_INSTALL ?= install

architecture := $(shell bash architecture.sh)
platform = $(shell sh platform.sh)

.PHONY: reformat check venv venv-init dist

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

venv-init: mitlm-0.4.2-$(architecture).tar.gz phonetisaurus-2019-$(architecture).tar.gz
	rm -rf .venv/
	python3 -m venv .venv
	.venv/bin/pip3 $(PIP_INSTALL) --upgrade pip
	.venv/bin/pip3 $(PIP_INSTALL) wheel setuptools
	.venv/bin/pip3 $(PIP_INSTALL) -r requirements.txt
	.venv/bin/pip3 $(PIP_INSTALL) -r requirements_dev.txt
	scripts/install-mitlm.sh \
        "${CURDIR}/mitlm-0.4.2-$(architecture).tar.gz" \
        "${CURDIR}/${PYTHON_NAME}"
	scripts/install-phonetisaurus.sh \
        "${CURDIR}/phonetisaurus-2019-$(architecture).tar.gz" \
        "${CURDIR}/${PYTHON_NAME}"

venv: venv-init kaldiroot etc/kaldi_flat_files.txt etc/kaldi_dir_files.txt
	scripts/install-kaldi.sh \
        "$(shell cat kaldiroot | envsubst)" \
        "${CURDIR}/etc/kaldi_flat_files.txt" \
        "${CURDIR}/etc/kaldi_dir_files.txt" \
        "${CURDIR}/${PYTHON_NAME}"
	.venv/bin/python3 kaldi_setup.py build_ext
	find build/ -type f -name 'nnet3*.so' -exec cp {} "$(PYTHON_NAME)/" \;
	find "$(PYTHON_NAME)" -type f -name 'nnet3*.so' -exec patchelf --set-rpath '$$ORIGIN' {} \;

dist: kaldiroot
	rm -rf dist/
	python3 setup.py bdist_wheel --plat-name $(platform)

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

# Download pre-built MITLM binaries.
mitlm-0.4.2-$(architecture).tar.gz:
	curl -sSfL -o $@ "https://github.com/synesthesiam/docker-mitlm/releases/download/v0.4.2/mitlm-0.4.2-$(architecture).tar.gz"

# Download pre-built Phonetisaurus binaries.
phonetisaurus-2019-$(architecture).tar.gz:
	curl -sSfL -o $@ "https://github.com/synesthesiam/docker-phonetisaurus/releases/download/v2019.1/phonetisaurus-2019-$(architecture).tar.gz"
