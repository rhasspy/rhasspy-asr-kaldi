SHELL := bash
PYTHON_NAME = rhasspyasr_kaldi

PYTHON_FILES = $(PYTHON_NAME)/*.py *.py
SHELL_FILES = bin/* *.sh

architecture := $(shell bash architecture.sh)
platform = $(shell sh platform.sh)

.PHONY: reformat check venv venv-init dist

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
	.venv/bin/pip3 install --upgrade pip
	.venv/bin/pip3 install wheel setuptools
	.venv/bin/pip3 install -r requirements.txt
	tar -C $(PYTHON_NAME)/ -xvf mitlm-0.4.2-$(architecture).tar.gz \
      --strip-components=2 \
      mitlm/bin/estimate-ngram mitlm/lib/libmitlm.so.1
	patchelf --set-rpath '$$ORIGIN' $(PYTHON_NAME)/estimate-ngram
	tar -C $(PYTHON_NAME)/ -xvf phonetisaurus-2019-$(architecture).tar.gz \
	  --strip-components=2 \
      ./bin/phonetisaurus-apply ./bin/phonetisaurus-g2pfst \
      ./lib/libfst.so.13.0.0 ./lib/libfstfar.so.13.0.0 ./lib/libfstngram.so.13.0.0
	patchelf --set-rpath '$$ORIGIN' $(PYTHON_NAME)/phonetisaurus-g2pfst
	for f in libfst.so.13 libfstfar.so.13 libfstngram.so.13; do \
      mv $(PYTHON_NAME)/$${f}.0.0 $(PYTHON_NAME)/$${f}; \
      patchelf --set-rpath '$$ORIGIN' $(PYTHON_NAME)/$${f}; done

venv: venv-init kaldiroot etc/kaldi_dir_files.txt etc/kaldi_flat_files.txt
	rm -rf $(PYTHON_NAME)/kaldi
	mkdir -p $(PYTHON_NAME)/kaldi
	while read -r path_part; do \
      input_path=$(shell cat kaldiroot)/$${path_part}; \
      output_path=$(PYTHON_NAME)/kaldi/; \
      if [[ -d $${input_path} ]]; then \
        cp --recursive --dereference $${input_path}/* $${output_path}/; \
      else \
        cp $${input_path} $${output_path}/; \
      fi; done < etc/kaldi_flat_files.txt
	find $(PYTHON_NAME)/kaldi -type f -exec patchelf --set-rpath '$$ORIGIN' {} \;
	while read -r path_part; do \
      input_path=$(shell cat kaldiroot)/$${path_part}; \
      output_path=$(PYTHON_NAME)/kaldi/$${path_part}/; \
      mkdir -p $${output_path}/; \
      if [[ -d $${input_path} ]]; then \
        cp --recursive --dereference $${input_path}/* $${output_path}/; \
      else \
        cp $${input_path} $${output_path}/; \
      fi; done < etc/kaldi_dir_files.txt


dist:
	rm -rf dist/
	python3 kaldi_setup.py build_ext
	python3 setup.py bdist_wheel --plat-name $(platform)
	bash bin/fix_rpath.sh "$(shell cat kaldiroot | envsubst)" dist/rhasspy*.whl

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

# Download pre-built MITLM binaries.
mitlm-0.4.2-$(architecture).tar.gz:
	curl -sSfL -o $@ "https://github.com/synesthesiam/docker-mitlm/releases/download/v0.4.2/mitlm-0.4.2-$(architecture).tar.gz"

# Download pre-built Phonetisaurus binaries.
phonetisaurus-2019-$(architecture).tar.gz:
	curl -sSfL -o $@ "https://github.com/synesthesiam/docker-phonetisaurus/releases/download/v2019.1/phonetisaurus-2019-$(architecture).tar.gz"
