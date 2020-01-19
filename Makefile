SHELL := bash

PYTHON_FILES = rhasspyasr_kaldi/*.py *.py
SHELL_FILES = bin/* *.sh

.PHONY: reformat check virtualenv dist

reformat:
	black .
	isort $(PYTHON_FILES)

venv:
	rm -rf .venv/
	python3 -m venv .venv
	.venv/bin/pip3 install --upgrade pip	
	.venv/bin/pip3 install wheel setuptools
	.venv/bin/pip3 install -r requirements.txt
	.venv/bin/pip3 install -r requirements_dev.txt

check:
	flake8 $(PYTHON_FILES)
	pylint $(PYTHON_FILES)
	mypy $(PYTHON_FILES)
	black --check .
	isort --check-only $(PYTHON_FILES)
	bashate $(SHELL_FILES)
	yamllint .
	pip list --outdated

dist:
	rm -rf dist/
	python3 kaldi_setup.py build_ext
	python3 setup.py bdist_wheel
	bash bin/fix_rpath.sh "$(shell cat kaldiroot | envsubst)" dist/rhasspy*.whl
