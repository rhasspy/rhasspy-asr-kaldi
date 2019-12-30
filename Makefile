SHELL := bash

.PHONY: check virtualenv dist

venv:
	rm -rf .venv/
	python3 -m venv .venv
	.venv/bin/pip3 install wheel setuptools
	.venv/bin/pip3 install -r requirements_all.txt

check:
	flake8 rhasspyasr_kaldi/*.py
	pylint rhasspyasr_kaldi/*.py

dist:
	rm -rf dist/
	python3 kaldi_setup.py build_ext
	python3 setup.py bdist_wheel
	bash bin/fix_rpath.sh "$(shell cat kaldiroot | envsubst)" dist/rhasspy*.whl
