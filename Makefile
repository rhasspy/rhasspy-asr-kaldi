SHELL := bash

.PHONY: check virtualenv dist

virtualenv:
	python3 -m venv .venv
	source .venv/bin/activate
	python3 -m pip install wheel setuptools
	python3 -m pip install -r requirements

check:
	flake8 rhasspyasr_kaldi/*.py
	pylint rhasspyasr_kaldi/*.py

dist:
	python3 setup.py bdist_wheel
	bash bin/fix_rpath.sh `cat kaldiroot` dist/rhasspy*.whl
