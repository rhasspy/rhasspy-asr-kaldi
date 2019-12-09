SHELL := bash

.PHONY: check virtualenv

virtualenv:
	python3 -m venv .venv
	source .venv/bin/activate
	python3 -m pip install wheel setuptools
	python3 -m pip -r requirements
