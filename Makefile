SHELL := bash

.PHONY: check clean reformat dist docker amd64 index test

all: dist

venv:
	scripts/create-venv.sh

check:
	scripts/check-code.sh

reformat:
	scripts/format-code.sh

dist:
	python3 setup.py sdist

test:
	scripts/run-tests.sh

docker:
	scripts/build-docker.sh

index:
	bin/make_sample_html.py local/ > index.html
