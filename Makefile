SHELL := bash

.PHONY: check clean reformat dist docker amd64

all: dist

venv:
	scripts/create-venv.sh

check:
	scripts/check-code.sh

reformat:
	scripts/format-code.sh

dist:
	python3 setup.py sdist

docker:
	scripts/build-docker.sh

amd64:
	NOBUILDX=1 scripts/build-docker.sh
