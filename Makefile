SHELL := bash

.PHONY: check clean reformat dist docker index test genders

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

genders:
	scripts/get-genders.sh > larynx/VOICE_GENDERS
