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
	for lang in de-de en-us es-es fr-fr it-it nl ru-ru sv-se sw; do \
        LARYNX_LANGUAGE=$$lang scripts/build-docker.sh; \
    done

amd64:
	NOBUILDX=1 scripts/build-docker.sh

index:
	bin/make_sample_html.py local/ > index.html
