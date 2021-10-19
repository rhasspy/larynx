PLATFORMS ?= linux/amd64,linux/arm64,linux/arm/v7

all:
	docker buildx build . -f Dockerfile.debian.test_bullseye --output=type=local,dest=output --platform $(PLATFORMS)
