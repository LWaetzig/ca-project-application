#!/bin/bash

echo "Make sure to be logged into the github package registry"

REGISTRY="ghcr.io/open-discourse/open-discourse/open-discourse"

docker buildx create \
  --name container \
  --driver=docker-container 2>/dev/null

cd database
: ${REVISION_DATABASE:="$(node -p "require('./package.json').version")"}
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.prod \
  --tag "$REGISTRY/database:$REVISION_DATABASE" --tag "$REGISTRY/database:latest" \
  --builder=container \
  --push \
  --progress plain .
