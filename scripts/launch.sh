#!/bin/bash
cd "$(dirname "$(dirname "$0")")" || exit 1
docker build -t "notatory-learning" docker

docker run -it --rm \
  --net=host \
  -u "$(id -u):$(id -g)" \
  -v "$(pwd):/workspace" \
  --env-file .env \
  notatory-learning bash scripts/serve.sh
