#!/bin/bash
cd "$(dirname "$(dirname "$0")")" || exit 1

uv venv --allow-existing
uv sync

host="${NOTATORY_LEARNING_HOST:-localhost}"
keyfile="${NOTATORY_LEARNING_SSL_KEYFILE:-key.pem}"
certfile="${NOTATORY_LEARNING_SSL_CERTFILE:-cert.pem}"

if [[ ! -f "$keyfile" || ! -f "$certfile" ]]; then
  openssl req -x509 -newkey rsa:2048 \
    -keyout "$keyfile" -out "$certfile" \
    -days 365 -nodes \
    -subj "/CN=$host"
fi

uv run python notatory_learning/main.py \
  --host "$host" \
  --port "${NOTATORY_LEARNING_PORT:-8100}" \
  --voicevox_url "${VOICEVOX_URL:-http://localhost:50021}" \
  --ssl_keyfile "$keyfile" \
  --ssl_certfile "$certfile"
