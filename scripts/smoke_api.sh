#!/usr/bin/env bash
# Phase 6 smoke test (make smoke-api): start the Flask dev server, send real requests with curl, stop it.
# Sentences: MASSIVE test ids 0 and 8 (the same sentences in English, Hindi and Kannada).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH="$PWD/src:$PWD"
PORT=${PORT:-5001}
URL="http://127.0.0.1:$PORT"

$PY -m flask --app "app.app:make_app()" run --port "$PORT" > logs/06_flask_server.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf "$URL/health" >/dev/null && break; sleep 1; done

echo "\$ curl -s $URL/health"
curl -s "$URL/health"
for id in 0 8; do  # MASSIVE ids are unique across train/dev/test, so not every number is a test id
  for lang in en hi kn; do
    body=$($PY - "$lang" "$id" <<'EOF'
import json, sys
lang, id_ = sys.argv[1], sys.argv[2]
for line in open(f"data/processed/{lang}_test.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == id_:
        print(json.dumps({"text": r["text"]}, ensure_ascii=False))
        break
else:
    sys.exit(f"test id {id_} not found for {lang}")
EOF
)
    echo
    echo "\$ curl -s -X POST $URL/predict -d '$body'"
    curl -s -X POST "$URL/predict" -H 'Content-Type: application/json' -d "$body"
  done
done
echo
echo "\$ curl -s -X POST $URL/predict -d '{\"text\": \"\"}'"
curl -s -w '  -> HTTP %{http_code}' -X POST "$URL/predict" -H 'Content-Type: application/json' -d '{"text": ""}'
echo
echo "\$ curl -s -X POST $URL/predict -d '{\"text\": \"aaa...(501 characters)\"}'"
curl -s -w '  -> HTTP %{http_code}' -X POST "$URL/predict" -H 'Content-Type: application/json' \
  -d "{\"text\": \"$(printf 'a%.0s' $(seq 501))\"}"
echo
