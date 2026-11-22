#!/usr/bin/env bash
# Space entry point: NGINX on 0.0.0.0:7860 (the Space's public port) -> gunicorn on 127.0.0.1:8000.
# Same config files as the local stack; only the NGINX listen address changes.
set -euo pipefail
cd "$(dirname "$0")"
N=/tmp/nginx
mkdir -p "$N/run" "$N/temp" "$N/logs"
sed -e 's/listen 127.0.0.1:8080;/listen 0.0.0.0:7860;/' deploy/nginx/nginx.conf > "$N/nginx.conf"
grep -q "listen 0.0.0.0:7860;" "$N/nginx.conf"  # fail loudly if the listen line ever changes

python -m gunicorn -c deploy/gunicorn.conf.py &
GUNICORN=$!
for _ in $(seq 300); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  kill -0 "$GUNICORN" 2>/dev/null || { echo "gunicorn exited during start-up"; exit 1; }
  sleep 1
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "gunicorn not healthy after 300 s"; exit 1; }

nginx -p "$N/" -c nginx.conf -e /dev/stderr -g "daemon off;" &
echo "serving on :7860"
wait -n  # if either process dies, exit so the Space restarts the container
