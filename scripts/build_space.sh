#!/usr/bin/env bash
# Assemble the Hugging Face Space folder (make space): only what the server needs, nothing else.
#   bash scripts/build_space.sh [target dir, default ../hybrid-intent-router-space]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-../hybrid-intent-router-space}
for f in models/small/model.pt models/small/meta.json results/routing.json results/prompt_examples.json \
         data/processed/labels.json; do
  [ -e "$f" ] || { echo "missing $f (train/route first, and run: python -m router.llm_baseline examples)"; exit 1; }
done
rm -rf "$OUT"
mkdir -p "$OUT/src" "$OUT/deploy/nginx" "$OUT/results" "$OUT/data/processed" "$OUT/models"
cp deploy/space/Dockerfile deploy/space/start.sh deploy/space/requirements.txt deploy/space/README.md "$OUT/"
cp -r app "$OUT/app"
cp -r src/router "$OUT/src/router"
cp deploy/gunicorn.conf.py "$OUT/deploy/"
cp deploy/nginx/nginx.conf "$OUT/deploy/nginx/"
cp results/routing.json results/prompt_examples.json "$OUT/results/"
cp data/processed/labels.json "$OUT/data/processed/"
cp -r models/small "$OUT/models/small"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
chmod +x "$OUT/start.sh"
echo "space folder ready: $OUT ($(du -sh "$OUT" | cut -f1))"
