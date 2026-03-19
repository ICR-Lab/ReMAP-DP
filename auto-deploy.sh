#!/usr/bin/env bash
# auto-deploy.sh — watches src/, public/, and config files for changes,
# then auto-commits and pushes to trigger GitHub Pages deployment.

set -euo pipefail
cd "$(dirname "$0")"

BRANCH="master"
DEBOUNCE=5  # seconds to wait after last change before pushing

echo "[auto-deploy] Watching for changes on branch '$BRANCH'..."
echo "[auto-deploy] Will commit & push ${DEBOUNCE}s after last save."
echo "[auto-deploy] Press Ctrl+C to stop."
echo ""

npx chokidar-cli \
  'src/**/*' 'public/**/*' 'index.html' 'vite.config.js' \
  --debounce $((DEBOUNCE * 1000)) \
  --command \
  'echo "[auto-deploy] Change detected — pushing..." && \
   git add -A && \
   git diff --cached --quiet && echo "[auto-deploy] No changes to commit." || \
   (git commit -m "auto: update site $(date +%H:%M:%S)" && \
    git push origin '"$BRANCH"' && \
    echo "[auto-deploy] Pushed successfully.")'
