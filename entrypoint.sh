#!/usr/bin/env bash
set -euo pipefail

APP_CMD="${APP_CMD:-python src/complete_pipeline.py}"
MAX_TRIES="${MAX_TRIES:-3}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-10}"

echo "🚀 Starting Predictive Churn Pipeline..."
echo "--------------------------------------"

TRY=0
while [ "$TRY" -lt "$MAX_TRIES" ]; do
  TRY=$((TRY+1))
  echo "Attempt ${TRY}/${MAX_TRIES}..."
  if ${APP_CMD}; then
    echo "Pipeline finished successfully!"
    exit 0
  else
    echo "Pipeline failed. Retrying in ${SLEEP_BETWEEN}s..."
    sleep "${SLEEP_BETWEEN}"
  fi
done

echo "❌ All ${MAX_TRIES} attempts failed."
exit 1
