#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== evaluation ==="
python -m moelab evaluate

echo
echo "=== gates (achievable set: must pass) ==="
python -m moelab gates --max-mse 0.06 --min-purity 0.95 --max-expert-share 0.26

echo
echo "=== gates (impossible purity bar above 1.0: must fail) ==="
if python -m moelab gates --min-purity 1.5; then
    echo "ERROR: impossible gate set passed"
    exit 1
else
    echo "impossible gate set failed as expected"
fi
