#!/usr/bin/env bash
# Thin wrapper — see scripts/evaluate_all.py for usage and logic.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m scripts.evaluate_all "$@"
