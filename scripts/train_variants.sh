#!/usr/bin/env bash
# Thin wrapper — see scripts/train_variants.py for usage and logic.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m scripts.train_variants "$@"
