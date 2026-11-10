#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python || ! -x .venv-ocr/bin/python ]]; then
  echo "Run bash setup_m4.sh first." >&2
  exit 1
fi
.venv/bin/python download_models.py
.venv-ocr/bin/python download_models.py --ocr
printf '\nAll 3 components have been downloaded and cached. Run: bash run_all.sh\n'
