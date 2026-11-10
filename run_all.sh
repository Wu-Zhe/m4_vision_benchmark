#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python || ! -x .venv-ocr/bin/python ]]; then
  echo "Run bash setup_m4.sh first." >&2
  exit 1
fi
if [[ ! -f samples/manifest.json ]]; then
  .venv/bin/python make_samples.py
fi
if [[ ! -f models/embeddinggemma-2/config.json || ! -f models/qwen3.5-9b-4bit/config.json ]]; then
  echo "Run .venv/bin/python download_models.py first." >&2
  exit 1
fi
# Separate Python processes release RAM/Metal memory between models.
.venv/bin/python test_embeddings.py
.venv/bin/python test_qwen.py
.venv-ocr/bin/python test_ocr.py
printf '\nFinished: inspect results/*.json, *.csv, and *.npy\n'
