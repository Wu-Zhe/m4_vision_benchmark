#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: This setup targets native Apple Silicon macOS (Darwin arm64)."
fi
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] not in [(3,11), (3,12)]:
    raise SystemExit("Use Python 3.11 or 3.12. Example: PYTHON_BIN=python3.12 bash setup_m4.sh")
print("Python", sys.version.split()[0])
PY
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-main.txt
"$PYTHON_BIN" -m venv .venv-ocr
.venv-ocr/bin/python -m pip install --upgrade pip setuptools wheel
.venv-ocr/bin/python -m pip install -r requirements-ocr.txt
.venv/bin/python make_samples.py
cat <<'MSG'

Setup completed. Next:
  .venv/bin/python download_models.py
  .venv-ocr/bin/python download_models.py --ocr
  bash run_all.sh

Both model downloads and first OCR initialization require Internet access.
MSG
