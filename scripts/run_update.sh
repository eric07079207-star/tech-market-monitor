#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ".venv/bin/python" ]]; then
  echo "找不到 .venv/bin/python，請先建立虛擬環境並安裝 requirements.txt" >&2
  exit 1
fi

.venv/bin/python scripts/update_data.py
