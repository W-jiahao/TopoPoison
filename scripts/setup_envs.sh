#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3.10}"
SKIP_VLLM="${SKIP_VLLM:-0}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi
if [[ "$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
  echo "TopoPoison environments require Python 3.10" >&2
  exit 1
fi
if [[ "$SKIP_VLLM" != "0" && "$SKIP_VLLM" != "1" ]]; then
  echo "SKIP_VLLM must be 0 or 1" >&2
  exit 1
fi

"$PYTHON" -m venv "$ROOT/.venvs/graphrag"
"$ROOT/.venvs/graphrag/bin/python" -m pip install --upgrade pip wheel
"$ROOT/.venvs/graphrag/bin/python" -m pip install -r "$ROOT/requirements-graphrag.txt"

"$PYTHON" -m venv "$ROOT/.venvs/hipporag"
"$ROOT/.venvs/hipporag/bin/python" -m pip install --upgrade pip wheel
"$ROOT/.venvs/hipporag/bin/python" -m pip install -r "$ROOT/requirements-hipporag.txt"

if [[ "$SKIP_VLLM" == "0" ]]; then
  "$PYTHON" -m venv "$ROOT/.venvs/vllm"
  "$ROOT/.venvs/vllm/bin/python" -m pip install --upgrade pip wheel
  "$ROOT/.venvs/vllm/bin/python" -m pip install -r "$ROOT/requirements-vllm.txt"
  echo "Created GraphRAG, HippoRAG2, and vLLM environments under $ROOT/.venvs"
else
  echo "Created GraphRAG and HippoRAG2 environments under $ROOT/.venvs"
fi
