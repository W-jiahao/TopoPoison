#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-microsoft/phi-4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${VLLM_API_KEY:-sk-vllm}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec "$ROOT/.venvs/vllm/bin/vllm" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
