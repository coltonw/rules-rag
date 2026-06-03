#!/usr/bin/env bash
# Launch the GPU reranker sidecar.
# Puts the pip-installed CUDA 12 / cuDNN 9 runtime libs (and the WSL CUDA
# driver) on LD_LIBRARY_PATH so onnxruntime-gpu's CUDA provider can load.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "no venv found — run ./setup.sh first" >&2
  exit 1
fi

# Collect every nvidia/*/lib dir from the venv, plus the WSL GPU driver libs.
NVIDIA_LIBS="$(find "$PWD/.venv" -type d -path '*/nvidia/*/lib' | paste -sd: -)"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec ./.venv/bin/python server.py
