#!/usr/bin/env bash
# One-time setup: create a venv and install the GPU reranker deps.
# Re-runnable; pip reuses its download cache so it's quick after the first run.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
./.venv/bin/pip install --upgrade pip >/dev/null
./.venv/bin/pip install -r requirements.txt

# download the model and build the fp16 copy the server runs
./.venv/bin/python prepare_model.py

echo
echo "setup complete. start the sidecar with:  ./run.sh"
