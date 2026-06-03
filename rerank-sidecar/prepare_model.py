#!/usr/bin/env python3
"""Fetch the reranker model and build the fp16 copy the server runs.

The sidecar owns its model now (the Rust app no longer downloads it). This
script, run once by setup.sh, downloads BGE-reranker-v2-m3 from Hugging Face
into `rerank-sidecar/model/` and converts it to fp16.

Why fp16: it halves GPU memory (~2.2 GB -> ~1.1 GB). The 3080 is shared with
Ollama, and the fp32 model squeezes gemma's layers onto the CPU, which slows
every query rewrite. fp16 is also faster on the GPU and doesn't change ranking.
We delete the fp32 weights after converting — the server only needs the fp16
model and the tokenizer.

Idempotent: if `model_fp16.onnx` already exists it does nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import onnx
from huggingface_hub import snapshot_download
from onnxconverter_common import float16

MODEL_DIR = Path(__file__).resolve().parent / "model"
REPO_ID = "rozgo/bge-reranker-v2-m3"
# fp16 needs only the model + tokenizer; skip the rest of the repo.
PATTERNS = ["model.onnx", "model.onnx.data", "tokenizer.json"]


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fp16 = MODEL_DIR / "model_fp16.onnx"
    if fp16.exists():
        print(f"already prepared: {fp16}")
        return

    print(f"downloading {REPO_ID} -> {MODEL_DIR} ...")
    snapshot_download(repo_id=REPO_ID, local_dir=str(MODEL_DIR), allow_patterns=PATTERNS)

    print("converting to fp16 (~40s) ...")
    model = onnx.load(str(MODEL_DIR / "model.onnx"))
    model16 = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=True
    )
    onnx.save(
        model16,
        str(fp16),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_fp16.onnx.data",
    )

    # The fp32 weights (~2.2 GB) were only needed for the conversion.
    for leftover in ("model.onnx", "model.onnx.data"):
        (MODEL_DIR / leftover).unlink(missing_ok=True)
    shutil.rmtree(MODEL_DIR / ".cache", ignore_errors=True)  # hf-hub bookkeeping

    print(f"ready: {fp16}")


if __name__ == "__main__":
    main()
