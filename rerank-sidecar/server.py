#!/usr/bin/env python3
"""GPU reranker sidecar for rules-rag.

Why this exists: the Rust app talks to a BGE-reranker-v2-m3 cross-encoder. Run
on CPU (fastembed) that's ~19s for 30 chunks. The GPU does the same work in
~0.5s, but the Rust `ort` 2.0.0-rc.12 CUDA EP deadlocks on this WSL box during
session init. So instead of reranking in-process, the Rust `HttpReranker` POSTs
to this tiny Python service, which runs the *identical* ONNX model on the GPU
via onnxruntime-gpu (proven working here) and returns scores.

Parity: we load the same `model.onnx` fastembed uses, tokenize the same
`(query, document)` pairs, and return `logits[:, 0]` sorted descending — exactly
what fastembed's reranker does. So ranking order (and therefore eval
recall/MRR) matches the CPU path; only the latency changes.

Run it via run.sh (sets up the CUDA library path). Talks plain JSON over HTTP:

    POST /rerank  {"query": "...", "documents": ["...", "..."]}
      -> {"results": [{"index": 2, "score": 7.31}, ...], "device": "cuda"}
    GET  /health  -> {"status": "ok", "device": "cuda"}
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

SIDECAR_DIR = Path(__file__).resolve().parent
MAX_LENGTH = 512  # reranker max sequence length (matches the model's training)
HOST = os.environ.get("RERANK_HOST", "127.0.0.1")
PORT = int(os.environ.get("RERANK_PORT", "8071"))


def find_model_dir() -> Path:
    """The sidecar's own model dir, populated by prepare_model.py."""
    model_dir = Path(os.environ.get("RERANK_MODEL_DIR", SIDECAR_DIR / "model"))
    if not (model_dir / "model_fp16.onnx").exists() and not (model_dir / "model.onnx").exists():
        sys.exit(
            f"no reranker model in {model_dir}\n"
            "Run:  ./.venv/bin/python prepare_model.py   (or ./setup.sh)"
        )
    return model_dir


class Reranker:
    def __init__(self) -> None:
        model_dir = find_model_dir()
        # Prefer the fp16 model if present: it's ~half the VRAM (1.1 vs 2.2 GB),
        # which matters because the GPU is shared with Ollama — fp32 squeezes
        # gemma's layers onto the CPU and slows every rewrite. fp16 is also
        # faster on the GPU and doesn't change ranking. Build it with
        # `prepare_model.py`. Override the choice with RERANK_MODEL_FILE.
        default_file = "model_fp16.onnx" if (model_dir / "model_fp16.onnx").exists() else "model.onnx"
        model_file = os.environ.get("RERANK_MODEL_FILE", default_file)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.tokenizer.enable_padding()

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(model_dir / model_file), providers=providers
        )
        self.device = "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"
        self.input_names = {i.name for i in self.session.get_inputs()}
        # ORT sessions are safe for concurrent Run(), but the GPU serializes the
        # work anyway, so a lock keeps things simple and deterministic.
        self.lock = threading.Lock()
        print(f"[sidecar] model: {model_dir / model_file}", flush=True)
        print(f"[sidecar] device: {self.device} ({self.session.get_providers()})", flush=True)
        self._warmup()

    def _run(self, query: str, documents: list[str]) -> list[dict]:
        if not documents:
            return []
        encodings = self.tokenizer.encode_batch([(query, d) for d in documents])
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}
        with self.lock:
            logits = self.session.run(["logits"], feeds)[0]
        scores = logits[:, 0].astype(float)
        order = np.argsort(-scores)  # descending, matches fastembed
        return [{"index": int(i), "score": float(scores[i])} for i in order]

    def _warmup(self) -> None:
        self._run("warmup", ["warmup document one", "warmup document two"])
        print("[sidecar] warmup complete", flush=True)


class Handler(BaseHTTPRequestHandler):
    reranker: Reranker  # set on the server instance below

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path == "/health":
            self._send(200, {"status": "ok", "device": self.server.reranker.device})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if self.path != "/rerank":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            query = req["query"]
            documents = req["documents"]
        except (KeyError, ValueError) as exc:
            self._send(400, {"error": f"bad request: {exc}"})
            return
        try:
            results = self.server.reranker._run(query, documents)
        except Exception as exc:  # noqa: BLE001 - report any inference failure
            self._send(500, {"error": f"rerank failed: {exc}"})
            return
        self._send(200, {"results": results, "device": self.server.reranker.device})

    def log_message(self, *args) -> None:  # quiet the default per-request logging
        pass


def main() -> None:
    reranker = Reranker()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.reranker = reranker
    print(f"[sidecar] listening on http://{HOST}:{PORT}  (POST /rerank)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sidecar] shutting down", flush=True)


if __name__ == "__main__":
    main()
