# GPU reranker sidecar

A tiny HTTP service that runs the **BGE-reranker-v2-m3** cross-encoder on the
GPU and scores `(query, document)` pairs for the Rust app's `HttpReranker`.

## Why this is a separate process

The reranker is the dominant cost in the MultiQuery pipeline: ~19s for 30
chunks on CPU vs ~0.5s on the GPU (RTX 3080). We can't run it on the GPU
*inside* the Rust app because `ort` 2.0.0-rc.12's CUDA execution provider
deadlocks during session init on this WSL box (the underlying CUDA stack is
fine — `onnxruntime-gpu` from Python works). So the GPU work lives here, in
Python, and the Rust side talks to it over HTTP — the same pattern the app
already uses for Ollama.

**Parity:** this loads the *same* `model.onnx` fastembed uses, tokenizes the
same `(query, document)` pairs, and returns `logits[:, 0]` sorted descending —
exactly fastembed's behavior. So ranking order (and eval recall/MRR) is
identical to the in-process CPU reranker; only latency changes.

## Setup & run

```sh
cd rerank-sidecar
./setup.sh      # one-time: .venv + onnxruntime-gpu + CUDA libs + downloads the
                #           model and builds the fp16 copy (prepare_model.py)
./run.sh        # starts the service on 127.0.0.1:8071
```

`setup.sh` downloads BGE-reranker-v2-m3 from Hugging Face into `model/` (gitignored)
and converts it to fp16. The fp32 weights are deleted after conversion; the
server runs `model/model_fp16.onnx`.

### Why fp16 (important on a shared GPU)

The 3080's 10 GB VRAM is shared with Ollama (gemma + embedder ≈ 6 GB). The
fp32 reranker is 2.2 GB, which pushes the total to ~9.4 GB and forces Ollama to
offload gemma layers to the CPU — that made every query **rewrite** ~2–4× slower
and *erased most of the GPU rerank win* (eval p50 went from 6.7 s with a CPU
reranker to 28 s). The fp16 model is ~1.1 GB; total VRAM drops to ~6.4 GB, gemma
stays fully on-GPU, rewrites return to ~2.2 s, and rerank is even faster (~0.2 s
vs 0.5 s). Ranking order is unchanged, so eval quality is unaffected.

`setup.sh` installs the CUDA 12 / cuDNN 9 runtime as pip wheels (no system CUDA
install, no sudo); `run.sh` puts them on `LD_LIBRARY_PATH` alongside the WSL GPU
driver. On startup it prints `device: cuda` — if it says `cpu`, the CUDA libs
didn't load (check `run.sh`'s `LD_LIBRARY_PATH`).

The model lives in `model/` (populated by `prepare_model.py`). Override the
location with `RERANK_MODEL_DIR` if needed.

## API

```
POST /rerank   {"query": "...", "documents": ["...", "..."]}
            -> {"results": [{"index": 2, "score": 7.31}, ...], "device": "cuda"}
GET  /health   -> {"status": "ok", "device": "cuda"}
```

`results` is sorted by score descending; `index` refers into the request's
`documents` array. Env knobs: `RERANK_HOST` (default `127.0.0.1`),
`RERANK_PORT` (default `8071`), `RERANK_MODEL_DIR`.

## Notes

- The app's `--retriever multi-query` path requires this sidecar to be running.
  Start it before `cli ask`/`cli eval` with the MultiQuery retriever.
- The GPU is shared with Ollama (also on the 3080); both fit in 10 GB VRAM.
