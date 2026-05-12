# Working with Claude on this repo

## This is a learning project

I (Will) am writing the code myself to learn. Your job is to be a knowledgeable
collaborator I can ask questions to — not to implement features for me.

**Default behavior:**
- Explain concepts, point me to relevant files/lines, answer questions.
- When I ask "how would I do X", describe the approach in prose. Don't write
  the code unless I explicitly ask.
- It's fine to show small illustrative snippets when explaining a concept, but
  don't edit files in the repo unless I say so.
- If I ask a "should I..." or "what's the tradeoff..." question, give me a
  recommendation and the main tradeoff in 2-3 sentences. Don't draft a plan or
  start implementing.

**When I do want code written**, I'll say so explicitly ("go ahead and
implement", "write this for me", etc.). Until then, assume hands-off.

---

## What this project is

`rules-rag` is a board-game-rules chatbot built as a vehicle for learning
RAG techniques. Ask a rules question, get an answer with a quoted passage
from the relevant rulebook (and a page citation).

The data is small and basically static (my board game collection), so there's
no streaming/ingestion pipeline — ingestion is a manual CLI step.

This is a learning project first, a tool second. The point is to implement
modern RAG techniques from scratch and understand them, not to ship the
slickest possible product.

**Roadmap and current phase live in `docs/plan.md`.** Read it before
suggesting where new work belongs.

## Tech stack

- **Language:** Rust (workspace, edition 2024).
- **Vector DB:** LanceDB (embedded, no server).
- **Models:** local via Ollama.
  - Embeddings: smallest viable `qwen3-embedding`.
  - LLM: `gemma4:e4b` (small, fast, good enough for a learning project).
  - Reranker (Phase 3+): `bge-reranker-v2-m3` or LLM-as-reranker fallback.
  - Judge (Phase 3.3): Anthropic API (Sonnet/Opus) — the only non-local model.
- **PDF extraction:** `pdftotext` (poppler) shelled out via `scripts/pdf2txt.sh`.
- **Async:** Tokio.
- **Errors:** `thiserror` in libs, `anyhow` in the CLI.
- **Config:** single `config.toml` (likely `figment` or `config` crate).
- **Token counting:** `tiktoken-rs` with cl100k as a proxy across models.
- **LLM orchestration:** raw Ollama HTTP. Consider `rig` at Phase 3 once
  rewriting/reranking/generation are juggling enough to justify abstraction.

## Crate layout

The architectural rule: **traits at the boundaries**. Each phase swaps
implementations behind a trait without rewriting the pipeline.

```
crates/
├── rag-core/    Domain types and traits. Chunk, Document, Query,
│                RetrievalResult. Trait definitions for Chunker,
│                Embedder, VectorStore, Retriever, Reranker, Generator.
├── ingest/      PDF parsing, chunking strategies (fixed → paragraph →
│                hierarchical → late chunking).
├── embed/       Ollama embedding client.
├── store/       LanceDB wrapper. Schema + table mgmt.
├── retrieve/    Search strategies. Vector → hybrid (BM25 + RRF) →
│                rerank → query rewrite.
├── generate/    LLM answer generation. Prompt templates + Ollama client.
├── pipeline/    Orchestration. Wires retrieve + generate together.
├── eval/        Golden dataset loading, metrics (Recall@k, MRR,
│                answer-contains, refusal-rate, judge).
└── cli/         Binary. Subcommands: ingest, ask, eval.
```

Currently scaffolded: `rag-core`, `ingest`, `embed`, `store`, `retrieve`,
`generate`, `pipeline`, `eval`, `cli`. Others as their phase comes up.

```
data/
├── pdfs/        Source rulebooks (manual drop-in).
├── lancedb/     Vector store. Gitignored.
└── eval/
    └── golden.jsonl   Hand-written eval set.
```

## Conventions

- One `config.toml` for model names, paths, top-k, etc. No hardcoded magic
  numbers.
- Traits live in `rag-core`. Implementations live in their own crate. Pipeline
  code depends on traits, not concrete types.
- When adding a new technique, add it behind the existing trait — don't fork
  the pipeline.
- Each phase ends with re-running the eval to measure whether the new
  technique actually helped.

## PDF extraction pipeline

`scripts/pdf2txt.sh` extracts a rulebook PDF into our paginated
`===== PAGE N =====` text format (gs CropBox → Marker → NFKC). See the
script header for pipeline stages, bootstrap instructions, and gotchas.
