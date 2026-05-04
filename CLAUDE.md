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

## Tech stack

- **Language:** Rust (workspace, edition 2024).
- **Vector DB:** LanceDB (embedded, no server).
- **Models:** local via Ollama.
  - Embeddings: smallest viable `qwen3-embedding`.
  - LLM: `gemma4:e2b` (small, fast, good enough for a learning project).
  - Reranker (Phase 3+): `bge-reranker-v2-m3` or LLM-as-reranker fallback.
- **PDF extraction:** `pdftotext` (poppler) shelled out, or `pdf-extract` crate.
- **Async:** Tokio.
- **Errors:** `thiserror` in libs, `anyhow` in the CLI.
- **Config:** single `config.toml` (likely `figment` or `config` crate).
- **Token counting:** `tiktoken-rs` with cl100k as a proxy across models.
- **LLM orchestration:** raw Ollama HTTP for Phases 1–2; consider `rig` at
  Phase 3 once rewriting/reranking/generation are juggling enough to justify
  an abstraction.

## Crate layout

The architectural rule: **traits at the boundaries**. Each phase swaps
implementations behind a trait without rewriting the pipeline.

```
crates/
├── core/        Domain types and traits. Chunk, Document, Query,
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
│                answer-contains).
└── cli/         Binary. Subcommands: ingest, ask, eval.
```

Currently scaffolded: `core`, `ingest`, `cli`. The rest will be added as
their phase comes up.

```
data/
├── pdfs/        Source rulebooks (manual drop-in).
├── lancedb/     Vector store. Gitignored.
└── eval/
    └── golden.jsonl   Hand-written eval set.
```

## Phased build plan

Each phase ends with a re-run of the eval harness so we can measure whether
the new technique actually helped.

### Phase 1 — Naive RAG end-to-end
Smallest possible version of every component. Fixed-size chunker
(512 tokens, 64 overlap), single-table LanceDB, top-k cosine search,
one-shot answer prompt with citation.

CLI: `bgrag ingest <pdf> --game <name>` and `bgrag ask <question> --game <name>`.

Goal: working pipeline. Quality will be mediocre; that's expected.

### Phase 1.1 — Ingest manifest
A `data/pdfs/manifest.toml` (array-of-tables) describes every document so
re-ingestion doesn't require retyping `--game` and friends every time.

```toml
[[document]]
file = "pandemic-rules.txt"
game = "Pandemic"
doc_type = "rules"
```

Deserializes to `Vec<DocMeta>` via `serde` + `toml`. CLI changes:
`bgrag ingest` with no args ingests every entry; `bgrag ingest <path>`
looks up that file's metadata in the manifest. `--game` becomes an
optional override. `doc_type` threads through `Chunk` and the LanceDB
schema so it's filterable later (rules vs FAQ vs transcript).

Pipeline knobs (chunk size, overlap, top-k) stay out of the manifest —
those belong in the eventual `config.toml`. The manifest is per-document
facts only.

### Phase 1.2 — Eval harness (do this BEFORE Phase 2)
Hand-write 20–30 golden questions across 2–3 well-known games. Implement
Recall@k, MRR, and a "answer contains expected phrases" check. Add
`bgrag eval`.

Without this, Phase 2+ changes are flying blind.

### Phase 2 — Hybrid search + multi-game
- BM25 via LanceDB's Tantivy-backed full-text search (start here; standalone
  `tantivy` crate is the fallback if needed).
- Reciprocal Rank Fusion to combine vector + BM25.
- Metadata filter on `game` per query.
- Paragraph-aware chunker that keeps section headers attached.
- Ingest the full collection.

### Phase 3 — Query rewriting + reranking
- LLM query rewriting (consider multi-query: 3 variants → fuse).
- Cross-encoder reranker. Pipeline: retrieve top-20 → rerank → top-5 → generate.
- Glossary index: extract defined terms into their own table, inject
  definitions into context when terms appear in a query.

### Phase 4 — Advanced chunking
- Hierarchical chunking (small-to-big retrieval): retrieve at fine
  granularity, return parent paragraph/section as context.
- Late chunking using qwen3's long context: embed whole document, slice
  embeddings at chunk boundaries afterward.
- Cross-reference resolver: detect "see page X" / "see [Section]" patterns,
  auto-pull the referenced chunk alongside.

## Out of scope

- **YouTube transcripts.** Tempting but cut for now.
- **Web UI / productization.** CLI is the frontend. Maybe later, not now.
- **BGG scraping.** Official FAQ PDFs are higher quality; revisit only if needed.
- **Ingestion automation** (watchers, queues, schedulers). Manual CLI is fine.
- **Agent frameworks.** Rust ecosystem isn't there; would be fighting abstractions.

## Conventions

- One `config.toml` for model names, paths, top-k, etc. No hardcoded magic numbers.
- Traits live in `core`. Implementations live in their own crate. Pipeline
  code depends on traits, not concrete types.
- When adding a new technique, add it behind the existing trait — don't
  fork the pipeline.

## PDF extraction pipeline

`scripts/pdf2txt.sh` is the entry point. Internally it chains gs CropBox
normalization → Marker (ML-based layout analysis + OCR via surya) →
NFKC perl normalization. Marker outputs paginated Markdown; the script
rewrites its `{N}===PAGE_BREAK===` tokens to our `===== PAGE M =====`
format (1-indexed) that the Rust ingest crate parses.

Marker runs from a venv at `scripts/.venv/` (gitignored). Bootstrap once:

```
python3 -m venv scripts/.venv
scripts/.venv/bin/pip install marker-pdf
```

First run downloads ~1 GB of model weights into `~/.cache/datalab/`.
Subsequent runs reuse them. Marker auto-detects CUDA and uses the GPU
when available; on a 3080, ~5s/page with `--force_ocr`.

Output quality is dramatically better than the previous hand-rolled
`pdftotext + XY-cut` pipeline — Marker's layout model handles
multi-column flow, inline figures splitting wrapped lines, and even
labels-on-artwork pages (Quacks contents page) reasonably well.

Gotchas already paid for:

- **gs `-dUseCropBox` is load-bearing.** Some rulebooks (Pandemic) are
  2-up scans where each PDF page is the same scan with a CropBox selecting
  only the left or right half. Skipping the CropBox step makes the
  extractor read from the MediaBox and emit duplicated content from both
  halves. Always normalize first.
- **`--force_ocr` is load-bearing.** Marker's default text-extraction
  path uses pypdfium, which silently drops glyphs whose ToUnicode CMap is
  partial — Pandemic's stylized initial-cap "Th" got swallowed, producing
  "e clock is ticking" instead of "The clock is ticking". Forcing OCR
  via surya re-reads the rendered glyph and recovers the character.
  Slower (~5s/page on GPU vs ~1s) but robust against the kinds of font
  weirdness rulebook designers love.
- **Inline icons render as emoji approximations** under OCR (e.g. the
  Challengers fans icon comes through as 👚). Harmless for retrieval —
  surrounding words still match — and arguably useful as a marker that
  an icon was there.
