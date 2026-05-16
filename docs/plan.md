# RAG plan

Where we are: Phase 1 done (naive end-to-end), all phases 1.x done (eval
improvements), and phase 2.1 done. Phase 2.2 is up next.

Conventions: each phase ends with re-running the eval so we can measure
whether the new technique actually helped. Each subphase should be small
enough to ship and re-eval independently. Threads marked *(mine)* are
mechanical work for Claude; *(yours)* is the learning work.

---

## Phase 2 — Hybrid search + multi-game

Better retrieval quality, scale to the whole collection.

### 2.2 — BM25 via LanceDB FTS

LanceDB has native full-text search via Tantivy. Add it as a second
`Retriever` impl. Standalone `tantivy` crate is the fallback if needed.

### 2.3 — RRF + per-game filter

`HybridRetriever` runs vector + BM25 in parallel, fuses with Reciprocal
Rank Fusion (`score = Σ 1/(k + rank_i)`, k≈60). Per-query metadata filter
on `game`. Hybrid search should meaningfully improve recall on queries
with specific game terms ("Longest Road", "Knight card") while vector
search wins on paraphrased questions — the eval will show which.

---

## Phase 3 — Rewriting, reranking, judge

The two highest-leverage improvements in modern RAG (rewriting and
reranking), plus the judge metric that the deferred multi-hop /
enumerative questions need.

### 3.1 — Query rewriter

LLM rewrites the query before search: expand abbreviations, add synonyms
for game terms, split multi-part questions. Consider multi-query
rewriting: 3 variants → fuse with RRF.

### 3.2 — Cross-encoder reranker

Pull a cross-encoder. First choice: `bge-reranker-v2-m3` via Ollama or
llama.cpp. Fallback: LLM-as-reranker prompt scoring 0–10. Pipeline
becomes retrieve top-20 → rerank → keep top-5 for generation. Typically
10-20% recall improvement on hard queries.

### 3.3 — LLM-as-judge metric

A stronger model (Sonnet/Opus via Anthropic API — the only place we'd
need a non-local model) evaluates whether the candidate answer matches
`expected_answer`, tolerating paraphrase. Returns Y/N + brief
justification.

New `Judge` trait with `judge(question, gold, candidate) -> Verdict`.
`AnthropicJudge` impl. Eval calls judge after existing checks;
`judge_match: bool` joins `quote_match` and `chunk_match`.
Prompt-cache the system prompt across all 60+ judgments.

This is also the moment to add **multi-hop synthesis** and **enumerative
list** questions to the golden set — the judge handles those naturally
where quote-grep can't.

### 3.4 — HyDE experiment *(optional)*

Hypothetical Document Embeddings: LLM generates a hypothetical answer
paragraph for the query, embed *that*, search with it. Mechanism is
distinct from query rewriting (a hypothetical *answer* vs. a better
*query*) but for a learning project the distinction is small. One-day
experiment, A/B against rewriter alone. Skip if it bores you.

---

## Phase 4 — Advanced chunking

The genuinely hard stuff. Ordered roughly by leverage on the corpus
we'll have by then.

### 4.1 — Hierarchical / small-to-big retrieval

Build a tree per rulebook: document → section → paragraph → sentence.
Store chunks at multiple granularities. Retrieve at fine grain, return
parent paragraph/section as generation context.

### 4.2 — Cross-reference resolver

Detect "see page X" / "see [Section Name]" patterns. When a chunk with
such a reference is retrieved, auto-pull the referenced chunk too. Add
cross-ref questions to the eval — these need the long rulebooks from
1.3.3 to actually exist (8-page rulebooks rarely cross-reference).

### 4.3 — Late chunking

Embed whole document (or large sections) in one pass with qwen3's long
context, then slice embeddings at chunk boundaries afterward. Preserves
cross-chunk context. **Caveat**: qwen3-embedding's window bounds this —
fine for ≤30-page rulebooks, awkward beyond. Note the limit when running
the experiment.

### 4.4 — Contextual retrieval comparison

Anthropic's late-2024 technique: prepend a 1–2 sentence LLM-generated
context to each chunk before embedding. Promising but expensive at
ingest time and weaker with small local LLMs. A Microsoft paper found
limited benefit vs. other methods. Worth measuring against late
chunking under our eval rather than adopting on faith.

---

## Deferred (with triggers)

### FAQ / errata as separate `doc_type`

Ingest official FAQ documents as `doc_type=faq` separately from `rules`.
The `doc_type` field already threads through the schema; this is a data
ingestion task waiting for source material.

**Trigger**: when a question demonstrably requires FAQ content the
rulebook doesn't cover.

### Web UI

Out of the RAG-learning scope. Will (a full-time webdev) will revisit
once Phases 1-4 are done. No notes needed here — the webdev part is the
easy part.

**Trigger**: after Phase 4 ships.

---

## Out of scope

- **YouTube transcripts.** Tempting but cut.
- **BGG scraping.** Official FAQ PDFs are higher quality; revisit only if
  needed.
- **Ingestion automation** (watchers, queues, schedulers). Manual CLI is
  fine for static data.
- **Agent frameworks.** Rust ecosystem isn't there; would be fighting
  abstractions.
- **Self-corrective / agentic RAG.** Marginal gains until base retrieval
  is good. Not for a learning project.
