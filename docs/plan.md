# RAG plan

Where we are: Phase 1 done (naive end-to-end), Phase 1.1 done (ingest
manifest), Phase 1.2 done (eval harness with 62 hand-written goldens
across Pandemic, Challengers!, Quacks). Phase 1.3 done — eval V2,
which gates Phase 2. 1.3.1 (eval cleanup), 1.3.2 (retrieval-only mode +
`retrieve` crate + Recall@k / MRR / latency), 1.3.3 (hardness expansion:
+4 rulebooks, +53 goldens, `--no-game-filter` flag), and 1.3.4 (cheap
LLM-side metrics: refusal rate, latency p50/p95, input/output token
counts via cl100k proxy) are all done. Phase 2 is up next.

Conventions: each phase ends with re-running the eval so we can measure
whether the new technique actually helped. Each subphase should be small
enough to ship and re-eval independently. Threads marked *(mine)* are
mechanical work for Claude; *(yours)* is the learning work.

---

## Phase 1.3 — eval V2 (gates Phase 2)

### 1.3.4 — Cheap LLM-side metrics *(mine, mechanical)*

The cheap parts of LLM-side instrumentation. Each is straightforward once
1.3.1 is in.

- **Refusal rate**: % of answers containing any `forbidden_phrases`
  value. Falls out of 1.3.1 essentially for free.
- **Answer latency** p50 / p95 of `pipeline.ask()`.
- **Token counts**: input/output tokens per question via `tiktoken-rs`
  (already in stack).

Descriptive metrics, not pass/fail. Move as you change models, prompts,
or retrieval.

---

## Phase 1.4 — when annoying

Quality-of-life and measurement work that doesn't gate Phase 2 but is
worth doing whenever the friction shows up.

### 1.4.1 — Filtering + parallelism

The 30-min full eval runtime is already annoying. 1.3.2's retrieval-only
mode covers most iteration; this brings the slow path down too.

```
bgrag eval --only hard,reasoning      # tag-filtered subset
bgrag eval --only easy --limit 5      # quick smoke test
```

Filters apply after loading the golden set, before the eval loop. Tag
predicate: `--only A,B` matches questions whose tags include A OR B.

**Parallelism.** Current `for example in examples` loop is sequential.
Ollama can handle small concurrency; `futures::stream::iter(...)
.buffered(N)` with N=4 should give 3-4x with no quality cost. Bound is
conservative because local Ollama swaps if N is too high. Brings full
eval to ~8 minutes.

### 1.4.2 — Long-context baseline

For a small static corpus, "stuff the whole rulebook in context" can
beat RAG outright. Establish the ceiling number now so all Phase 2+
improvements are measured against something honest.

Implementation: a `FullContextPipeline` impl behind the `Pipeline`
trait that skips retrieval and dumps the entire game's rulebook into the
prompt. Run the eval against it.

Real RAGs do compare against this — partly as sanity, partly because for
small corpora the answer might genuinely be "don't bother retrieving."

## Phase 2 — Hybrid search + multi-game

Better retrieval quality, scale to the whole collection.

### 2.1 — Paragraph-aware chunker + full-collection ingest

`ParagraphChunker` that respects paragraph boundaries, keeps section
headers attached, and includes a `section_title` field in metadata.
Rulebooks have structure — use it. Add as a new entry under the
multi-table LanceDB layout from 1.3.2 so vs. fixed-size is a clean A/B.

Ingest the full PDF collection. Adds the cross-game disambiguation
pressure that 1.3.3's `--no-game-filter` mode measures.

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
