# RAG plan

Where we are: Phase 1 done (naive end-to-end), Phase 1.1 done (ingest
manifest), Phase 1.2 done (eval harness with 62 hand-written goldens
across Pandemic, Challengers!, Quacks). Phase 1.3 is in flight — eval V2,
which gates Phase 2 because Phase 2's whole point is moving retrieval
quality numbers we can't currently measure cleanly.

A representative Phase 1.2 eval run:
- Quote match: ~88-94% (depends whether you count known false-negatives)
- Chunk match: ~98%
- Wall-clock: ~30 minutes (Ollama is the bottleneck)

The eval works but has three problems for "number go up" iteration:
chunk-match has no headroom on 8-page rulebooks at top-5; there's only
one pass/fail signal per metric, no curve to push on; 30-minute runtime
kills iteration speed.

Conventions: each phase ends with re-running the eval so we can measure
whether the new technique actually helped. Each subphase should be small
enough to ship and re-eval independently. Threads marked *(mine)* are
mechanical work for Claude; *(yours)* is the learning work.

---

## Phase 1.3 — eval V2 (gates Phase 2)

These four are sequential within themselves but 1.3.1 and 1.3.3 can run in
parallel with 1.3.2.

### 1.3.1 — Eval cleanup *(mine, fast)*

Eliminates known false-negatives in the current eval. Pure quality bump on
the existing question set, no new architecture.

Schema additions:

```rust
pub struct EvalExample {
    // ...
    pub expected_quote: Vec<String>,        // was String; ANY of these counts as a match
    pub forbidden_phrases: Vec<String>,     // answer must NOT contain any of these
}
```

ANY-match semantics for `expected_quote`: a hit on any one entry passes.
Lets us cover "rule sentence OR worked example OR parallel phrasing on a
different page" cases that produce 4 of our 7 current failures
(pandemic-004, pandemic-014, quacks-020, quacks-021).

`forbidden_phrases` is a flat blacklist: answer (after normalization)
contains any of these → fail regardless of quote match. Default seed:
`"no information"`, `"cannot determine"`, `"unable to determine"`,
`"no chunk supports"`, `"not specified"`. Catches silent-refusal failures
(pandemic-006, challengers-003) that quote-grep currently misses.

Backfill: update JSONL to add alternative quotes for the 4 known
false-negatives and any others that turn up. The validation test
(`every_expected_substring_appears_in_source`) catches any quote that
isn't actually in the source.

Out of scope here: changing what entries exist, adding new questions,
or changing the chunk-match check.

### 1.3.2 — Retrieval-only mode + `retrieve` crate + Recall@k / MRR *(yours)*

The big architectural change. Splits the Evaluator so retrieval can be
benchmarked independently of generation, unlocking cheap iteration on
chunkers/embedders/retrievers and exposing measurements that are
currently invisible.

**`retrieve` crate.** Pull retrieval out of `pipeline` into its own crate.
Trait: `Retriever::retrieve(query, opts) -> Vec<RetrievalResult>`. First
impl: `VectorRetriever` (what's inline in pipeline today). This is the
moment to add the crate because (a) retrieval-only mode needs to construct
retrieval without `Generator`, and (b) Phase 2 will add a second impl.

**Multi-table LanceDB layout.** Parameterize the table name on chunker
identity (`chunks_fixed_512_64`, `chunks_paragraph`, etc.) so different
chunking strategies coexist without rebuild. Schema is shared; the table
name is a function of `(chunker_name, chunker_config_hash)`. Eval picks
which table to point at via config.

**CLI surface.**

```
bgrag eval                     # full pipeline (slow, ~30 min)
bgrag eval --retrieval-only    # retrieval only (~10 sec for 62 questions)
```

**Internal dispatch.** The flag selects between two evaluator types:

```rust
pub struct RetrievalEvaluator<S: VectorStore, E: Embedder> { ... }
pub struct FullEvaluator<P: Pipeline> { ... }
```

`RetrievalEvaluator` doesn't hold a `Generator` and doesn't construct one
(avoids the Ollama health-check on `OllamaGenerator::new()`). Each
returns its own report shape — the retrieval-only report has no
`answer` field, so the type system enforces the asymmetry. They share a
`RetrievalMetrics` substruct so formatting helpers reuse. No `Evaluator`
trait yet — YAGNI until there's a third evaluator.

**Retrieval metrics.** All pure functions of `(question, top_k_chunks)`,
free to compute once you have the retrieval results:

- **Recall@k curve** at k=1, 3, 5, 10. Reveals the *shape* of retrieval
  quality.
- **MRR** (mean reciprocal rank): single number summarizing the curve.
- **Retrieval latency** p50 / p95.

**Schema decision.** `expected_chunk_contains` becomes `Vec<String>` with
**ANY-match** semantics, mirroring `expected_quote`. Triggered by
pandemic-014 (rule appears on pages 5 and 7, either passage is acceptable
grounding). ALL-match semantics for multi-hop synthesis are deferred —
when multi-hop lands, add a separate field `expected_chunks_all:
Vec<String>` rather than overloading.

**Expected finding.** Top-1 chunk-match is much worse than top-5
chunk-match, which is the actual signal Phase 2's hybrid + reranker will
move.

### 1.3.3 — Hardness expansion *(mostly mine, design with you)*

Data-side work. Expands what the eval can stress.

**More rulebooks.** Mix of long and medium for different pressure types:

- One **long** (>20 pages): Spirit Island or Ark Nova. Stresses chunking
  — fixed 512-token chunks across a 30-page rulebook will start losing
  context that 8-page rulebooks didn't expose. Justifies Phase 2's
  paragraph-aware chunker and Phase 4's hierarchical chunking.
- Two-three **medium** (8-15 pages, dense rules): Wingspan, Azul,
  Cascadia, or similar. Stresses disambiguation — six rulebooks with
  overlapping vocabulary makes "What's the deck phase?" actually
  ambiguous.

Process: you pick games. I run `scripts/pdf2txt.sh`, do a smoke read to
confirm parse quality (no contents-list / icon-emoji / Robot-section
breakage like in Challengers), then generate ~10-15 golden questions per
new game in batches for you to approve. You don't write questions; you
review.

**New question shapes.**

- **Adversarial / wrong-premise**: "When I draw two Epidemics in a row,
  what happens?" — answer is "nothing special, just resolve each." Tests
  whether the model invents rules.
- **Threshold questions** (like quacks-017): exact-direction precision
  ("more than" vs "at least" vs "exceeds"). Cheap to write, real signal.
- **No-game-filter flag** (`--no-game-filter`) on existing questions:
  tests whether the embedding alone disambiguates across games. Makes
  cross-game pressure measurable.

Out of scope: multi-hop synthesis questions and enumerative questions —
land those alongside the LLM judge in Phase 3.3.

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

### 1.4.3 — Quote-faithfulness ratio *(yours, design)*

Of the answer's character count, how much is verbatim from retrieved
chunks vs invented. Definition isn't obvious — minimum quote length,
allowed normalization, attribution of "the rulebook says" prefixes —
that's the design work.

---

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
