# RAG plan

Where we are: Phases 1 and 2 done (naive end-to-end, eval scaffolding,
hybrid search, multi-game with game classifier). Phase 3 is up next.

This project teaches two things: *what* to build for modern RAG, and
*when not to call the LLM*. Phase 4 covers the second explicitly, and
short notes in earlier phases flag the same lesson where it first
applies.

Conventions: each phase ends with re-running the eval so we can measure
whether the new technique actually helped. Each subphase should be small
enough to ship and re-eval independently. Threads marked *(mine)* are
mechanical work for Claude; *(yours)* is the learning work.

---

## Phase 3 — Rewriting, reranking, judge

The two highest-leverage improvements in modern RAG (rewriting and
reranking), plus the judge metric that the deferred multi-hop /
enumerative questions need.

### 3.1 — Query rewriter (multi-query + RRF)

LLM generates ~3 reformulations of the query; retrieve top-k for each
(including the original); fuse the result lists with RRF. Same RRF
already in use for hybrid search. Structured output via Ollama's JSON
Schema mode — parse failures aren't a concern, but a sanity pass
(dedupe, drop empties, fall back to original if zero survive) handles
semantic misses.

New `rewrite/` crate, mirroring `route/`: one Ollama-backed impl behind
a `QueryRewriter` trait in `rag-core`. `NoOpRewriter` for the baseline.
Pipeline calls the rewriter, loops the retriever over the returned
queries, fuses with RRF.

**Eval methodology**
- Re-run the golden set with and without the rewriter. Compare Recall@k,
  MRR, answer-contains rate.
- Per-query diff: which queries got better, which got worse? Rewriting
  almost always helps some and hurts others; the question is the net.
- Ablate: original alone vs. N=3 rewrites alone vs. original + 2
  rewrites fused. "Include original" is the safest baseline — it can't
  underperform no-rewrite on any query where the rewrites are bad.
- Log the rewriter LLM call's latency separately. Usually the dominant
  contributor to p50 in a small-corpus RAG.

A "skip the rewriter when it isn't needed" heuristic is deferred to 4.4
— design it after the per-query diff shows where rewriting hurts.

### 3.2 — Cross-encoder reranker

Pull a cross-encoder. First choice: `bge-reranker-v2-m3` via Ollama or
llama.cpp. Fallback: LLM-as-reranker prompt scoring 0–10. Pipeline
becomes retrieve top-20 → rerank → keep top-5 for generation. Typically
10-20% recall improvement on hard queries.

The cross-encoder is itself the "don't use the LLM" answer for
reranking — it's a small purpose-built model, not a general LLM. The
LLM-as-reranker fallback exists for capability when no good
cross-encoder is available; in production the cross-encoder wins on
cost and latency.

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

## Phase 4 — When NOT to call the LLM

A class of techniques that get the routing job done cheaper and faster
than the shipped LLM classifier. Each technique below is compared
against that classifier's baseline on FPR and end-of-pipeline retrieval
recall, at materially lower p50 latency.

The lesson is the architecture: layer cheap deterministic checks first,
fall to expensive smart checks last. This pattern recurs in production
RAG well beyond routing.

### 4.1 — Substring prefilter

For each game name in the collection (plus a small table of unambiguous
shortenings — "Quacks" → "The Quacks of Quedlinburg", "Lorcana" →
"Disney Lorcana", "Arnak" → "Lost Ruins of Arnak"), check if it appears
as a substring of the question. Exactly one match → return it; zero
matches → null; multiple matches → defer to a later layer.

Latency: effectively 0ms. Handles every case where the user names the
game outright. Hooks in front of the existing classifier so the LLM
call is skipped on substring hits.

### 4.2 — Embedding-based router

Pre-embed a one-paragraph description per game (manual or auto-generated
from the rulebook). At query time, embed the question and take cosine
similarity against each game. If the top match is meaningfully ahead of
the second (margin threshold), return it; otherwise null and fall
through.

Latency: ~30ms (one embedding call). Handles paraphrased / mechanic-only
references that substring matching can't ("the bag-drawing game" →
Quacks). Has its own failure modes (close semantic neighbors, threshold
choice) which are themselves the learning point.

### 4.3 — Multi-stage cascade

Substring → embedding → LLM disambiguator. The LLM only runs when
earlier layers report ambiguity (multiple substring matches, or two
embedding scores within margin of each other). On most queries, the LLM
never fires.

Phase end: compare end-to-end retrieval recall and routing latency
against the LLM-classifier baseline. Win condition is matching or
beating that classifier's FPR and recall while moving the p50 routing
latency well below the LLM call cost.

### 4.4 — Skip the rewriter when it isn't needed

Phase 3.1 calls the LLM rewriter unconditionally. The per-query diff
from 3.1's eval will show queries where rewriting *hurt* recall — that
data drives this subphase.

Two techniques, A/B both against 3.1's always-rewrite baseline:

**Length heuristic.** Skip rewriting when the query is already long
(threshold around 20 tokens, tune from the data). Long queries are
usually specific and rewriting tends to dilute them. Zero added
latency; only skips the obvious cases.

**Confidence-gated.** Run cheap initial retrieval first. If the top
hit's similarity is above some threshold, return early without
rewriting. Adds one retrieval pass on every query but skips the LLM
call on the easy ones. Threshold choice is the learning point.

Phase end: compare end-to-end recall and p50 latency vs. the
always-rewrite baseline. Same shape of win condition as 4.3 — match
recall, win on latency.

---

## Phase 5 — Advanced chunking

(Was Phase 4; the old "Contextual retrieval comparison" sub-phase moved
to Out of scope — beyond a learning project, because it requires a
strong LLM across every chunk at ingest.)

The genuinely hard stuff. Ordered roughly by leverage on the corpus
we'll have by then.

### 5.1 — Hierarchical / small-to-big retrieval

Build a tree per rulebook: document → section → paragraph → sentence.
Store chunks at multiple granularities. Retrieve at fine grain, return
parent paragraph/section as generation context.

### 5.2 — Cross-reference resolver

Detect "see page X" / "see [Section Name]" patterns. When a chunk with
such a reference is retrieved, auto-pull the referenced chunk too. Add
cross-ref questions to the eval — these need the long rulebooks from
1.3.3 to actually exist (8-page rulebooks rarely cross-reference).

### 5.3 — Late chunking

Embed whole document (or large sections) in one pass with qwen3's long
context, then slice embeddings at chunk boundaries afterward. Preserves
cross-chunk context. **Caveat**: qwen3-embedding's window bounds this —
fine for ≤30-page rulebooks, awkward beyond. Note the limit when running
the experiment.

---

## Phase 6 — Conversational context

Real chatbots are multi-turn. Once base retrieval is solid, add session
state and study how it changes routing and retrieval — and how eval has
to change to measure it.

### 6.1 — `chat` REPL subcommand

Add `cargo run -- chat` opening an interactive session. State carried
turn-to-turn: most recently identified game, last few turns of
question/answer for follow-up context. Stateless `ask` stays for
one-shot use and for the existing eval loop.

### 6.2 — Multi-turn golden set + eval harness

Extend the golden set with a multi-turn entry type: a list of turns,
each with its own expected_chunk_contains/expected_quote. Eval walks the
session turn-by-turn against a session-aware pipeline, carrying
prior-turn state forward. Write ~10-15 entries that specifically test
inheritance behavior (turn 1 names the game, turn 2 asks a follow-up
without naming it; turn 3 might switch games via explicit naming).

### 6.3 — Router uses prior-turn game

The router checks the new question for a game name first (substring or
embedding); if none, it carries forward the prior turn's game. This is
where the Phase 4 routing work most pays off — the cheap checks gate
whether to inherit context vs. trigger a fresh route.

### 6.4 — Reference resolution *(optional)*

"What about it?" / "How does that work?" rewrites to inject the implied
subject from the prior turn before retrieval. Reuses Phase 3's query
rewriter pattern.

---

## Deferred (with triggers)

### FAQ / errata as separate `doc_type`

Ingest official FAQ documents as `doc_type=faq` separately from `rules`.
The `doc_type` field already threads through the schema; this is a data
ingestion task waiting for source material.

**Trigger**: when a question demonstrably requires FAQ content the
rulebook doesn't cover.

---

## Out of scope — deliberate learning-project boundaries

- **Streaming answers.** Useful in chatbots with long-form output; rule
  questions are short and CLI users see the full answer on completion.
- **YouTube transcripts.** Tempting but cut.
- **BGG scraping.** Official FAQ PDFs are higher quality; revisit only if
  needed.
- **Ingestion automation** (watchers, queues, schedulers). Manual CLI is
  fine for static data.
- **Agent frameworks.** Rust ecosystem isn't there; would be fighting
  abstractions.
- **Self-corrective / agentic RAG.** Marginal gains until base retrieval
  is good. Not for a learning project.

## Out of scope — beyond a learning project

If this project ever expanded past learning, these are the things a
production version would need. Listed so the learning project's gaps
are deliberate, not accidental.

- **Caching layers.** Embedding cache, response cache, prompt caching.
  Biggest cost/latency lever in real RAG; no useful signal in a
  single-machine local project.
- **Continuous eval / drift detection.** Production runs eval samples
  against shipped traffic and alerts on regression. Our one-shot eval
  is enough for learning.
- **Cost tracking.** $/query budgets and dashboards. Local Ollama is
  free, so no signal.
- **Web UI.** A full-time webdev problem; out of the RAG-learning scope.
  Will (a full-time webdev) will revisit once Phase 6 ships.
- **Contextual retrieval (Anthropic 2024).** Prepending an
  LLM-generated 1–2 sentence context to each chunk before embedding.
  Real technique with mixed evidence, but requires a strong LLM run
  across every chunk at ingest time — doesn't fit the "free and local"
  constraint. Worth knowing about; not worth running here.
