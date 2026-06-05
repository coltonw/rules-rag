# RAG plan

Where we are: Phases 1 and 2 done (naive end-to-end, eval scaffolding,
hybrid search, multi-game with game classifier). Phase 3.1 (query
rewriter: multi-query + RRF) and 3.2 (cross-encoder reranker) are also
done. HyDE was tried and rejected (see "Experiments run and rejected");
MMR (3.4) is up next.

This project teaches two things: *what* to build for modern RAG, and
*when not to call the LLM*. Phase 4 covers the second explicitly, and
short notes in earlier phases flag the same lesson where it first
applies.

Conventions: each phase ends with re-running the eval so we can measure
whether the new technique actually helped. Each subphase should be small
enough to ship and re-eval independently. Threads marked *(mine)* are
mechanical work for Claude; *(yours)* is the learning work.

---

## Phase 3 — Rewrite and Rerank

Query rewriting (3.1) and cross-encoder reranking (3.2) are done — the
two highest-leverage improvements in modern RAG. What remains are two
query-/candidate-shaping experiments: MMR diversity (3.4) and step-back
prompting (3.5, low priority). (HyDE was tried and rejected — see
"Experiments run and rejected". The LLM-as-judge metric moved to
Deferred — it needs the paid Anthropic API.)

### 3.4 — MMR / diversity

Maximal Marginal Relevance: greedily pick the next chunk by
`λ·relevance − (1−λ)·max-similarity-to-already-selected`, so each
selection trades relevance against redundancy with what's already in the
context. λ is the knob (1.0 = pure relevance, the current behavior;
lower = more diverse).

This is the one technique family with no representation elsewhere in the
plan. The reranker (3.2) reorders candidates by relevance but does
nothing about redundancy — several near-duplicate chunks from the same
sub-section can crowd out the second passage a question actually needs.
MMR is pure vector math: no LLM call, ~no added latency, free and local.
It also reinforces the Phase 4 lesson — a cheap deterministic step beats
an expensive one.

Design fork (the learning): does MMR run over the dense candidate pool,
or *after* the cross-encoder rerank? Post-rerank diversifies an
already-relevance-sorted list; pre-rerank changes what the reranker even
sees. A/B both.

Eval caveat: MMR's payoff is only visible on questions that need more
than one passage. The current golden set is single-quote /
answer-contains, and the multi-hop + enumerative questions are deferred
(see LLM-as-judge). On single-answer questions MMR can only break even
or *hurt* (it may demote the single best chunk). So either pull a
handful of diversity-demanding questions forward into the golden set
first, or accept that the first eval may read flat and treat that as the
finding. A/B against the reranker-alone baseline.

### 3.5 — Step-back prompting *(low priority)*

Generate a more general/abstract version of the question ("can my worker
move through an enemy piece?" → "what are the movement rules?"), retrieve
for both, fuse. Same family as 3.1 and 3.3 — a query transform — just
abstracting *upward* instead of paraphrasing (3.1) or hypothesizing an
answer (3.3). By the time HyDE lands the "transform the query before
retrieval" lesson is already taught twice, so this is a third point on
the same axis and the most droppable item in the phase. One-day A/B like
HyDE if it interests you; otherwise skip.

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

The shipped rewriter calls the LLM unconditionally. The per-query diff
from the rewriter's eval will show queries where rewriting *hurt*
recall — that data drives this subphase.

Two techniques, A/B both against the always-rewrite baseline:

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

## Experiments run and rejected

Techniques we built, measured, and decided against. Kept here so the
findings outlive the phase that produced them.

### HyDE (was Phase 3.3)

Hypothetical Document Embeddings: have the LLM generate a hypothetical
answer paragraph, embed *that*, and search with it (matching answer↔answer
instead of question↔answer). Built as a single hypothetical generated as
a raw string (no JSON grammar, to keep the prose in rulebook register),
fused with the original query, A/B'd against the multi-query rewriter
alone.

Rejected on both axes. It lost on eval recall, *and* ran ~2.5x slower
(~30s vs. ~12s/query): free-form generation produced ~435 output tokens
(incl. hidden reasoning) at ~16s, vs. the rewriter's grammar-constrained
56 tokens at ~2s. Dropping the JSON grammar to improve prose quality is
exactly what unbounded the generation — the two goals are in tension.

Net: it added an unconstrained LLM call to the critical path and didn't
earn it — the Phase 4 lesson in miniature. A cross-encoder reranker (3.2)
already attacks the same question↔answer asymmetry HyDE targets, from
downstream, without the extra call.

## Deferred (with triggers)

### LLM-as-judge metric

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

**Trigger**: deferred for now to avoid paid API token spend. Pick up
when the multi-hop / enumerative questions need it and the API cost is
acceptable.

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
- **Context compression.** An extractive or LLM pass over retrieved
  chunks to strip irrelevant tokens before generation. Solves a problem
  this corpus doesn't have — tiny merged paragraph chunks with a small
  top-k don't strain the context budget — and a per-chunk compression
  pass on a local LLM is the worst latency trade available. Same shape
  of cut as contextual retrieval: real technique, needs an LLM pass per
  chunk, doesn't fit free-and-local.
