# Future work

Items planned but not yet started, plus items deliberately deferred. The
eval-improvement plan lives at the top; deferred items at the bottom with
the reason they're deferred and the trigger to revisit.

---

## Eval improvement plan

### Where things stand

After Phase 1.2, the eval has 62 hand-written golden questions across three
short rulebooks (Pandemic, Challengers!, Quacks). A representative run:

- Quote match: ~88-94% (depends on whether you count known false-negatives)
- Chunk match: ~98%
- Wall-clock: ~30 minutes (Ollama is the bottleneck)

The eval works but has three problems for "number go up" iteration:

1. Chunk-match has no headroom on 8-page rulebooks with top-5 retrieval.
2. There's only one pass/fail signal per metric, no curve to push on.
3. 30-minute runtime kills iteration speed.

The plan below is five threads. Threads 1 and 4 are mostly mine (mechanical).
Thread 2 is the big architecture chunk. Threads 3, 5 are design + a mix.

### Thread 1 — Eval cleanup *(mine, fast)*

Eliminates known false-negatives in the current eval. Pure quality bump on
the existing question set, no new architecture.

**Schema additions**

```rust
pub struct EvalExample {
    // ...
    pub expected_quote: Vec<String>,        // was String; ANY of these counts as a match
    pub forbidden_phrases: Vec<String>,     // answer must NOT contain any of these
}
```

ANY-match semantics for `expected_quote`: a hit on any one entry passes the
check. Lets us cover the "rule sentence OR worked example OR parallel
phrasing on a different page" cases that produce 4 of our 7 current
failures (pandemic-004, pandemic-014, quacks-020, quacks-021).

`forbidden_phrases` is a flat blacklist: if the answer (after normalization)
contains any of these, the entry fails regardless of quote match. Default
list to seed: `"no information"`, `"cannot determine"`, `"unable to determine"`,
`"no chunk supports"`, `"not specified"`. Catches the silent-refusal
failures (pandemic-006, challengers-003) that quote-grep currently misses.

**Backfill**: I update the JSONL to add alternative quotes for the 4 known
false-negatives and any others that turn up. JSONL change is mechanical;
the validation test (`every_expected_substring_appears_in_source`) catches
any quote that isn't actually in the source.

**Out of scope here**: changing what entries exist, adding new questions,
or changing the chunk-match check. This thread is purely "make the existing
metric more accurate."

### Thread 2 — Retrieval-only mode + retrieval measurements *(yours)*

The big architectural change. Splits the Evaluator so retrieval can be
benchmarked independently of generation, unlocking cheap iteration on
chunkers/embedders/retrievers and exposing the measurements that are
currently invisible.

**CLI surface**

```
bgrag eval                     # full pipeline (slow, ~30 min)
bgrag eval --retrieval-only    # retrieval only (~10 sec for 62 questions)
```

**Internal dispatch**

The flag selects between two evaluator types:

```rust
pub struct RetrievalEvaluator<S: VectorStore, E: Embedder> { ... }
pub struct FullEvaluator<P: Pipeline> { ... }
```

`RetrievalEvaluator` doesn't hold a `Generator` and doesn't construct one
(avoids the Ollama health-check on `OllamaGenerator::new()`). Each evaluator
returns its own report shape — the retrieval-only report doesn't have an
`answer` field, so the type system enforces the asymmetry.

The two reports share a `RetrievalMetrics` substruct so the formatting
helpers can be reused. No `Evaluator` trait needed yet — YAGNI until
there's a third evaluator.

**Retrieval metrics computed in this mode**

These are all pure functions of `(question, top_k_retrieved_chunks)`, so
they're free to compute once you have the retrieval results:

- **Recall@k curve**: at k=1, 3, 5, 10. For each k, fraction of questions
  where the right chunk (any chunk containing `expected_chunks` substrings)
  appears in top-k. Reveals the *shape* of retrieval quality.
- **MRR (Mean Reciprocal Rank)**: average of `1/rank_of_first_relevant_chunk`.
  Single number summarizing the curve. Goes up as relevant chunks move
  toward the top.
- **Retrieval latency**: p50 / p95. Will move when you swap embedders or
  add reranking.

**Schema decision (resolved during thread 1)**

`expected_chunk_contains` is now `Vec<String>` with **ANY-match** semantics,
mirroring `expected_quote`. The trigger was pandemic-014, where the rule
appears in two places (page 5 and page 7) and either retrieved passage is
acceptable grounding.

ALL-match semantics for multi-hop synthesis questions (originally noted
here as the rationale for the change) are deferred. When multi-hop lands,
add a separate field — `expected_chunks_all: Vec<String>` — rather than
overloading the existing field. This keeps the two semantics cleanly
distinguishable and avoids retrofitting parallel-passage entries.

**What you'll learn**: probably that top-1 chunk-match is much worse than
top-5 chunk-match, which is the actual signal Phase 2's hybrid search +
reranker will move.

### Thread 3 — Filtering + parallelism *(yours)*

Smaller architecture. Speeds up the slow path so the full eval is usable
during iteration, not just for pre-commit checks.

**CLI**

```
bgrag eval --only hard,reasoning      # tag-filtered subset
bgrag eval --only easy --limit 5      # quick smoke test
```

Filters apply after loading the golden set, before the eval loop. Tag
predicate: `--only A,B` matches questions whose tags include `A` OR `B`;
multiple flags would be AND. Probably overengineered for now — just OR.

**Parallelism**

The current `for example in examples` loop is sequential. `Ollama` can
handle a small number of concurrent requests; `futures::stream::iter(...)
.buffered(N)` with N=4 should give 3-4x speedup with no quality cost.
Bound is conservative because local Ollama starts swapping if N is too
high.

This is mostly a `futures` ergonomics exercise. Worth doing because it
brings the full eval down to ~8 minutes, which changes how often you'll
run it.

### Thread 4 — LLM-side measurements *(mine + yours)*

Once thread 2 is done, the slow path has the LLM. Add measurements that
require generation:

- **Refusal rate** *(mine, mechanical)*: % of answers containing any
  `forbidden_phrases` value. Falls out of thread 1 essentially for free.
- **Answer latency** *(mine, mechanical)*: p50 / p95 of `pipeline.ask()`.
- **Token counts** *(mine, mechanical)*: input tokens, output tokens per
  question. Uses `tiktoken-rs` (already in stack).
- **Quote-faithfulness ratio** *(yours, design)*: of the answer's character
  count, how much is verbatim from retrieved chunks vs invented. Definition
  isn't obvious — minimum quote length, allowed normalization, how to
  attribute "the rulebook says" prefixes — that's the design work.

These don't need to be pass/fail; they're descriptive metrics that move as
you change models, prompts, or retrieval.

### Thread 5 — Hardness expansion *(mostly mine, design with you)*

The data-side work. Expands what the eval can stress.

**More rulebooks**

Mix of long and medium for different pressure types:

- **One long** (>20 pages): Spirit Island or Ark Nova. Stresses chunking
  — fixed 512-token chunks across a 30-page rulebook will start losing
  context that 8-page rulebooks didn't expose. Justifies Phase 2's
  paragraph-aware chunker and Phase 4's hierarchical chunking.
- **Two-three medium** (8-15 pages, dense rules): Wingspan, Azul,
  Cascadia, or similar. Stresses disambiguation — six rulebooks with
  overlapping vocabulary makes "What's the deck phase?" actually
  ambiguous.

**Process**: you pick games. I run `scripts/pdf2txt.sh`, do a smoke read
to confirm parse quality (no contents-list / icon-emoji / Robot-section
breakage like in Challengers), then generate ~10-15 golden questions per
new game in batches for you to approve. You don't write questions; you
review them.

**New question shapes**

Once we have new rulebooks, add some:

- **Adversarial / wrong-premise questions**: "When I draw two Epidemics in
  a row, what happens?" — answer is "nothing special, just resolve each."
  Tests whether the model invents rules.
- **Threshold questions**: like quacks-017. Test exact-direction precision
  ("more than" vs "at least" vs "exceeds"). Cheap to write, real signal.
- **No-game-filter questions**: not new questions, but a flag
  (`--no-game-filter`) that runs existing questions without
  `game_filter`. Tests whether the embedding alone can disambiguate
  across games. Makes the cross-game pressure measurable.

**Out of scope**: multi-hop synthesis questions and questions that
naturally enumerate (like the original pandemic-003). See deferred section.

---

## Plan summary

| Thread | Owner | Effort | Unlocks |
|--------|-------|--------|---------|
| 1. Eval cleanup | Claude | Small | Removes ~4 known false-negatives; surfaces refusals |
| 2. Retrieval-only mode + metrics | Will | Large | 30 min → 10 sec for retrieval iteration; Recall@k curve, MRR |
| 3. Filtering + parallelism | Will | Small-medium | Full eval 30 min → 8 min |
| 4. LLM-side measurements | Mixed | Medium | Refusal rate, latency, tokens, faithfulness |
| 5. Hardness expansion | Claude (with Will's review) | Medium-large | Headroom on chunk-match; cross-game disambiguation; chunking pressure |

Threads 1 and 5 can run in parallel with threads 2/3/4. Sequential within
2 → 3 → 4 because each builds on the previous (retrieval-only mode is the
prerequisite for everything else).

---

## Deferred

### Multi-hop synthesis questions

**What**: questions whose answer requires combining information from 2+
chunks. Example: "What's the maximum number of cubes that could be on the
board at once?" — needs "24 per color" and "4 colors" from different
passages.

**Why deferred**: chunk-match works fine for these (`expected_chunks` lists
all required substrings, all must hit somewhere in top-k). Answer-side
validation does *not* — the model says "96" without quoting either source.
Quote-grep is the wrong tool. We could add `key_phrases: Vec<String>` for
synthesized claims, but that's a weak signal compared to the existing
quote-grep for single-passage questions.

**Trigger to revisit**: LLM-as-judge wired up (next item). Multi-hop
questions are the natural test target for the judge; doing them in concert
makes more sense than doing them with a half-strength metric now.

### LLM-as-judge metric

**What**: a stronger model (Sonnet/Opus) evaluating whether the candidate
answer matches the gold reference (`expected_answer`), tolerating
paraphrase. Returns Y/N + a brief justification.

**Why deferred**: cost and complexity. Current quote-grep covers ~94% of
question types in our golden set. The judge becomes essential when we add
synthesis / enumerative / paraphrase-heavy questions, which we're also
deferring.

**Implementation sketch**:
- New `Judge` trait with `judge(question, gold, candidate) -> Verdict`.
- An `AnthropicJudge` impl using the API (the only place we'd need a non-local model).
- Eval framework calls judge after the existing checks; `judge_match: bool`
  joins `quote_match` and `chunk_match`.
- Prompt-cache the system prompt across all 60+ judgments to control cost.

**Trigger to revisit**: Phase 3, when query rewriting / reranking start
producing answer-quality changes that quote-grep can't see. Or when we
add the first batch of multi-hop / enumerative questions.

### LLM output caching for harness work

**What**: hash `(prompt_template, question, retrieved_chunks)` → answer.
Replay LLM outputs from cache when re-running the eval after harness
changes.

**Why deferred**: narrow win. Any pipeline change (chunker, embedder,
retriever, prompt template) invalidates the cache. Only useful for
iterating on the eval *harness* (metrics, normalization, scoring) without
re-running the LLM. Thread 2 (retrieval-only mode) plus thread 3
(parallelism) cover most of the same use cases without the cache-
invalidation complexity.

**Trigger to revisit**: only if we end up doing significant work on the
eval harness in isolation (rare — usually harness changes accompany
pipeline changes).

### Pandemic-style enumerative / list-shaped questions

**What**: questions like the original pandemic-003 ("What are the special
actions on my turn?") whose answer is a list with no spine quote.

**Why deferred**: same as multi-hop — quote-grep doesn't fit. Could be
handled with `key_phrases: Vec<String>` (all four action names must
appear) but that's the same compromise.

**Trigger to revisit**: after LLM judge lands. The judge handles
list-shaped answers naturally given a gold reference.

### Phase 4 cross-reference questions

**What**: questions that exercise "see page 23" / "see [Section]" patterns
that Phase 4's cross-reference resolver is supposed to handle.

**Why deferred**: needs longer rulebooks where these patterns actually
exist (8-page rulebooks rarely cross-reference). Thread 5 adds the
rulebooks; the questions follow naturally when we have a 30-page rulebook
to ask them about.

**Trigger to revisit**: Phase 4 entry, post-thread-5.

### FAQ / errata as separate `doc_type`

**What**: ingest official FAQ documents as `doc_type=faq` separately from
`rules`. Eval questions that should specifically retrieve from FAQ.

**Why deferred**: we don't have FAQ documents in the manifest yet. The
`doc_type` field already threads through the schema; this is a data
ingestion task waiting for source material.

**Trigger to revisit**: when a question demonstrably requires FAQ content
that the rulebook doesn't cover.
