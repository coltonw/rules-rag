# Multi-query rewriter — findings (Phase 3.1)

Notes from the 2026-05-31 analysis of why MQ slightly under-performs hybrid
on R@3/R@10 despite winning R@1. Kept so we don't re-derive the mechanism;
the raw eval artifacts are in `analysis/` (gitignored) and reproducible with
the instrumentation listed at the bottom.

## Mechanism: RRF dilution of lexical bridges

MQ runs original + 3 rewrites, each as a hybrid query (dense + FTS, RRF-fused,
top-30). Those four result lists are then RRF-fused into the final top-10.
A chunk's final score is `Σ 1/(60+rank_i)`, so a chunk that's mid-pack in *all
four* sub-queries beats a chunk that's #1 in only the original.

`gemma4:e4b` is conservative about preserving game-specific terms (the prompt
explicitly says "do not invent game-specific terms you are not certain of"),
so rewrites tend to euphemize the exact words the user typed. Whenever the
original was an FTS hit on a rare token, the rewrites drop that token and FTS
finds different chunks; RRF averages the bridge away.

`paleo-008` is the clean illustration: question is "deck is empty, went to
sleep, can I still help?". The answer chunk ranks **#1** under the original
(FTS hits "sleep") and 8/11/12 across the three rewrites, which all drop
"sleep". The distractor `Paleo-rules-6-1` ("HELPING") ranks #0 in all four
queries because they all talk about "help / abilities / players in trouble",
and wins the RRF sum (`4/60 ≈ 0.067` vs the answer's `~0.059`).

## Weighting the original in RRF

In an instrumented experiment with cached rewrites (so every variant sees
identical sub-queries), weighting the original by N — i.e. summing
`N/(60+rank_orig)` instead of `1/(60+rank_orig)` — improves R@1 and MRR up
to N≈6, then degrades back toward "use only the original":

| Variant         |  R@1  |  R@3  |  R@5  | R@10   |  MRR  |
|-----------------|------:|------:|------:|-------:|------:|
| Hybrid          | 65.2% | **94.8%** | 96.5% | **100.0%** | 0.797 |
| MQ W=1          | 67.0% | 92.2% | 96.5% | 98.3%  | 0.802 |
| MQ **W=6**      | **71.3%** | 93.0% | 96.5% | 98.3%  | **0.824** |
| MQ W=10         | 67.0% | 93.9% | 96.5% | 98.3%  | 0.799 |

W=6 essentially says "treat rewrites as a tie-breaker when the original
isn't decisive." Defensible for *this* rewriter but bakes in the weakness
of `gemma4:e4b`. Not shipping it as a default — the right fix is the
cross-encoder rerank from Phase 3.2: rerank the union of all sub-query
top-30s against the **original** question, which bypasses RRF dilution
entirely and lets the rewrites do their actual job (diversifying the
candidate pool, not voting on the final order).

## Chunk-boundary failures to recheck in Phase 5

Two consistent losers are *not* retrieval bugs. The answer sentence is
buried inside a chunk whose **leading paragraph** embeds to something
else, so the chunk never ranks high in any sub-query. Weighting can't
fix these; hierarchical chunking (5.1) should. Re-run both once 5.1 lands.

- **`spirit-island-001`** — "phases of a turn". Answer chunk
  `Spirit Island-rules-6-3` contains "Each turn has the following phases:"
  but *starts* with "INVADERS' STARTING ACTION". Embedding sees
  Setup/Invaders, not turn-phases. Under hybrid the FTS half rescues it
  via the rare word "phases"; MQ averages that bridge away.
- **`pandemic-006`** — "do roles get better abilities throughout the game?".
  Answer is a Medic-specific sentence ("if the Medic at any time finds
  herself…") buried inside the "Eradicating a Disease" chunk. No query
  names "Medic" or "cured", so FTS can't bridge; embeddings ceiling out
  at rank ~10 across every sub-query.

## Instrumentation worth re-adding if digging in again

These patches were temporary and reverted; recreate them if revisiting:

- `crates/rewrite/src/lib.rs` — bump `rewrites` log from `debug` → `info`
  with the `query` field, and add a `REWRITES_CACHE=path` env var that
  bypasses the LLM and reads pre-recorded rewrites from JSON. The cache
  is what makes fusion-variant comparisons apples-to-apples.
- `crates/eval/src/lib.rs` — add a `question` field to the per-row
  `info!("ok")` log so eval output can be joined to rewriter logs.
- `crates/retrieve/src/lib.rs` — log every sub-query's top-30 chunk IDs
  and text; add an `MQ_ORIGINAL_WEIGHT` env var to control the RRF
  weighting of the original query.
