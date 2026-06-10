# MMR / diversity — findings (Phase 3.4, not shipped)

Notes from the 2026-06-09 probe of whether Maximal Marginal Relevance buys
anything on this corpus. Conclusion up front: **it doesn't — MMR reads flat to
marginally negative across every configuration the plan's design fork names, so
3.4 is dropped.** Kept so we don't re-derive it. The throwaway probe
(`crates/retrieve/tests/mmr_candidates.rs`) was removed after this writeup;
"Reconstructing the probe" at the bottom says how to rebuild it.

## What MMR is supposed to fix, and why it can't here

MMR greedily picks the next chunk by `λ·rel(d) − (1−λ)·max_sim(d, picked)`, so
it spends a fixed top-k budget on *distinct* information instead of near-
duplicates. Its win only exists when **two conditions both hold**: (a) the top-k
without MMR wastes slots on near-duplicate chunks, and (b) the passage those
duplicates are crowding out is close behind on relevance. On this corpus neither
reliably holds:

1. **Recall is saturated before MMR runs.** With a game filter, our rulebooks
   are 8–32 pages → ~15–70 chunks. A two-passage question's second passage is
   almost always already inside the top ~8, and the eval runs `top_k=10` /
   `recall@10` any-match, so it never even *sees* the difficulty. The signal,
   if any, lives at k≤5.
2. **The hard cases are hard on *relevance*, not redundancy.** When the second
   passage is deep, it's because it's a weak lexical/semantic match for the
   question — not because duplicates bury it. MMR trades relevance for
   diversity, so it cannot promote a low-relevance chunk no matter how redundant
   the chunks above it are.

### The `burial` diagnostic

To separate the two, the probe measured `burial` = max pairwise cosine among
the chunks ranked *strictly above* the deeper of the two needles. High burial
("there ARE duplicates above the buried passage") is necessary but **not**
sufficient: e.g. `stoneage-food-produce-starve` on the rerank pool sits at
rank 6 with burial 0.74, and no λ moves it — the second passage is just too far
down on relevance. That row is the whole finding in miniature.

## The numbers

Two-passage probe questions (Quacks, Stone Age, Spirit Island), coverage rank =
deepest rank you must read to have *both* passages. MMR run over each candidate
pool the plan names, λ swept 1.0 → 0.7:

| Pool (relevance source)     | MMR lowered coverage rank |
|-----------------------------|--------------------------:|
| Dense (vector only)         | 0 / 12 |
| Hybrid (vector + FTS + RRF) | 0 / 12 |
| **Rerank (cross-encoder)**  | **1 / 12** |

The single positive was `quacks-rubies-earn-spend` on the rerank pool
(burial 0.66): MMR at λ≤0.8 pulled the second passage from rank 3 → 2 — a
1-slot nudge on a question that already passed at k=3. Lower λ that might catch
the genuinely-deep cases just scrambles the ranking and *hurts* other questions
(e.g. `spiritisland-blight-cascade-and-remove`, hybrid, 4 → 5 at λ<1).

Two structural notes that fell out of the probe:

- **The cross-encoder, not MMR, fixes multi-chunk coverage.** Reranking alone
  moved `quacks-move-droplet` 8 → 4 and `quacks-explode` 4 → 3; MMR on top of
  the rerank added ≈nothing. If multi-passage coverage matters, spend the budget
  on the reranker (already shipped, 3.2), not MMR.
- **Pure-dense-cosine MMR is actively worse.** An earlier config that recomputed
  relevance as query↔chunk cosine (instead of reusing the hybrid/rerank score)
  and used λ=0.5 *raised* coverage on 9/10 — it discards the FTS signal that
  surfaces exact rule keywords. If MMR is ever revisited, relevance must be the
  existing fused/rerank score, not a fresh embedding cosine.

## Why this matches the plan's prediction

`docs/plan.md` 3.4 already flagged: "On single-answer questions MMR can only
break even or *hurt*… the first eval may read flat and treat that as the
finding." It does, and now across all three pools with a λ sweep rather than as
a guess. The diversity-family lesson is still worth knowing; it just has no
purchase on a small, game-filtered, deduplicated corpus. The multi-passage
*questions* the probe produced are still useful — but for the distinct-chunk
eval metric and the Phase 5 techniques (small-to-big 5.1, cross-ref resolver
5.2) that actually target multi-passage retrieval, not for MMR.

## Reconstructing the probe

If revisited, the harness was an `#[ignore]` integration test in `retrieve`
that, per candidate `{question, needle_a, needle_b}`:

1. Retrieves a pool (`top_k=15`) from dense / hybrid / rerank retrievers with a
   game filter.
2. Batch-embeds the pooled chunk texts (qwen3) for the redundancy term.
3. Reorders with MMR using `rel` = min-max-normalized existing pool score and
   redundancy = chunk↔chunk cosine, λ ∈ {1.0, 0.9, 0.8, 0.7}.
4. Reports `coverage_rank` (rank for both needles) before/after, plus `burial`.

Needs the real LanceDB index, a running Ollama embedder, and (for the rerank
pool) the `:8071` rerank sidecar up.
