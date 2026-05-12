# Backlog

Items that aren't on the critical path for the next phase but are worth
revisiting when their pain point shows up. No ordering implied.

---

## Quote-faithfulness ratio *(design)*

Of the answer's character count, how much is verbatim from retrieved
chunks vs invented. Definition isn't obvious — minimum quote length,
allowed normalization, attribution of "the rulebook says" prefixes —
that's the design work.

**Trigger**: when refusal rate and quote-match alone stop telling you
whether the model is grounding vs paraphrasing freely.

---

## Broaden `expected_chunk_contains` to accept glossary/iconography hits

The 1.3.3 eval run surfaced several "Chunk not found" cases where the
model found and quoted the *correct* answer — just from the glossary or
iconography page rather than the main-text page the golden was anchored
to. So the chunk_contains check fails on goldens where the answer is
actually fine.

Known affected: `res-arcana-001`, `spirit-island-003`,
`spirit-island-006`, `spirit-island-007`, `spirit-island-012`. Likely
others. Fix: add a second `expected_chunk_contains` entry per golden
matching the glossary/iconography phrasing.

**Trigger**: when chasing recall numbers up and false negatives from
narrow goldens become noise against real failures.

---

## Investigate generator derailments

In the 1.3.3 eval, `spirit-island-008` produced gibberish about
"Scythe" and "Town Town characters" — full context loss, not a fact
hallucination. Separately, `spirit-island-009` leaked a `<|channel|>`
token into the output. Likely either prompt-template formatting or
local Gemma struggling with long multi-part scenarios.

**Trigger**: if/when these patterns repeat across the eval rather than
being one-offs. Could be diagnosed alongside any future generator
swap.
