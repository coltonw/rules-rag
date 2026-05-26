# bgrag

Board-game rules chatbot. Learning vehicle for RAG techniques. See
[`docs/plan.md`](docs/plan.md) for the roadmap and [`CLAUDE.md`](CLAUDE.md)
for working conventions.

## Useful commands

```bash
# Ask a question (classifies the game automatically)
cargo run -- ask "When does the infection rate go up in Pandemic?"

# Default eval: oracle game filter (no classifier), hybrid retriever, naive pipeline
cargo run -- eval

# Fastest retrieval-iteration loop: retrieval only, 5 examples
cargo run -- eval -r --limit 5

# Full eval with classifier on (measures routing accuracy too — slower)
cargo run -- eval --filter-mode classifier

# Cross-game pressure: no game filter at all (how well does retrieval cope?)
cargo run -- eval -r --filter-mode none

# Compare retrievers (swap hybrid/dense/sparse)
cargo run -- eval -r --retriever sparse
cargo run -- eval -r --retriever dense

# Compare chunkers (global -c flag)
cargo run -- -c fixed-512-64 eval -r

# Tag-filtered eval (only run rows matching one of these tags)
cargo run -- eval --only pandemic,challengers
```
