use futures::stream::{self, StreamExt};
use rag_core::{Answer, GameClassifier, Pipeline, QueryOptions, RetrievalResult, Retrieve};
use serde::{Deserialize, Serialize};
use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
    time::Instant,
};
use tiktoken_rs::cl100k_base_singleton;

/// cl100k token count for `text`. Used as a model-agnostic proxy per
/// CLAUDE.md's convention.
fn count_tokens(text: &str) -> usize {
    cl100k_base_singleton()
        .encode_with_special_tokens(text)
        .len()
}

// Anything with heavy comments was made or at least modified by an LLM. Actually kind of a nice marker for what I did vs what I didn't do.

/// Default phrases that mark a refused / hedged answer. Checked against
/// every answer in addition to per-example `forbidden_phrases`.
pub const DEFAULT_REFUSAL_PHRASES: &[&str] = &[
    "no information",
    "cannot determine",
    "unable to determine",
    "no chunk supports",
    "not specified",
];

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("failed to read eval file at {path}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse eval file at {path} on line {line}")]
    ParseFile {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
}

/// One required passage, matched by ANY of its acceptable phrasings.
///
/// In the golden JSON a passage is written either as a bare string (single
/// phrasing) or as an array of strings (the rule is stated in more than one
/// place, or a worked example is also acceptable grounding — match any one):
///
/// ```json
/// "expected_chunks": [
///   "A player whose pot explodes must stop",
///   ["chose between Evaluation Phase D or E", "does not get to roll the die"]
/// ]
/// ```
///
/// A list of passages is ALL-of: every passage must be satisfied (see
/// [`EvalExample::expected_chunks`] / [`expected_quotes`]).
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum Passage {
    One(String),
    AnyOf(Vec<String>),
}

impl Passage {
    /// The acceptable phrasings for this passage (always at least one).
    pub fn phrasings(&self) -> &[String] {
        match self {
            Self::One(s) => std::slice::from_ref(s),
            Self::AnyOf(v) => v,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct EvalExample {
    pub id: String,
    pub game: Option<String>,
    pub question: String,
    /// Passages the answer must quote, verbatim. ALL-of: the answer passes the
    /// quote check only if it contains a quote from EVERY passage (each passage
    /// matched by ANY of its phrasings, after normalization). A single-quote
    /// question is just one passage; a multi-rule answer needs one per rule.
    pub expected_quotes: Vec<Passage>,
    /// Passages that must appear among the retrieved chunks. ALL-of: every
    /// passage must appear in some retrieved chunk (each matched by ANY of its
    /// phrasings, after normalization). Passages may share a chunk or land in
    /// different chunks — distinct chunks are NOT required, so a full-context
    /// retriever still passes. Retrieval recall scores on the COVERAGE RANK:
    /// the deepest rank at which the last required passage first appears.
    pub expected_chunks: Vec<Passage>,
    pub expected_answer: String,
    /// Per-example refusal phrases, on top of `DEFAULT_REFUSAL_PHRASES`.
    /// Use when the question has its own way of being refused that the
    /// global list doesn't cover.
    #[serde(default)]
    pub forbidden_phrases: Vec<String>,
    pub tags: Vec<String>,
}

#[derive(Serialize)]
pub struct FullEvaluation {
    pub evals: Vec<FullEval>,
    pub routing_ratios: Option<RoutingRatios>,
    pub retrieval_ratios: RetrievalRatios,
    pub generation_ratios: GenerationRatios,
}

#[derive(Serialize)]
pub struct RoutingRatios {
    pub accuracy: f32,
    /// Fraction of examples where the classifier returned `Some(g)` but `g`
    /// didn't match the expected game. Covers both wrong-game picks and
    /// extraneous filtering of examples whose expected game is `None`.
    pub false_positive_rate: f32,
    pub elapsed_millis_p50: u64,
    pub elapsed_millis_p95: u64,
}

#[derive(Serialize)]
pub struct RetrievalRatios {
    pub recall_at_1: f32,
    pub recall_at_3: f32,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    /// Fraction of questions that hit their best achievable coverage rank
    /// (`rank == floor`). The difficulty-normalized "recall@1": single-chunk
    /// questions must land at rank 0, N-chunk questions must pack into the top
    /// N slots. Always achievable, unlike raw recall@1 for multi-chunk rows.
    pub perfect_coverage: f32,
    pub mrr_mean: f32,
    pub elapsed_millis_p50: u64,
    pub elapsed_millis_p95: u64,
}

#[derive(Serialize)]
pub struct GenerationRatios {
    pub quote: f32,
    pub refusal: f32,
    pub total_elapsed_millis_p50: u64,
    pub total_elapsed_millis_p95: u64,
    pub input_tokens_p50: usize,
    pub input_tokens_p95: usize,
    pub output_tokens_p50: usize,
    pub output_tokens_p95: usize,
}

#[derive(Serialize)]
pub struct FullEval {
    pub example: EvalExample,
    pub outcome: FullOutcome,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FullOutcome {
    Ok {
        answer: Answer,
        routing_metrics: RoutingMetrics,
        retrieval_metrics: RetrievalMetrics,
        generation_metrics: GenerationMetrics,
    },
    Errored {
        error: Vec<String>,
    },
}

#[derive(Serialize, Default, Debug)]
pub struct RoutingMetrics {
    pub correct: bool,
    pub classified: Option<String>,
    pub elapsed_millis: u64,
}

#[derive(Serialize, Default, Debug)]
#[allow(clippy::struct_excessive_bools)] // bools are each meaningful
pub struct RetrievalMetrics {
    pub recall_at_1: bool,
    pub recall_at_3: bool,
    pub recall_at_5: bool,
    pub recall_at_10: bool,
    /// `true` when the retriever achieved the best coverage rank this question
    /// allows ([`Coverage::rank`] == [`Coverage::floor`]). For single-chunk
    /// questions this is identical to `recall_at_1`; for multi-chunk questions
    /// it's the fair "recall@1 equivalent" — perfect packing of the required
    /// chunks at the top, rather than the impossible bar of fitting them all
    /// into the single top slot.
    pub perfect_coverage: bool,
    pub mrr: f32,
    pub found_at: usize,
    pub elapsed_millis: u64,
}

impl RetrievalMetrics {
    fn from(coverage: Option<Coverage>, elapsed_millis: u64) -> Self {
        coverage.map_or(
            Self {
                recall_at_1: false,
                recall_at_3: false,
                recall_at_5: false,
                recall_at_10: false,
                perfect_coverage: false,
                mrr: 0.0,
                found_at: 0,
                elapsed_millis,
            },
            |Coverage { rank, floor }| Self {
                recall_at_1: rank < 1,
                recall_at_3: rank < 3,
                recall_at_5: rank < 5,
                recall_at_10: rank < 10,
                perfect_coverage: rank == floor,
                mrr: 1.0 / (rank as f32 + 1.0),
                found_at: rank + 1,
                elapsed_millis,
            },
        )
    }
}

#[derive(Serialize, Default)]
pub struct GenerationMetrics {
    pub quote_match: bool,
    pub refused: bool,
    pub total_elapsed_millis: u64,
    /// cl100k token count for the proxy prompt (question + concatenated
    /// retrieved chunk texts). Not the exact prompt the generator builds,
    /// but a stable cross-model proxy per CLAUDE.md's convention.
    pub input_tokens: usize,
    /// cl100k token count for the generated answer.
    pub output_tokens: usize,
}

impl FullOutcome {
    pub const fn metrics(&self) -> Option<MetricsRef<'_>> {
        match self {
            Self::Ok {
                routing_metrics,
                retrieval_metrics,
                generation_metrics,
                ..
            } => Some(MetricsRef {
                routing_metrics,
                retr_metrics: retrieval_metrics,
                gen_metrics: generation_metrics,
            }),
            Self::Errored { .. } => None,
        }
    }
}

pub struct MetricsRef<'a> {
    pub routing_metrics: &'a RoutingMetrics,
    pub retr_metrics: &'a RetrievalMetrics,
    pub gen_metrics: &'a GenerationMetrics,
}

#[derive(Serialize)]
pub struct RetrievalEvaluation {
    pub evals: Vec<RetrievalEval>,
    pub routing_ratios: Option<RoutingRatios>,
    pub ratios: RetrievalRatios,
}

#[derive(Serialize)]
pub struct RetrievalEval {
    pub example: EvalExample,
    pub outcome: RetrievalOutcome,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RetrievalOutcome {
    Ok {
        routing_metrics: RoutingMetrics,
        retrieval: Vec<RetrievalResult>,
        metrics: RetrievalMetrics,
    },
    Errored {
        error: Vec<String>,
    },
}

impl RetrievalOutcome {
    pub const fn metrics(&self) -> Option<&RetrievalMetrics> {
        match self {
            Self::Ok { metrics, .. } => Some(metrics),
            Self::Errored { .. } => None,
        }
    }

    pub const fn routing_metrics(&self) -> Option<&RoutingMetrics> {
        match self {
            Self::Ok {
                routing_metrics, ..
            } => Some(routing_metrics),
            Self::Errored { .. } => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum FilterMode {
    Classifier,
    Oracle,
    None,
}

pub struct PipelineEvaluator<P: Pipeline, C: GameClassifier> {
    pipeline: P,
    classifier: C,
    games: Vec<String>,
    filter_mode: FilterMode,
    tag_filters: Vec<String>,
    limit: Option<usize>,
}

impl<P: Pipeline, C: GameClassifier> PipelineEvaluator<P, C> {
    pub const fn new(
        pipeline: P,
        classifier: C,
        games: Vec<String>,
        filter_mode: FilterMode,
        tag_filters: Vec<String>,
        limit: Option<usize>,
    ) -> Self {
        Self {
            pipeline,
            classifier,
            games,
            filter_mode,
            tag_filters,
            limit,
        }
    }

    #[allow(clippy::too_many_lines)]
    #[tracing::instrument(
        level = "info",
        name = "pipeline_eval",
        skip(self),
        fields(
            filter_mode = ?self.filter_mode,
            n_tags = self.tag_filters.len(),
            limit = ?self.limit,
        ),
    )]
    pub async fn run(&self) -> Result<FullEvaluation, EvalError> {
        let examples = get_golden_set(Path::new("./data/eval/golden.jsonl"))?
            .into_iter()
            .filter(|example| {
                self.tag_filters.is_empty()
                    || example
                        .tags
                        .iter()
                        .any(|tag| self.tag_filters.contains(tag))
            })
            .take(self.limit.unwrap_or(usize::MAX));

        let mut evals: Vec<FullEval> = stream::iter(examples).map(|example| async move {
            let start = Instant::now();
            let (routing_metrics, game_filter): (RoutingMetrics, Option<String>) = match classify_with_metrics(&self.classifier, &example, self.filter_mode, &self.games).await {
                Ok(results) => results,
                Err(e) => {
                    tracing::warn!(id = %example.id, error = %e, "errored");
                    return FullEval {
                        example,
                        outcome: FullOutcome::Errored {
                            error: flatten_error_chain(&e),
                        },
                    }
                }
            };
            let retrieval_start = Instant::now();
            let (retrieval_results, elapsed_millis_retrieval) = match self
                .pipeline
                .retrieve(
                    &example.question,
                    &QueryOptions {
                        top_k: 10,
                        game_filter,
                    },
                )
                .await
            {
                Ok(results) => {
                    let elapsed = u64::try_from(retrieval_start.elapsed().as_millis()).unwrap_or(u64::MAX);
                    (results, elapsed)
                }
                Err(e) => {
                    tracing::warn!(id = %example.id, error = %e, "errored");
                    return FullEval {
                        example,
                        outcome: FullOutcome::Errored {
                            error: flatten_error_chain(&e),
                        },
                    }
                }
            };
            let outcome = match self
                .pipeline
                .ask_with(&example.question, &retrieval_results)
                .await
            {
                Ok(text) => {
                    let retrieval_metrics = RetrievalMetrics::from(
                        chunk_coverage(&example, &retrieval_results),
                        elapsed_millis_retrieval,
                    );
                    let quote_match = check_expected_quotes(&example, &text);
                    let refused = check_refused(&example, &text);
                    let elapsed_millis = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                    let input_tokens = count_tokens(&example.question)
                        + retrieval_results[..5.min(retrieval_results.len())]
                            .iter()
                            .map(|r| count_tokens(&r.chunk.text))
                            .sum::<usize>();
                    let output_tokens = count_tokens(&text);
                    let generation_metrics = GenerationMetrics {
                        quote_match,
                        refused,
                        total_elapsed_millis: elapsed_millis,
                        input_tokens,
                        output_tokens,
                    };
                    tracing::info!(id = %example.id, ?routing_metrics, ?retrieval_metrics, quote_match, refused, "ok");
                    FullOutcome::Ok {
                        answer: Answer {
                            text,
                            retrieval: retrieval_results,
                        },
                        routing_metrics,
                        retrieval_metrics,
                        generation_metrics,
                    }
                }
                Err(e) => {
                    tracing::warn!(id = %example.id, error = %e, "errored");
                    FullOutcome::Errored {
                        error: flatten_error_chain(&e),
                    }
                }
            };

            FullEval { example, outcome }
        })
            .buffer_unordered(4) // 4 at once
            .collect()
            .await;
        // buffer_unordered scrambles order; sort by id so verbose runs are line-diffable.
        evals.sort_unstable_by(|a, b| a.example.id.cmp(&b.example.id));

        let metrics: Vec<MetricsRef> = evals.iter().filter_map(|e| e.outcome.metrics()).collect();
        let routing_metrics: Vec<&RoutingMetrics> =
            metrics.iter().map(|m| m.routing_metrics).collect();
        let retrieval_metrics: Vec<&RetrievalMetrics> =
            metrics.iter().map(|m| m.retr_metrics).collect();
        let total = metrics.len();
        let quote_passed = metrics.iter().filter(|m| m.gen_metrics.quote_match).count();
        let refused_count = metrics.iter().filter(|m| m.gen_metrics.refused).count();

        let quote = ratio(quote_passed as f32, total);
        let refusal = ratio(refused_count as f32, total);

        let mut generation_elapsed_sorted: Vec<u64> = metrics
            .iter()
            .map(|m| m.gen_metrics.total_elapsed_millis)
            .collect();
        generation_elapsed_sorted.sort_unstable();
        let p50_generation = generation_elapsed_sorted
            .get(generation_elapsed_sorted.len() / 2)
            .copied()
            .unwrap_or_default();
        let p95_generation = generation_elapsed_sorted
            .get(generation_elapsed_sorted.len() * 19 / 20)
            .copied()
            .unwrap_or_default();

        let (input_tokens_p50, input_tokens_p95) =
            percentiles(metrics.iter().map(|m| m.gen_metrics.input_tokens));
        let (output_tokens_p50, output_tokens_p95) =
            percentiles(metrics.iter().map(|m| m.gen_metrics.output_tokens));

        let retrieval_ratios = summarize_retrieval(&retrieval_metrics);
        let routing_ratios = matches!(self.filter_mode, FilterMode::Classifier)
            .then(|| summarize_routing(&routing_metrics));

        Ok(FullEvaluation {
            evals,
            routing_ratios,
            retrieval_ratios,
            generation_ratios: GenerationRatios {
                quote,
                refusal,
                total_elapsed_millis_p50: p50_generation,
                total_elapsed_millis_p95: p95_generation,
                input_tokens_p50,
                input_tokens_p95,
                output_tokens_p50,
                output_tokens_p95,
            },
        })
    }
}

pub struct RetrievalEvaluator<R: Retrieve, C: GameClassifier> {
    retriever: R,
    classifier: C,
    games: Vec<String>,
    filter_mode: FilterMode,
    tag_filters: Vec<String>,
    limit: Option<usize>,
}

impl<R: Retrieve, C: GameClassifier> RetrievalEvaluator<R, C> {
    pub const fn new(
        retriever: R,
        classifier: C,
        games: Vec<String>,
        filter_mode: FilterMode,
        tag_filters: Vec<String>,
        limit: Option<usize>,
    ) -> Self {
        Self {
            retriever,
            classifier,
            games,
            filter_mode,
            tag_filters,
            limit,
        }
    }

    #[tracing::instrument(
        level = "info",
        name = "retrieval_eval",
        skip(self),
        fields(
            filter_mode = ?self.filter_mode,
            n_tags = self.tag_filters.len(),
            limit = ?self.limit,
        ),
    )]
    pub async fn run(&self) -> Result<RetrievalEvaluation, EvalError> {
        let examples = get_golden_set(Path::new("./data/eval/golden.jsonl"))?
            .into_iter()
            .filter(|example| {
                self.tag_filters.is_empty()
                    || example
                        .tags
                        .iter()
                        .any(|tag| self.tag_filters.contains(tag))
            })
            .take(self.limit.unwrap_or(usize::MAX));

        let mut evals: Vec<RetrievalEval> = stream::iter(examples)
            .map(|example| async move {
                let (routing_metrics, game_filter): (RoutingMetrics, Option<String>) =
                    match classify_with_metrics(
                        &self.classifier,
                        &example,
                        self.filter_mode,
                        &self.games,
                    )
                    .await
                    {
                        Ok(results) => results,
                        Err(e) => {
                            tracing::warn!(id = %example.id, error = %e, "errored");
                            return RetrievalEval {
                                example,
                                outcome: RetrievalOutcome::Errored {
                                    error: flatten_error_chain(&e),
                                },
                            };
                        }
                    };

                let retrieval_start = Instant::now();
                let outcome = match self
                    .retriever
                    .retrieve(
                        &example.question,
                        &QueryOptions {
                            top_k: 10,
                            game_filter,
                        },
                    )
                    .await
                {
                    Ok(retrieval) => {
                        let elapsed_millis = u64::try_from(retrieval_start.elapsed().as_millis())
                            .unwrap_or(u64::MAX);
                        let metrics = RetrievalMetrics::from(
                            chunk_coverage(&example, &retrieval),
                            elapsed_millis,
                        );
                        tracing::info!(id = %example.id, ?routing_metrics, ?metrics, "ok");
                        RetrievalOutcome::Ok {
                            routing_metrics,
                            retrieval,
                            metrics,
                        }
                    }
                    Err(e) => {
                        tracing::warn!(id = %example.id, error = %e, "errored");
                        RetrievalOutcome::Errored {
                            error: flatten_error_chain(&e),
                        }
                    }
                };

                RetrievalEval { example, outcome }
            })
            .buffer_unordered(4) // 4 at once
            .collect()
            .await;
        // buffer_unordered scrambles order; sort by id so verbose runs are line-diffable.
        evals.sort_unstable_by(|a, b| a.example.id.cmp(&b.example.id));

        let metrics: Vec<&RetrievalMetrics> =
            evals.iter().filter_map(|e| e.outcome.metrics()).collect();
        let ratios = summarize_retrieval(&metrics);
        let routing_metrics: Vec<&RoutingMetrics> = evals
            .iter()
            .filter_map(|e| e.outcome.routing_metrics())
            .collect();
        let routing_ratios = matches!(self.filter_mode, FilterMode::Classifier)
            .then(|| summarize_routing(&routing_metrics));

        Ok(RetrievalEvaluation {
            evals,
            routing_ratios,
            ratios,
        })
    }
}

pub fn get_golden_set(path: &Path) -> Result<Vec<EvalExample>, EvalError> {
    let text = read_to_string(path).map_err(|e| EvalError::ReadFile {
        path: path.to_path_buf(),
        source: e,
    })?;
    let lines = text.lines();
    let mut examples: Vec<EvalExample> = Vec::new();
    for (line_number, line) in lines.into_iter().enumerate() {
        let example =
            serde_json::from_str::<EvalExample>(line).map_err(|e| EvalError::ParseFile {
                path: path.to_path_buf(),
                line: line_number,
                source: e,
            })?;
        examples.push(example);
    }
    Ok(examples)
}

/// Normalize a string for substring matching during evaluation.
///
/// Lowercases, strips markdown emphasis chars (`*`, `_`), strips HTML tag
/// artifacts left over from PDF parsing (`<sup>`, `</sup>`, `<sub>`, `</sub>`),
/// and collapses whitespace runs to a single space.
///
/// Apply this to both the haystack (chunk text or answer) and the needle
/// (expected substring) before checking `contains`.
pub fn normalize(s: &str) -> String {
    let lowered = s.to_lowercase();
    let stripped = lowered
        .replace("<sup>", "")
        .replace("</sup>", "")
        .replace("<sub>", "")
        .replace("</sub>", "");

    let mut out = String::with_capacity(stripped.len());
    let mut last_was_space = true;
    for c in stripped.chars() {
        if c == '*' || c == '_' {
            continue;
        }
        if c.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(c);
            last_was_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Did the model's answer include a quote from EVERY required passage?
/// (Each passage is satisfied by any one of its phrasings.) If
/// `expected_quotes` is empty, returns true (no expectations to satisfy).
pub fn check_expected_quotes(example: &EvalExample, answer: &str) -> bool {
    let normalized_answer = normalize(answer);
    example.expected_quotes.iter().all(|passage| {
        passage
            .phrasings()
            .iter()
            .any(|q| normalized_answer.contains(&normalize(q)))
    })
}

/// Does the answer contain a refusal/hedge phrase from either the global
/// default list or the per-example overrides?
pub fn check_refused(example: &EvalExample, answer: &str) -> bool {
    let normalized_answer = normalize(answer);
    DEFAULT_REFUSAL_PHRASES
        .iter()
        .any(|p| normalized_answer.contains(&normalize(p)))
        || example
            .forbidden_phrases
            .iter()
            .any(|p| normalized_answer.contains(&normalize(p)))
}

/// How well a retrieval covered a question's required passages, paired with the
/// best result the question could possibly get.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coverage {
    /// Coverage rank: the deepest (0-based) rank you must read down to before
    /// every required passage has appeared. This is what recall@k and MRR
    /// score against.
    pub rank: usize,
    /// The smallest `rank` this question could possibly achieve: one less than
    /// the number of distinct chunks the required passages occupy. A retriever
    /// hits `floor` exactly when it packs every required chunk into the top
    /// slots with nothing wasted in between. `rank == floor` is the
    /// per-question "perfect" result; for a single-chunk question `floor` is 0,
    /// so it coincides with recall@1.
    pub floor: usize,
}

/// Coverage of a question's required passages by a retrieval.
///
/// `rank` is the max, over passages, of the earliest (0-based) chunk containing
/// any of that passage's phrasings. `floor` is `(distinct required chunks) - 1`
/// — the best `rank` achievable — so `rank == floor` means the required chunks
/// were packed as tightly at the top as this question allows. `None` if any
/// passage is absent from the retrieval entirely.
///
/// Passages may share a chunk (distinct chunks are not required), so a
/// whole-rulebook retriever covers everything at rank 0. A single-passage
/// question has `floor == 0` and reduces to the old `found_at`. If
/// `expected_chunks` is empty, returns `None` (and warns — every entry should
/// expect at least one passage).
pub fn chunk_coverage(example: &EvalExample, retrieval: &[RetrievalResult]) -> Option<Coverage> {
    if example.expected_chunks.is_empty() {
        tracing::warn!(id = example.id, "unexpected empty expected_chunks");
        return None;
    }
    let normalized_chunks: Vec<String> =
        retrieval.iter().map(|r| normalize(&r.chunk.text)).collect();

    let mut earliest_per_passage = Vec::with_capacity(example.expected_chunks.len());
    for passage in &example.expected_chunks {
        let earliest = normalized_chunks.iter().position(|chunk| {
            passage
                .phrasings()
                .iter()
                .any(|needle| chunk.contains(&normalize(needle)))
        })?;
        earliest_per_passage.push(earliest);
    }

    // Non-empty: expected_chunks was non-empty and every passage matched (else
    // the `?` above returned None).
    let rank = earliest_per_passage.iter().copied().max().unwrap_or(0);
    let mut distinct = earliest_per_passage;
    distinct.sort_unstable();
    distinct.dedup();
    let floor = distinct.len() - 1;
    Some(Coverage { rank, floor })
}

/// Coverage rank only — the value recall@k and MRR score against. See
/// [`chunk_coverage`] for the full picture (including the per-question floor).
pub fn check_expected_chunks(
    example: &EvalExample,
    retrieval: &[RetrievalResult],
) -> Option<usize> {
    chunk_coverage(example, retrieval).map(|c| c.rank)
}

/// Routes one eval row through the classifier (only when filter_mode requires
/// it) and resolves the game_filter the retriever should see. Oracle and None
/// modes skip the classifier entirely — the classifier output isn't used as
/// the filter, so paying for the LLM call would just slow eval iteration with
/// no signal gained. If you want routing accuracy numbers, run with
/// `--filter-mode classifier`.
async fn classify_with_metrics<C: GameClassifier>(
    classifier: &C,
    example: &EvalExample,
    filter_mode: FilterMode,
    games: &[String],
) -> Result<(RoutingMetrics, Option<String>), C::Error> {
    match filter_mode {
        FilterMode::Oracle => Ok((RoutingMetrics::default(), example.game.clone())),
        FilterMode::None => Ok((RoutingMetrics::default(), None)),
        FilterMode::Classifier => {
            let start = Instant::now();
            let game_filter = classifier.classify(&example.question, games).await?;
            let elapsed_millis = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
            let routing_metrics = RoutingMetrics {
                correct: game_filter == example.game,
                classified: game_filter.clone(),
                elapsed_millis,
            };
            Ok((routing_metrics, game_filter))
        }
    }
}

fn flatten_error_chain(e: &(dyn std::error::Error + 'static)) -> Vec<String> {
    let mut chain = vec![e.to_string()];
    let mut current = e.source();
    while let Some(src) = current {
        chain.push(src.to_string());
        current = src.source();
    }
    chain
}

fn ratio(numerator: f32, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f32
    }
}

fn percentiles<I: IntoIterator<Item = usize>>(values: I) -> (usize, usize) {
    let mut sorted: Vec<usize> = values.into_iter().collect();
    sorted.sort_unstable();
    let p50 = sorted.get(sorted.len() / 2).copied().unwrap_or_default();
    let p95 = sorted
        .get(sorted.len() * 19 / 20)
        .copied()
        .unwrap_or_default();
    (p50, p95)
}

fn summarize_routing(metrics: &[&RoutingMetrics]) -> RoutingRatios {
    let total = metrics.len();
    let correct = metrics.iter().filter(|m| m.correct).count();
    let false_positives = metrics
        .iter()
        .filter(|m| !m.correct && m.classified.is_some())
        .count();
    let accuracy = ratio(correct as f32, total);
    let false_positive_rate = ratio(false_positives as f32, total);
    let mut elapsed_sorted: Vec<u64> = metrics.iter().map(|m| m.elapsed_millis).collect();
    elapsed_sorted.sort_unstable();
    let elapsed_millis_p50 = elapsed_sorted
        .get(elapsed_sorted.len() / 2)
        .copied()
        .unwrap_or_default();
    let elapsed_millis_p95 = elapsed_sorted
        .get(elapsed_sorted.len() * 19 / 20)
        .copied()
        .unwrap_or_default();
    RoutingRatios {
        accuracy,
        false_positive_rate,
        elapsed_millis_p50,
        elapsed_millis_p95,
    }
}

#[allow(clippy::similar_names)]
fn summarize_retrieval(metrics: &[&RetrievalMetrics]) -> RetrievalRatios {
    let total = metrics.len();
    let recall_at_1_passed = metrics.iter().filter(|m| m.recall_at_1).count();
    let recall_at_3_passed = metrics.iter().filter(|m| m.recall_at_3).count();
    let recall_at_5_passed = metrics.iter().filter(|m| m.recall_at_5).count();
    let recall_at_10_passed = metrics.iter().filter(|m| m.recall_at_10).count();
    let perfect_coverage_passed = metrics.iter().filter(|m| m.perfect_coverage).count();
    let mrr_mean = ratio(metrics.iter().map(|m| m.mrr).sum::<f32>(), total);

    let recall_at_1 = ratio(recall_at_1_passed as f32, total);
    let recall_at_3 = ratio(recall_at_3_passed as f32, total);
    let recall_at_5 = ratio(recall_at_5_passed as f32, total);
    let recall_at_10 = ratio(recall_at_10_passed as f32, total);
    let perfect_coverage = ratio(perfect_coverage_passed as f32, total);

    let mut elapsed_sorted: Vec<u64> = metrics.iter().map(|m| m.elapsed_millis).collect();
    elapsed_sorted.sort_unstable();
    let elapsed_millis_p50 = elapsed_sorted
        .get(elapsed_sorted.len() / 2)
        .copied()
        .unwrap_or_default();
    let elapsed_millis_p95 = elapsed_sorted
        .get(elapsed_sorted.len() * 19 / 20)
        .copied()
        .unwrap_or_default();

    RetrievalRatios {
        recall_at_1,
        recall_at_3,
        recall_at_5,
        recall_at_10,
        perfect_coverage,
        mrr_mean,
        elapsed_millis_p50,
        elapsed_millis_p95,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // tests compare against exact 0.0 / 1.0 sentinel values

    use super::*;

    #[test]
    fn normalize_strips_markdown_emphasis() {
        assert_eq!(normalize("**bold** and *italic*"), "bold and italic");
        assert_eq!(normalize("__under__ and _emph_"), "under and emph");
    }

    #[test]
    fn normalize_strips_sup_tags() {
        assert_eq!(normalize("the 2<sup>nd</sup> round"), "the 2nd round");
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize("hello\n\n  world"), "hello world");
        assert_eq!(normalize("  trim  me  "), "trim me");
    }

    #[test]
    fn normalize_lowercases() {
        assert_eq!(normalize("HELLO World"), "hello world");
    }

    #[test]
    fn normalize_handles_combined_artifacts() {
        let src = "The **Operations Expert** does *not* have to play\nthe card";
        let needle = "Operations Expert does not have to play the card";
        assert!(normalize(src).contains(&normalize(needle)));
    }

    fn make_example(quotes: Vec<&str>, forbidden: Vec<&str>) -> EvalExample {
        // Treat the quote list as a SINGLE passage with any-match phrasings,
        // matching the any-of-alternatives semantics these tests were written
        // against. Multi-passage (all-of) behavior is covered by its own tests.
        let expected_quotes = if quotes.is_empty() {
            vec![]
        } else {
            vec![Passage::AnyOf(quotes.into_iter().map(String::from).collect())]
        };
        EvalExample {
            id: "test".into(),
            game: Some("Pandemic".into()),
            question: "q".into(),
            expected_quotes,
            expected_chunks: vec![Passage::One("x".into())],
            expected_answer: "x".into(),
            forbidden_phrases: forbidden.into_iter().map(String::from).collect(),
            tags: vec![],
        }
    }

    fn make_chunk(text: &str) -> RetrievalResult {
        RetrievalResult {
            chunk: rag_core::Chunk {
                id: "id".into(),
                text: text.into(),
                game: "Pandemic".into(),
                doc_type: rag_core::DocType::Rules,
                page: Some(1),
                embedding: None,
            },
            score: 1.0,
        }
    }

    #[test]
    fn check_chunks_matches_any_phrasing_within_a_passage() {
        // One required passage with two acceptable phrasings (any-of).
        let mut example = make_example(vec!["x"], vec![]);
        example.expected_chunks = vec![Passage::AnyOf(vec![
            "no effect when drawn on the Infector's turn".into(),
            "of a color that has been eradicated, do not add a cube".into(),
        ])];
        // Only the second phrasing is in the retrieved chunk, at rank 0.
        let retrieval = vec![make_chunk(
            "If, however, the pictured city is of a color that has been eradicated, do not add a cube.",
        )];
        assert_eq!(check_expected_chunks(&example, &retrieval), Some(0));

        // Match at rank 2 (third chunk): coverage is 2.
        let retrieval = vec![
            make_chunk("Unrelated chunk one."),
            make_chunk("Unrelated chunk two."),
            make_chunk(
                "If, however, the pictured city is of a color that has been eradicated, do not add a cube.",
            ),
        ];
        assert_eq!(check_expected_chunks(&example, &retrieval), Some(2));

        // Neither phrasing present anywhere: None.
        let retrieval = vec![make_chunk("Some unrelated chunk text.")];
        assert_eq!(check_expected_chunks(&example, &retrieval), None);
    }

    #[test]
    fn check_chunks_coverage_rank_is_the_deepest_required_passage() {
        // Two distinct required passages (all-of).
        let mut example = make_example(vec!["x"], vec![]);
        example.expected_chunks = vec![
            Passage::One("first required rule".into()),
            Passage::One("second required rule".into()),
        ];

        // A at rank 0, B at rank 2 => coverage is the deeper one, 2.
        let retrieval = vec![
            make_chunk("here is the first required rule, stated plainly"),
            make_chunk("unrelated"),
            make_chunk("and here is the second required rule"),
        ];
        assert_eq!(check_expected_chunks(&example, &retrieval), Some(2));

        // Both passages in the SAME chunk at rank 1: distinct chunks are not
        // required, so coverage is 1 (a full-context retriever would pass @1).
        let retrieval = vec![
            make_chunk("unrelated"),
            make_chunk("the first required rule and also the second required rule together"),
        ];
        assert_eq!(check_expected_chunks(&example, &retrieval), Some(1));

        // One passage present, the other missing entirely: None.
        let retrieval = vec![make_chunk("only the first required rule is here")];
        assert_eq!(check_expected_chunks(&example, &retrieval), None);
    }

    #[test]
    fn check_chunks_empty_expected_returns_none() {
        let mut example = make_example(vec!["x"], vec![]);
        example.expected_chunks = vec![];
        assert_eq!(check_expected_chunks(&example, &[]), None);
    }

    #[test]
    fn retrieval_metrics_from_rank() {
        // Rank 0, floor 0: every recall@k passes, mrr = 1.0, and it's perfect.
        let m = RetrievalMetrics::from(Some(Coverage { rank: 0, floor: 0 }), 0);
        assert!(m.recall_at_1 && m.recall_at_3 && m.recall_at_5 && m.recall_at_10);
        assert!(m.perfect_coverage);
        assert_eq!(m.mrr, 1.0);

        // Rank 2: @1 misses, @3/@5/@10 hit, mrr = 1/3.
        let m = RetrievalMetrics::from(Some(Coverage { rank: 2, floor: 0 }), 0);
        assert!(!m.recall_at_1);
        assert!(m.recall_at_3 && m.recall_at_5 && m.recall_at_10);
        assert!((m.mrr - 1.0 / 3.0).abs() < 1e-6);

        // A 3-chunk question packed into the top 3 slots: rank 2 == floor 2, so
        // recall@1 is (correctly) false but perfect_coverage is true.
        let m = RetrievalMetrics::from(Some(Coverage { rank: 2, floor: 2 }), 0);
        assert!(!m.recall_at_1);
        assert!(m.perfect_coverage);

        // Same floor, one wasted slot (rank 3 > floor 2): not perfect.
        let m = RetrievalMetrics::from(Some(Coverage { rank: 3, floor: 2 }), 0);
        assert!(!m.perfect_coverage);

        // Rank 9: only @10 hits.
        let m = RetrievalMetrics::from(Some(Coverage { rank: 9, floor: 0 }), 0);
        assert!(!m.recall_at_1 && !m.recall_at_3 && !m.recall_at_5);
        assert!(m.recall_at_10);
        assert!(!m.perfect_coverage);
        assert!((m.mrr - 0.1).abs() < 1e-6);

        // No match: all false, mrr = 0, not perfect.
        let m = RetrievalMetrics::from(None, 0);
        assert!(!m.recall_at_1 && !m.recall_at_3 && !m.recall_at_5 && !m.recall_at_10);
        assert!(!m.perfect_coverage);
        assert_eq!(m.mrr, 0.0);
    }

    #[test]
    fn chunk_coverage_reports_floor() {
        let mut example = make_example(vec!["x"], vec![]);
        example.expected_chunks = vec![
            Passage::One("first required rule".into()),
            Passage::One("second required rule".into()),
        ];

        // Two distinct chunks packed at the top (ranks 0 and 1): floor 1, and
        // rank 1 == floor, so this is the perfect result for a 2-chunk question.
        let retrieval = vec![
            make_chunk("here is the first required rule"),
            make_chunk("and here is the second required rule"),
        ];
        assert_eq!(
            chunk_coverage(&example, &retrieval),
            Some(Coverage { rank: 1, floor: 1 })
        );

        // A wasted slot between them (ranks 0 and 2): floor stays 1, rank is 2.
        let retrieval = vec![
            make_chunk("here is the first required rule"),
            make_chunk("unrelated"),
            make_chunk("and here is the second required rule"),
        ];
        assert_eq!(
            chunk_coverage(&example, &retrieval),
            Some(Coverage { rank: 2, floor: 1 })
        );

        // Both passages in one shared chunk: a single distinct chunk, floor 0.
        let retrieval = vec![make_chunk(
            "the first required rule and the second required rule together",
        )];
        assert_eq!(
            chunk_coverage(&example, &retrieval),
            Some(Coverage { rank: 0, floor: 0 })
        );
    }

    #[test]
    fn check_quotes_matches_any_phrasing_within_a_passage() {
        let example = make_example(
            vec![
                "A player gets 4 actions to spend on her turn",
                "Each player takes 4 actions",
            ],
            vec![],
        );
        // Matches first phrasing.
        assert!(check_expected_quotes(
            &example,
            "Per the rulebook: A player gets **4** actions to spend on her turn."
        ));
        // Matches second phrasing even when first is absent.
        assert!(check_expected_quotes(
            &example,
            "The rules say: Each player takes 4 actions per turn."
        ));
        // Neither present.
        assert!(!check_expected_quotes(
            &example,
            "Players have lots of options on their turn."
        ));
    }

    #[test]
    fn check_quotes_requires_a_quote_from_every_passage() {
        // Two distinct required passages (all-of): the answer must quote both.
        let example = EvalExample {
            expected_quotes: vec![
                Passage::One("destroying a Town generates 1 Fear".into()),
                Passage::One("Terror Level 3: No Cities on the island".into()),
            ],
            ..make_example(vec![], vec![])
        };
        // Only the first passage quoted: fails.
        assert!(!check_expected_quotes(
            &example,
            "Destroying a Town generates 1 Fear, so press the attack."
        ));
        // Both passages quoted: passes.
        assert!(check_expected_quotes(
            &example,
            "Destroying a Town generates 1 Fear. At Terror Level 3: No Cities on the island wins."
        ));
        // Neither: fails.
        assert!(!check_expected_quotes(&example, "Fear is good and cities are bad."));
    }

    #[test]
    fn check_quotes_empty_expected_passes() {
        let example = make_example(vec![], vec![]);
        assert!(check_expected_quotes(&example, "any answer at all"));
    }

    #[test]
    fn check_refused_default_phrases() {
        let example = make_example(vec!["x"], vec![]);
        assert!(check_refused(
            &example,
            "I am unable to determine the answer."
        ));
        assert!(check_refused(&example, "No chunk supports this answer."));
        assert!(check_refused(
            &example,
            "The exact behavior is not specified in the rules."
        ));
        assert!(!check_refused(&example, "The rule clearly states X."));
    }

    #[test]
    fn check_refused_per_example_phrases() {
        let example = make_example(vec!["x"], vec!["definitely wrong"]);
        assert!(check_refused(&example, "this is definitely wrong"));
        assert!(!check_refused(&example, "this is correct"));
    }

    #[tokio::test]
    async fn check_golden_questions() {
        let golden_qs = get_golden_set(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/eval/golden.jsonl"),
        )
        .unwrap();

        let q = golden_qs.first().unwrap();
        assert_eq!(q.id, "pandemic-001", "game id should be loaded properly");
        assert_eq!(
            q.game,
            Some("Pandemic".to_string()),
            "game should be loaded properly"
        );
        let q = golden_qs.get(1).unwrap();
        assert_eq!(
            q.id, "pandemic-002",
            "second game id should be loaded properly"
        );
        assert_eq!(
            q.game,
            Some("Pandemic".to_string()),
            "second game should be loaded properly"
        );
    }

    /// Every phrasing in every passage (chunks and quotes) should be a real
    /// substring of its source rulebook (after normalization). If this
    /// regresses, the eval will silently report 0% chunk-match for that entry.
    /// Covers all games present in the golden set — an unmapped game is a
    /// failure, not a silent skip, so new games can't slip through unverified.
    #[test]
    fn every_expected_substring_appears_in_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let golden = get_golden_set(&root.join("data/eval/golden.jsonl")).unwrap();
        let sources: std::collections::HashMap<&str, String> = [
            ("Pandemic", "data/pdfs/pandemic.txt"),
            ("Challengers!", "data/pdfs/challengers-rulebook.txt"),
            (
                "The Quacks of Quedlinburg",
                "data/pdfs/the-quacks-of-quedlinburg-rulebook.txt",
            ),
            ("Stone Age", "data/pdfs/stone-age-rulebook.txt"),
            ("Res Arcana", "data/pdfs/res-arcana-rulebook.txt"),
            ("Paleo", "data/pdfs/paleo-rulebook.txt"),
            ("Spirit Island", "data/pdfs/spirit-island-rulebook.txt"),
        ]
        .iter()
        .map(|(g, p)| (*g, normalize(&read_to_string(root.join(p)).unwrap())))
        .collect();

        let mut failures = Vec::new();
        for ex in &golden {
            let game = ex.game.as_deref().unwrap_or("");
            let Some(src) = sources.get(game) else {
                failures.push(format!("{}: game {game:?} has no source mapping", ex.id));
                continue;
            };
            let check = |kind: &str, passages: &[Passage], failures: &mut Vec<String>| {
                for (i, passage) in passages.iter().enumerate() {
                    for phrasing in passage.phrasings() {
                        if !src.contains(&normalize(phrasing)) {
                            failures.push(format!(
                                "{}: {kind}[{i}] not in source: {phrasing:?}",
                                ex.id
                            ));
                        }
                    }
                }
            };
            check("expected_chunks", &ex.expected_chunks, &mut failures);
            check("expected_quotes", &ex.expected_quotes, &mut failures);
        }
        assert!(failures.is_empty(), "verification failures: {failures:#?}");
    }
}
