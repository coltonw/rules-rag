use rag_core::{Answer, Pipeline, QueryOptions, RetrievalResult, Retriever};
use serde::{Deserialize, Serialize};
use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
    time::Instant,
};

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

#[derive(Serialize, Deserialize)]
pub struct EvalExample {
    pub id: String,
    pub game: Option<String>,
    pub question: String,
    /// Acceptable verbatim quotes from the rulebook. ANY-match: the answer
    /// passes the quote check if it contains any one of these (after
    /// normalization). Use multiple entries when the same rule is stated
    /// in more than one place, or when the worked example is also an
    /// acceptable grounding source.
    pub expected_quote: Vec<String>,
    /// Substrings expected to appear in retrieved chunks. ANY-match: passes
    /// if at least one of these substrings appears in any one of the
    /// retrieved chunks (after normalization). Use multiple entries when
    /// the rule appears in more than one place in the rulebook and either
    /// retrieved passage is acceptable grounding.
    pub expected_chunk_contains: Vec<String>,
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
    pub retrieval_ratios: RetrievalRatios,
    pub generation_ratios: GenerationRatios,
}

#[derive(Serialize)]
pub struct RetrievalRatios {
    pub recall_at_1: f32,
    pub recall_at_3: f32,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub mrr_mean: f32,
    pub elapsed_millis_p50: u64,
    pub elapsed_millis_p95: u64,
}

#[derive(Serialize)]
pub struct GenerationRatios {
    pub quote: f32,
    pub refusal: f32,
    pub elapsed_millis_p50: u64,
    pub elapsed_millis_p95: u64,
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
        retrieval_metrics: RetrievalMetrics,
        generation_metrics: GenerationMetrics,
    },
    Errored {
        error: Vec<String>,
    },
}

#[derive(Serialize, Default, Debug)]
pub struct RetrievalMetrics {
    pub recall_at_1: bool,
    pub recall_at_3: bool,
    pub recall_at_5: bool,
    pub recall_at_10: bool,
    pub mrr: f32,
    pub found_at: usize,
    pub elapsed_millis: u64,
}

impl RetrievalMetrics {
    fn from(found: Option<usize>, elapsed_millis: u64) -> Self {
        match found {
            None => Self {
                recall_at_1: false,
                recall_at_3: false,
                recall_at_5: false,
                recall_at_10: false,
                mrr: 0.0,
                found_at: 0,
                elapsed_millis,
            },
            Some(idx) => Self {
                recall_at_1: idx < 1,
                recall_at_3: idx < 3,
                recall_at_5: idx < 5,
                recall_at_10: idx < 10,
                mrr: 1.0 / (idx as f32 + 1.0),
                found_at: idx + 1,
                elapsed_millis,
            },
        }
    }
}

#[derive(Serialize, Default)]
pub struct GenerationMetrics {
    pub quote_match: bool,
    pub refused: bool,
    pub elapsed_millis: u64,
}

impl FullOutcome {
    pub fn metrics<'s>(&'s self) -> Option<MetricsRef<'s>> {
        match self {
            Self::Ok {
                retrieval_metrics,
                generation_metrics,
                ..
            } => Some(MetricsRef {
                retr_metrics: retrieval_metrics,
                gen_metrics: generation_metrics,
            }),
            _ => None,
        }
    }
}

pub struct MetricsRef<'a> {
    pub retr_metrics: &'a RetrievalMetrics,
    pub gen_metrics: &'a GenerationMetrics,
}

#[derive(Serialize)]
pub struct RetrievalEvaluation {
    pub evals: Vec<RetrievalEval>,
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
        retrieval: Vec<RetrievalResult>,
        metrics: RetrievalMetrics,
    },
    Errored {
        error: Vec<String>,
    },
}

impl RetrievalOutcome {
    pub fn metrics(&self) -> Option<&RetrievalMetrics> {
        match self {
            Self::Ok { metrics, .. } => Some(metrics),
            _ => None,
        }
    }
}

pub struct PipelineEvaluator<P: Pipeline> {
    pipeline: P,
}

impl<P: Pipeline> PipelineEvaluator<P> {
    pub fn new(pipeline: P) -> Self {
        Self { pipeline }
    }

    pub async fn run(&self) -> Result<FullEvaluation, EvalError> {
        let examples = get_golden_set(Path::new("./data/eval/golden.jsonl"))?;
        let mut evals: Vec<FullEval> = Vec::new();

        for example in examples {
            let start = Instant::now();
            let (retrieval_results, elapsed_millis_retrieval) = match self
                .pipeline
                .retrieve(
                    &example.question,
                    &QueryOptions {
                        top_k: 5,
                        game_filter: example.game.clone(),
                    },
                )
                .await
            {
                Ok(results) => {
                    let elapsed = start.elapsed().as_millis() as u64;
                    (results, elapsed)
                }
                Err(e) => {
                    tracing::warn!(id = %example.id, error = %e, "errored");
                    evals.push(FullEval {
                        example,
                        outcome: FullOutcome::Errored {
                            error: flatten_error_chain(&e),
                        },
                    });
                    continue;
                }
            };
            let outcome = match self
                .pipeline
                .ask_with(&example.question, &retrieval_results)
                .await
            {
                Ok(text) => {
                    let retrieval_metrics = RetrievalMetrics::from(
                        check_expected_chunk_contains(&example, &retrieval_results),
                        elapsed_millis_retrieval,
                    );
                    let quote_match = check_expected_quote(&example, &text);
                    let refused = check_refused(&example, &text);
                    let elapsed_millis = start.elapsed().as_millis() as u64;
                    let generation_metrics = GenerationMetrics {
                        quote_match,
                        refused,
                        elapsed_millis,
                    };
                    tracing::info!(id = %example.id, quote_match, ?retrieval_metrics, refused, "ok");
                    FullOutcome::Ok {
                        answer: Answer {
                            text,
                            retrieval: retrieval_results,
                        },
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

            evals.push(FullEval { example, outcome });
        }

        let metrics: Vec<MetricsRef> = evals.iter().filter_map(|e| e.outcome.metrics()).collect();
        let total = metrics.len();
        let recall_at_1_passed = metrics
            .iter()
            .filter(|m| m.retr_metrics.recall_at_1)
            .count();
        let recall_at_3_passed = metrics
            .iter()
            .filter(|m| m.retr_metrics.recall_at_3)
            .count();
        let recall_at_5_passed = metrics
            .iter()
            .filter(|m| m.retr_metrics.recall_at_5)
            .count();
        let recall_at_10_passed = metrics
            .iter()
            .filter(|m| m.retr_metrics.recall_at_10)
            .count();
        let mrr_mean = metrics.iter().map(|m| m.retr_metrics.mrr).sum::<f32>() / total as f32;
        let quote_passed = metrics.iter().filter(|m| m.gen_metrics.quote_match).count();
        let refused_count = metrics.iter().filter(|m| m.gen_metrics.refused).count();

        let recall_at_1 = ratio(recall_at_1_passed, total);
        let recall_at_3 = ratio(recall_at_3_passed, total);
        let recall_at_5 = ratio(recall_at_5_passed, total);
        let recall_at_10 = ratio(recall_at_10_passed, total);
        let quote = ratio(quote_passed, total);
        let refusal = ratio(refused_count, total);

        let mut retrieval_elapsed_sorted: Vec<u64> = metrics
            .iter()
            .map(|m| m.retr_metrics.elapsed_millis)
            .collect();
        retrieval_elapsed_sorted.sort();
        let p50_retrieval = retrieval_elapsed_sorted[retrieval_elapsed_sorted.len() / 2];
        let p95_retrieval = retrieval_elapsed_sorted[retrieval_elapsed_sorted.len() * 19 / 20];
        let mut generation_elapsed_sorted: Vec<u64> = metrics
            .iter()
            .map(|m| m.gen_metrics.elapsed_millis)
            .collect();
        generation_elapsed_sorted.sort();
        let p50_generation = generation_elapsed_sorted[generation_elapsed_sorted.len() / 2];
        let p95_generation = generation_elapsed_sorted[generation_elapsed_sorted.len() * 19 / 20];

        Ok(FullEvaluation {
            evals,
            retrieval_ratios: RetrievalRatios {
                recall_at_1,
                recall_at_3,
                recall_at_5,
                recall_at_10,
                mrr_mean,
                elapsed_millis_p50: p50_retrieval,
                elapsed_millis_p95: p95_retrieval,
            },
            generation_ratios: GenerationRatios {
                quote,
                refusal,
                elapsed_millis_p50: p50_generation,
                elapsed_millis_p95: p95_generation,
            },
        })
    }
}

pub struct RetrievalEvaluator<R: Retriever> {
    retriever: R,
}

impl<R: Retriever> RetrievalEvaluator<R> {
    pub fn new(retriever: R) -> Self {
        Self { retriever }
    }

    pub async fn run(&self) -> Result<RetrievalEvaluation, EvalError> {
        let examples = get_golden_set(Path::new("./data/eval/golden.jsonl"))?;
        let mut evals: Vec<RetrievalEval> = Vec::new();

        for example in examples {
            let start = Instant::now();
            let outcome = match self
                .retriever
                .retrieve(
                    &example.question,
                    &QueryOptions {
                        top_k: 5,
                        game_filter: example.game.clone(),
                    },
                )
                .await
            {
                Ok(retrieval) => {
                    let elapsed_millis = start.elapsed().as_millis() as u64;
                    let metrics = RetrievalMetrics::from(
                        check_expected_chunk_contains(&example, &retrieval),
                        elapsed_millis,
                    );
                    tracing::info!(id = %example.id, ?metrics, "ok");
                    RetrievalOutcome::Ok { retrieval, metrics }
                }
                Err(e) => {
                    tracing::warn!(id = %example.id, error = %e, "errored");
                    RetrievalOutcome::Errored {
                        error: flatten_error_chain(&e),
                    }
                }
            };

            evals.push(RetrievalEval { example, outcome });
        }

        let metrics: Vec<&RetrievalMetrics> =
            evals.iter().filter_map(|e| e.outcome.metrics()).collect();
        let total = metrics.len();
        let recall_at_1_passed = metrics.iter().filter(|m| m.recall_at_1).count();
        let recall_at_3_passed = metrics.iter().filter(|m| m.recall_at_3).count();
        let recall_at_5_passed = metrics.iter().filter(|m| m.recall_at_5).count();
        let recall_at_10_passed = metrics.iter().filter(|m| m.recall_at_10).count();
        let mrr_mean = metrics.iter().map(|m| m.mrr).sum::<f32>() / total as f32;

        let recall_at_1 = ratio(recall_at_1_passed, total);
        let recall_at_3 = ratio(recall_at_3_passed, total);
        let recall_at_5 = ratio(recall_at_5_passed, total);
        let recall_at_10 = ratio(recall_at_10_passed, total);

        let mut elapsed_sorted: Vec<u64> = metrics.iter().map(|m| m.elapsed_millis).collect();
        elapsed_sorted.sort();
        let elapsed_millis_p50 = elapsed_sorted[elapsed_sorted.len() / 2];
        let elapsed_millis_p95 = elapsed_sorted[elapsed_sorted.len() * 19 / 20];

        Ok(RetrievalEvaluation {
            evals,
            ratios: RetrievalRatios {
                recall_at_1,
                recall_at_3,
                recall_at_5,
                recall_at_10,
                mrr_mean,
                elapsed_millis_p50,
                elapsed_millis_p95,
            },
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

/// Did the model's answer include any one of the expected verbatim quotes?
/// If `expected_quote` is empty, returns true (no expectations to satisfy).
pub fn check_expected_quote(example: &EvalExample, answer: &str) -> bool {
    if example.expected_quote.is_empty() {
        return true;
    }
    let normalized_answer = normalize(answer);
    example
        .expected_quote
        .iter()
        .any(|q| normalized_answer.contains(&normalize(q)))
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

/// Did any retrieved chunk contain at least one of the expected substrings?
/// If `expected_chunk_contains` is empty, returns default retrieval result.
pub fn check_expected_chunk_contains(
    example: &EvalExample,
    retrieval: &[RetrievalResult],
) -> Option<usize> {
    if example.expected_chunk_contains.is_empty() {
        tracing::warn!(id = example.id, "unexpected empty expected_chunk_contains");
        return None;
    }
    let normalized_chunks: Vec<String> =
        retrieval.iter().map(|r| normalize(&r.chunk.text)).collect();
    normalized_chunks.iter().position(|chunk| {
        example.expected_chunk_contains.iter().any(|needle| {
            let normalized_needle = normalize(needle);
            chunk.contains(&normalized_needle)
        })
    })
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

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
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
        EvalExample {
            id: "test".into(),
            game: Some("Pandemic".into()),
            question: "q".into(),
            expected_quote: quotes.into_iter().map(String::from).collect(),
            expected_chunk_contains: vec!["x".into()],
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
    fn check_chunk_contains_matches_any_alternative() {
        let mut example = make_example(vec!["x"], vec![]);
        example.expected_chunk_contains = vec![
            "no effect when drawn on the Infector's turn".into(),
            "of a color that has been eradicated, do not add a cube".into(),
        ];
        // Only the second alternative is in the retrieved chunk, at rank 0.
        let retrieval = vec![make_chunk(
            "If, however, the pictured city is of a color that has been eradicated, do not add a cube.",
        )];
        assert_eq!(check_expected_chunk_contains(&example, &retrieval), Some(0));

        // Match at rank 2 (third chunk): position is 2.
        let retrieval = vec![
            make_chunk("Unrelated chunk one."),
            make_chunk("Unrelated chunk two."),
            make_chunk(
                "If, however, the pictured city is of a color that has been eradicated, do not add a cube.",
            ),
        ];
        assert_eq!(check_expected_chunk_contains(&example, &retrieval), Some(2));

        // Neither alternative present anywhere: None.
        let retrieval = vec![make_chunk("Some unrelated chunk text.")];
        assert_eq!(check_expected_chunk_contains(&example, &retrieval), None);
    }

    #[test]
    fn check_chunk_contains_empty_expected_returns_none() {
        let mut example = make_example(vec!["x"], vec![]);
        example.expected_chunk_contains = vec![];
        assert_eq!(check_expected_chunk_contains(&example, &[]), None);
    }

    #[test]
    fn retrieval_metrics_from_rank() {
        // Rank 0: every recall@k passes, mrr = 1.0.
        let m = RetrievalMetrics::from(Some(0), 0);
        assert!(m.recall_at_1 && m.recall_at_3 && m.recall_at_5 && m.recall_at_10);
        assert_eq!(m.mrr, 1.0);

        // Rank 2: @1 misses, @3/@5/@10 hit, mrr = 1/3.
        let m = RetrievalMetrics::from(Some(2), 0);
        assert!(!m.recall_at_1);
        assert!(m.recall_at_3 && m.recall_at_5 && m.recall_at_10);
        assert!((m.mrr - 1.0 / 3.0).abs() < 1e-6);

        // Rank 9: only @10 hits.
        let m = RetrievalMetrics::from(Some(9), 0);
        assert!(!m.recall_at_1 && !m.recall_at_3 && !m.recall_at_5);
        assert!(m.recall_at_10);
        assert!((m.mrr - 0.1).abs() < 1e-6);

        // No match: all false, mrr = 0.
        let m = RetrievalMetrics::from(None, 0);
        assert!(!m.recall_at_1 && !m.recall_at_3 && !m.recall_at_5 && !m.recall_at_10);
        assert_eq!(m.mrr, 0.0);
    }

    #[test]
    fn check_quote_matches_any_alternative() {
        let example = make_example(
            vec![
                "A player gets 4 actions to spend on her turn",
                "Each player takes 4 actions",
            ],
            vec![],
        );
        // Matches first alternative.
        assert!(check_expected_quote(
            &example,
            "Per the rulebook: A player gets **4** actions to spend on her turn."
        ));
        // Matches second alternative even when first is absent.
        assert!(check_expected_quote(
            &example,
            "The rules say: Each player takes 4 actions per turn."
        ));
        // Neither present.
        assert!(!check_expected_quote(
            &example,
            "Players have lots of options on their turn."
        ));
    }

    #[test]
    fn check_quote_empty_expected_passes() {
        let example = make_example(vec![], vec![]);
        assert!(check_expected_quote(&example, "any answer at all"));
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

    /// Every entry's expected_chunk_contains and expected_quote should be
    /// a real substring of its source rulebook (after normalization).
    /// If this regresses, the eval will silently report 0% chunk-match.
    #[test]
    fn every_expected_substring_appears_in_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let golden = get_golden_set(&root.join("data/eval/golden.jsonl")).unwrap();
        let sources: std::collections::HashMap<&str, String> = [
            ("Pandemic", "data/pdfs/pandemic.txt"),
            ("Challengers!", "data/pdfs/challengers-rulebook.txt"),
            (
                "The Quacks of Quedlinberg",
                "data/pdfs/the-quacks-of-quedlinburg-rulebook.txt",
            ),
        ]
        .iter()
        .map(|(g, p)| (*g, normalize(&read_to_string(root.join(p)).unwrap())))
        .collect();

        let mut failures = Vec::new();
        for ex in &golden {
            let game = ex.game.as_deref().unwrap_or("");
            let Some(src) = sources.get(game) else {
                continue;
            };
            for (i, chunk_needle_str) in ex.expected_chunk_contains.iter().enumerate() {
                let needle = normalize(chunk_needle_str);
                if !src.contains(&needle) {
                    failures.push(format!(
                        "{}: expected_chunk_contains[{}] not in source: {:?}",
                        ex.id, i, chunk_needle_str
                    ));
                }
            }
            for (i, quote) in ex.expected_quote.iter().enumerate() {
                let needle = normalize(quote);
                if !src.contains(&needle) {
                    failures.push(format!(
                        "{}: expected_quote[{}] not in source: {:?}",
                        ex.id, i, quote
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "verification failures: {failures:#?}");
    }
}
