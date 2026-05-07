use rag_core::{Answer, Pipeline, QueryOptions, RetrievalResult};
use serde::{Deserialize, Serialize};
use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
};

/// Default phrases that mark a refused / hedged answer. Checked against
/// every answer in addition to per-example `forbidden_phrases`.
pub const DEFAULT_REFUSAL_PHRASES: &[&str] = &[
    "no information",
    "cannot determine",
    "unable to determine",
    "no chunk supports",
    "not specified",
];

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
/// If `expected_chunk_contains` is empty, returns true (no expectations).
pub fn check_expected_chunk_contains(
    example: &EvalExample,
    retrieval: &[RetrievalResult],
) -> bool {
    if example.expected_chunk_contains.is_empty() {
        return true;
    }
    let normalized_chunks: Vec<String> = retrieval
        .iter()
        .map(|r| normalize(&r.chunk.text))
        .collect();
    example.expected_chunk_contains.iter().any(|needle| {
        let normalized_needle = normalize(needle);
        normalized_chunks
            .iter()
            .any(|chunk| chunk.contains(&normalized_needle))
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

#[derive(Serialize, Default)]
pub struct ExampleMetrics {
    pub quote_match: bool,
    pub chunk_match: bool,
    pub refused: bool,
}

#[derive(Serialize)]
pub struct SingleEval {
    pub example: EvalExample,
    pub outcome: ExampleOutcome,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExampleOutcome {
    Ok {
        answer: Answer,
        metrics: ExampleMetrics,
    },
    Errored {
        error: Vec<String>,
    },
}

impl ExampleOutcome {
    pub fn metrics(&self) -> Option<&ExampleMetrics> {
        match self {
            Self::Ok { metrics, .. } => Some(metrics),
            _ => None,
        }
    }
}

#[derive(Serialize)]
pub struct Evaluation {
    pub evals: Vec<SingleEval>,
    pub quote_ratio: f32,
    pub chunk_ratio: f32,
    pub refusal_ratio: f32,
}

pub struct Evaluator<P: Pipeline> {
    pipeline: P,
}

impl<P: Pipeline> Evaluator<P> {
    pub fn new(pipeline: P) -> Self {
        Self { pipeline }
    }

    pub async fn run(&self) -> Result<Evaluation, EvalError> {
        let examples = get_golden_set(Path::new("./data/eval/golden.jsonl"))?;
        let mut evals: Vec<SingleEval> = Vec::new();

        for example in examples {
            let outcome = match self
                .pipeline
                .ask(
                    &example.question,
                    &QueryOptions {
                        top_k: 5,
                        game_filter: example.game.clone(),
                    },
                )
                .await
            {
                Ok(answer) => {
                    let quote_match = check_expected_quote(&example, &answer.text);
                    let chunk_match =
                        check_expected_chunk_contains(&example, &answer.retrieval);
                    let refused = check_refused(&example, &answer.text);
                    let metrics = ExampleMetrics {
                        quote_match,
                        chunk_match,
                        refused,
                    };
                    tracing::info!(id = %example.id, quote_match, chunk_match, refused, "ok");
                    ExampleOutcome::Ok { answer, metrics }
                }
                Err(e) => {
                    tracing::warn!(id = %example.id, error = %e, "errored");
                    ExampleOutcome::Errored {
                        error: flatten_error_chain(&e),
                    }
                }
            };

            evals.push(SingleEval { example, outcome });
        }

        let metrics: Vec<&ExampleMetrics> =
            evals.iter().filter_map(|e| e.outcome.metrics()).collect();
        let total = metrics.len();
        let quote_passed = metrics.iter().filter(|m| m.quote_match).count();
        let chunk_passed = metrics.iter().filter(|m| m.chunk_match).count();
        let refused_count = metrics.iter().filter(|m| m.refused).count();

        let quote_ratio = if total == 0 {
            0.0
        } else {
            quote_passed as f32 / total as f32
        };
        let chunk_ratio = if total == 0 {
            0.0
        } else {
            chunk_passed as f32 / total as f32
        };
        let refusal_ratio = if total == 0 {
            0.0
        } else {
            refused_count as f32 / total as f32
        };

        Ok(Evaluation {
            evals,
            quote_ratio,
            chunk_ratio,
            refusal_ratio,
        })
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
        // Only the second alternative is in the retrieved chunk: still passes.
        let retrieval = vec![make_chunk(
            "If, however, the pictured city is of a color that has been eradicated, do not add a cube.",
        )];
        assert!(check_expected_chunk_contains(&example, &retrieval));
        // Neither alternative present: fails.
        let retrieval = vec![make_chunk("Some unrelated chunk text.")];
        assert!(!check_expected_chunk_contains(&example, &retrieval));
    }

    #[test]
    fn check_chunk_contains_empty_expected_passes() {
        let mut example = make_example(vec!["x"], vec![]);
        example.expected_chunk_contains = vec![];
        assert!(check_expected_chunk_contains(&example, &[]));
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
