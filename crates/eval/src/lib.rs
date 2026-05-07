use rag_core::{Answer, Pipeline, QueryOptions};
use serde::{Deserialize, Serialize};
use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
pub struct EvalExample {
    pub id: String,
    pub game: Option<String>,
    pub question: String,
    pub expected_answer_contains: Vec<String>,
    pub expected_pages: Vec<u32>,
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

pub fn does_answer_contain(example: &EvalExample, answer: &str) -> Option<bool> {
    if example.expected_answer_contains.is_empty() {
        return None;
    }
    let answer_lower = answer.to_ascii_lowercase();
    Some(
        example
            .expected_answer_contains
            .iter()
            .all(|s| answer_lower.contains(&s.to_ascii_lowercase())),
    )
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
    pub answer_contains: Option<bool>,
    pub recall_at_k: Option<f32>, // future
    pub mrr: Option<f32>,         // future
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
    pub ratio: f32,
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
                    let ac = does_answer_contain(&example, &answer.text);
                    let metrics = ExampleMetrics {
                        answer_contains: ac,
                        ..Default::default()
                    };
                    tracing::info!(id = %example.id, answer_contains = ?ac, "ok");
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

        let scored: Vec<bool> = evals
            .iter()
            .filter_map(|e| e.outcome.metrics()?.answer_contains)
            .collect();
        let total = scored.len();
        let passed = scored.iter().filter(|&&p| p).count();

        let ratio = if total == 0 {
            0.0
        } else {
            passed as f32 / total as f32
        };
        Ok(Evaluation { evals, ratio })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
