use rag_core::{Answer, Pipeline};
use serde::{Deserialize, Serialize};
use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
pub struct EvalExample {
    pub id: String,
    pub game: String,
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
    #[error("pipeline failed for example {id}")]
    PipelineFailure {
        id: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
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

pub fn does_answer_contain(example: &EvalExample, answer: &str) -> bool {
    for answer_should_contain in &example.expected_answer_contains {
        if !answer
            .to_ascii_lowercase()
            .trim()
            .contains(answer_should_contain.to_ascii_lowercase().trim())
        {
            return false;
        }
    }
    true
}

#[derive(Serialize)]
pub struct ExampleMetrics {
    pub answer_contains: Option<bool>,
    pub recall_at_k: Option<f32>, // future
    pub mrr: Option<f32>,         // future
}

#[derive(Serialize)]
pub struct SingleEval {
    pub example: EvalExample,
    pub answer: Answer,
    pub metrics: ExampleMetrics,
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
            let answer = self.pipeline.ask(&example.question).await.map_err(|e| {
                EvalError::PipelineFailure {
                    id: example.id.clone(),
                    source: Box::new(e),
                }
            })?;
            let answer_contains = does_answer_contain(&example, &answer.text);

            evals.push(SingleEval {
                example,
                answer,
                metrics: ExampleMetrics {
                    answer_contains: Some(answer_contains),
                    recall_at_k: None,
                    mrr: None,
                },
            });
        }

        let total = evals
            .iter()
            .filter(|e| e.metrics.answer_contains.is_some())
            .count();
        let passed = evals
            .iter()
            .filter(|e| e.metrics.answer_contains == Some(true))
            .count();

        let ratio = passed as f32 / (total as f32).max(1.0);
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
        assert_eq!(q.game, "Pandemic", "game should be loaded properly");
        let q = golden_qs.get(1).unwrap();
        assert_eq!(
            q.id, "pandemic-002",
            "second game id should be loaded properly"
        );
        assert_eq!(q.game, "Pandemic", "second game should be loaded properly");
    }
}
