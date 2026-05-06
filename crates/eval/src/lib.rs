use pipeline::Pipeline;
use rag_core::Answer;
use serde::Deserialize;
use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
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
    #[error("failed to parse eval file at {path}")]
    ParseFile {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

pub async fn get_golden_set(path: &Path) -> Result<Vec<EvalExample>, EvalError> {
    let text = read_to_string(path).map_err(|e| EvalError::ReadFile {
        path: path.to_path_buf(),
        source: e,
    })?;
    let lines = text.lines();
    let mut examples: Vec<EvalExample> = Vec::new();
    for line in lines {
        let example =
            serde_json::from_str::<EvalExample>(line).map_err(|e| EvalError::ParseFile {
                path: path.to_path_buf(),
                source: e,
            })?;
        examples.push(example);
    }
    Ok(examples)
}

pub fn answer_contains(example: &EvalExample, answer: &str) -> bool {
    for answer_should_contain in &example.expected_answer_contains {
        if !answer.contains(answer_should_contain) {
            return false;
        }
    }
    true
}

pub struct SingleEval {
    pub example: EvalExample,
    pub answer: Answer,
    pub correct: bool,
}

pub struct Evaluation {
    pub evals: Vec<SingleEval>,
    pub ratio: f32,
}

pub struct Evaluator {
    pipeline: Pipeline,
}

impl Evaluator {
    pub fn new(pipeline: Pipeline) -> Self {
        Self { pipeline }
    }

    pub async fn run(&self) -> Result<Evaluation, EvalError> {
        let examples = get_golden_set(Path::new("./data/eval/golden.jsonl")).await?;
        let mut evals: Vec<SingleEval> = Vec::new();

        for example in examples {
            let answer = self.pipeline.ask(&example.question).await.unwrap();
            let correct = answer_contains(&example, &answer.text);

            evals.push(SingleEval {
                example,
                answer,
                correct,
            });
        }

        let correct = evals.iter().filter(|e| e.correct).count();

        let ratio = correct as f32 / (evals.len() as f32);
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
        .await
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
