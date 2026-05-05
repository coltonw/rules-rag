use serde::Deserialize;
use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
pub struct EvalQuestion {
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

pub async fn get_golden_questions(path: &Path) -> Result<Vec<EvalQuestion>, EvalError> {
    let text = read_to_string(path).map_err(|e| EvalError::ReadFile {
        path: path.to_path_buf(),
        source: e,
    })?;
    let lines = text.lines();
    let mut questions: Vec<EvalQuestion> = Vec::new();
    for line in lines {
        let q = serde_json::from_str::<EvalQuestion>(line).map_err(|e| EvalError::ParseFile {
            path: path.to_path_buf(),
            source: e,
        })?;
        questions.push(q);
    }
    Ok(questions)
}
