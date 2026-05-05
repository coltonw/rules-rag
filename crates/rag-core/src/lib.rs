#![allow(async_fn_in_trait)]
use std::path::Path;

use serde::{Deserialize, Serialize};

pub const EMBED_DIM: i32 = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub game: String,
    pub doc_type: DocType,
    pub page: Option<u32>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct RawChunk {
    pub text: String,
    pub page: Option<u32>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocType {
    Rules,
    Reference,
    Faq,
}

pub struct RetrievalResult {
    pub chunk: Chunk,
    pub score: f32,
}

pub struct Answer {
    pub text: String,
    pub retrieval: Vec<RetrievalResult>,
}

pub trait VectorStore: Sized {
    type Error: std::error::Error + Send + Sync + 'static;
    async fn connect(path: &Path) -> Result<Self, Self::Error>;
    async fn insert(&self, chunks: &[Chunk]) -> Result<(), Self::Error>;
    async fn query(&self, embedding: &[f32], k: usize)
    -> Result<Vec<RetrievalResult>, Self::Error>;
}

pub trait Embedder {
    type Error: std::error::Error + Send + Sync + 'static;
    fn new() -> Self;
    async fn generate(&self, inputs: &[impl AsRef<str>]) -> Result<Vec<Vec<f32>>, Self::Error>;
    async fn generate_one(&self, input: &str) -> Result<Vec<f32>, Self::Error>;
}

pub trait Generator: Sized {
    type Error: std::error::Error + Send + Sync + 'static;
    fn new() -> Self;
    async fn generate(
        &self,
        query: &str,
        retrieval: &[RetrievalResult],
    ) -> Result<String, Self::Error>;
}
