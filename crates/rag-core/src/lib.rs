#![allow(async_fn_in_trait)]
use std::path::Path;

use serde::{Deserialize, Serialize};

pub const EMBED_DIM: i32 = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub game: String,
    pub source: String, // e.g., "pandemic_rules.pdf"
    pub page: Option<u32>,
    pub embedding: Option<Vec<f32>>,
}

pub struct RetrievalResult {
    pub chunk: Chunk,
    pub score: f32,
}

pub trait VectorStore: Sized {
    type Error: std::error::Error + Send + Sync + 'static;
    async fn connect(path: &Path) -> Result<Self, Self::Error>;
    async fn insert(&self, chunks: &[Chunk]) -> Result<(), Self::Error>;
    async fn query(&self, embedding: &[f32], k: usize)
    -> Result<Vec<RetrievalResult>, Self::Error>;
}

pub trait Embedder: Sized {
    fn new() -> Self;
    async fn generate(&self, inputs: &[impl AsRef<str>]) -> Vec<Vec<f32>>;
    async fn generate_one(&self, input: &str) -> Vec<f32>;
}

pub trait Generator: Sized {
    fn new() -> Self;
    async fn generate(&self, query: &str, retrieval: &[RetrievalResult]) -> String;
}
