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

pub trait Chunker {
    fn chunk(&self, text_path: &Path) -> Vec<Chunk>;
}

pub struct RetrievalResult {
    pub chunk: Chunk,
    pub score: f32,
}

pub trait VectorStore {
    async fn connect(path: &Path) -> Self;
    async fn insert(&self, chunks: &[Chunk]);
    async fn query(&self, embedding: &[f32], k: usize) -> Vec<RetrievalResult>;
}

pub trait Embedder {
    fn new() -> Self;
    async fn generate(&self, inputs: &[impl AsRef<str>]) -> Vec<Vec<f32>>;
    async fn generate_one(&self, input: &str) -> Vec<f32>;
}

pub trait Generator {
    fn new() -> Self;
    async fn generate(&self, query: &str, retrieval: &[RetrievalResult]) -> String;
}
