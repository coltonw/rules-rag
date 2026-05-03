use embed::OllamaEmbedder;
use generate::OllamaGenerator;
use ingest::FixedSizeChunker;
use rag_core::{Chunker as _, Embedder as _, Generator as _, VectorStore as _};
use std::{env, path::Path};
use store::LanceStore;

#[tokio::main]
async fn main() {
    let mut args: Vec<String> = env::args().collect();
    let query: String = args
        .pop()
        .unwrap_or("In Pandemic, how do you win?".to_string());

    let chunker = FixedSizeChunker {
        size: 512,
        overlap: 64,
    };
    let mut chunks = chunker.chunk(Path::new("./data/pdfs/pandemic.txt"));

    let to_embed: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    let embedder = OllamaEmbedder::new();
    let embeddings = embedder.generate(&to_embed).await;
    for (i, embedding) in embeddings.into_iter().enumerate() {
        chunks[i].embedding = Some(embedding);
    }

    let store = LanceStore::connect(Path::new("./data/lancedb")).await;
    store.insert(&chunks).await;

    let results = store.query(&embedder.generate_one(&query).await, 2).await;

    let generator = OllamaGenerator::new();
    println!("{}", generator.generate(&query, &results).await)
}
