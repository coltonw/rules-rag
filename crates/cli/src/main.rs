use embed::OllamaEmbedder;
use ingest::FixedSizeChunker;
use rag_core::{Chunker as _, Embedder as _, VectorStore as _};
use std::path::Path;
use store::LanceStore;

#[tokio::main]
async fn main() {
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

    let results = store
        .query(&embedder.generate_one("How do you win?").await, 2)
        .await;

    for result in results {
        println!("Score: {}", result.score);
        println!("{}", result.chunk.text);
    }
}
