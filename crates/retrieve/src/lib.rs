#![allow(async_fn_in_trait)]
use embed::OllamaEmbedder;
use rag_core::{Embedder, QueryOptions, RetrievalResult, Retriever, VectorStore};
use store::LanceStore;

#[derive(thiserror::Error, Debug)]
pub enum RetrieveError {
    #[error("embedding failed")]
    Embed(#[from] embed::EmbedError),
    #[error("retrieval failed")]
    Store(#[from] store::StoreError),
}

pub struct FixedChunkRetriever {
    store: LanceStore,
    embedder: OllamaEmbedder,
}

impl FixedChunkRetriever {
    pub fn new(store: LanceStore, embedder: OllamaEmbedder) -> Self {
        Self { store, embedder }
    }
}

impl Retriever for FixedChunkRetriever {
    type Error = RetrieveError;
    async fn retrieve(
        &self,
        question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        let results = self
            .store
            .query(&self.embedder.generate_one(question).await?, options)
            .await?;

        Ok(results)
    }
}
