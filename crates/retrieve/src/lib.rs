#![allow(async_fn_in_trait)]
use embed::OllamaEmbedder;
use rag_core::{Embedder, QueryOptions, RetrievalResult, Retrieve, Store};
use store::LanceStore;

#[derive(thiserror::Error, Debug)]
pub enum RetrieveError {
    #[error("embedding failed")]
    Embed(#[from] embed::EmbedError),
    #[error("retrieval failed")]
    Store(#[from] store::StoreError),
}

pub struct DenseRetriever {
    store: LanceStore,
    embedder: OllamaEmbedder,
}

impl DenseRetriever {
    pub fn new(store: LanceStore, embedder: OllamaEmbedder) -> Self {
        Self { store, embedder }
    }
}

impl Retrieve for DenseRetriever {
    type Error = RetrieveError;
    async fn retrieve(
        &self,
        question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        let results = self
            .store
            .query_vector(&self.embedder.embed_one(question).await?, options)
            .await?;

        Ok(results)
    }
}

pub struct SparseRetriever {
    store: LanceStore,
}

impl SparseRetriever {
    pub fn new(store: LanceStore) -> Self {
        Self { store }
    }
}

impl Retrieve for SparseRetriever {
    type Error = RetrieveError;
    async fn retrieve(
        &self,
        question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        let results = self.store.query_fts(question, options).await?;

        Ok(results)
    }
}

pub enum Retriever {
    Dense(DenseRetriever),
    Sparse(SparseRetriever),
}

impl Retrieve for Retriever {
    type Error = RetrieveError;
    async fn retrieve(
        &self,
        question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        match self {
            Self::Dense(retriever) => retriever.retrieve(question, options).await,
            Self::Sparse(retriever) => retriever.retrieve(question, options).await,
        }
    }
}
