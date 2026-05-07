#![allow(async_fn_in_trait)]
use embed::OllamaEmbedder;
use generate::OllamaGenerator;
use rag_core::{Answer, Embedder, Generator, Pipeline, RetrievalResult, VectorStore};
use store::LanceStore;

#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("embedding failed")]
    Embed(#[from] embed::EmbedError),
    #[error("retrieval failed")]
    Store(#[from] store::StoreError),
    #[error("generation failed")]
    Generate(#[from] generate::GenerateError),
}

pub struct NaivePipeline {
    store: LanceStore,
    embedder: OllamaEmbedder,
    generator: OllamaGenerator,
}

impl NaivePipeline {
    pub fn new(store: LanceStore, embedder: OllamaEmbedder, generator: OllamaGenerator) -> Self {
        Self {
            store,
            embedder,
            generator,
        }
    }
}

impl Pipeline for NaivePipeline {
    type Error = PipelineError;
    async fn retrieve(&self, question: &str) -> Result<Vec<RetrievalResult>, PipelineError> {
        let results = self
            .store
            .query(&self.embedder.generate_one(question).await?, 5)
            .await?;

        Ok(results)
    }

    async fn ask(&self, question: &str) -> Result<Answer, PipelineError> {
        let results = self.retrieve(question).await?;
        let answer = self.generator.generate(question, &results).await?;

        Ok(Answer {
            text: answer,
            retrieval: results,
        })
    }
}
