#![allow(async_fn_in_trait)]
use generate::OllamaGenerator;
use rag_core::{Answer, Generator, Pipeline, QueryOptions, RetrievalResult, Retriever as _};
use retrieve::FixedChunkRetriever;

#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("retrieval failed")]
    Embed(#[from] retrieve::RetrieveError),
    #[error("generation failed")]
    Generate(#[from] generate::GenerateError),
}

pub struct NaivePipeline {
    retriever: FixedChunkRetriever,
    generator: OllamaGenerator,
}

impl NaivePipeline {
    pub fn new(retriever: FixedChunkRetriever, generator: OllamaGenerator) -> Self {
        Self {
            retriever,
            generator,
        }
    }
}

impl Pipeline for NaivePipeline {
    type Error = PipelineError;
    async fn retrieve(
        &self,
        question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, PipelineError> {
        let results = self.retriever.retrieve(question, options).await?;

        Ok(results)
    }

    async fn ask(&self, question: &str, options: &QueryOptions) -> Result<Answer, PipelineError> {
        let results = self.retrieve(question, options).await?;
        let answer = self.generator.generate(question, &results).await?;

        Ok(Answer {
            text: answer,
            retrieval: results,
        })
    }
}
