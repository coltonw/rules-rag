#![allow(async_fn_in_trait)]
use std::path::{Path, PathBuf};

use generate::OllamaGenerator;
use ingest::manifest::read_manifest;
use rag_core::{Chunk, Generator, Pipeline, QueryOptions, RetrievalResult, Retriever as _};
use retrieve::FixedChunkRetriever;
use std::fs::read_to_string;

#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("retrieval failed")]
    Embed(#[from] retrieve::RetrieveError),
    #[error("generation failed")]
    Generate(#[from] generate::GenerateError),
    #[error("loading manifest failed")]
    Manifest(#[from] ingest::IngestError),
    #[error("failed to read text file at {path}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
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

    async fn ask_with(
        &self,
        question: &str,
        results: &[RetrievalResult],
    ) -> Result<String, PipelineError> {
        let answer = self.generator.generate(question, results).await?;

        Ok(answer)
    }
}

pub struct FullContextPipeline {
    generator: OllamaGenerator,
}

impl FullContextPipeline {
    pub fn new(generator: OllamaGenerator) -> Self {
        Self { generator }
    }
}

impl Pipeline for FullContextPipeline {
    type Error = PipelineError;
    async fn retrieve(
        &self,
        _question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, PipelineError> {
        let manifest = read_manifest(Path::new("./data/pdfs/manifest.toml"))?;
        let metadata = match options
            .game_filter
            .as_ref()
            .and_then(|game| manifest.into_iter().find(|manifest| manifest.game == *game))
        {
            Some(metadata) => metadata,
            None => {
                panic!("FullContextPipeline requires game filter matching a game in the manifest")
            }
        };

        let text = read_to_string(&metadata.file).map_err(|e| PipelineError::ReadFile {
            path: metadata.file.clone(),
            source: e,
        })?;

        Ok(vec![RetrievalResult {
            chunk: Chunk {
                id: metadata.game.clone(),
                text,
                game: metadata.game.clone(),
                doc_type: metadata.doc_type,
                page: None,
                embedding: None,
            },
            score: 1.0,
        }])
    }

    async fn ask_with(
        &self,
        question: &str,
        results: &[RetrievalResult],
    ) -> Result<String, PipelineError> {
        let answer = self.generator.generate(question, results).await?;

        Ok(answer)
    }
}
