#![allow(async_fn_in_trait)]
use std::path::{Path, PathBuf};

use generate::OllamaGenerator;
use ingest::manifest::read_manifest;
use rag_core::{Chunk, Generator, Pipeline, QueryOptions, RetrievalResult, Retrieve as _};
use retrieve::Retriever;
use std::fs::read_to_string;
use tracing::{debug, info, instrument};

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
    retriever: Retriever,
    generator: OllamaGenerator,
}

impl NaivePipeline {
    pub const fn new(retriever: Retriever, generator: OllamaGenerator) -> Self {
        Self {
            retriever,
            generator,
        }
    }
}

impl Pipeline for NaivePipeline {
    type Error = PipelineError;
    #[instrument(
        level = "info",
        name = "naive_retrieve",
        skip_all,
        fields(
            q_len = question.len(),
            top_k = options.top_k,
            game = options.game_filter.as_deref().unwrap_or(""),
        ),
    )]
    async fn retrieve(
        &self,
        question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, PipelineError> {
        let results = self.retriever.retrieve(question, options).await?;

        info!(n_results = results.len(), "retrieved");
        Ok(results)
    }

    #[instrument(
        level = "info",
        name = "naive_ask_with",
        skip_all,
        fields(q_len = question.len(), n_results = results.len()),
    )]
    async fn ask_with(
        &self,
        question: &str,
        results: &[RetrievalResult],
    ) -> Result<String, PipelineError> {
        let answer = self.generator.generate(question, results).await?;

        info!(answer_len = answer.len(), "generated");
        Ok(answer)
    }
}

pub struct FullContextPipeline {
    generator: OllamaGenerator,
}

impl FullContextPipeline {
    pub const fn new(generator: OllamaGenerator) -> Self {
        Self { generator }
    }
}

impl Pipeline for FullContextPipeline {
    type Error = PipelineError;
    #[instrument(
        level = "info",
        name = "full_context_retrieve",
        skip_all,
        fields(game = options.game_filter.as_deref().unwrap_or("")),
    )]
    async fn retrieve(
        &self,
        _question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, PipelineError> {
        let manifest = read_manifest(Path::new("./data/pdfs/manifest.toml"))?;
        let Some(metadata) = options
            .game_filter
            .as_ref()
            .and_then(|game| manifest.into_iter().find(|manifest| manifest.game == *game))
        else {
            panic!("FullContextPipeline requires game filter matching a game in the manifest")
        };

        let text = read_to_string(&metadata.file).map_err(|e| PipelineError::ReadFile {
            path: metadata.file.clone(),
            source: e,
        })?;

        debug!(file = %metadata.file.display(), text_len = text.len(), "loaded full context");
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

    #[instrument(
        level = "info",
        name = "full_context_ask_with",
        skip_all,
        fields(q_len = question.len(), n_results = results.len()),
    )]
    async fn ask_with(
        &self,
        question: &str,
        results: &[RetrievalResult],
    ) -> Result<String, PipelineError> {
        let answer = self.generator.generate(question, results).await?;

        info!(answer_len = answer.len(), "generated");
        Ok(answer)
    }
}
