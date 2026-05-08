use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand};
use embed::OllamaEmbedder;
use eval::{
    GenerationMetrics, PipelineEvaluator, PipelineOutcome, RetrievalEvaluator, RetrievalMetrics,
    RetrievalOutcome,
};
use generate::OllamaGenerator;
use ingest::manifest::DocMeta;
use ingest::{Chunker as _, FixedSizeChunker, manifest::read_manifest};
use pipeline::NaivePipeline;
use rag_core::{
    Chunk, Embedder as _, Generator as _, Pipeline as _, QueryOptions, VectorStore as _,
};
use retrieve::FixedChunkRetriever;
use std::path::Path;
use std::path::PathBuf;
use store::LanceStore;

/// A board game rules chatbot
#[derive(Parser)]
#[command(name = "bgrag", version, about)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Ingest parsed rulebooks into the RAG.
    Ingest {
        /// The files to ingest.
        paths: Vec<PathBuf>,
        /// Game name.
        #[arg(short, long)]
        game: Option<String>,
    },
    /// Ask the chatbot a rules question.
    Ask { question: String },
    /// Run the chatbot eval.
    Eval {
        #[arg(short, long)]
        retrieval_only: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let default_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let embedder = OllamaEmbedder::new();
    let store = LanceStore::connect(Path::new("./data/lancedb")).await?;

    match cli.command {
        Command::Ingest { paths, game } => {
            let manifest = read_manifest(Path::new("./data/pdfs/manifest.toml"))?;
            let chunker = FixedSizeChunker {
                size: 512,
                overlap: 64,
            };
            let to_ingest: Vec<DocMeta> = if paths.is_empty() {
                manifest
            } else {
                paths
                    .into_iter()
                    .map(|path| {
                        let single_manifest = manifest.iter().find(|meta| meta.file == path);
                        let game = game
                            .as_ref()
                            .or(single_manifest.map(|m| &m.game))
                            .ok_or_else(|| {
                                anyhow!(
                                    "{} not found in the manifest and no --game parameter",
                                    path.display()
                                )
                            })?
                            .to_string();
                        let doc_type = single_manifest.map(|m| m.doc_type).ok_or_else(|| {
                            anyhow!("{} not found in the manifest", path.display())
                        })?;
                        Ok(DocMeta {
                            file: path,
                            game,
                            doc_type,
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
            };
            for doc_meta in &to_ingest {
                let raw_chunks = chunker
                    .chunk(&doc_meta.file)
                    .with_context(|| format!("chunking {}", &doc_meta.file.display()))?;

                let to_embed: Vec<&str> =
                    raw_chunks.iter().map(|chunk| chunk.text.as_str()).collect();
                let embeddings = embedder
                    .generate(&to_embed)
                    .await
                    .with_context(|| format!("embedding {}", &doc_meta.file.display()))?;
                let mut chunks: Vec<Chunk> = Vec::new();
                for (raw_chunk, embedding) in raw_chunks.into_iter().zip(embeddings) {
                    chunks.push(Chunk {
                        id: "TODO".to_string(),
                        text: raw_chunk.text,
                        game: doc_meta.game.to_string(),
                        doc_type: doc_meta.doc_type,
                        page: raw_chunk.page,
                        embedding: Some(embedding),
                    })
                }
                store
                    .insert(&chunks)
                    .await
                    .with_context(|| format!("inserting {}", &doc_meta.file.display()))?;
            }
            println!("{} rulebooks ingested", to_ingest.len());
        }
        Command::Ask { question } => {
            let retriever = FixedChunkRetriever::new(store, embedder);
            let generator = OllamaGenerator::new();
            let pipeline = NaivePipeline::new(retriever, generator);

            let answer = pipeline
                .ask(
                    &question,
                    &QueryOptions {
                        top_k: 5,
                        ..Default::default()
                    },
                )
                .await?;
            println!("{}", answer.text);
        }
        Command::Eval { retrieval_only } => {
            let retriever = FixedChunkRetriever::new(store, embedder);
            if retrieval_only {
                let evaluator = RetrievalEvaluator::new(retriever);
                let evaluation = evaluator.run().await?;
                println!("Chunk match:  {:.1}%", evaluation.chunk_ratio * 100.0);
                if evaluation.chunk_ratio < 1.0 {
                    println!("\nWrong answers:\n")
                }
                for wrong in evaluation
                    .evals
                    .iter()
                    .filter(|e| e.outcome.metrics().is_some_and(|m| !m.chunk_match))
                {
                    println!("Question:\n{}", wrong.example.question);
                    // This pattern match is unecessary because you can only get here if chunk_match was false, but very soon
                    // we will be adding more retrieval metrics and this already being set up will make that much easier
                    #[allow(clippy::collapsible_if)]
                    if let RetrievalOutcome::Ok {
                        metrics: RetrievalMetrics { chunk_match },
                    } = &wrong.outcome
                    {
                        if !chunk_match {
                            println!("Chunk failure");
                            println!("Expected chunk(s):");
                            for c in &wrong.example.expected_chunk_contains {
                                println!("  - {}", c);
                            }
                        }
                    }
                    println!();
                }
            } else {
                let generator = OllamaGenerator::new();
                let pipeline = NaivePipeline::new(retriever, generator);

                let evaluator = PipelineEvaluator::new(pipeline);
                let evaluation = evaluator.run().await?;
                println!("Quote match:  {:.1}%", evaluation.quote_ratio * 100.0);
                println!("Chunk match:  {:.1}%", evaluation.chunk_ratio * 100.0);
                println!("Refusal rate: {:.1}%", evaluation.refusal_ratio * 100.0);
                let any_failures = evaluation.quote_ratio < 1.0
                    || evaluation.chunk_ratio < 1.0
                    || evaluation.refusal_ratio > 0.0;
                if any_failures {
                    println!("\nWrong answers:\n")
                }
                for wrong in evaluation.evals.iter().filter(|e| {
                    e.outcome.metrics().is_some_and(|m| {
                        !m.retr_metrics.chunk_match
                            || !m.gen_metrics.quote_match
                            || m.gen_metrics.refused
                    })
                }) {
                    println!("Question:\n{}", wrong.example.question);
                    if let PipelineOutcome::Ok {
                        retrieval_metrics: RetrievalMetrics { chunk_match },
                        generation_metrics:
                            GenerationMetrics {
                                quote_match,
                                refused,
                            },
                        answer,
                    } = &wrong.outcome
                    {
                        if !chunk_match {
                            println!("Chunk failure");
                            println!("Expected chunk(s):");
                            for c in &wrong.example.expected_chunk_contains {
                                println!("  - {}", c);
                            }
                        } else if *refused {
                            println!("Refusal");
                        } else if !quote_match {
                            println!("Quote failure");
                            println!("Expected quote(s):");
                            for q in &wrong.example.expected_quote {
                                println!("  - {}", q);
                            }
                        }
                        println!("Answer:\n{}", answer.text);
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}
