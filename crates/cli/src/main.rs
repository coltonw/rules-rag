use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand};
use embed::OllamaEmbedder;
use eval::Evaluator;
use generate::OllamaGenerator;
use ingest::manifest::DocMeta;
use ingest::{Chunker as _, FixedSizeChunker, manifest::read_manifest};
use pipeline::NaivePipeline;
use rag_core::{Chunk, Embedder as _, Generator as _, Pipeline as _, VectorStore as _};
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
    Eval,
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
            let generator = OllamaGenerator::new();
            let pipeline = NaivePipeline::new(store, embedder, generator);

            let answer = pipeline.ask(&question).await?;
            println!("{}", answer.text);
        }
        Command::Eval => {
            let generator = OllamaGenerator::new();
            let pipeline = NaivePipeline::new(store, embedder, generator);

            let evaluator = Evaluator::new(pipeline);
            let evaluation = evaluator.run().await?;
            println!("{:.1}%", evaluation.ratio * 100.0);
            if evaluation.ratio < 1.0 {
                println!("Wrong answers:")
            }
            for wrong in evaluation
                .evals
                .iter()
                .filter(|e| e.metrics.answer_contains == Some(false))
            {
                println!("Question: {}", wrong.example.question);
                println!("Expected: {:?}", wrong.example.expected_answer_contains);
                println!("Answer: {}", wrong.answer.text);
            }
        }
    }

    Ok(())
}
