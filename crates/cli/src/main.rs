use anyhow::Context;
use clap::{Parser, Subcommand};
use embed::OllamaEmbedder;
use generate::OllamaGenerator;
use ingest::{Chunker as _, FixedSizeChunker};
use rag_core::{Embedder as _, Generator as _, VectorStore as _};
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
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// Game name.
        #[arg(short, long)]
        game: String,
    },
    /// Ask the chatbot a rules question.
    Ask { question: String },
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
            let chunker = FixedSizeChunker {
                size: 512,
                overlap: 64,
            };
            for path in &paths {
                let mut chunks = chunker
                    .chunk(path, &game)
                    .with_context(|| format!("chunking {}", path.display()))?;

                let to_embed: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
                let embeddings = embedder
                    .generate(&to_embed)
                    .await
                    .with_context(|| format!("embedding {}", path.display()))?;
                for (chunk, embedding) in chunks.iter_mut().zip(embeddings) {
                    chunk.embedding = Some(embedding);
                }
                store
                    .insert(&chunks)
                    .await
                    .with_context(|| format!("inserting {}", path.display()))?;
            }
            println!("{} rulebooks ingested", paths.len());
        }
        Command::Ask { question } => {
            let results = store
                .query(&embedder.generate_one(&question).await?, 2)
                .await?;

            let generator = OllamaGenerator::new();
            let answer = generator.generate(&question, &results).await?;

            println!("{}", answer);
        }
    }

    Ok(())
}
