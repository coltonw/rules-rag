use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand};
use embed::OllamaEmbedder;
use eval::{
    FullEvaluation, FullOutcome, GenerationMetrics, PipelineEvaluator, RetrievalEvaluator,
    RetrievalMetrics, RetrievalOutcome, RetrievalRatios,
};
use generate::OllamaGenerator;
use ingest::ParagraphChunker;
use ingest::manifest::DocMeta;
use ingest::{Chunker as _, FixedSizeChunker, manifest::read_manifest};
use pipeline::{FullContextPipeline, NaivePipeline};
use rag_core::{Chunk, Embedder as _, Generator as _, Pipeline, QueryOptions, Store as _};
use retrieve::{DenseRetriever, Retriever, SparseRetriever};
use std::collections::HashMap;
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

    #[arg(short = 'c', long, value_enum, global = true, default_value_t = Chunker::Paragraph)]
    chunker: Chunker,

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
        /// Disable the per-question game metadata filter so retrieval runs across
        /// all games. Measures cross-game disambiguation pressure.
        #[arg(long)]
        no_game_filter: bool,

        #[arg(long, value_enum, default_value_t = RetrieverKind::Dense)]
        retriever: RetrieverKind,

        #[arg(short = 'p', long, value_enum, default_value_t = PipelineOption::Naive)]
        pipeline: PipelineOption,

        /// Only run evals that match one of the comma separated tags
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,

        /// Do a limited number for smaller tests
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[derive(Copy, Clone, clap::ValueEnum)]
enum Chunker {
    #[value(name = "fixed-512-64", alias = "f")]
    Fixed51264,
    #[value(alias = "p")]
    Paragraph,
}

#[derive(Copy, Clone, clap::ValueEnum)]
enum RetrieverKind {
    #[value(alias = "d")]
    Dense,
    #[value(alias = "s")]
    Sparse,
}

#[derive(Copy, Clone, clap::ValueEnum)]
enum PipelineOption {
    Naive,
    FullContext,
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

    let table_name = match cli.chunker {
        Chunker::Fixed51264 => "chunks_fixed_512_64", // default
        Chunker::Paragraph => "chunks_paragraph",
    };

    let embedder = OllamaEmbedder::new();
    let store = LanceStore::connect(Path::new("./data/lancedb"), table_name).await?;

    match cli.command {
        Command::Ingest { paths, game } => {
            run_ingest(cli.chunker, embedder, store, paths, game).await
        }
        Command::Ask { question } => run_ask(embedder, store, question).await,
        Command::Eval {
            retrieval_only,
            no_game_filter,
            retriever: retriever_kind,
            pipeline,
            only,
            limit,
        } => {
            if matches!(pipeline, PipelineOption::FullContext) && (no_game_filter || retrieval_only)
            {
                return Err(anyhow!(
                    "Incompatible options. --pipeline full-context cannot be used with --no-game-filter or --retrieval-only."
                ));
            }
            let apply_game_filter = !no_game_filter;
            let retriever = match retriever_kind {
                RetrieverKind::Dense => Retriever::Dense(DenseRetriever::new(store, embedder)),
                RetrieverKind::Sparse => Retriever::Sparse(SparseRetriever::new(store)),
            };
            if retrieval_only {
                run_retrieval_eval(retriever, apply_game_filter, only, limit, cli.verbose).await
            } else {
                run_pipeline_eval(retriever, pipeline, apply_game_filter, only, limit).await
            }
        }
    }
}

async fn run_ingest(
    chunker_choice: Chunker,
    embedder: OllamaEmbedder,
    store: LanceStore,
    paths: Vec<PathBuf>,
    game: Option<String>,
) -> anyhow::Result<()> {
    let manifest = read_manifest(Path::new("./data/pdfs/manifest.toml"))?;
    let to_ingest: Vec<DocMeta> = if paths.is_empty() {
        manifest
    } else {
        paths
            .into_iter()
            .map(|path| {
                let single_manifest = manifest.iter().find(|meta| meta.file == path);
                let game = game
                    .as_ref()
                    .or_else(|| single_manifest.map(|m| &m.game))
                    .ok_or_else(|| {
                        anyhow!(
                            "{} not found in the manifest and no --game parameter",
                            path.display()
                        )
                    })?
                    .clone();
                let doc_type = single_manifest
                    .map(|m| m.doc_type)
                    .ok_or_else(|| anyhow!("{} not found in the manifest", path.display()))?;
                Ok(DocMeta {
                    file: path,
                    game,
                    doc_type,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    for doc_meta in &to_ingest {
        let raw_chunks = match chunker_choice {
            Chunker::Fixed51264 => {
                let chunker = FixedSizeChunker {
                    size: 512,
                    overlap: 64,
                };
                chunker
                    .chunk(&doc_meta.file)
                    .with_context(|| format!("chunking {}", &doc_meta.file.display()))?
            }
            Chunker::Paragraph => {
                let chunker = ParagraphChunker {
                    min_size: 150,
                    target_size: 350,
                    max_size: 500,
                };
                chunker
                    .chunk(&doc_meta.file)
                    .with_context(|| format!("chunking {}", &doc_meta.file.display()))?
            }
        };

        let to_embed: Vec<&str> = raw_chunks.iter().map(|chunk| chunk.text.as_str()).collect();
        let embeddings = embedder
            .embed(&to_embed)
            .await
            .with_context(|| format!("embedding {}", &doc_meta.file.display()))?;
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut page_counters: HashMap<Option<u32>, u32> = HashMap::new();
        for (raw_chunk, embedding) in raw_chunks.into_iter().zip(embeddings) {
            let counter = page_counters.entry(raw_chunk.page).or_insert(0);
            let page_str = raw_chunk
                .page
                .map_or_else(|| "none".to_string(), |p| p.to_string());
            let id = format!(
                "{}-{}-{}-{}",
                doc_meta.game, doc_meta.doc_type, page_str, *counter
            );
            *counter += 1;
            chunks.push(Chunk {
                id,
                text: raw_chunk.text,
                game: doc_meta.game.clone(),
                doc_type: doc_meta.doc_type,
                page: raw_chunk.page,
                embedding: Some(embedding),
            });
        }
        store
            .insert(&chunks)
            .await
            .with_context(|| format!("inserting {}", &doc_meta.file.display()))?;
    }
    store
        .update_indices()
        .await
        .with_context(|| "updating indices")?;
    println!("{} rulebooks ingested", to_ingest.len());
    Ok(())
}

async fn run_ask(
    embedder: OllamaEmbedder,
    store: LanceStore,
    question: String,
) -> anyhow::Result<()> {
    let retriever = Retriever::Dense(DenseRetriever::new(store, embedder));
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
    Ok(())
}

async fn run_retrieval_eval(
    retriever: Retriever,
    apply_game_filter: bool,
    only: Vec<String>,
    limit: Option<usize>,
    verbose: u8,
) -> anyhow::Result<()> {
    let evaluator = RetrievalEvaluator::new(retriever, apply_game_filter, only, limit);
    let evaluation = evaluator.run().await?;
    println!("Evals run: {}", evaluation.evals.len());
    print_retrieval_ratios(&evaluation.ratios);
    if verbose > 0 {
        if evaluation.ratios.recall_at_1 < 1.0 {
            println!("\nMissed Recall@1:\n");
        }
        for wrong in evaluation
            .evals
            .iter()
            .filter(|e| e.outcome.metrics().is_some_and(|m| !m.recall_at_1))
        {
            println!("ID: {}", wrong.example.id);
            println!("Question:\n{}", wrong.example.question);
            // This pattern match is unecessary because you can only get here if chunk_match was false, but very soon
            // we will be adding more retrieval metrics and this already being set up will make that much easier
            #[allow(clippy::collapsible_if)]
            if let RetrievalOutcome::Ok {
                retrieval,
                metrics:
                    RetrievalMetrics {
                        recall_at_3,
                        recall_at_5,
                        recall_at_10,
                        found_at,
                        ..
                    },
            } = &wrong.outcome
            {
                if !recall_at_10 {
                    println!("Chunk(s) failed Recall@10");
                } else if !recall_at_5 {
                    println!("Chunk(s) passed Recall@10 but failed Recall@5");
                } else if !recall_at_3 {
                    println!("Chunk(s) passed Recall@5 but failed Recall@3");
                } else {
                    println!("Chunk(s) passed Recall@3 but failed Recall@1");
                }
                println!("Expected chunk(s):");
                for c in &wrong.example.expected_chunk_contains {
                    println!("  - {c}");
                }
                if verbose > 1 {
                    println!("Actual failed chunks:\n");
                    let to_take = if *found_at > 0 {
                        *found_at - 1
                    } else {
                        retrieval.len()
                    };
                    for rr in retrieval.iter().take(to_take) {
                        println!(
                            "Failed chunk {}:\n{}\n",
                            rr.chunk.id,
                            rr.chunk.text.replace("\n\n", "\n").trim()
                        );
                    }
                }
            }
            println!();
        }
    }
    Ok(())
}

async fn run_pipeline_eval(
    retriever: Retriever,
    pipeline: PipelineOption,
    apply_game_filter: bool,
    only: Vec<String>,
    limit: Option<usize>,
) -> anyhow::Result<()> {
    let generator = OllamaGenerator::new();
    let evaluation = match pipeline {
        PipelineOption::Naive => {
            let pipeline = NaivePipeline::new(retriever, generator);
            let evaluator = PipelineEvaluator::new(pipeline, apply_game_filter, only, limit);
            evaluator.run().await?
        }
        PipelineOption::FullContext => {
            let pipeline = FullContextPipeline::new(generator);
            let evaluator = PipelineEvaluator::new(pipeline, apply_game_filter, only, limit);
            evaluator.run().await?
        }
    };
    print_pipeline_summary(&evaluation);
    print_pipeline_failures(&evaluation);
    Ok(())
}

fn print_retrieval_ratios(ratios: &RetrievalRatios) {
    println!("Recall@1 match:  {:.1}%", ratios.recall_at_1 * 100.0);
    println!("Recall@3 match:  {:.1}%", ratios.recall_at_3 * 100.0);
    println!("Recall@5 match:  {:.1}%", ratios.recall_at_5 * 100.0);
    println!("Recall@10 match:  {:.1}%", ratios.recall_at_10 * 100.0);
    println!("MRR mean:  {:.3}", ratios.mrr_mean);
    println!("Retrieval latency:");
    println!("  - p50: {:.1}ms", ratios.elapsed_millis_p50);
    println!("  - p95: {:.1}ms", ratios.elapsed_millis_p95);
}

fn print_pipeline_summary(evaluation: &FullEvaluation) {
    println!("Evals run: {}", evaluation.evals.len());
    print_retrieval_ratios(&evaluation.retrieval_ratios);
    let gen_ratios = &evaluation.generation_ratios;
    println!("Quote match:  {:.1}%", gen_ratios.quote * 100.0);
    println!("Refusal rate: {:.1}%", gen_ratios.refusal * 100.0);
    println!("Total latency:");
    println!("  - p50: {:.1}ms", gen_ratios.total_elapsed_millis_p50);
    println!("  - p95: {:.1}ms", gen_ratios.total_elapsed_millis_p95);
    println!("Input tokens (approx):");
    println!("  - p50: {}", gen_ratios.input_tokens_p50);
    println!("  - p95: {}", gen_ratios.input_tokens_p95);
    println!("Output tokens (approx):");
    println!("  - p50: {}", gen_ratios.output_tokens_p50);
    println!("  - p95: {}", gen_ratios.output_tokens_p95);
}

fn print_pipeline_failures(evaluation: &FullEvaluation) {
    let any_failures = evaluation.retrieval_ratios.recall_at_1 < 1.0
        || evaluation.generation_ratios.quote < 1.0
        || evaluation.generation_ratios.refusal > 0.0;
    if any_failures {
        println!("\nWrong answers:\n");
    }
    for wrong in evaluation.evals.iter().filter(|e| {
        e.outcome.metrics().is_some_and(|m| {
            !m.retr_metrics.recall_at_1 || !m.gen_metrics.quote_match || m.gen_metrics.refused
        })
    }) {
        println!("ID: {}", wrong.example.id);
        println!("Question:\n{}", wrong.example.question);
        if let FullOutcome::Ok {
            retrieval_metrics:
                RetrievalMetrics {
                    recall_at_3,
                    recall_at_5,
                    ..
                },
            generation_metrics:
                GenerationMetrics {
                    quote_match,
                    refused,
                    ..
                },
            answer,
        } = &wrong.outcome
        {
            if !recall_at_5 {
                println!("Chunk not found");
                println!("Expected chunk(s):");
                for c in &wrong.example.expected_chunk_contains {
                    println!("  - {c}");
                }
            } else if *refused {
                println!("Refusal");
            } else if !quote_match {
                println!("Quote failure");
                println!("Expected quote(s):");
                for q in &wrong.example.expected_quote {
                    println!("  - {q}");
                }
            } else {
                if *recall_at_3 {
                    println!("Recall@1 failed");
                } else {
                    println!("Recall@3 failed");
                }
                println!("Expected chunk(s):");
                for c in &wrong.example.expected_chunk_contains {
                    println!("  - {c}");
                }
            }
            println!("Answer:\n{}", answer.text);
        }
        println!();
    }
}
