use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand};
use embed::OllamaEmbedder;
use eval::{
    FullEvaluation, FullOutcome, GenerationMetrics, PipelineEvaluator, RetrievalEvaluator,
    RetrievalMetrics, RetrievalOutcome, RetrievalRatios, RoutingRatios,
};
use generate::OllamaGenerator;
use ingest::ParagraphChunker;
use ingest::manifest::DocMeta;
use ingest::{Chunker as _, FixedSizeChunker, manifest::read_manifest};
use pipeline::{FullContextPipeline, NaivePipeline};
use rag_core::{
    Chunk, Embedder as _, GameClassifier, Generator as _, Pipeline, QueryOptions, Reranker as _,
    Rewriter as _, Store as _,
};
use rerank::HttpReranker;
use retrieve::{DenseRetriever, HybridRetriever, MultiQueryRetriever, Retriever, SparseRetriever};
use rewrite::OllamaRewriter;
use route::OllamaGameClassifier;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use store::LanceStore;
use tracing::{info, instrument};
use tracing_subscriber::fmt::format::FmtSpan;

/// Noisy third-party crates kept at `warn` regardless of `-v`. Adding to this
/// list is cheaper than adding `RUST_LOG=…=warn` to every invocation. If a new
/// dependency starts flooding the log, append it here.
const NOISY_LIBS: &str = concat!(
    "lance=warn,lancedb=warn,lance_core=warn,lance_table=warn,lance_index=warn,",
    "lance_io=warn,lance_encoding=warn,lance_file=warn,lance_arrow=warn,",
    "lance_datafusion=warn,datafusion=warn,datafusion_common=warn,",
    "datafusion_execution=warn,datafusion_expr=warn,datafusion_optimizer=warn,",
    "datafusion_physical_expr=warn,datafusion_physical_plan=warn,",
    "datafusion_sql=warn,datafusion_functions=warn,arrow=warn,arrow_array=warn,",
    "arrow_buffer=warn,arrow_data=warn,arrow_schema=warn,arrow_select=warn,",
    "arrow_cast=warn,arrow_ipc=warn,arrow_string=warn,object_store=warn,",
    "hyper=warn,hyper_util=warn,reqwest=warn,h2=warn,rustls=warn,tower=warn,",
    "tokio_util=warn,mio=warn",
);

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
    Ask {
        question: String,

        /// Game name.
        #[arg(short, long)]
        game: Option<String>,
    },
    /// Run the chatbot eval.
    Eval {
        #[arg(short, long)]
        retrieval_only: bool,

        #[arg(long, value_enum, default_value_t = CliFilterMode::Oracle)]
        filter_mode: CliFilterMode,

        #[arg(long, value_enum, default_value_t = RetrieverKind::Hybrid)]
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
    #[value(alias = "h")]
    Hybrid,
    #[value(alias = "d")]
    Dense,
    #[value(alias = "s")]
    Sparse,
    #[value(alias = "m")]
    MultiQuery,
}

#[derive(Copy, Clone, clap::ValueEnum)]
enum PipelineOption {
    Naive,
    FullContext,
}

#[derive(Copy, Clone, clap::ValueEnum)]
enum CliFilterMode {
    Classifier,
    Oracle,
    None,
}

impl From<CliFilterMode> for eval::FilterMode {
    fn from(mode: CliFilterMode) -> Self {
        match mode {
            CliFilterMode::Classifier => Self::Classifier,
            CliFilterMode::Oracle => Self::Oracle,
            CliFilterMode::None => Self::None,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let default_level = match cli.verbose {
        0 => format!("warn,{NOISY_LIBS}"),
        1 => format!("info,{NOISY_LIBS}"),
        2 => format!("debug,{NOISY_LIBS}"),
        _ => format!("trace,{NOISY_LIBS}"),
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    let table_name = match cli.chunker {
        Chunker::Fixed51264 => "chunks_fixed_512_64",
        Chunker::Paragraph => "chunks_paragraph",
    };

    let embedder = OllamaEmbedder::new();
    let store = LanceStore::connect(Path::new("./data/lancedb"), table_name).await?;

    match cli.command {
        Command::Ingest { paths, game } => {
            run_ingest(cli.chunker, embedder, store, paths, game).await
        }
        Command::Ask { question, game } => run_ask(embedder, store, question, game).await,
        Command::Eval {
            retrieval_only,
            filter_mode,
            retriever: retriever_kind,
            pipeline,
            only,
            limit,
        } => {
            if matches!(pipeline, PipelineOption::FullContext)
                && (!matches!(filter_mode, CliFilterMode::None) || retrieval_only)
            {
                return Err(anyhow!(
                    "Incompatible options. --pipeline full-context must be used with --filter-mode oracle and cannot be used with --retrieval-only."
                ));
            }
            let filter_mode: eval::FilterMode = filter_mode.into();
            let classifier = OllamaGameClassifier::new();
            let games = store.games().await?;
            let retriever = match retriever_kind {
                RetrieverKind::Hybrid => Retriever::Hybrid(HybridRetriever::new(store, embedder)),
                RetrieverKind::Dense => Retriever::Dense(DenseRetriever::new(store, embedder)),
                RetrieverKind::Sparse => Retriever::Sparse(SparseRetriever::new(store)),
                RetrieverKind::MultiQuery => {
                    let rewriter = OllamaRewriter::new();
                    let reranker = HttpReranker::new()?;
                    Retriever::MultiQuery(MultiQueryRetriever::new(
                        store, embedder, rewriter, reranker,
                    ))
                }
            };
            if retrieval_only {
                run_retrieval_eval(
                    retriever,
                    classifier,
                    games,
                    filter_mode,
                    only,
                    limit,
                    cli.verbose,
                )
                .await
            } else {
                run_pipeline_eval(
                    retriever,
                    classifier,
                    pipeline,
                    games,
                    filter_mode,
                    only,
                    limit,
                )
                .await
            }
        }
    }
}

#[instrument(skip_all, fields(n_paths = paths.len(), game = game.as_deref().unwrap_or("")))]
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
    info!(n_docs = to_ingest.len(), "starting ingest");
    for doc_meta in &to_ingest {
        info!(
            game = %doc_meta.game,
            doc_type = %doc_meta.doc_type,
            file = %doc_meta.file.display(),
            "ingesting doc"
        );
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
        info!(n_chunks = chunks.len(), "doc chunked and embedded");
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

#[instrument(skip(embedder, store), fields(q_len = question.len(), game = game.as_deref().unwrap_or("")))]
async fn run_ask(
    embedder: OllamaEmbedder,
    store: LanceStore,
    question: String,
    game: Option<String>,
) -> anyhow::Result<()> {
    let game_classifier = OllamaGameClassifier::new();

    let game = if let Some(g) = game {
        Some(g)
    } else {
        let games = store.games().await?;
        let classified = game_classifier.classify(&question, &games).await?;
        info!(
            classified = classified.as_deref().unwrap_or("<none>"),
            "game classified"
        );
        classified
    };

    let rewriter = OllamaRewriter::new();
    let reranker = HttpReranker::new()?;
    let retriever = Retriever::MultiQuery(MultiQueryRetriever::new(
        store, embedder, rewriter, reranker,
    ));
    let generator = OllamaGenerator::new();
    let pipeline = NaivePipeline::new(retriever, generator);
    let answer = pipeline
        .ask(
            &question,
            &QueryOptions {
                top_k: 5,
                game_filter: game,
            },
        )
        .await?;
    println!("{}", answer.text);
    Ok(())
}

#[instrument(skip(retriever, classifier, games), fields(?filter_mode, n_tags = only.len(), limit))]
async fn run_retrieval_eval(
    retriever: Retriever,
    classifier: OllamaGameClassifier,
    games: Vec<String>,
    filter_mode: eval::FilterMode,
    only: Vec<String>,
    limit: Option<usize>,
    verbose: u8,
) -> anyhow::Result<()> {
    let evaluator = RetrievalEvaluator::new(retriever, classifier, games, filter_mode, only, limit);
    let evaluation = evaluator.run().await?;
    println!("Evals run: {}", evaluation.evals.len());
    if let Some(routing_ratios) = &evaluation.routing_ratios {
        print_routing_ratios(routing_ratios);
    }
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
                ..
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

#[instrument(skip(retriever, classifier, pipeline, games), fields(?filter_mode, n_tags = only.len(), limit))]
async fn run_pipeline_eval(
    retriever: Retriever,
    classifier: OllamaGameClassifier,
    pipeline: PipelineOption,
    games: Vec<String>,
    filter_mode: eval::FilterMode,
    only: Vec<String>,
    limit: Option<usize>,
) -> anyhow::Result<()> {
    let generator = OllamaGenerator::new();
    let evaluation = match pipeline {
        PipelineOption::Naive => {
            let pipeline = NaivePipeline::new(retriever, generator);
            let evaluator =
                PipelineEvaluator::new(pipeline, classifier, games, filter_mode, only, limit);
            evaluator.run().await?
        }
        PipelineOption::FullContext => {
            let pipeline = FullContextPipeline::new(generator);
            let evaluator =
                PipelineEvaluator::new(pipeline, classifier, games, filter_mode, only, limit);
            evaluator.run().await?
        }
    };
    print_pipeline_summary(&evaluation);
    print_pipeline_failures(&evaluation);
    Ok(())
}

fn print_routing_ratios(ratios: &RoutingRatios) {
    println!("Classifier accuracy:  {:.1}%", ratios.accuracy * 100.0);
    println!(
        "Classifier false positive rate:  {:.1}%",
        ratios.false_positive_rate * 100.0
    );
    println!("Routing latency:");
    println!("  - p50: {:.1}ms", ratios.elapsed_millis_p50);
    println!("  - p95: {:.1}ms", ratios.elapsed_millis_p95);
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
    if let Some(routing_ratios) = &evaluation.routing_ratios {
        print_routing_ratios(routing_ratios);
    }
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
            ..
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
