#![allow(async_fn_in_trait)]
use std::{collections::HashMap, iter};

use embed::OllamaEmbedder;
use futures::future::try_join_all;
use rag_core::{Embedder, QueryOptions, RetrievalResult, Retrieve, Rewriter, Store};
use rewrite::OllamaRewriter;
use store::LanceStore;
use tracing::{debug, instrument};

#[derive(thiserror::Error, Debug)]
pub enum RetrieveError {
    #[error("embedding failed")]
    Embed(#[from] embed::EmbedError),
    #[error("retrieval failed")]
    Store(#[from] store::StoreError),
    #[error("rewrite failed")]
    Rewrite(#[from] rewrite::RewriteError),
}

pub struct DenseRetriever {
    store: LanceStore,
    embedder: OllamaEmbedder,
}

impl DenseRetriever {
    pub const fn new(store: LanceStore, embedder: OllamaEmbedder) -> Self {
        Self { store, embedder }
    }
}

impl Retrieve for DenseRetriever {
    type Error = RetrieveError;
    #[instrument(
        level = "debug",
        name = "dense_retrieve",
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
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        let results = self
            .store
            .query_vector(&self.embedder.embed_one(question).await?, options)
            .await?;

        debug!(n_results = results.len(), "dense retrieve done");
        Ok(results)
    }
}

pub struct SparseRetriever {
    store: LanceStore,
}

impl SparseRetriever {
    pub const fn new(store: LanceStore) -> Self {
        Self { store }
    }
}

impl Retrieve for SparseRetriever {
    type Error = RetrieveError;
    #[instrument(
        level = "debug",
        name = "sparse_retrieve",
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
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        let results = self.store.query_fts(question, options).await?;

        debug!(n_results = results.len(), "sparse retrieve done");
        Ok(results)
    }
}

pub struct HybridRetriever {
    store: LanceStore,
    embedder: OllamaEmbedder,
}

impl HybridRetriever {
    pub const fn new(store: LanceStore, embedder: OllamaEmbedder) -> Self {
        Self { store, embedder }
    }
}

impl Retrieve for HybridRetriever {
    type Error = RetrieveError;
    #[instrument(
        level = "debug",
        name = "hybrid_retrieve",
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
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        hybrid_one(&self.store, &self.embedder, question, options).await
    }
}

pub struct MultiQueryRetriever {
    store: LanceStore,
    embedder: OllamaEmbedder,
    rewriter: OllamaRewriter,
}

impl MultiQueryRetriever {
    pub const fn new(
        store: LanceStore,
        embedder: OllamaEmbedder,
        rewriter: OllamaRewriter,
    ) -> Self {
        Self {
            store,
            embedder,
            rewriter,
        }
    }
}

impl Retrieve for MultiQueryRetriever {
    type Error = RetrieveError;
    #[instrument(
        level = "debug",
        name = "multi-query_retrieve",
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
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        let rewrites = self.rewriter.rewrite(question).await?;
        debug!(n_rewrites = rewrites.len(), "rewrites generated");

        let fanout = 3;
        let inner_take = options.top_k * fanout;

        let queries = iter::once(question)
            .chain(rewrites.iter().map(String::as_str))
            .enumerate()
            .map(
                async |(idx, q)| -> Result<Vec<RetrievalResult>, RetrieveError> {
                    debug!(idx, q_len = q.len(), "querying rewrite");
                    hybrid_one(
                        &self.store,
                        &self.embedder,
                        q,
                        &options.with_top_k(inner_take),
                    )
                    .await
                },
            );
        let results: Vec<Vec<RetrievalResult>> = try_join_all(queries).await?;
        let results: Vec<RetrievalResult> = rrf(results).into_iter().take(options.top_k).collect();

        debug!(n_results = results.len(), "multi-query retrieve done");
        Ok(results)
    }
}

pub enum Retriever {
    Dense(DenseRetriever),
    Sparse(SparseRetriever),
    Hybrid(HybridRetriever),
    MultiQuery(MultiQueryRetriever),
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
            Self::Hybrid(retriever) => retriever.retrieve(question, options).await,
            Self::MultiQuery(retriever) => retriever.retrieve(question, options).await,
        }
    }
}

/// Fuses vectors with Reciprocal Rank Fusion (`score = Σ 1/(k + rank_i)`, k≈60)
fn rrf(results_list: Vec<Vec<RetrievalResult>>) -> Vec<RetrievalResult> {
    let mut results_map: HashMap<String, RetrievalResult> = HashMap::new();

    // This should technically be a constant at the top of the file,
    // but I am not sure I will ever dig in enough to care so for now leaving it here
    let rrf_k: f32 = 60.0;
    for results in results_list {
        for (index, mut result) in results.into_iter().enumerate() {
            let score = 1.0 / (rrf_k + index as f32);
            if let Some(result) = results_map.get_mut(&result.chunk.id) {
                result.score += score;
            } else {
                result.score = score;
                results_map.insert(result.chunk.id.clone(), result);
            }
        }
    }

    let mut results: Vec<RetrievalResult> = results_map.into_values().collect();

    results.sort_by(|a, b| b.score.total_cmp(&a.score));
    results
}

#[instrument(
    level = "debug",
    name = "hybrid_one",
    skip_all,
    fields(q_len = question.len(), take_n),
)]
async fn hybrid_one(
    store: &LanceStore,
    embedder: &OllamaEmbedder,
    question: &str,
    options: &QueryOptions,
) -> Result<Vec<RetrievalResult>, RetrieveError> {
    let embedding = embedder.embed_one(question).await?;
    let vector_query = store.query_vector(&embedding, options);
    let fts_query = store.query_fts(question, options);
    let (vector_results, fts_results) = tokio::try_join!(vector_query, fts_query)?;

    let n_vector = vector_results.len();
    let n_fts = fts_results.len();
    let results: Vec<RetrievalResult> = rrf(vec![vector_results, fts_results])
        .into_iter()
        .take(options.top_k)
        .collect();

    debug!(
        n_vector,
        n_fts,
        n_results = results.len(),
        "hybrid_one done"
    );
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_core::{Chunk, DocType};

    const K: f32 = 60.0;
    const EPS: f32 = 1e-6;

    fn result(id: &str) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk {
                id: id.to_string(),
                text: String::new(),
                game: String::new(),
                doc_type: DocType::Rules,
                page: None,
                embedding: None,
            },
            score: 0.0,
        }
    }

    fn ids(results: &[RetrievalResult]) -> Vec<&str> {
        results.iter().map(|r| r.chunk.id.as_str()).collect()
    }

    fn find<'a>(results: &'a [RetrievalResult], id: &str) -> &'a RetrievalResult {
        results
            .iter()
            .find(|r| r.chunk.id == id)
            .unwrap_or_else(|| panic!("missing id {id}"))
    }

    #[test]
    fn empty_inputs_produce_empty_output() {
        let out = rrf(vec![vec![], vec![]]);
        assert!(out.is_empty());
    }

    #[test]
    fn vector_only_scored_by_rank() {
        let out = rrf(vec![vec![result("a"), result("b"), result("c")], vec![]]);
        assert_eq!(ids(&out), vec!["a", "b", "c"]);
        assert!((out[0].score - 1.0 / K).abs() < EPS);
        assert!((out[1].score - 1.0 / (K + 1.0)).abs() < EPS);
        assert!((out[2].score - 1.0 / (K + 2.0)).abs() < EPS);
    }

    #[test]
    fn fts_only_scored_by_rank() {
        let out = rrf(vec![vec![], vec![result("a"), result("b"), result("c")]]);
        assert_eq!(ids(&out), vec!["a", "b", "c"]);
        assert!((out[0].score - 1.0 / K).abs() < EPS);
        assert!((out[1].score - 1.0 / (K + 1.0)).abs() < EPS);
        assert!((out[2].score - 1.0 / (K + 2.0)).abs() < EPS);
    }

    #[test]
    fn overlapping_chunk_sums_scores() {
        let out = rrf(vec![vec![result("a")], vec![result("a")]]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].chunk.id, "a");
        let expected = 1.0 / K + 1.0 / K;
        assert!((out[0].score - expected).abs() < EPS);
    }

    #[test]
    fn overlap_at_different_ranks_uses_each_list_rank() {
        let vector = vec![result("x"), result("y"), result("a")];
        let fts = vec![result("a"), result("y"), result("x")];
        let out = rrf(vec![vector, fts]);
        assert_eq!(out.len(), 3);

        let expected_a = 1.0 / (K + 2.0) + 1.0 / K;
        let expected_y = 1.0 / (K + 1.0) + 1.0 / (K + 1.0);
        let expected_x = 1.0 / K + 1.0 / (K + 2.0);

        assert!((find(&out, "a").score - expected_a).abs() < EPS);
        assert!((find(&out, "y").score - expected_y).abs() < EPS);
        assert!((find(&out, "x").score - expected_x).abs() < EPS);
    }

    #[test]
    fn disjoint_lists_include_all_chunks() {
        let vector = vec![result("a"), result("b")];
        let fts = vec![result("c"), result("d")];
        let out = rrf(vec![vector, fts]);
        assert_eq!(out.len(), 4);
        let mut got = ids(&out);
        got.sort_unstable();
        assert_eq!(got, vec!["a", "b", "c", "d"]);

        assert!((find(&out, "a").score - 1.0 / K).abs() < EPS);
        assert!((find(&out, "b").score - 1.0 / (K + 1.0)).abs() < EPS);
        assert!((find(&out, "c").score - 1.0 / K).abs() < EPS);
        assert!((find(&out, "d").score - 1.0 / (K + 1.0)).abs() < EPS);
    }

    #[test]
    fn output_is_sorted_descending_by_score() {
        let vector = vec![result("a"), result("b"), result("c"), result("d")];
        let fts = vec![result("d"), result("c"), result("b"), result("a")];
        let out = rrf(vec![vector, fts]);
        for w in out.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "out of order: {} >= {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn chunk_ranked_first_in_both_lists_wins() {
        let vector = vec![result("top"), result("mid"), result("bot")];
        let fts = vec![result("top"), result("bot"), result("mid")];
        let out = rrf(vec![vector, fts]);
        assert_eq!(out[0].chunk.id, "top");
    }

    #[test]
    fn no_duplicate_chunks_in_output() {
        let vector = vec![result("a"), result("b"), result("c")];
        let fts = vec![result("b"), result("c"), result("d")];
        let out = rrf(vec![vector, fts]);
        let mut seen = std::collections::HashSet::new();
        for r in &out {
            assert!(
                seen.insert(r.chunk.id.clone()),
                "duplicate id {}",
                r.chunk.id
            );
        }
        assert_eq!(out.len(), 4);
    }
}
