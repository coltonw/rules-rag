#![allow(async_fn_in_trait)]
use std::collections::HashMap;

use embed::OllamaEmbedder;
use rag_core::{Embedder, QueryOptions, RetrievalResult, Retrieve, Store};
use store::LanceStore;

#[derive(thiserror::Error, Debug)]
pub enum RetrieveError {
    #[error("embedding failed")]
    Embed(#[from] embed::EmbedError),
    #[error("retrieval failed")]
    Store(#[from] store::StoreError),
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
    async fn retrieve(
        &self,
        question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        let results = self
            .store
            .query_vector(&self.embedder.embed_one(question).await?, options)
            .await?;

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
    async fn retrieve(
        &self,
        question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        let results = self.store.query_fts(question, options).await?;

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
    async fn retrieve(
        &self,
        question: &str,
        options: &QueryOptions,
    ) -> Result<Vec<RetrievalResult>, RetrieveError> {
        let embedding = &self.embedder.embed_one(question).await?;
        let vector_query = self.store.query_vector(embedding, options);
        let fts_query = self.store.query_fts(question, options);
        let (vector_results, fts_results) = tokio::try_join!(vector_query, fts_query)?;

        let results: Vec<RetrievalResult> = rrf(vector_results, fts_results)
            .into_iter()
            .take(options.top_k)
            .collect();

        Ok(results)
    }
}

pub enum Retriever {
    Dense(DenseRetriever),
    Sparse(SparseRetriever),
    Hybrid(HybridRetriever),
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
        }
    }
}

/// Fuses vectors with Reciprocal Rank Fusion (`score = Σ 1/(k + rank_i)`, k≈60)
fn rrf(
    vector_results: Vec<RetrievalResult>,
    fts_results: Vec<RetrievalResult>,
) -> Vec<RetrievalResult> {
    let mut results_map: HashMap<String, RetrievalResult> = HashMap::new();

    // This should technically be a constant at the top of the file,
    // but I am not sure I will ever dig in enough to care so for now leaving it here
    let rrf_k: f32 = 60.0;
    for (vector_index, mut vector_result) in vector_results.into_iter().enumerate() {
        let score = 1.0 / (rrf_k + vector_index as f32);
        vector_result.score = score;
        results_map.insert(vector_result.chunk.id.clone(), vector_result);
    }
    for (fts_index, mut fts_result) in fts_results.into_iter().enumerate() {
        let score = 1.0 / (rrf_k + fts_index as f32);
        if let Some(result) = results_map.get_mut(&fts_result.chunk.id) {
            result.score += score;
        } else {
            fts_result.score = score;
            results_map.insert(fts_result.chunk.id.clone(), fts_result);
        }
    }

    let mut results: Vec<RetrievalResult> = results_map.into_values().collect();

    results.sort_by(|a, b| b.score.total_cmp(&a.score));
    results
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
        let out = rrf(vec![], vec![]);
        assert!(out.is_empty());
    }

    #[test]
    fn vector_only_scored_by_rank() {
        let out = rrf(vec![result("a"), result("b"), result("c")], vec![]);
        assert_eq!(ids(&out), vec!["a", "b", "c"]);
        assert!((out[0].score - 1.0 / K).abs() < EPS);
        assert!((out[1].score - 1.0 / (K + 1.0)).abs() < EPS);
        assert!((out[2].score - 1.0 / (K + 2.0)).abs() < EPS);
    }

    #[test]
    fn fts_only_scored_by_rank() {
        let out = rrf(vec![], vec![result("a"), result("b"), result("c")]);
        assert_eq!(ids(&out), vec!["a", "b", "c"]);
        assert!((out[0].score - 1.0 / K).abs() < EPS);
        assert!((out[1].score - 1.0 / (K + 1.0)).abs() < EPS);
        assert!((out[2].score - 1.0 / (K + 2.0)).abs() < EPS);
    }

    #[test]
    fn overlapping_chunk_sums_scores() {
        let out = rrf(vec![result("a")], vec![result("a")]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].chunk.id, "a");
        let expected = 1.0 / K + 1.0 / K;
        assert!((out[0].score - expected).abs() < EPS);
    }

    #[test]
    fn overlap_at_different_ranks_uses_each_list_rank() {
        let vector = vec![result("x"), result("y"), result("a")];
        let fts = vec![result("a"), result("y"), result("x")];
        let out = rrf(vector, fts);
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
        let out = rrf(vector, fts);
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
        let out = rrf(vector, fts);
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
        let out = rrf(vector, fts);
        assert_eq!(out[0].chunk.id, "top");
    }

    #[test]
    fn no_duplicate_chunks_in_output() {
        let vector = vec![result("a"), result("b"), result("c")];
        let fts = vec![result("b"), result("c"), result("d")];
        let out = rrf(vector, fts);
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
