use std::time::{Duration, Instant};

use rag_core::{Chunk, Reranker, RetrievalResult};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

#[derive(Debug, thiserror::Error)]
pub enum RerankError {
    #[error(
        "reranker sidecar request failed at {op} — is it running? start it with `rerank-sidecar/run.sh`"
    )]
    Http {
        op: &'static str,
        #[source]
        source: reqwest::Error,
    },
}

#[derive(Serialize)]
struct RerankRequest<'a> {
    query: &'a str,
    documents: Vec<&'a str>,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<ScoredIndex>,
}

#[derive(Deserialize)]
struct ScoredIndex {
    index: usize,
    score: f32,
}

/// Reranker backed by the GPU sidecar in `rerank-sidecar/` (see its README).
///
/// The sidecar runs the BGE-reranker-v2-m3 ONNX model on the GPU and returns
/// `{index, score}` already sorted descending, so we just map indices back to
/// chunks. ~0.5s for 30 chunks vs ~19s for the same model on CPU.
pub struct HttpReranker {
    client: Client,
    base_url: String,
}

impl Reranker for HttpReranker {
    type Error = RerankError;

    fn new() -> Result<Self, RerankError> {
        // Short connect timeout so a stopped sidecar fails in ~2s instead of
        // hanging ~30s on the OS TCP SYN-retransmit timeout.
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| RerankError::Http {
                op: "build client",
                source: e,
            })?;
        Ok(Self {
            client,
            // TODO: cargo.config for stuff like this (mirrors OllamaRewriter)
            base_url: "http://127.0.0.1:8071".to_string(),
        })
    }

    #[instrument(
        level = "debug",
        name = "rerank",
        skip_all,
        fields(q_len = query.len(), n_chunks = chunks.len()),
    )]
    async fn rerank(
        &self,
        query: &str,
        chunks: Vec<Chunk>,
    ) -> Result<Vec<RetrievalResult>, RerankError> {
        let documents: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();

        let started = Instant::now();
        let resp: RerankResponse = self
            .client
            .post(format!("{}/rerank", self.base_url))
            .json(&RerankRequest { query, documents })
            .send()
            .await
            .map_err(|e| RerankError::Http {
                op: "send request",
                source: e,
            })?
            .error_for_status()
            .map_err(|e| RerankError::Http {
                op: "check response status",
                source: e,
            })?
            .json()
            .await
            .map_err(|e| RerankError::Http {
                op: "parse response",
                source: e,
            })?;
        let elapsed = started.elapsed();

        // map indices from the sidecar back into chunks (order is preserved)
        let results: Vec<RetrievalResult> = resp
            .results
            .into_iter()
            .filter_map(|sr| {
                chunks.get(sr.index).map(|chunk| RetrievalResult {
                    chunk: chunk.clone(),
                    score: sr.score,
                })
            })
            .collect();

        debug!(
            n_results = results.len(),
            elapsed = elapsed.as_secs_f64(),
            "rerank done"
        );
        Ok(results)
    }
}
