use rag_core::Embedder;
use reqwest::Client;
use std::time::Duration;
use tracing::debug;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("embed request failed at {op}")]
    Reqwest {
        op: &'static str,
        #[source]
        source: reqwest::Error,
    },
}

pub struct OllamaEmbedder {
    client: Client,
    base_url: String, // e.g. "http://localhost:11434"
    model: String,    // e.g. "qwen3-embedding:0.6b"
}

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
    total_duration: u64,
    load_duration: u64,
    prompt_eval_count: u32,
}

impl Embedder for OllamaEmbedder {
    type Error = EmbedError;
    fn new() -> Self {
        let client = Client::new();
        OllamaEmbedder {
            client,
            // TODO: cargo.config for stuff like this
            base_url: "http://localhost:11434".to_string(),
            model: "qwen3-embedding:0.6b".to_string(),
        }
    }

    async fn generate(&self, inputs: &[impl AsRef<str>]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let inputs: Vec<&str> = inputs.iter().map(AsRef::as_ref).collect();
        let resp: EmbedResponse = self
            .client
            .post(format!("{}/api/embed", self.base_url))
            .json(&EmbedRequest {
                model: &self.model,
                input: &inputs,
            })
            .send()
            .await
            .map_err(|e| EmbedError::Reqwest {
                op: "send request",
                source: e,
            })?
            .error_for_status()
            .map_err(|e| EmbedError::Reqwest {
                op: "check response status",
                source: e,
            })?
            .json()
            .await
            .map_err(|e| EmbedError::Reqwest {
                op: "parse response",
                source: e,
            })?;

        debug!(
            load_duration = Duration::from_nanos(resp.load_duration).as_secs_f64(),
            total_duration = Duration::from_nanos(resp.total_duration).as_secs_f64(),
            prompt_eval_count = resp.prompt_eval_count,
            "embed call complete"
        );

        assert_eq!(
            inputs.len(),
            resp.embeddings.len(),
            "There should be an embedding for each input"
        );
        Ok(resp.embeddings)
    }

    async fn generate_one(&self, input: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self
            .generate(&[input.to_string()])
            .await?
            .pop()
            .expect("There should always be an embed result"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "Depends on ollama running"]
    #[tokio::test]
    async fn generate_one_embed() {
        let embedder = OllamaEmbedder::new();
        let embed = embedder.generate_one("Hello").await.unwrap();
        assert_eq!(embed.len(), 1024);
    }
}
