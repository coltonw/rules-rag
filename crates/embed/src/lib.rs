use std::time::Duration;

use rag_core::Embedder;
use reqwest::Client;

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
    fn new() -> Self {
        let client = Client::new();
        OllamaEmbedder {
            client,
            // TODO: cargo.config for stuff like this
            base_url: "http://localhost:11434".to_string(),
            model: "qwen3-embedding:0.6b".to_string(),
        }
    }

    // TODO: actual error handling
    async fn generate(&self, inputs: &[impl AsRef<str>]) -> Vec<Vec<f32>> {
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
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();

        println!(
            "Embedder load duration: {:.3} s",
            Duration::from_nanos(resp.load_duration).as_secs_f64()
        );
        println!(
            "Embedder total duration: {:.3} s",
            Duration::from_nanos(resp.total_duration).as_secs_f64()
        );
        println!("Embedder prompt eval count: {}", resp.prompt_eval_count);

        resp.embeddings
    }

    async fn generate_one(&self, input: &str) -> Vec<f32> {
        self.generate(&[input.to_string()]).await.pop().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "Depends on ollama running"]
    #[tokio::test]
    async fn generate_one_embed() {
        let embedder = OllamaEmbedder::new();
        let embed = embedder.generate_one("Hello").await;
        assert_eq!(embed.len(), 1024);
    }
}
