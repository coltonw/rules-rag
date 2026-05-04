use indoc::formatdoc;
use rag_core::{Generator, RetrievalResult};
use reqwest::Client;
use std::time::Duration;
use tracing::debug;

#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error("generate request failed at {op}")]
    Reqwest {
        op: &'static str,
        #[source]
        source: reqwest::Error,
    },
}

pub struct OllamaGenerator {
    client: Client,
    base_url: String, // e.g. "http://localhost:11434"
    model: String,    // e.g. "gemma4:e2b"
}

#[derive(serde::Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
}

// Example responses:
// {"model":"gemma4:e2b","created_at":"2026-05-03T14:06:34.6943474Z","response":" pink","done":false}
// {"model":"gemma4:e2b","created_at":"2026-05-03T14:06:34.8352684Z","response":"","done":true,"done_reason":"stop","context":[...],
//   "total_duration":33802973200,"load_duration":236411800,"prompt_eval_count":22,"prompt_eval_duration":96124400,"eval_count":1263,
//   "eval_duration":32965842900}
#[derive(serde::Deserialize)]
struct GenerateResponse {
    response: String,
    done: bool,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u32>,
    eval_duration: Option<u64>,
}

fn prompt(query: &str, retrieval: &[RetrievalResult]) -> String {
    let chunks: Vec<String> = retrieval
        .iter()
        .map(|r| {
            let chunk = &r.chunk;
            let page_line = chunk
                .page
                .map(|p| format!("\n- Page number: {p}"))
                .unwrap_or_default();
            formatdoc! {"
            ### Chunk {id}

            - Game: {game}
            - Source: {source}{page_line}
            - Chunk search score: {score}
            - Text:
            ```
            {text}
            ```
            ",
                id = chunk.id,
                game = chunk.game,
                source = chunk.source,
                page_line = page_line,
                text = chunk.text,
                score = r.score
            }
        })
        .collect();
    let chunks: String = chunks.join("");
    // TODO: do some cleansing of the user query to prevent prompt injection
    formatdoc! {"
        # Answer Board Game Rules Questions

        You are a chatbot built for the sole purpose of answering rules questions

        - ONLY answer rules questions. For unrelated questions answer some version of \"I'm not sure I can answer that\"
        - ONLY give answers you can determine from provided rules chunks. If you cannot answer the users question, respond honestly.

        ## Relevant rules chunks

        {chunks}

        ## User query

        Here is the user query:

        ```
        {query}
        ```
        ",
        query = query,
        chunks = chunks
    }
}

impl Generator for OllamaGenerator {
    type Error = GenerateError;
    fn new() -> Self {
        let client = Client::new();
        OllamaGenerator {
            client,
            // TODO: cargo.config for stuff like this
            base_url: "http://localhost:11434".to_string(),
            model: "gemma4:e2b".to_string(),
        }
    }

    async fn generate(
        &self,
        query: &str,
        retrieval: &[RetrievalResult],
    ) -> Result<String, GenerateError> {
        let resp: GenerateResponse = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&GenerateRequest {
                model: &self.model,
                prompt: &prompt(query, retrieval),
                stream: false,
            })
            .send()
            .await
            .map_err(|e| GenerateError::Reqwest {
                op: "send request",
                source: e,
            })?
            .error_for_status()
            .map_err(|e| GenerateError::Reqwest {
                op: "check response status",
                source: e,
            })?
            .json()
            .await
            .map_err(|e| GenerateError::Reqwest {
                op: "parse response",
                source: e,
            })?;

        debug!(
            total_duration =
                Duration::from_nanos(resp.total_duration.unwrap_or_default()).as_secs_f64(),
            load_duration =
                Duration::from_nanos(resp.load_duration.unwrap_or_default()).as_secs_f64(),
            prompt_eval_count = resp.prompt_eval_count.unwrap_or_default(),
            prompt_eval_duration =
                Duration::from_nanos(resp.prompt_eval_duration.unwrap_or_default()).as_secs_f64(),
            eval_count = resp.eval_count.unwrap_or_default(),
            eval_duration =
                Duration::from_nanos(resp.eval_duration.unwrap_or_default()).as_secs_f64(),
            "generate call complete"
        );

        assert!(
            resp.done,
            "Generate response should be done before we return it"
        );

        Ok(resp.response)
    }
}
