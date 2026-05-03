use indoc::formatdoc;
use rag_core::{Chunk, Generator, RetrievalResult};
use reqwest::Client;
use std::time::Duration;

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
            formatdoc! {"
            ### Chunk {id}

            - Game: {game}
            - Source: {source}
            - Page number: {page}
            - Chunk search score: {score}
            - Text:
            ```
            {text}
            ```
            ",
                id = chunk.id,
                game = chunk.game,
                source = chunk.source,
                page = chunk.page.unwrap_or_default(),
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
    fn new() -> Self {
        let client = Client::new();
        OllamaGenerator {
            client,
            // TODO: cargo.config for stuff like this
            base_url: "http://localhost:11434".to_string(),
            model: "gemma4:e2b".to_string(),
        }
    }

    // TODO: actual error handling
    async fn generate(&self, query: &str, retrieval: &[RetrievalResult]) -> String {
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
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();

        if let Some(load_duration) = resp.load_duration {
            println!(
                "Generator load duration: {:.3} s",
                Duration::from_nanos(load_duration).as_secs_f64()
            );
        }
        if let Some(total_duration) = resp.total_duration {
            println!(
                "Generator total duration: {:.3} s",
                Duration::from_nanos(total_duration).as_secs_f64()
            );
        }

        resp.response
    }
}
