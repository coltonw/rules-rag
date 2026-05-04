use rag_core::Chunk;
use regex::Regex;
use std::collections::HashSet;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use tiktoken_rs::cl100k_base_singleton;

pub trait Chunker {
    fn chunk(&self, text_path: &Path, game: &str) -> Result<Vec<Chunk>, IngestError>;
}

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("failed to read text file at {path}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub struct FixedSizeChunker {
    pub size: usize,
    pub overlap: usize,
}

static NEW_PAGE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^=+ PAGE \d+ =+$").unwrap());

impl Chunker for FixedSizeChunker {
    fn chunk(&self, text_path: &Path, game: &str) -> Result<Vec<Chunk>, IngestError> {
        let full_text = read_to_string(text_path).map_err(|e| IngestError::ReadFile {
            path: text_path.to_path_buf(),
            source: e,
        })?;

        let tokenizer = cl100k_base_singleton();
        let lines = full_text.lines();
        let mut pages: Vec<Vec<u32>> = Vec::new();
        let allowed_specials = &HashSet::new();
        for line in lines {
            if NEW_PAGE_RE.is_match(line) {
                pages.push(Vec::new());
                continue;
            }
            if let Some(page) = pages.last_mut() {
                // TODO: figure out what last_piece_token_len is and if I need to use it
                page.extend(tokenizer.encode(&format!("{}\n", line), allowed_specials).0);
            }
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        for (page_num, page) in pages.iter().enumerate() {
            // TODO: have chunks go through page boundaries
            let mut index = 0;
            while index < page.len() {
                let text = tokenizer
                    .decode(&page[index..(index + self.size).min(page.len())])
                    .expect("Error decoding what I JUST encoded should never happen");
                // TODO: id should not be hardcoded
                chunks.push(Chunk {
                    id: "id".to_string(),
                    game: game.to_string(),
                    text,
                    source: "pdf".to_string(),
                    page: Some((page_num + 1) as u32), // page numbers start from 1
                    embedding: None,
                });

                index += self.size;

                // if this chunk didn't go to the end of the page yet, subtract overlap
                if index < page.len() {
                    index -= self.overlap;
                }
            }
        }
        Ok(chunks)
    }
}
