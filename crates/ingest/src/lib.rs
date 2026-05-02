use rag_core::{Chunk, Chunker};
use regex::Regex;
use std::collections::HashSet;
use std::fs::read_to_string;
use std::path::Path;
use tiktoken_rs::cl100k_base_singleton;

pub struct FixedSizeChunker {
    pub size: usize,
    pub overlap: usize,
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, text_path: &Path) -> Vec<Chunk> {
        let Ok(full_text) = read_to_string(text_path) else {
            // TODO: better error handling
            panic!("Failed to read file {}", text_path.display());
        };

        let tokenizer = cl100k_base_singleton();
        let lines = full_text.lines();
        let mut pages: Vec<Vec<u32>> = Vec::new();
        let new_page_regex = Regex::new(r"^=+ PAGE \d+ =+$").unwrap();
        let allowed_specials = &HashSet::new();
        for line in lines {
            if new_page_regex.is_match(line) {
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
            let mut index = 0;
            while index < page.len() {
                let Ok(text) = tokenizer.decode(&page[index..(index + self.size).min(page.len())])
                else {
                    println!("Error decoding the thing I literally JUST encoded??");
                    break;
                };
                // TODO: id and game should not be hardcoded
                chunks.push(Chunk {
                    id: "id".to_string(),
                    game: "Pandemic".to_string(),
                    text,
                    source: "pdf".to_string(),
                    page: Some((page_num + 1) as u32), // page numbers start from 1
                    embedding: None,
                });
                index += self.size - self.overlap;
            }
        }
        chunks
    }
}
