use core::Chunk;
use regex::Regex;
use std::collections::HashSet;
use std::fs::read_to_string;
use std::path::Path;
use tiktoken_rs::cl100k_base_singleton;

pub trait Chunker {
    fn chunk(&self, text_path: &str) -> Vec<Chunk>;
}

pub struct FixedSizeChunker {
    pub size: usize,
    pub overlap: usize,
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, text_path: &str) -> Vec<Chunk> {
        let path = Path::new(text_path);
        let Ok(full_text) = read_to_string(path) else {
            panic!("Failed to read file {}", text_path);
        };

        let tokenizer = cl100k_base_singleton();

        // Read the file contents into a string, returns `io::Result<usize>`
        let lines = full_text.lines();
        let mut pages: Vec<Vec<u32>> = Vec::new();
        let new_page_regex = Regex::new(r"^=+ PAGE \d+ =+$").unwrap();
        for line in lines {
            if new_page_regex.is_match(line) {
                pages.push(Vec::new());
                continue;
            }
            if let Some(page) = pages.last_mut() {
                // TODO: figure out what last_piece_token_len is and if I need to use it
                let (mut t, _) = tokenizer.encode(&format!("{}\n", line), &HashSet::new());
                page.append(&mut t);
            }
        }

        let mut index = 0;
        let mut chunks: Vec<Chunk> = Vec::new();
        for (page_num, page) in pages.iter().enumerate() {
            while index < page.len() {
                let Ok(text) = tokenizer.decode(&page[index..(index + self.size).min(page.len())])
                else {
                    println!("Error decoding the thing I literally JUST encoded??");
                    continue;
                };
                // TODO: id and game should not be hardcoded
                // Page numbers generally start from 1
                let page_num_u32 = u32::try_from(page_num + 1);
                chunks.push(Chunk {
                    id: "id".to_string(),
                    game: "Pandemic".to_string(),
                    text,
                    source: "pdf".to_string(),
                    page: page_num_u32.ok(),
                    embedding: None,
                });
                index += self.size - self.overlap;
            }
        }
        chunks
    }
}
