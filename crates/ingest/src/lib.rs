use core::Chunk;
use std::cmp::min;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
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
        let display = path.display();
        let file = match File::open(path) {
            Err(err) => panic!("couldn't open {}: {}", display, err),
            Ok(file) => file,
        };

        let tokenizer = cl100k_base_singleton();

        // Read the file contents into a string, returns `io::Result<usize>`
        let lines = BufReader::new(file).lines();
        let mut tokens: Vec<u32> = Vec::new();
        for line in lines.map_while(Result::ok) {
            // TODO: figure out the last_piece_token_len
            let (mut t, _) = tokenizer.encode(line.as_str(), &HashSet::new());
            tokens.append(&mut t);
        }

        let mut index = 0;
        let mut chunks: Vec<Chunk> = Vec::new();
        while index < tokens.len() {
            let Ok(text) = tokenizer.decode(&tokens[index..min(index+self.size, tokens.len())]) else {
                println!("Error decoding the thing I literally JUST encoded??");
                continue;
            };
            // TODO: fix this
            chunks.push(Chunk {
                id: "id".to_string(),
                game: "Pandemic".to_string(),
                text,
                source: "pdf".to_string(),
                page: Some(1),
                embedding: None,
            });
            index += self.size - self.overlap;
        }
        chunks
    }
}

