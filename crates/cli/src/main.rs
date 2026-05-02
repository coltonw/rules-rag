use core::Chunker as _;
use ingest::FixedSizeChunker;
use std::path::Path;

fn main() {
    let chunker = FixedSizeChunker {
        size: 512,
        overlap: 64,
    };
    let chunks = chunker.chunk(Path::new("./data/pdfs/pandemic.txt"));
    println!("{}", chunks.first().unwrap().text);
}
