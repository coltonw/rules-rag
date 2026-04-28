use ingest::{Chunker as _, FixedSizeChunker};

fn main() {
    let chunker = FixedSizeChunker {
        size: 512,
        overlap: 64,
    };
    let chunks = chunker.chunk("./data/pdfs/pandemic.txt");
    println!("{}", chunks.first().unwrap().text);
}
