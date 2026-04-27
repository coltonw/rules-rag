use tiktoken_rs::cl100k_base;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::BufReader;

struct Chunker;

impl Chunker {
  
}

struct FixedSizeChunker {
  size: usize,
  overlap: usize,
}

pub fn chunk(text_path: &String) {
    let path = Path::new(text_path);
    let display = path.display();

    // Open the path in read-only mode, returns `io::Result<File>`
    let mut file = match File::open(path) {
        Err(err) => panic!("couldn't open {}: {}", display, err),
        Ok(file) => file,
    };

    // Read the file contents into a string, returns `io::Result<usize>`
    let lines = BufReader::new(file).lines();
    for line in lines.map_while(Result::ok) {
          println!("{}", line);
      }
}
