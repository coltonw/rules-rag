use rag_core::RawChunk;
use regex::Regex;
use std::collections::HashSet;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use tiktoken_rs::cl100k_base_singleton;

pub mod manifest;

pub trait Chunker {
    fn chunk(&self, text_path: &Path) -> Result<Vec<RawChunk>, IngestError>;
}

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("failed to read text file at {path}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read manifest.toml file at {path}")]
    ReadManifest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse manifest.toml file at {path}")]
    ParseManifest {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

pub struct FixedSizeChunker {
    pub size: usize,
    pub overlap: usize,
}

static NEW_PAGE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^=+ PAGE \d+ =+$").unwrap());

impl FixedSizeChunker {
    pub fn chunk_text(&self, text: &str) -> Vec<RawChunk> {
        let tokenizer = cl100k_base_singleton();
        let lines = text.lines();
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

        let mut chunks: Vec<RawChunk> = Vec::new();
        for (page_num, page) in pages.iter().enumerate() {
            // TODO: have chunks go through page boundaries
            let mut index = 0;
            while index < page.len() {
                let text = tokenizer
                    .decode(&page[index..(index + self.size).min(page.len())])
                    .expect("Error decoding what I JUST encoded should never happen");
                // TODO: id should not be hardcoded
                chunks.push(RawChunk {
                    text,
                    page: Some((page_num + 1) as u32), // page numbers start from 1
                });

                index += self.size;

                // if this chunk didn't go to the end of the page yet, subtract overlap
                if index < page.len() {
                    index -= self.overlap;
                }
            }
        }
        chunks
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, text_path: &Path) -> Result<Vec<RawChunk>, IngestError> {
        let text = read_to_string(text_path).map_err(|e| IngestError::ReadFile {
            path: text_path.to_path_buf(),
            source: e,
        })?;
        Ok(self.chunk_text(&text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_before_first_page_marker_are_dropped() {
        let chunker = FixedSizeChunker {
            size: 100,
            overlap: 0,
        };
        let chunks = chunker.chunk_text("preamble line\nanother line\n");
        assert!(chunks.is_empty());
    }

    #[test]
    fn each_page_is_chunked_separately_with_one_indexed_numbers() {
        let chunker = FixedSizeChunker {
            size: 1000,
            overlap: 0,
        };
        let chunks = chunker.chunk_text(
            "=== PAGE 1 ===\nfirst page\n\
             === PAGE 2 ===\nsecond page\n\
             === PAGE 3 ===\nthird page\n",
        );

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].page, Some(1));
        assert!(chunks[0].text.contains("first page"));
        assert_eq!(chunks[1].page, Some(2));
        assert!(chunks[1].text.contains("second page"));
        assert_eq!(chunks[2].page, Some(3));
        assert!(chunks[2].text.contains("third page"));
    }

    #[test]
    fn page_numbers_come_from_order_not_marker_digits() {
        let chunker = FixedSizeChunker {
            size: 1000,
            overlap: 0,
        };
        let chunks =
            chunker.chunk_text("=== PAGE 7 ===\nseven\n=== PAGE 13 ===\nthirteen\n");
        assert_eq!(chunks[0].page, Some(1));
        assert_eq!(chunks[1].page, Some(2));
    }

    #[test]
    fn chunks_never_span_page_boundaries() {
        let chunker = FixedSizeChunker {
            size: 10_000,
            overlap: 0,
        };
        let chunks = chunker.chunk_text("=== PAGE 1 ===\nalpha\n=== PAGE 2 ===\nomega\n");
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("alpha"));
        assert!(!chunks[0].text.contains("omega"));
        assert!(chunks[1].text.contains("omega"));
        assert!(!chunks[1].text.contains("alpha"));
    }

    #[test]
    fn long_page_splits_with_overlapping_tokens() {
        let chunker = FixedSizeChunker {
            size: 4,
            overlap: 2,
        };
        let body = (0..20)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&format!("=== PAGE 1 ===\n{body}\n"));

        assert!(chunks.len() > 1, "should split into multiple chunks");
        for chunk in &chunks {
            assert_eq!(chunk.page, Some(1));
        }

        let tokenizer = cl100k_base_singleton();
        let allowed = HashSet::new();
        for window in chunks.windows(2) {
            let a = tokenizer.encode(&window[0].text, &allowed).0;
            let b = tokenizer.encode(&window[1].text, &allowed).0;
            // Final chunk may be shorter than `size`, with no overlap into a successor.
            // For chunks at full size, the last `overlap` tokens must equal the next chunk's first `overlap`.
            if a.len() == chunker.size {
                let tail = &a[a.len() - chunker.overlap..];
                let head = &b[..chunker.overlap.min(b.len())];
                assert_eq!(tail, head, "consecutive full-size chunks should overlap");
            }
        }
    }
}
