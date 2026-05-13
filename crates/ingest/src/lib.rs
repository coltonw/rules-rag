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

pub struct ParagraphChunker {
    pub target_size: usize,
    pub max_size: usize,
}

struct Section {
    paragraphs: Vec<String>,
    page: u32,
}

impl Section {
    fn new(page: u32) -> Self {
        Self {
            paragraphs: Vec::new(),
            page,
        }
    }
    fn heading_level(&self) -> Option<u8> {
        let first_line = self.paragraphs.first()?.lines().next()?;
        let count = first_line.bytes().take_while(|&b| b == b'#').count();
        (count > 0).then_some(count.min(6) as u8)
    }
}

impl ParagraphChunker {
    pub fn chunk_text(&self, text: &str) -> Vec<RawChunk> {
        let sections = self.parse_sections(text);
        self.pack_sections(&sections)
    }

    fn parse_sections(&self, text: &str) -> Vec<Section> {
        let lines = text.lines();
        let mut page_num = 0;
        let mut prev_blank = false;
        let mut current_paragraph = "".to_string();
        let mut current_section: Section = Section::new(0);
        let mut sections: Vec<Section> = Vec::new();
        // First iterate through lines and break it down into sections and paragraphs
        for line in lines {
            if line.trim().is_empty() {
                prev_blank = true;
                continue;
            }
            if NEW_PAGE_RE.is_match(line) {
                if !current_paragraph.is_empty() {
                    current_section.paragraphs.push(current_paragraph);
                }
                if !current_section.paragraphs.is_empty() {
                    sections.push(current_section);
                }
                page_num += 1;
                current_paragraph = "".to_string();
                current_section = Section::new(page_num);
                prev_blank = false;
                continue;
            }
            if prev_blank {
                if !current_paragraph.is_empty() {
                    current_section.paragraphs.push(current_paragraph);
                }
                current_paragraph = "".to_string();
                if line.starts_with("#") {
                    if !current_section.paragraphs.is_empty() {
                        sections.push(current_section);
                    }
                    current_section = Section::new(page_num);
                }
            }
            prev_blank = false;
            current_paragraph += &format!("{}\n", line);
        }
        if !current_paragraph.is_empty() {
            current_section.paragraphs.push(current_paragraph);
        }
        if !current_section.paragraphs.is_empty() {
            sections.push(current_section);
        }
        sections
    }

    fn pack_sections(&self, sections: &[Section]) -> Vec<RawChunk> {
        let tokenizer = cl100k_base_singleton();
        let allowed_specials = &HashSet::new();
        let mut chunks: Vec<RawChunk> = Vec::new();
        if sections.is_empty() {
            return chunks;
        }
        let mut chunk = "".to_string();
        let mut chunk_token_len = 0;
        let mut chunk_page: u32 = 1;
        let mut chunk_heading_level: Option<u8> = None;
        for section in sections {
            let section_text = section.paragraphs.join("\n");
            let section_token_len = tokenizer
                .encode(&format!("{}\n", &section_text), allowed_specials)
                .0
                .len();
            let section_heading_level = section.heading_level();
            tracing::trace!(
                tokens = section_token_len,
                page = section.page,
                heading_level = ?section_heading_level,
                "section"
            );
            if section_token_len > self.max_size {
                tracing::warn!(
                    tokens = section_token_len,
                    max = self.max_size,
                    page = section.page,
                    heading_level = ?section_heading_level,
                    "section exceeds max_size; emitting as a single oversized chunk"
                );
            }
            let section_heading_flush = if let (Some(chunk_level), Some(section_level)) =
                (chunk_heading_level, section_heading_level)
            {
                section_level <= chunk_level
            } else {
                false
            };
            if section_token_len + chunk_token_len > self.max_size || section_heading_flush {
                flush(
                    &mut chunks,
                    &mut chunk,
                    &mut chunk_token_len,
                    &mut chunk_heading_level,
                    chunk_page,
                );
                // TODO: check if this one section is too big for a single chunk and break on paragraphs if it is
                chunk += &format!("{}\n", &section_text);
                chunk_token_len += section_token_len;
                chunk_page = section.page;
                chunk_heading_level = section_heading_level;
            } else if section_token_len + chunk_token_len >= self.target_size {
                if chunk.is_empty() {
                    chunk_page = section.page;
                }
                chunk += &format!("{}\n", &section_text);
                chunk_token_len += section_token_len;
                chunk_heading_level = chunk_heading_level.or(section_heading_level);
                flush(
                    &mut chunks,
                    &mut chunk,
                    &mut chunk_token_len,
                    &mut chunk_heading_level,
                    chunk_page,
                );
            } else {
                if chunk.is_empty() {
                    chunk_page = section.page;
                }
                chunk += &format!("{}\n", &section_text);
                chunk_token_len += section_token_len;
                chunk_heading_level = chunk_heading_level.or(section_heading_level);
            }
        }
        flush(
            &mut chunks,
            &mut chunk,
            &mut chunk_token_len,
            &mut chunk_heading_level,
            chunk_page,
        );
        tracing::debug!(n_chunks = chunks.len(), n_sections = sections.len(), "packed");
        chunks
    }
}

impl Chunker for ParagraphChunker {
    fn chunk(&self, text_path: &Path) -> Result<Vec<RawChunk>, IngestError> {
        let text = read_to_string(text_path).map_err(|e| IngestError::ReadFile {
            path: text_path.to_path_buf(),
            source: e,
        })?;
        Ok(self.chunk_text(&text))
    }
}

fn flush(
    chunks: &mut Vec<RawChunk>,
    chunk: &mut String,
    chunk_token_len: &mut usize,
    chunk_heading_level: &mut Option<u8>,
    chunk_page: u32,
) {
    if !chunk.is_empty() {
        tracing::debug!(
            tokens = *chunk_token_len,
            page = chunk_page,
            heading_level = ?*chunk_heading_level,
            "emit chunk"
        );
        chunks.push(RawChunk {
            text: std::mem::take(chunk),
            page: Some(chunk_page),
        });
    }
    *chunk_token_len = 0;
    *chunk_heading_level = None;
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
        let chunks = chunker.chunk_text("=== PAGE 7 ===\nseven\n=== PAGE 13 ===\nthirteen\n");
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

    #[test]
    fn paragraph_chunker_empty_input_produces_no_chunks() {
        let chunker = ParagraphChunker {
            target_size: 50,
            max_size: 100,
        };
        assert!(chunker.chunk_text("").is_empty());
    }

    #[test]
    fn paragraph_chunker_nested_sections_accumulate_into_one_chunk() {
        // Headings at progressively deeper levels nest within a single chunk:
        // the chunk anchors at the highest-priority level and deeper headings
        // never trigger a flush.
        let chunker = ParagraphChunker {
            target_size: 1000,
            max_size: 2000,
        };
        let text = "=== PAGE 1 ===\n\
                    intro paragraph\n\n\
                    # Top\n\nalpha content\n\n\
                    ## Subsection\n\nbravo content\n\n\
                    ### Deeper\n\ncharlie content\n";
        let chunks = chunker.chunk_text(text);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("alpha"));
        assert!(chunks[0].text.contains("bravo"));
        assert!(chunks[0].text.contains("charlie"));
        assert_eq!(chunks[0].page, Some(1));
    }

    #[test]
    fn paragraph_chunker_sibling_headings_split_into_separate_chunks() {
        // Two `##` siblings under the same implicit parent should split:
        // the second heading's level (2) is <= the chunk anchor (2), so it
        // forces a flush before being added.
        let chunker = ParagraphChunker {
            target_size: 1000,
            max_size: 2000,
        };
        let text = "=== PAGE 1 ===\n\
                    intro paragraph\n\n\
                    ## Sibling A\n\nalpha content\n\n\
                    ## Sibling B\n\nbravo content\n";
        let chunks = chunker.chunk_text(text);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("Sibling A"));
        assert!(chunks[0].text.contains("alpha"));
        assert!(!chunks[0].text.contains("Sibling B"));
        assert!(chunks[1].text.contains("Sibling B"));
        assert!(chunks[1].text.contains("bravo"));
    }

    #[test]
    fn paragraph_chunker_subsection_stays_with_parent_chunk() {
        // The user's motivating case: a `## Parent` followed immediately by a
        // `### Sub` should not orphan the parent into a tiny chunk. The
        // subsection's level (3) > the chunk anchor (2), so no flush.
        let chunker = ParagraphChunker {
            target_size: 1000,
            max_size: 2000,
        };
        let text = "=== PAGE 1 ===\n\
                    intro paragraph\n\n\
                    ## Parent\n\n\
                    ### Sub\n\nsubsection content\n";
        let chunks = chunker.chunk_text(text);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Parent"));
        assert!(chunks[0].text.contains("Sub"));
        assert!(chunks[0].text.contains("subsection content"));
    }

    #[test]
    fn paragraph_chunker_anchor_resets_after_heading_flush() {
        // After a heading-driven flush, the new chunk's anchor must reset so
        // unrelated subsequent headings pack correctly. Without the reset,
        // a stale anchor from a previous chunk would cause extra spurious
        // flushes.
        //
        // Sequence:
        //   intro + ## A   -> chunk 1 (anchor 2)
        //   ## B           -> flush (2 <= 2), new chunk starts with B (anchor 2)
        //   # Big          -> flush (1 <= 2), new chunk starts with Big (anchor 1)
        //   ## After Big   -> 2 > 1, NO flush; packs with Big (chunk 3)
        //
        // If the anchor doesn't reset, "## After Big" sees a stale anchor of 2
        // and triggers an extra flush, producing 4 chunks instead of 3.
        let chunker = ParagraphChunker {
            target_size: 10_000,
            max_size: 20_000,
        };
        let text = "=== PAGE 1 ===\n\
                    intro paragraph\n\n\
                    ## A\n\na content\n\n\
                    ## B\n\nb content\n\n\
                    # Big\n\nbig content\n\n\
                    ## After Big\n\nafter content\n";
        let chunks = chunker.chunk_text(text);
        assert_eq!(chunks.len(), 3, "expected 3 chunks, got {}", chunks.len());
        assert!(chunks[0].text.contains("A"));
        assert!(chunks[1].text.contains("B"));
        assert!(chunks[2].text.contains("Big"));
        assert!(
            chunks[2].text.contains("After Big"),
            "## After Big should pack with # Big, not start a new chunk"
        );
    }

    #[test]
    fn paragraph_chunker_flushes_when_target_reached() {
        let chunker = ParagraphChunker {
            target_size: 1,
            max_size: 1000,
        };
        let text = "=== PAGE 1 ===\n\
                    first content\n\n\
                    ## Heading Two\n\n\
                    second content\n";
        let chunks = chunker.chunk_text(text);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("first content"));
        assert!(!chunks[0].text.contains("Heading Two"));
        assert!(chunks[1].text.contains("Heading Two"));
        assert!(chunks[1].text.contains("second content"));
    }

    #[test]
    fn paragraph_chunker_heading_without_blank_line_does_not_split() {
        let chunker = ParagraphChunker {
            target_size: 1,
            max_size: 1000,
        };
        let text = "=== PAGE 1 ===\n\
                    content here\n\
                    ## Inline Heading\n\
                    more content\n";
        let chunks = chunker.chunk_text(text);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Inline Heading"));
        assert!(chunks[0].text.contains("content here"));
        assert!(chunks[0].text.contains("more content"));
    }

    #[test]
    fn paragraph_chunker_oversized_section_emits_its_own_chunk() {
        let chunker = ParagraphChunker {
            target_size: 5,
            max_size: 10,
        };
        let body = (0..50)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let text = format!(
            "=== PAGE 1 ===\n\
             small first\n\n\
             ## Big\n\n{body}\n\n\
             ## Tail\n\nsmall last\n"
        );
        let chunks = chunker.chunk_text(&text);
        // The big section alone exceeds max, so it gets its own chunk.
        let big_chunk = chunks
            .iter()
            .find(|c| c.text.contains("word49"))
            .expect("oversized section should appear in some chunk");
        assert!(big_chunk.text.contains("word0"));
        assert!(!big_chunk.text.contains("small first"));
        assert!(!big_chunk.text.contains("small last"));
    }

    #[test]
    fn paragraph_chunker_chunk_page_is_first_sections_page() {
        let chunker = ParagraphChunker {
            target_size: 10_000,
            max_size: 20_000,
        };
        let text = "=== PAGE 1 ===\nstart\n\
                    === PAGE 2 ===\nmiddle\n\
                    === PAGE 3 ===\nend\n";
        let chunks = chunker.chunk_text(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].page, Some(1));
        assert!(chunks[0].text.contains("start"));
        assert!(chunks[0].text.contains("middle"));
        assert!(chunks[0].text.contains("end"));
    }

    #[test]
    fn paragraph_chunker_trailing_paragraph_is_flushed() {
        let chunker = ParagraphChunker {
            target_size: 1,
            max_size: 1000,
        };
        // No trailing newline, no closing page marker — the final paragraph
        // and section still need to be emitted.
        let chunks = chunker.chunk_text("=== PAGE 1 ===\nonly content");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("only content"));
        assert_eq!(chunks[0].page, Some(1));
    }

    #[test]
    fn paragraph_chunker_consecutive_blank_lines_are_collapsed() {
        let chunker = ParagraphChunker {
            target_size: 1,
            max_size: 1000,
        };
        let text = "=== PAGE 1 ===\nfirst\n\n\n\nsecond\n";
        let chunks = chunker.chunk_text(text);
        // Both paragraphs in one section (no heading between), so one chunk.
        assert_eq!(chunks.len(), 1);
        // No runs of three or more newlines from the multi-blank input.
        assert!(
            !chunks[0].text.contains("\n\n\n"),
            "got: {:?}",
            chunks[0].text
        );
    }
}
