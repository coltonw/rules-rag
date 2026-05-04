use rag_core::DocType;
use serde::Deserialize;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};

use crate::IngestError;

#[derive(Deserialize)]
pub struct DocMeta {
    pub file: PathBuf,
    pub game: String,
    pub doc_type: DocType,
}

#[derive(Deserialize)]
struct Documents {
    document: Vec<DocMeta>,
}

pub fn read_manifest(path: &Path) -> Result<Vec<DocMeta>, IngestError> {
    let manifest = read_to_string(path).map_err(|e| IngestError::ReadManifest {
        path: path.to_path_buf(),
        source: e,
    })?;
    toml::from_str::<Documents>(&manifest)
        .map_err(|e| IngestError::ParseManifest {
            path: path.to_path_buf(),
            source: e,
        })
        .map(|d| d.document)
}
