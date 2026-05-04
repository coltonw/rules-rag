use std::path::Path;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use futures::TryStreamExt as _;
use lancedb::query::{ExecutableQuery as _, QueryBase as _};
use lancedb::{DistanceType, Table, connect};
use rag_core::{Chunk, EMBED_DIM, RetrievalResult, VectorStore};
use serde_arrow::schema::{SchemaLike, TracingOptions};
use serde_arrow::{from_record_batch, to_record_batch};

pub use arrow_array;
pub use arrow_schema;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("LanceDB failure at {op}")]
    Lance {
        op: &'static str,
        #[source]
        source: lancedb::Error,
    },
    #[error("schema in DB does not match expected schema")]
    SchemaMismatch {
        expected: Arc<Schema>,
        actual: Arc<Schema>,
    },
}

pub struct LanceStore {
    table: Table,
    schema: Arc<Schema>,
}

fn col_as<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> &'a T {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("query result missing column '{name}'"))
        .as_any()
        .downcast_ref::<T>()
        .unwrap_or_else(|| panic!("query result column '{name}' has wrong Arrow type"))
}

fn records_to_results(batches: Vec<RecordBatch>) -> Vec<RetrievalResult> {
    let mut results: Vec<RetrievalResult> = Vec::new();

    for batch in batches {
        let embedding_index = batch
            .schema()
            .index_of("embedding")
            .expect("embedding column missing");
        let trimmed_indices: Vec<usize> = (0..batch.num_columns())
            .filter(|i| *i != embedding_index)
            .collect();
        let batch = batch
            .project(&trimmed_indices)
            .expect("batch projection should never fail");
        let chunks = from_record_batch::<Vec<Chunk>>(&batch)
            .expect("failed to deserialize records into chunks even after schema validation");
        let distances: &Float32Array = col_as(&batch, "_distance");
        for (i, chunk) in chunks.into_iter().enumerate() {
            results.push(RetrievalResult {
                chunk,
                score: 1.0 - distances.value(i),
            })
        }
    }

    results
}

impl LanceStore {
    fn schema() -> Arc<Schema> {
        let fields = Vec::<FieldRef>::from_type::<Chunk>(
            TracingOptions::default()
                .enums_without_data_as_strings(true)
                .strings_as_large_utf8(false)
                .overwrite(
                    "embedding",
                    Field::new(
                        "embedding",
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            EMBED_DIM,
                        ),
                        false,
                    ),
                )
                .expect("embedding field overwrite should never fail")
                .overwrite("doc_type", Field::new("doc_type", DataType::Utf8, false))
                .expect("doc_type field overwrite should never fail"),
        )
        .expect("Schema should never fail to be created from Chunk object");
        Arc::new(Schema::new(fields))
    }

    fn chunks_to_records(&self, rows: &[Chunk]) -> RecordBatch {
        let rows: Vec<&Chunk> = rows
            .iter()
            .filter(|chunk| chunk.embedding.is_some())
            .collect();
        to_record_batch(self.schema.fields(), &rows)
            .expect("Unexpected error converting chunks to an Arrow RecordBatch")
    }
}

impl VectorStore for LanceStore {
    type Error = StoreError;

    async fn connect(path: &Path) -> Result<Self, StoreError> {
        let connection = connect(path.to_str().expect("DB path must be UTF-8"))
            .execute()
            .await
            .map_err(|e| StoreError::Lance {
                op: "connect",
                source: e,
            })?;
        let schema = Self::schema();

        let table_name = "rules_chunks";
        let table = match connection.open_table(table_name).execute().await {
            Ok(table) => table,
            Err(lancedb::Error::TableNotFound { .. }) => connection
                .create_empty_table(table_name, schema.clone())
                .execute()
                .await
                .map_err(|e| StoreError::Lance {
                    op: "create table",
                    source: e,
                })?,
            Err(unknown) => {
                return Err(StoreError::Lance {
                    op: "open table",
                    source: unknown,
                });
            }
        };

        let table_schema = table.schema().await.map_err(|e| StoreError::Lance {
            op: "read table schema",
            source: e,
        })?;

        if table_schema != schema {
            return Err(StoreError::SchemaMismatch {
                expected: schema,
                actual: table_schema,
            });
        }

        Ok(LanceStore { table, schema })
    }

    async fn insert(&self, chunks: &[Chunk]) -> Result<(), StoreError> {
        self.table
            .add(self.chunks_to_records(chunks))
            .execute()
            .await
            .map_err(|e| StoreError::Lance {
                op: "insert",
                source: e,
            })?;
        Ok(())
    }

    async fn query(&self, embedding: &[f32], k: usize) -> Result<Vec<RetrievalResult>, StoreError> {
        let results = self
            .table
            .query()
            .nearest_to(embedding)
            .map_err(|e| StoreError::Lance {
                op: "build vector query",
                source: e,
            })?
            .distance_type(DistanceType::Cosine)
            .limit(k)
            .execute()
            .await
            .map_err(|e| StoreError::Lance {
                op: "execute query",
                source: e,
            })?;

        let batches: Vec<RecordBatch> =
            results.try_collect().await.map_err(|e| StoreError::Lance {
                op: "collect query results",
                source: e,
            })?;

        Ok(records_to_results(batches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_core::DocType;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn unit_embedding(hot_index: usize) -> Vec<f32> {
        let mut e = vec![0.0; EMBED_DIM as usize];
        e[hot_index] = 1.0;
        e
    }

    #[tokio::test]
    async fn roundtrip_through_lancedb() {
        let dir = TempDir::new().unwrap();
        let store = LanceStore::connect(dir.path()).await.unwrap();

        let chunks = vec![
            Chunk {
                id: "rules-1".into(),
                text: "rules text".into(),
                game: "Pandemic".into(),
                doc_type: DocType::Rules,
                page: Some(3),
                embedding: Some(unit_embedding(0)),
            },
            Chunk {
                id: "ref-1".into(),
                text: "reference text".into(),
                game: "Pandemic".into(),
                doc_type: DocType::Reference,
                page: None,
                embedding: Some(unit_embedding(1)),
            },
            Chunk {
                id: "faq-1".into(),
                text: "faq text".into(),
                game: "Pandemic".into(),
                doc_type: DocType::Faq,
                page: Some(7),
                embedding: Some(unit_embedding(2)),
            },
            Chunk {
                id: "no-emb".into(),
                text: "should be filtered".into(),
                game: "Pandemic".into(),
                doc_type: DocType::Rules,
                page: Some(1),
                embedding: None,
            },
        ];
        store.insert(&chunks).await.unwrap();

        let results = store.query(&unit_embedding(0), 10).await.unwrap();

        assert_eq!(
            results.len(),
            3,
            "chunk without embedding should be filtered out at insert"
        );

        assert_eq!(results[0].chunk.id, "rules-1");
        assert!(
            results[0].score > results[1].score,
            "exact-match chunk should rank above orthogonal ones"
        );

        let top = &results[0].chunk;
        assert_eq!(top.text, "rules text");
        assert_eq!(top.game, "Pandemic");
        assert!(matches!(top.doc_type, DocType::Rules));
        assert_eq!(top.page, Some(3));
        assert!(
            top.embedding.is_none(),
            "embedding should be projected out before deserialization"
        );

        let by_id: HashMap<&str, &Chunk> = results
            .iter()
            .map(|r| (r.chunk.id.as_str(), &r.chunk))
            .collect();
        assert!(matches!(by_id["ref-1"].doc_type, DocType::Reference));
        assert!(matches!(by_id["faq-1"].doc_type, DocType::Faq));
        assert_eq!(by_id["ref-1"].page, None);
    }

    #[tokio::test]
    async fn data_persists_across_reconnects() {
        let dir = TempDir::new().unwrap();

        {
            let store = LanceStore::connect(dir.path()).await.unwrap();
            store
                .insert(&[Chunk {
                    id: "a".into(),
                    text: "text a".into(),
                    game: "Pandemic".into(),
                    doc_type: DocType::Rules,
                    page: Some(1),
                    embedding: Some(unit_embedding(0)),
                }])
                .await
                .unwrap();
        }

        let store = LanceStore::connect(dir.path()).await.unwrap();
        let results = store.query(&unit_embedding(0), 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "a");
    }
}
