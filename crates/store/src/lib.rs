use std::path::Path;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt as _;
use lancedb::query::{ExecutableQuery as _, QueryBase as _};
use lancedb::{DistanceType, Table, connect};
use rag_core::{Chunk, EMBED_DIM, RetrievalResult, VectorStore};

pub use arrow_array;
pub use arrow_schema;

pub struct LanceStore {
    table: Table,
    schema: Arc<Schema>,
}

fn chunks_to_records(schema: Arc<Schema>, rows: &[Chunk]) -> RecordBatch {
    let rows_iter = rows.iter().filter(|chunk| chunk.embedding.is_some());
    let ids = StringArray::from_iter_values(rows_iter.clone().map(|row| row.id.as_str()));
    let texts = StringArray::from_iter_values(rows_iter.clone().map(|row| row.text.as_str()));
    let games = StringArray::from_iter_values(rows_iter.clone().map(|row| row.game.as_str()));
    let sources = StringArray::from_iter_values(rows_iter.clone().map(|row| row.source.as_str()));
    let pages = UInt32Array::from_iter(rows_iter.clone().map(|row| row.page));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        rows_iter.clone().map(|row| {
            Some(
                row.embedding
                    .as_ref()
                    .unwrap()
                    .iter()
                    .copied()
                    .map(Some)
                    .collect::<Vec<_>>(),
            )
        }),
        EMBED_DIM,
    );

    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids),
            Arc::new(texts),
            Arc::new(games),
            Arc::new(sources),
            Arc::new(pages),
            Arc::new(vectors),
        ],
    )
    .unwrap()
}

fn records_to_results(batches: Vec<RecordBatch>) -> Vec<RetrievalResult> {
    let mut results: Vec<RetrievalResult> = Vec::new();

    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let texts = batch
            .column_by_name("text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let games = batch
            .column_by_name("game")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let sources = batch
            .column_by_name("source")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let pages = batch
            .column_by_name("page")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let distances = batch
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            results.push(RetrievalResult {
                chunk: Chunk {
                    id: ids.value(i).to_string(),
                    text: texts.value(i).to_string(),
                    game: games.value(i).to_string(),
                    source: sources.value(i).to_string(),
                    page: (!pages.is_null(i)).then(|| pages.value(i)),
                    embedding: None,
                },
                score: 1.0 - distances.value(i),
            })
        }
    }

    results
}

impl LanceStore {
    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("game", DataType::Utf8, false),
            Field::new("source", DataType::Utf8, false),
            Field::new("page", DataType::UInt32, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBED_DIM,
                ),
                false,
            ),
        ]))
    }
}

impl VectorStore for LanceStore {
    // TODO: actual error handling
    async fn connect(path: &Path) -> Self {
        let connection = connect(path.to_str().unwrap()).execute().await.unwrap();
        let schema = Self::schema();

        let table_name = "rules_chunks";
        let table = match connection.open_table(table_name).execute().await {
            Ok(table) => table,
            Err(lancedb::Error::TableNotFound { .. }) => connection
                .create_empty_table(table_name, schema.clone())
                .execute()
                .await
                .unwrap(),
            Err(unknown) => panic!("Unknown error getting table: {:?}", unknown),
        };

        if table.schema().await.unwrap() != schema {
            panic!("Schema in DB doesn't match!")
        }

        LanceStore { table, schema }
    }

    async fn insert(&self, chunks: &[Chunk]) {
        self.table
            .add(chunks_to_records(self.schema.clone(), chunks))
            .execute()
            .await
            .unwrap();
    }

    async fn query(&self, embedding: &[f32], k: usize) -> Vec<RetrievalResult> {
        let results = self
            .table
            .query()
            .nearest_to(embedding)
            .unwrap()
            .distance_type(DistanceType::Cosine)
            .limit(k)
            .execute()
            .await
            .unwrap();

        let batches: Vec<RecordBatch> = results.try_collect().await.unwrap();

        records_to_results(batches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_chunk(id: &str, text: &str, embedding: Option<Vec<f32>>) -> Chunk {
        Chunk {
            id: id.to_string(),
            text: text.to_string(),
            game: "pandemic".to_string(),
            source: "rules.pdf".to_string(),
            page: Some(1),
            embedding,
        }
    }

    fn unit_embedding(hot_index: usize) -> Vec<f32> {
        let mut e = vec![0.0; EMBED_DIM as usize];
        e[hot_index] = 1.0;
        e
    }

    #[test]
    fn chunks_to_records_filters_chunks_without_embeddings() {
        let schema = LanceStore::schema();
        let chunks = vec![
            sample_chunk("a", "alpha", Some(unit_embedding(0))),
            sample_chunk("b", "beta", None),
            sample_chunk("c", "gamma", Some(unit_embedding(1))),
        ];
        let batch = chunks_to_records(schema, &chunks);

        assert_eq!(batch.num_rows(), 2);
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ids.value(0), "a");
        assert_eq!(ids.value(1), "c");
    }

    #[test]
    fn chunks_to_records_preserves_columns() {
        let schema = LanceStore::schema();
        let mut chunk = sample_chunk("x", "hello", Some(unit_embedding(3)));
        chunk.page = Some(42);
        let batch = chunks_to_records(schema, &[chunk]);

        let texts = batch
            .column_by_name("text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let pages = batch
            .column_by_name("page")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(texts.value(0), "hello");
        assert_eq!(pages.value(0), 42);
    }

    #[test]
    fn records_to_results_extracts_chunks_and_scores() {
        let chunks = vec![
            sample_chunk("x", "first", Some(unit_embedding(0))),
            sample_chunk("y", "second", Some(unit_embedding(1))),
        ];
        let base = chunks_to_records(LanceStore::schema(), &chunks);

        let mut fields: Vec<Field> = base
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new("_distance", DataType::Float32, false));
        let schema_with_distance = Arc::new(Schema::new(fields));
        let mut columns = base.columns().to_vec();
        columns.push(Arc::new(Float32Array::from(vec![0.5_f32, 0.8])));
        let batch = RecordBatch::try_new(schema_with_distance, columns).unwrap();

        let results = records_to_results(vec![batch]);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, "x");
        assert_eq!(results[0].chunk.text, "first");
        assert_eq!(results[0].score, 0.5);
        assert_eq!(results[1].chunk.id, "y");
        assert_eq!(results[1].score, 1.0 - 0.8);
        assert!(results[0].chunk.embedding.is_none());
    }

    #[tokio::test]
    async fn insert_and_query_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = LanceStore::connect(dir.path()).await;

        let chunks = vec![
            sample_chunk("a", "text a", Some(unit_embedding(0))),
            sample_chunk("b", "text b", Some(unit_embedding(1))),
            sample_chunk("c", "text c", Some(unit_embedding(2))),
        ];
        store.insert(&chunks).await;

        let query_vec = unit_embedding(0);
        let results = store.query(&query_vec, 2).await;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, "a");
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn connect_is_idempotent_across_calls() {
        let dir = TempDir::new().unwrap();
        let store = LanceStore::connect(dir.path()).await;
        store
            .insert(&[sample_chunk("a", "text a", Some(unit_embedding(0)))])
            .await;
        drop(store);

        let store2 = LanceStore::connect(dir.path()).await;
        let results = store2.query(&unit_embedding(0), 1).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "a");
    }
}
