use std::path::Path;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use core::{Chunk, EMBED_DIM, RetrievalResult, VectorStore};
use futures::TryStreamExt as _;
use lancedb::query::{ExecutableQuery as _, QueryBase as _};
use lancedb::{Connection, Table, connect};

pub use arrow_array;
pub use arrow_schema;

pub struct LanceStore {
    connection: Connection,
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
            .downcast_ref::<StringArray>();
    }

    results
}

impl VectorStore for LanceStore {
    async fn connect(path: &Path) -> Self {
        let connection = connect(path.to_str().unwrap()).execute().await.unwrap();
        let table = connection
            .open_table("rules_chunks")
            .execute()
            .await
            .unwrap();
        let schema = Arc::new(Schema::new(vec![
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
        ]));
        LanceStore {
            connection,
            table,
            schema,
        }
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
            .limit(k)
            .execute()
            .await
            .unwrap();

        let batches: Vec<RecordBatch> = results.try_collect().await.unwrap();

        records_to_results(batches)
    }
}
