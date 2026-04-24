pub struct Chunk {
    pub id: String,
    pub text: String,
    pub game: String,
    pub source: String, // e.g., "pandemic_rules.pdf"
    pub page: Option<u32>,
    pub embedding: Option<Vec<f32>>,
}
