use ingest::chunk;

fn main() {
    let test_chunk = core::Chunk {
        id: String::from("1"),
        text: String::from("You need to defeat all the diseases"),
        game: String::from("Pandemic"),
        source: String::from("pandemic_rules.pdf"),
        page: None,
        embedding: None,
    };
    println!("Hello world! {}", test_chunk.text);

    chunk(&"./data/pdfs/pandemic.txt".to_string());
}
