fn main() {
    let chunk = core::Chunk {
        id: String::from("1"),
        text: String::from("You need to defeat all the diseases"),
        game: String::from("Pandemic"),
        source: String::from("pandemic_rules.pdf"),
        page: None,
        embedding: None,
    };
    println!("Hello world! {}", chunk.text);
}
