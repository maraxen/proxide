use std::env;
use proxide_wasm::smiles;
use proxide_wasm::gaff2;

fn main() {
    let smiles_str = env::args()
        .nth(1)
        .expect("usage: param_cli <smiles>");

    let mol = smiles::parse(&smiles_str)
        .unwrap_or_else(|e| {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        });

    let params = gaff2::assign_params(&mol);

    match serde_json::to_string_pretty(&params) {
        Ok(json) => println!("{}", json),
        Err(e) => {
            eprintln!("JSON serialization error: {}", e);
            std::process::exit(1);
        }
    }
}
