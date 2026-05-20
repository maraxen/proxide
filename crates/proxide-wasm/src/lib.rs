use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn hello() -> String {
    "proxide-wasm ok".to_string()
}
