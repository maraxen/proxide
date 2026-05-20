use wasm_bindgen::prelude::*;

// Re-export key types we'll use in the WASM API.
// Actual API functions will be added in subsequent tasks.
pub use proxide_core::AtomicSystem;

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
