use wasm_bindgen::prelude::*;

// Re-export key types we'll use in the WASM API.
// Actual API functions will be added in subsequent tasks.
pub use proxide_core::AtomicSystem;

pub mod smiles;
pub use smiles::{WasmAtom, WasmBond, WasmMol, BondOrder};

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
