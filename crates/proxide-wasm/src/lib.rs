use wasm_bindgen::prelude::*;

// Re-export key types we'll use in the WASM API.
// Actual API functions will be added in subsequent tasks.
pub use proxide_core::AtomicSystem;

pub mod smiles;
pub use smiles::{WasmAtom, WasmBond, WasmMol, BondOrder};

pub mod pdb;
pub use pdb::parse_pdb;

pub mod gaff2;
pub use gaff2::{assign_params, ParamSet, BondRecord, AngleRecord, TorsionRecord, AtomRecord};

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
