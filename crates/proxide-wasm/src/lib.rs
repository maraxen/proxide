use wasm_bindgen::prelude::*;

pub mod gaff2;
pub mod pdb;
pub mod smiles;

pub use gaff2::{assign_params, AngleRecord, AtomRecord, BondRecord, ParamSet, TorsionRecord};
pub use pdb::parse_pdb as parse_pdb_internal;
pub use smiles::{BondOrder, WasmAtom, WasmBond, WasmMol};

/// Parse a SMILES string into a molecular graph.
/// Returns the WasmMol as a JS object (JSON-serialized), or throws on error.
#[wasm_bindgen]
pub fn parse_smiles(smiles: &str) -> Result<JsValue, JsValue> {
    let mol = smiles::parse(smiles).map_err(|e| JsValue::from_str(&e))?;
    serde_wasm_bindgen::to_value(&mol)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Parse PDB text (as a string) into a molecular graph.
#[wasm_bindgen]
pub fn parse_pdb(text: &str) -> Result<JsValue, JsValue> {
    let mol = pdb::parse_pdb(text).map_err(|e| JsValue::from_str(&e))?;
    serde_wasm_bindgen::to_value(&mol)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Assign GAFF2 bonded parameters to a molecule (previously parsed by parse_smiles or parse_pdb).
/// mol_js: a WasmMol JS object (from parse_smiles or parse_pdb).
/// Returns ParamSet as a JSON string.
#[wasm_bindgen]
pub fn assign_params_js(mol_js: JsValue) -> Result<String, JsValue> {
    let mol: WasmMol = serde_wasm_bindgen::from_value(mol_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let params = gaff2::assign_params(&mol);
    serde_json::to_string(&params).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Combined pipeline: parse SMILES + assign params, return JSON string.
#[wasm_bindgen]
pub fn smiles_to_params(smiles: &str) -> Result<String, JsValue> {
    let mol = smiles::parse(smiles).map_err(|e| JsValue::from_str(&e))?;
    let params = gaff2::assign_params(&mol);
    serde_json::to_string(&params).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Return the crate version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Set the number of threads for parallel iterators.
/// Call before any parallel work: `init_parallel(navigator.hardwareConcurrency)`.
#[wasm_bindgen]
pub fn init_parallel(num_threads: usize) {
    proxide_parallel_rt::set_num_threads(num_threads);
}
