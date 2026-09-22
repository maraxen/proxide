#![allow(dead_code)]

use proxide_confind::coords::extract_f64_backbone;
use proxide_core::processing::residues::ProcessedStructure;
use proxide_fixer::models::Topology;
use std::path::PathBuf;

/// Returns the path to a committed fixture file under tests/data/
pub fn fixture(name: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .join("tests")
        .join("data")
        .join(name)
}

/// Load a PDB file and parse it into a Topology.
/// Panics if file does not exist, parsing fails, or ATOM/HETATM line count
/// does not match the parsed number of atoms.
pub fn load_topology(name: &str) -> Topology {
    let path = fixture(name);
    if !path.exists() {
        panic!("Fixture {} not found at {:?}", name, path);
    }

    // Read the file to count ATOM/HETATM lines
    let file_content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", name, e));
    let line_count = file_content
        .lines()
        .filter(|line| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .count();

    // Parse the PDB file
    match proxide_io::formats::pdb::parse_pdb_file(&path) {
        Ok((raw_data, _model_ids)) => {
            let num_atoms = raw_data.num_atoms;
            if num_atoms != line_count {
                panic!(
                    "Fixture {} atom count mismatch: file has {} ATOM/HETATM lines but parsed {} atoms",
                    name, line_count, num_atoms
                );
            }
            Topology::from_raw_atom_data(&raw_data)
        }
        Err(e) => panic!("Failed to parse fixture {}: {}", name, e),
    }
}

/// Load a PDB file into a ProcessedStructure (which includes residue processing).
/// Panics if file does not exist, parsing fails, or ATOM/HETATM line count
/// does not match the parsed number of atoms.
pub fn load_processed(name: &str) -> ProcessedStructure {
    let path = fixture(name);
    if !path.exists() {
        panic!("Fixture {} not found at {:?}", name, path);
    }

    // Read the file to count ATOM/HETATM lines
    let file_content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", name, e));
    let line_count = file_content
        .lines()
        .filter(|line| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .count();

    // Parse the PDB file
    let (raw_data, _model_ids) = match proxide_io::formats::pdb::parse_pdb_file(&path) {
        Ok((raw, models)) => {
            let num_atoms = raw.num_atoms;
            if num_atoms != line_count {
                panic!(
                    "Fixture {} atom count mismatch: file has {} ATOM/HETATM lines but parsed {} atoms",
                    name, line_count, num_atoms
                );
            }
            (raw, models)
        }
        Err(e) => panic!("Failed to parse fixture {}: {}", name, e),
    };

    match ProcessedStructure::from_raw(raw_data) {
        Ok(ps) => ps,
        Err(e) => panic!("Failed to process fixture {}: {}", name, e),
    }
}

/// Load a PDB file into a ProteinBackbone for precondition checking.
/// Panics if file does not exist, parsing fails, or backbone extraction fails.
pub fn load_backbone(name: &str) -> proxide_confind::coords::ProteinBackbone {
    let processed = load_processed(name);
    match extract_f64_backbone(&processed) {
        Ok(bb) => bb,
        Err(e) => panic!("Failed to extract backbone from fixture {}: {}", name, e),
    }
}

/// Load a PDB file and parse it into a Topology (Option variant for compatibility).
/// Returns None if file does not exist or parsing fails.
pub fn load_topology_opt(name: &str) -> Option<Topology> {
    let path = fixture(name);
    if !path.exists() {
        return None;
    }

    match proxide_io::formats::pdb::parse_pdb_file(&path) {
        Ok((raw_data, _model_ids)) => Some(Topology::from_raw_atom_data(&raw_data)),
        Err(_) => None,
    }
}

/// Helper to get fixture path as a string (useful for integration tests).
pub fn fixture_path_str(name: &str) -> String {
    fixture(name).to_string_lossy().to_string()
}
