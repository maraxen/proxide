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
/// Returns None if file does not exist or parsing fails.
pub fn load_topology(name: &str) -> Option<Topology> {
    let path = fixture(name);
    if !path.exists() {
        return None;
    }

    // Use proxide_io to parse the PDB file
    // parse_pdb_file returns (RawAtomData, Vec<usize>) where Vec<usize> are model indices
    match proxide_io::formats::pdb::parse_pdb_file(&path) {
        Ok((raw_data, _model_ids)) => Some(Topology::from_raw_atom_data(&raw_data)),
        Err(_) => None,
    }
}

/// Helper to get fixture path as a string (useful for integration tests).
pub fn fixture_path_str(name: &str) -> String {
    fixture(name).to_string_lossy().to_string()
}
