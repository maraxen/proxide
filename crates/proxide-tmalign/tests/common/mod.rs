//! Shared test helpers for loading USalign sample PDB fixtures.
//!
//! Fixtures are vendored byte-identical from commit 177cc8a of
//! https://github.com/pylelab/USalign (see tests/data/NOTICE-USalign.md).
//! Integration tests that depend on these fixtures fail if they are missing or
//! unparseable, rather than silently skipping.

use proxide_tmalign::CaTrace;

/// Load a USalign sample PDB fixture from tests/data/.
///
/// Panics if the fixture file does not exist or fails to parse.
/// Byte-length assertion provides integrity check against the vendored copies.
pub fn load_usalign_sample(name: &str) -> CaTrace {
    // Byte lengths of vendored fixtures from USalign commit 177cc8a.
    // Verify against: sha256sum tests/data/PDB{1,2}.pdb
    const PDB1_SIZE: u64 = 207441;
    const PDB2_SIZE: u64 = 113076;

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);

    let metadata = std::fs::metadata(&path)
        .unwrap_or_else(|e| panic!("fixture {} not found: {}", path.display(), e));

    let expected_size = match name {
        "PDB1.pdb" => PDB1_SIZE,
        "PDB2.pdb" => PDB2_SIZE,
        _ => panic!("unknown fixture {}", name),
    };

    assert_eq!(
        metadata.len(),
        expected_size,
        "fixture {} size mismatch: expected {}, got {} bytes",
        name,
        expected_size,
        metadata.len()
    );

    proxide_tmalign::load_pdb_ca_trace(&path)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {}", path.display(), e))
}
