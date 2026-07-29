//! Shared test helpers: env-gated skip-if-absent loaders for the
//! USalign-bundled sample PDBs, following the convention in
//! `proxide-confind/tests/common/mod.rs` (env var override, default path,
//! `Option`-returning, skip via early `return` in the `#[test]` body rather
//! than `#[ignore]`).

use proxide_tmalign::CaTrace;

/// Locate `~/repos/USalign` (or `$USALIGN_REPO` if set) and load one of its
/// bundled sample PDB files as a [`CaTrace`]. Returns `None` if the repo or
/// file isn't present locally.
pub fn load_usalign_sample(name: &str) -> Option<CaTrace> {
    let base = std::env::var("USALIGN_REPO")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            std::path::PathBuf::from(home).join("repos/USalign")
        });
    let path = base.join(name);
    if !path.exists() {
        return None;
    }
    proxide_tmalign::load_pdb_ca_trace(&path).ok()
}
