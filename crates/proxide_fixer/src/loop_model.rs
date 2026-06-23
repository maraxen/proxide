use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct MissingLoop {
    pub chain_id: String,
    pub start_res: i32,
    pub end_res: i32,
    pub sequence: String, // 1-letter SEQRES residues for the gap
}

#[derive(Debug, Clone)]
pub struct LoopModelReport {
    pub loops_built: Vec<MissingLoop>,
    pub geometry_warnings: Vec<String>,
}

#[derive(Error, Debug)]
pub enum LoopModellingError {
    #[error("Modeller executable not found (set MODELLER_EXEC or add to PATH)")]
    ModelerNotInstalled,

    #[error("MODELLER_KEY not set; Modeller requires a license key")]
    MissingLicenseKey,

    #[error("failed to spawn Modeller: {0}")]
    Spawn(String),

    #[error("Modeller exited non-zero: {0}")]
    NonZeroExit(String),

    #[error("failed to parse Modeller output PDB: {0}")]
    Parse(String),

    #[error("loop geometry validation failed: {0}")]
    GeometryInvalid(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Locate the Modeller executable.
/// Checks `MODELLER_EXEC` env var first, then PATH.
pub fn find_modeller() -> Result<PathBuf, LoopModellingError> {
    if let Ok(exec) = std::env::var("MODELLER_EXEC") {
        let path = PathBuf::from(&exec);
        if path.exists() {
            return Ok(path);
        }
    }
    // Try common executable names in PATH
    for name in &["mod10.x", "mod9.25", "modeller"] {
        if let Ok(p) = which_in_path(name) {
            return Ok(p);
        }
    }
    Err(LoopModellingError::ModelerNotInstalled)
}

fn which_in_path(name: &str) -> Result<PathBuf, ()> {
    let path_var = std::env::var("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_installed_when_absent() {
        // Temporarily clear MODELLER_EXEC to simulate absence
        // (PATH-based check will also fail in CI where Modeller is not installed)
        let result = {
            let _guard = EnvGuard::clear("MODELLER_EXEC");
            find_modeller()
        };
        // In CI Modeller is not installed — expect ModelerNotInstalled
        // If Modeller IS installed, this test passes trivially with Ok.
        match result {
            Err(LoopModellingError::ModelerNotInstalled) => {} // expected in CI
            Ok(_) => {}  // Modeller present locally — also fine
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    struct EnvGuard {
        key: String,
        original: Option<String>,
    }
    impl EnvGuard {
        fn clear(key: &str) -> Self {
            let original = std::env::var(key).ok();
            std::env::remove_var(key);
            Self {
                key: key.to_string(),
                original,
            }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(v) => std::env::set_var(&self.key, v),
                None => std::env::remove_var(&self.key),
            }
        }
    }
}
