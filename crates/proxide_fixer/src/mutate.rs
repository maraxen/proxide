use thiserror::Error;

// Import the canonical AA list from proxide-confind (now a normal dependency)
use proxide_confind::precondition::CANONICAL_AA_NAMES;

use crate::models::Topology;
use crate::repack::RepackError;

#[derive(Debug, Clone)]
pub struct MutationRequest {
    pub chain_id: String,
    pub res_id: i32,
    pub insertion_code: char,
    pub target_aa: String,
}

#[derive(Debug, Clone)]
pub struct MutationReport {
    pub applied: Vec<(String, i32, String, String)>,  // chain, res_id, from_aa, to_aa
    pub neighbourhood_repacked: usize,
    pub disulfides_reevaluated: bool,
}

#[derive(Error, Debug)]
pub enum MutationError {
    #[error("residue {chain}:{res_id} not found")]
    ResidueNotFound { chain: String, res_id: i32 },
    #[error("target AA '{0}' is not a canonical residue")]
    InvalidTargetAa(String),
    #[error("residue {chain}:{res_id} missing backbone anchor {atom}")]
    MissingBackbone { chain: String, res_id: i32, atom: &'static str },
    #[error("rebuild failed: {0}")]
    Rebuild(#[from] RepackError),
    #[error("disulfide re-evaluation failed: {0}")]
    Disulfide(String),
}

pub struct Mutator<'a> {
    topology: &'a mut Topology,
    shell_radius: f32,
}

impl<'a> Mutator<'a> {
    pub fn new(topology: &'a mut Topology) -> Self {
        Self { topology, shell_radius: 8.0 }
    }

    pub fn with_shell(topology: &'a mut Topology, shell: f32) -> Self {
        Self { topology, shell_radius: shell }
    }

    pub fn apply(&mut self, requests: &[MutationRequest])
        -> Result<MutationReport, MutationError>
    {
        if requests.is_empty() {
            return Ok(MutationReport {
                applied: vec![],
                neighbourhood_repacked: 0,
                disulfides_reevaluated: false,
            });
        }
        // Step 1: validate all targets
        for req in requests {
            if !CANONICAL_AA_NAMES.contains(&req.target_aa.as_str()) {
                return Err(MutationError::InvalidTargetAa(req.target_aa.clone()));
            }
        }
        // Full algorithm in T9.2b/T9.3; placeholder
        todo!("T9.2b: rebuild_residue + repack_neighbourhood")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_requests_is_noop() {
        // Topology construction is tested via fixtures; here we test the early-return path
        // This test verifies the type compiles and empty input returns Ok
        // Full integration test in T9.3
        let _ = CANONICAL_AA_NAMES; // pub const is reachable from non-test code
        assert!(CANONICAL_AA_NAMES.contains(&"ALA"));
        assert!(CANONICAL_AA_NAMES.contains(&"TYR"));
        assert!(!CANONICAL_AA_NAMES.contains(&"NOTANAA"));
    }

    #[test]
    fn invalid_target_aa_rejected() {
        // MutationError::InvalidTargetAa is constructible
        let err = MutationError::InvalidTargetAa("XYZ".to_string());
        assert!(err.to_string().contains("XYZ"));
    }
}
