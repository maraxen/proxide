use thiserror::Error;
use crate::models::Topology;

// Constants declared here (not hidden in algorithm) — these are test-assertion targets.
pub const ESP_GRID_SPACING_ANGSTROM: f32 = 1.0;
pub const MIN_ION_PROTEIN_SEPARATION: f32 = 4.0;
pub const MIN_ION_ION_SEPARATION: f32 = 6.0;
pub const ION_ION_SEPARATION_FLOOR: f32 = 4.0;

#[derive(Debug, Clone)]
pub struct SolvationConfig {
    pub keep_crystal_waters: bool,
    pub water_shell_radius: f32,
    pub neutralize: bool,
    pub build_solvation_box: bool,
    pub box_padding: f32,
}

impl Default for SolvationConfig {
    fn default() -> Self {
        Self {
            keep_crystal_waters: true,
            water_shell_radius: 3.5,
            neutralize: true,
            build_solvation_box: false,
            box_padding: 10.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SolvationReport {
    pub waters_kept: usize,
    pub waters_discarded: usize,
    pub na_added: usize,
    pub cl_added: usize,
    pub net_charge_before: f32,
    pub solvation_box_built: bool,
}

#[derive(Error, Debug)]
pub enum SolvationError {
    #[error("PDBFixer executable not found (set PDBFIXER_EXEC or add to PATH)")]
    PdbFixerNotInstalled,
    #[error("failed to spawn PDBFixer: {0}")]
    Spawn(String),
    #[error("PDBFixer exited non-zero: {0}")]
    NonZeroExit(String),
    #[error("failed to parse PDBFixer output: {0}")]
    Parse(String),
    #[error("ion placement infeasible: needed {needed} ions but could only place {placed} meeting separation constraints")]
    IonPlacementInfeasible { needed: usize, placed: usize },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct Solvator<'a> {
    topology: &'a mut Topology,
    config: SolvationConfig,
}

impl<'a> Solvator<'a> {
    pub fn new(topology: &'a mut Topology, config: SolvationConfig) -> Self {
        Self { topology, config }
    }

    pub fn run(&mut self) -> Result<SolvationReport, SolvationError> {
        let mut report = SolvationReport::default();
        report.net_charge_before = net_charge(self.topology);
        // Algorithm implemented in T11.2 (triage) and T11.3 (counterions)
        Ok(report)
    }
}

/// Compute net formal charge from residue protonation states.
/// ARG/LYS/HIP: +1. ASP/GLU: -1. All others: 0.
pub fn net_charge(topology: &Topology) -> f32 {
    // Full implementation in T11.3
    let _ = topology;
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let cfg = SolvationConfig::default();
        assert_eq!(cfg.water_shell_radius, 3.5);
        assert!(cfg.neutralize);
        assert!(!cfg.build_solvation_box);
        assert!(cfg.keep_crystal_waters);
    }

    #[test]
    fn constants_declared() {
        assert_eq!(ESP_GRID_SPACING_ANGSTROM, 1.0);
        assert_eq!(MIN_ION_PROTEIN_SEPARATION, 4.0);
        assert_eq!(MIN_ION_ION_SEPARATION, 6.0);
        assert_eq!(ION_ION_SEPARATION_FLOOR, 4.0);
    }
}
