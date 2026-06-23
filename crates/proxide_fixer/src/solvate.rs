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

        if self.config.keep_crystal_waters {
            let (kept, discarded) = triage_waters(self.topology, self.config.water_shell_radius);
            report.waters_kept = kept;
            report.waters_discarded = discarded;
        } else {
            // Discard all waters
            let discarded = discard_all_waters(self.topology);
            report.waters_discarded = discarded;
        }

        // Counterion neutralization — T11.3 fills this in
        // PDBFixer solvation box — T11.4 fills this in

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

/// Check if a residue name matches water (HOH or WAT).
fn is_water_residue(res_name: &str) -> bool {
    res_name == "HOH" || res_name == "WAT"
}

/// Check if an atom is a hydrogen.
fn is_hydrogen(element: &str) -> bool {
    element == "H" || element == "D"
}

/// Compute Euclidean distance between two 3D points.
fn euclidean_distance(p1: &[f32; 3], p2: &[f32; 3]) -> f32 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let dz = p2[2] - p1[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Keep waters within `shell_radius` Å of any protein heavy atom.
/// Returns (kept_count, discarded_count).
/// Mutates topology in-place to remove discarded waters.
fn triage_waters(topology: &mut Topology, shell_radius: f32) -> (usize, usize) {
    // Step 1: Collect positions of all non-water heavy atoms
    let mut protein_atom_coords = Vec::new();
    for chain in &topology.chains {
        for residue in &chain.residues {
            if !is_water_residue(&residue.name) {
                for atom in &residue.atoms {
                    // Only include heavy atoms (not H, not D)
                    if !is_hydrogen(&atom.element) {
                        protein_atom_coords.push(atom.coords);
                    }
                }
            }
        }
    }

    // Step 2: Mark water residues for removal if no protein heavy atom within shell_radius
    let mut residues_to_remove = Vec::new(); // (chain_idx, res_idx)

    for (chain_idx, chain) in topology.chains.iter().enumerate() {
        for (res_idx, residue) in chain.residues.iter().enumerate() {
            if is_water_residue(&residue.name) {
                // Find the oxygen atom (first non-H atom, or any atom if all are H)
                let oxygen_opt = residue
                    .atoms
                    .iter()
                    .find(|a| a.element == "O")
                    .or_else(|| residue.atoms.first());

                if let Some(oxygen) = oxygen_opt {
                    // Compute minimum distance to any protein heavy atom
                    let min_dist = protein_atom_coords
                        .iter()
                        .map(|coord| euclidean_distance(&oxygen.coords, coord))
                        .fold(f32::INFINITY, f32::min);

                    // Discard if min dist > shell_radius
                    if min_dist > shell_radius {
                        residues_to_remove.push((chain_idx, res_idx));
                    }
                }
            }
        }
    }

    // Step 3: Remove marked residues in reverse order (to avoid index shifting)
    let mut discarded_count = 0;
    for (chain_idx, res_idx) in residues_to_remove.iter().rev() {
        topology.chains[*chain_idx].residues.remove(*res_idx);
        discarded_count += 1;
    }

    // Collect remaining water count by scanning all remaining residues
    let mut kept_count = 0;
    for chain in &topology.chains {
        for residue in &chain.residues {
            if is_water_residue(&residue.name) {
                kept_count += 1;
            }
        }
    }

    (kept_count, discarded_count)
}

/// Discard all HOH/WAT residues from the topology.
/// Returns the count of residues removed.
fn discard_all_waters(topology: &mut Topology) -> usize {
    let mut total_removed = 0;

    for chain in &mut topology.chains {
        let initial_len = chain.residues.len();
        chain.residues.retain(|res| !is_water_residue(&res.name));
        total_removed += initial_len - chain.residues.len();
    }

    total_removed
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

    #[test]
    fn triage_keeps_close_water_discards_far() {
        use crate::models::{Atom, Chain, Residue};

        // Build a minimal test topology with:
        // - One protein residue (ALA) with CA at origin (0,0,0)
        // - One water (HOH) with O at (2.0, 0.0, 0.0) — 2.0 Å from CA → keep
        // - One water (HOH) with O at (10.0, 0.0, 0.0) — 10.0 Å from CA → discard
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    // ALA residue
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    // HOH residue — close (keep)
                    Residue {
                        name: "HOH".to_string(),
                        res_id: 101,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "O".to_string(),
                            element: "O".to_string(),
                            coords: [2.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: true,
                        }],
                    },
                    // HOH residue — far (discard)
                    Residue {
                        name: "HOH".to_string(),
                        res_id: 102,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "O".to_string(),
                            element: "O".to_string(),
                            coords: [10.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 3,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: true,
                        }],
                    },
                ],
            }],
        };

        // Run triage with shell_radius=3.5
        let (kept, discarded) = triage_waters(&mut topology, 3.5);

        // Verify counts
        assert_eq!(kept, 1, "Should keep 1 water (close)");
        assert_eq!(discarded, 1, "Should discard 1 water (far)");

        // Verify structure: should have ALA + 1 HOH
        assert_eq!(topology.chains[0].residues.len(), 2);
        assert_eq!(topology.chains[0].residues[0].name, "ALA");
        assert_eq!(topology.chains[0].residues[1].name, "HOH");
        // Verify it's the close water (res_id 101)
        assert_eq!(topology.chains[0].residues[1].res_id, 101);
    }

    #[test]
    fn discard_all_waters_removes_all_hoh_wat() {
        use crate::models::{Atom, Chain, Residue};

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "HOH".to_string(),
                        res_id: 101,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "O".to_string(),
                            element: "O".to_string(),
                            coords: [1.0, 1.0, 1.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: true,
                        }],
                    },
                    Residue {
                        name: "WAT".to_string(),
                        res_id: 102,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "O".to_string(),
                            element: "O".to_string(),
                            coords: [2.0, 2.0, 2.0],
                            alt_loc: ' ',
                            serial: 3,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: true,
                        }],
                    },
                ],
            }],
        };

        let discarded = discard_all_waters(&mut topology);

        assert_eq!(discarded, 2, "Should discard both HOH and WAT");
        assert_eq!(topology.chains[0].residues.len(), 1);
        assert_eq!(topology.chains[0].residues[0].name, "ALA");
    }
}
