use thiserror::Error;

// Import the canonical AA list from proxide-confind (now a normal dependency)
use proxide_confind::precondition::CANONICAL_AA_NAMES;
use proxide_rotlib::RotamerLibrary;

use crate::models::Topology;
use crate::repack::{RepackError, SidechainRepacker};

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
    lib: &'a RotamerLibrary,
    shell_radius: f32,
}

impl<'a> Mutator<'a> {
    pub fn new(topology: &'a mut Topology, lib: &'a RotamerLibrary) -> Self {
        Self { topology, lib, shell_radius: 8.0 }
    }

    pub fn with_shell(topology: &'a mut Topology, lib: &'a RotamerLibrary, shell: f32) -> Self {
        Self { topology, lib, shell_radius: shell }
    }

    pub fn apply(&mut self, requests: &[MutationRequest])
        -> Result<MutationReport, MutationError>
    {
        // Step 0: trivial case
        if requests.is_empty() {
            return Ok(MutationReport {
                applied: vec![],
                neighbourhood_repacked: 0,
                disulfides_reevaluated: false,
            });
        }

        // Step 1: validate all requests upfront
        let mut validated_mutations = Vec::new();
        for req in requests {
            // Check target AA is canonical
            if !CANONICAL_AA_NAMES.contains(&req.target_aa.as_str()) {
                return Err(MutationError::InvalidTargetAa(req.target_aa.clone()));
            }

            // Check target AA has rotlib entry
            if self.lib.sidechain_atom_names(&req.target_aa).is_err() {
                return Err(MutationError::InvalidTargetAa(req.target_aa.clone()));
            }

            // Find residue in topology
            let (chain_idx, res_idx) = self
                .topology
                .chains
                .iter()
                .enumerate()
                .find_map(|(c_idx, chain)| {
                    if chain.id == req.chain_id {
                        chain
                            .residues
                            .iter()
                            .enumerate()
                            .find_map(|(r_idx, residue)| {
                                if residue.res_id == req.res_id && residue.insertion_code == req.insertion_code {
                                    Some((c_idx, r_idx))
                                } else {
                                    None
                                }
                            })
                    } else {
                        None
                    }
                })
                .ok_or(MutationError::ResidueNotFound {
                    chain: req.chain_id.clone(),
                    res_id: req.res_id,
                })?;

            // Check for backbone atoms N, CA, C, O
            let residue = &self.topology.chains[chain_idx].residues[res_idx];
            for backbone_atom in &["N", "CA", "C", "O"] {
                if !residue.atoms.iter().any(|a| a.name == *backbone_atom) {
                    return Err(MutationError::MissingBackbone {
                        chain: req.chain_id.clone(),
                        res_id: req.res_id,
                        atom: backbone_atom,
                    });
                }
            }

            validated_mutations.push((req.clone(), chain_idx, res_idx));
        }

        // Step 2: apply name changes
        let mut applied_log = Vec::new();
        for (req, chain_idx, res_idx) in &validated_mutations {
            let old_aa = self.topology.chains[*chain_idx].residues[*res_idx].name.clone();
            self.topology.chains[*chain_idx].residues[*res_idx].name = req.target_aa.clone();
            applied_log.push((
                req.chain_id.clone(),
                req.res_id,
                old_aa,
                req.target_aa.clone(),
            ));
        }

        // Step 3: rebuild sidechains
        let mut repacker = SidechainRepacker::new(&mut *self.topology, self.lib);
        for (req, _chain_idx, _res_idx) in &validated_mutations {
            repacker.rebuild_residue(&req.chain_id, req.res_id, req.insertion_code)?;
        }

        // Step 4: repack neighbourhood
        let targets: Vec<(String, i32, char)> = validated_mutations
            .iter()
            .map(|(req, _, _)| (req.chain_id.clone(), req.res_id, req.insertion_code))
            .collect();
        let neighbourhood_count = targets.len();
        repacker.repack_neighbourhood(&targets, self.shell_radius)?;

        // Step 5: re-evaluate disulfides (CB-CB proxy)
        let mut cys_residues = Vec::new();
        for (chain_idx, chain) in self.topology.chains.iter().enumerate() {
            for (res_idx, residue) in chain.residues.iter().enumerate() {
                if residue.name == "CYS" {
                    cys_residues.push((chain_idx, res_idx));
                }
            }
        }

        // Check CB-CB distances and rename CYS to CYX if within threshold
        for i in 0..cys_residues.len() {
            for j in (i + 1)..cys_residues.len() {
                let (chain_i, res_i) = cys_residues[i];
                let (chain_j, res_j) = cys_residues[j];

                let residue_i = &self.topology.chains[chain_i].residues[res_i];
                let residue_j = &self.topology.chains[chain_j].residues[res_j];

                if let (Some(cb_i), Some(cb_j)) = (
                    residue_i.atoms.iter().find(|a| a.name == "CB"),
                    residue_j.atoms.iter().find(|a| a.name == "CB"),
                ) {
                    let dx = cb_i.coords[0] - cb_j.coords[0];
                    let dy = cb_i.coords[1] - cb_j.coords[1];
                    let dz = cb_i.coords[2] - cb_j.coords[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    let dist = dist_sq.sqrt();

                    // Threshold: 4.5 Å CB-CB distance
                    if dist <= 4.5 {
                        self.topology.chains[chain_i].residues[res_i].name = "CYX".to_string();
                        self.topology.chains[chain_j].residues[res_j].name = "CYX".to_string();
                    }
                }
            }
        }

        // Step 6: build report
        Ok(MutationReport {
            applied: applied_log,
            neighbourhood_repacked: neighbourhood_count,
            disulfides_reevaluated: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue};

    fn make_test_topology() -> Topology {
        // Simple 2-residue topology: A:1 ALA and A:2 ALA
        let residue_1 = Residue {
            name: "ALA".to_string(),
            res_id: 1,
            insertion_code: ' ',
            atoms: vec![
                Atom {
                    name: "N".to_string(),
                    element: "N".to_string(),
                    coords: [0.0, 0.0, 0.0],
                    alt_loc: ' ',
                    serial: 1,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "CA".to_string(),
                    element: "C".to_string(),
                    coords: [1.5, 0.0, 0.0],
                    alt_loc: ' ',
                    serial: 2,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "C".to_string(),
                    element: "C".to_string(),
                    coords: [2.5, 1.0, 0.0],
                    alt_loc: ' ',
                    serial: 3,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "O".to_string(),
                    element: "O".to_string(),
                    coords: [2.5, 2.0, 0.0],
                    alt_loc: ' ',
                    serial: 4,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "CB".to_string(),
                    element: "C".to_string(),
                    coords: [1.5, 1.0, 1.0],
                    alt_loc: ' ',
                    serial: 5,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
            ],
        };

        let residue_2 = Residue {
            name: "ALA".to_string(),
            res_id: 2,
            insertion_code: ' ',
            atoms: vec![
                Atom {
                    name: "N".to_string(),
                    element: "N".to_string(),
                    coords: [3.5, 1.0, 0.0],
                    alt_loc: ' ',
                    serial: 6,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "CA".to_string(),
                    element: "C".to_string(),
                    coords: [4.5, 1.0, 0.0],
                    alt_loc: ' ',
                    serial: 7,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "C".to_string(),
                    element: "C".to_string(),
                    coords: [5.5, 1.0, 0.0],
                    alt_loc: ' ',
                    serial: 8,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "O".to_string(),
                    element: "O".to_string(),
                    coords: [5.5, 2.0, 0.0],
                    alt_loc: ' ',
                    serial: 9,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "CB".to_string(),
                    element: "C".to_string(),
                    coords: [4.5, 1.0, 1.0],
                    alt_loc: ' ',
                    serial: 10,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
            ],
        };

        let chain = Chain {
            id: "A".to_string(),
            residues: vec![residue_1, residue_2],
        };

        Topology {
            chains: vec![chain],
        }
    }

    #[test]
    #[ignore] // Requires rotlib to be loaded
    fn apply_empty_returns_ok() {
        let mut topology = make_test_topology();
        let rotlib_path = std::path::PathBuf::from(
            std::env::var("ROTLIB_PATH")
                .unwrap_or_else(|_| "/home/marielle/repos/mosaist/testfiles/rotlib.bin".to_string()),
        );
        let lib = RotamerLibrary::load(&rotlib_path).expect("rotlib load");
        let mut mutator = Mutator::new(&mut topology, &lib);
        let requests = vec![];
        let result = mutator.apply(&requests);
        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.applied.len(), 0);
        assert_eq!(report.neighbourhood_repacked, 0);
    }

    #[test]
    #[ignore] // Requires rotlib to be loaded
    fn apply_invalid_target_aa_returns_error() {
        let mut topology = make_test_topology();
        let rotlib_path = std::path::PathBuf::from(
            std::env::var("ROTLIB_PATH")
                .unwrap_or_else(|_| "/home/marielle/repos/mosaist/testfiles/rotlib.bin".to_string()),
        );
        let lib = RotamerLibrary::load(&rotlib_path).expect("rotlib load");
        let mut mutator = Mutator::new(&mut topology, &lib);
        let requests = vec![MutationRequest {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
            target_aa: "XYZ".to_string(),
        }];
        let result = mutator.apply(&requests);
        assert!(result.is_err());
        if let Err(MutationError::InvalidTargetAa(aa)) = result {
            assert_eq!(aa, "XYZ");
        } else {
            panic!("Expected InvalidTargetAa error");
        }
    }

    #[test]
    #[ignore] // Requires rotlib to be loaded
    fn apply_missing_residue_returns_error() {
        let mut topology = make_test_topology();
        let rotlib_path = std::path::PathBuf::from(
            std::env::var("ROTLIB_PATH")
                .unwrap_or_else(|_| "/home/marielle/repos/mosaist/testfiles/rotlib.bin".to_string()),
        );
        let lib = RotamerLibrary::load(&rotlib_path).expect("rotlib load");
        let mut mutator = Mutator::new(&mut topology, &lib);
        let requests = vec![MutationRequest {
            chain_id: "A".to_string(),
            res_id: 999,
            insertion_code: ' ',
            target_aa: "LEU".to_string(),
        }];
        let result = mutator.apply(&requests);
        assert!(result.is_err());
        if let Err(MutationError::ResidueNotFound { chain, res_id }) = result {
            assert_eq!(chain, "A");
            assert_eq!(res_id, 999);
        } else {
            panic!("Expected ResidueNotFound error");
        }
    }

    #[test]
    #[ignore] // Requires rotlib to be loaded
    fn apply_missing_backbone_returns_error() {
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![Residue {
                    name: "ALA".to_string(),
                    res_id: 1,
                    insertion_code: ' ',
                    atoms: vec![
                        Atom {
                            name: "CA".to_string(), // Missing N
                            element: "C".to_string(),
                            coords: [1.5, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        },
                        Atom {
                            name: "C".to_string(),
                            element: "C".to_string(),
                            coords: [2.5, 1.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        },
                        Atom {
                            name: "O".to_string(),
                            element: "O".to_string(),
                            coords: [2.5, 2.0, 0.0],
                            alt_loc: ' ',
                            serial: 3,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        },
                    ],
                }],
            }],
        };
        let rotlib_path = std::path::PathBuf::from(
            std::env::var("ROTLIB_PATH")
                .unwrap_or_else(|_| "/home/marielle/repos/mosaist/testfiles/rotlib.bin".to_string()),
        );
        let lib = RotamerLibrary::load(&rotlib_path).expect("rotlib load");
        let mut mutator = Mutator::new(&mut topology, &lib);
        let requests = vec![MutationRequest {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
            target_aa: "LEU".to_string(),
        }];
        let result = mutator.apply(&requests);
        assert!(result.is_err());
        if let Err(MutationError::MissingBackbone { chain, res_id, atom }) = result {
            assert_eq!(chain, "A");
            assert_eq!(res_id, 1);
            assert_eq!(atom, "N");
        } else {
            panic!("Expected MissingBackbone error, got {:?}", result);
        }
    }

    #[test]
    #[ignore] // Requires full rotlib + topology rebuild; integration test
    fn apply_ala_to_leu_rebuilds_sidechain() {
        let mut topology = make_test_topology();
        let rotlib_path = std::path::PathBuf::from(
            std::env::var("ROTLIB_PATH")
                .unwrap_or_else(|_| "/home/marielle/repos/mosaist/testfiles/rotlib.bin".to_string()),
        );
        let lib = RotamerLibrary::load(&rotlib_path).expect("rotlib load");
        let mut mutator = Mutator::new(&mut topology, &lib);
        let requests = vec![MutationRequest {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
            target_aa: "LEU".to_string(),
        }];
        let result = mutator.apply(&requests);
        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.applied.len(), 1);
        assert_eq!(report.applied[0].2, "ALA"); // from_aa
        assert_eq!(report.applied[0].3, "LEU"); // to_aa
        // Check that residue name was updated
        let residue = &topology.chains[0].residues[0];
        assert_eq!(residue.name, "LEU");
    }

    #[test]
    fn canonical_aa_names_accessible() {
        assert!(CANONICAL_AA_NAMES.contains(&"ALA"));
        assert!(CANONICAL_AA_NAMES.contains(&"TYR"));
        assert!(CANONICAL_AA_NAMES.contains(&"CYS"));
        assert!(CANONICAL_AA_NAMES.contains(&"CYX"));
        assert!(!CANONICAL_AA_NAMES.contains(&"NOTANAA"));
    }

    #[test]
    fn mutation_error_display() {
        let err = MutationError::InvalidTargetAa("XYZ".to_string());
        assert!(err.to_string().contains("XYZ"));
    }
}
