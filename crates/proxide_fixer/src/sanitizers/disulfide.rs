use crate::models::Topology;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DisulfideError {
    #[error("disulfide detection failed: {0}")]
    General(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Disulfide {
    pub chain_a: String,
    pub res_a: i32,
    pub chain_b: String,
    pub res_b: i32,
    pub distance: f32,
}

pub struct DisulfideSanitizer<'a> {
    topology: &'a mut Topology,
    threshold: f32,
}

impl<'a> DisulfideSanitizer<'a> {
    pub fn new(topology: &'a mut Topology) -> Self {
        Self {
            topology,
            threshold: 2.5,
        }
    }

    pub fn with_threshold(topology: &'a mut Topology, threshold: f32) -> Self {
        Self {
            topology,
            threshold,
        }
    }

    /// Detect disulfide bonds without mutation.
    /// Returns a Vec of detected Disulfide candidates (unsorted, unfiltered by 1:1 pairing).
    pub fn detect(&self) -> Vec<Disulfide> {
        let mut sg_residues = Vec::new();

        // Step 1: Collect all CYS (and CYH/CYM variants) residues with SG atom coords
        for chain in &self.topology.chains {
            for residue in &chain.residues {
                let is_cys_variant = residue.name == "CYS"
                    || residue.name == "CYH"
                    || residue.name == "CYM"
                    || residue.name == "CYX";

                if is_cys_variant {
                    if let Some(sg_atom) = residue.atoms.iter().find(|a| a.name == "SG") {
                        sg_residues.push((chain.id.clone(), residue.res_id, sg_atom.coords));
                    }
                }
            }
        }

        // Step 2: Compute pairwise distances for all SG pairs
        let mut candidates = Vec::new();
        for i in 0..sg_residues.len() {
            for j in (i + 1)..sg_residues.len() {
                let (chain_a, res_a, coords_a) = &sg_residues[i];
                let (chain_b, res_b, coords_b) = &sg_residues[j];

                let distance = euclidean_distance(coords_a, coords_b);
                if distance <= self.threshold {
                    candidates.push(Disulfide {
                        chain_a: chain_a.clone(),
                        res_a: *res_a,
                        chain_b: chain_b.clone(),
                        res_b: *res_b,
                        distance,
                    });
                }
            }
        }

        // Step 3: Greedy 1:1 pairing
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        let mut bonded_residues = std::collections::HashSet::new();
        let mut result = Vec::new();

        for candidate in candidates {
            let key_a = (candidate.chain_a.clone(), candidate.res_a);
            let key_b = (candidate.chain_b.clone(), candidate.res_b);

            if !bonded_residues.contains(&key_a) && !bonded_residues.contains(&key_b) {
                bonded_residues.insert(key_a);
                bonded_residues.insert(key_b);
                result.push(candidate);
            }
        }

        result
    }

    /// Detect disulfide bonds and apply mutations to the topology.
    /// Returns the detected and bonded Disulfides.
    pub fn run(&mut self) -> Result<Vec<Disulfide>, DisulfideError> {
        let disulfides = self.detect();

        for disulfide in &disulfides {
            // Find and rename residues: CYS -> CYX
            for chain in self.topology.chains.iter_mut() {
                if chain.id == disulfide.chain_a {
                    if let Some(residue) = chain
                        .residues
                        .iter_mut()
                        .find(|r| r.res_id == disulfide.res_a)
                    {
                        residue.name = "CYX".to_string();
                        // Remove HG or HG1 if present
                        residue.atoms.retain(|a| a.name != "HG" && a.name != "HG1");
                    }
                }
                if chain.id == disulfide.chain_b {
                    if let Some(residue) = chain
                        .residues
                        .iter_mut()
                        .find(|r| r.res_id == disulfide.res_b)
                    {
                        residue.name = "CYX".to_string();
                        // Remove HG or HG1 if present
                        residue.atoms.retain(|a| a.name != "HG" && a.name != "HG1");
                    }
                }
            }

            // Log the bond
            log::info!(
                "disulfide {}:{}-{}:{} Sγ-Sγ {:.2} Å -> CYX",
                disulfide.chain_a,
                disulfide.res_a,
                disulfide.chain_b,
                disulfide.res_b,
                disulfide.distance
            );
        }

        Ok(disulfides)
    }
}

fn euclidean_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue, Topology};

    #[test]
    fn test_bonded_pair_detected_and_renamed() {
        // Two CYS residues with SG atoms 2.04 Å apart
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 6,
                        insertion_code: ' ',
                        atoms: vec![
                            Atom {
                                name: "CA".to_string(),
                                element: "C".to_string(),
                                coords: [0.0, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 1,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                            Atom {
                                name: "SG".to_string(),
                                element: "S".to_string(),
                                coords: [0.0, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 2,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                            Atom {
                                name: "HG".to_string(),
                                element: "H".to_string(),
                                coords: [0.5, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 3,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                        ],
                    },
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 127,
                        insertion_code: ' ',
                        atoms: vec![
                            Atom {
                                name: "CA".to_string(),
                                element: "C".to_string(),
                                coords: [5.0, 5.0, 5.0],
                                alt_loc: ' ',
                                serial: 4,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                            Atom {
                                name: "SG".to_string(),
                                element: "S".to_string(),
                                coords: [2.04, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 5,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                            Atom {
                                name: "HG".to_string(),
                                element: "H".to_string(),
                                coords: [2.54, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 6,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                        ],
                    },
                ],
            }],
        };

        let mut sanitizer = DisulfideSanitizer::new(&mut topology);
        let disulfides = sanitizer.run().unwrap();

        // Should have detected and bonded 1 disulfide
        assert_eq!(disulfides.len(), 1);
        assert_eq!(disulfides[0].chain_a, "A");
        assert_eq!(disulfides[0].res_a, 6);
        assert_eq!(disulfides[0].chain_b, "A");
        assert_eq!(disulfides[0].res_b, 127);
        assert!((disulfides[0].distance - 2.04).abs() < 0.01);

        // Both residues should be renamed to CYX
        assert_eq!(topology.chains[0].residues[0].name, "CYX");
        assert_eq!(topology.chains[0].residues[1].name, "CYX");

        // HG atoms should be removed
        assert!(!topology.chains[0].residues[0]
            .atoms
            .iter()
            .any(|a| a.name == "HG" || a.name == "HG1"));
        assert!(!topology.chains[0].residues[1]
            .atoms
            .iter()
            .any(|a| a.name == "HG" || a.name == "HG1"));
    }

    #[test]
    fn test_non_bonded_pair_not_detected() {
        // Two CYS residues with SG atoms 4.0 Å apart (beyond 2.5 threshold)
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 6,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 127,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [4.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        };

        let sanitizer = DisulfideSanitizer::new(&mut topology);
        let disulfides = sanitizer.detect();

        // Should not detect any disulfides (distance > threshold)
        assert_eq!(disulfides.len(), 0);
    }

    #[test]
    fn test_boundary_threshold_2_5() {
        // SG atoms exactly at 2.5 Å (should be detected)
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [2.5, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        };

        let sanitizer = DisulfideSanitizer::new(&mut topology);
        let disulfides = sanitizer.detect();
        assert_eq!(disulfides.len(), 1);
    }

    #[test]
    fn test_boundary_threshold_2_51() {
        // SG atoms at 2.51 Å (just over 2.5 threshold, should NOT be detected)
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [2.51, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        };

        let sanitizer = DisulfideSanitizer::new(&mut topology);
        let disulfides = sanitizer.detect();
        assert_eq!(disulfides.len(), 0);
    }

    #[test]
    fn test_greedy_1_1_pairing_three_cys_cluster() {
        // Three CYS residues: form closest pair, leave third unpaired
        // CYS1 at (0,0,0), CYS2 at (1.5,0,0), CYS3 at (2.0,0,0)
        // Distance 1->2: 1.5, Distance 2->3: 0.5, Distance 1->3: 2.0
        // Should pair 2->3 (closest), leave 1 unpaired
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [1.5, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 3,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [2.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 3,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        };

        let sanitizer = DisulfideSanitizer::new(&mut topology);
        let disulfides = sanitizer.detect();

        // Only 1 pair should be bonded (the closest)
        assert_eq!(disulfides.len(), 1);
        // The closest pair should be res 2 and 3
        assert!(
            (disulfides[0].res_a == 2 && disulfides[0].res_b == 3)
                || (disulfides[0].res_a == 3 && disulfides[0].res_b == 2)
        );
    }

    #[test]
    fn test_cys_missing_sg_atom_skipped() {
        // CYS residue without SG atom should be skipped gracefully
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![
                            Atom {
                                name: "CA".to_string(),
                                element: "C".to_string(),
                                coords: [0.0, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 1,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                            // Missing SG atom!
                        ],
                    },
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [2.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        };

        let sanitizer = DisulfideSanitizer::new(&mut topology);
        // Should not panic; just skip the CYS without SG
        let disulfides = sanitizer.detect();
        assert_eq!(disulfides.len(), 0);
    }

    #[test]
    fn test_cys_variant_cyh_detected() {
        // CYH (protonated cysteine) should be detected
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "CYH".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "SG".to_string(),
                            element: "S".to_string(),
                            coords: [2.04, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        };

        let sanitizer = DisulfideSanitizer::new(&mut topology);
        let disulfides = sanitizer.detect();
        assert_eq!(disulfides.len(), 1);
    }

    #[test]
    fn test_hg1_variant_removed() {
        // Some PDB files use HG1 instead of HG; test it gets removed
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![
                            Atom {
                                name: "SG".to_string(),
                                element: "S".to_string(),
                                coords: [0.0, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 1,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                            Atom {
                                name: "HG1".to_string(),
                                element: "H".to_string(),
                                coords: [0.5, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 2,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                        ],
                    },
                    Residue {
                        name: "CYS".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![
                            Atom {
                                name: "SG".to_string(),
                                element: "S".to_string(),
                                coords: [2.04, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 3,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                            Atom {
                                name: "HG1".to_string(),
                                element: "H".to_string(),
                                coords: [2.54, 0.0, 0.0],
                                alt_loc: ' ',
                                serial: 4,
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            },
                        ],
                    },
                ],
            }],
        };

        let mut sanitizer = DisulfideSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        // Both HG1 atoms should be removed
        assert!(!topology.chains[0].residues[0]
            .atoms
            .iter()
            .any(|a| a.name == "HG1"));
        assert!(!topology.chains[0].residues[1]
            .atoms
            .iter()
            .any(|a| a.name == "HG1"));
    }
}
