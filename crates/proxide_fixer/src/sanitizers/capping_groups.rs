use crate::models::{Topology, Atom, Residue};
use proxide_geometry::geometry::nerf::Nerf;
use thiserror::Error;

/// Error type for capping group operations
#[derive(Error, Debug)]
pub enum CapError {
    #[error("Missing backbone atom {0} in residue")]
    MissingBackboneAtom(String),
    #[error("Failed to cap: {0}")]
    General(String),
}

/// Kind of capping group
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapKind {
    Ace,
    Nme,
}

/// Record of a capping group that was applied
#[derive(Debug, Clone)]
pub struct AppliedCap {
    pub chain: String,
    pub res_id: i32,
    pub kind: CapKind,
}

/// Detects and inserts ACE (acetyl) and NME (N-methyl amide) caps at internal chain breaks.
/// ACE is inserted after the upstream residue, NME is inserted before the downstream residue.
pub struct CapSanitizer<'a> {
    topology: &'a mut Topology,
}

impl<'a> CapSanitizer<'a> {
    pub fn new(topology: &'a mut Topology) -> Self {
        Self { topology }
    }

    /// Run capping on internal breaks. Returns the list of applied caps.
    pub fn run(&mut self, breaks: &[crate::sanitizers::capping::BreakSite]) -> Result<Vec<AppliedCap>, CapError> {
        let mut applied_caps = Vec::new();

        for break_site in breaks {
            // Find chain and residue indices
            let chain_idx = self
                .topology
                .chains
                .iter()
                .position(|c| c.id == break_site.chain)
                .ok_or_else(|| CapError::General("Chain not found".to_string()))?;

            let upstream_res_idx = self.topology.chains[chain_idx]
                .residues
                .iter()
                .position(|r| r.res_id == break_site.res_id_upstream)
                .ok_or_else(|| CapError::General("Upstream residue not found".to_string()))?;

            let downstream_res_idx = self.topology.chains[chain_idx]
                .residues
                .iter()
                .position(|r| r.res_id == break_site.res_id_downstream)
                .ok_or_else(|| CapError::General("Downstream residue not found".to_string()))?;

            // Get the next serial number (max of all atoms in topology + 1)
            let max_serial = self.topology.chains.iter()
                .flat_map(|c| c.residues.iter().flat_map(|r| r.atoms.iter().map(|a| a.serial)))
                .max()
                .unwrap_or(0);

            // Insert ACE after upstream residue
            let ace_res = create_ace_residue(
                &self.topology.chains[chain_idx].residues[upstream_res_idx],
                break_site.res_id_upstream,
                &break_site.chain,
                max_serial + 1,
            )?;
            self.topology.chains[chain_idx].residues.insert(upstream_res_idx + 1, ace_res);
            applied_caps.push(AppliedCap {
                chain: break_site.chain.clone(),
                res_id: break_site.res_id_upstream,
                kind: CapKind::Ace,
            });

            // Get updated max_serial after ACE insertion
            let max_serial = self.topology.chains.iter()
                .flat_map(|c| c.residues.iter().flat_map(|r| r.atoms.iter().map(|a| a.serial)))
                .max()
                .unwrap_or(0);

            // Insert NME before downstream residue (now at downstream_res_idx + 1 due to ACE insertion)
            let nme_res = create_nme_residue(
                &self.topology.chains[chain_idx].residues[downstream_res_idx + 1],
                break_site.res_id_downstream,
                &break_site.chain,
                max_serial + 1,
            )?;
            self.topology.chains[chain_idx].residues.insert(downstream_res_idx + 1, nme_res);
            applied_caps.push(AppliedCap {
                chain: break_site.chain.clone(),
                res_id: break_site.res_id_downstream,
                kind: CapKind::Nme,
            });

            // Re-sort residues by res_id to maintain order
            self.topology.chains[chain_idx].residues.sort_by_key(|r| (r.res_id, r.insertion_code));
        }

        Ok(applied_caps)
    }
}

/// Create an ACE (acetyl) residue capping the C-terminal end of the upstream residue.
/// ACE has three atoms: C (CY), O (OY), CH3 (CAY).
/// Geometry: C-O ~1.25 Å (double bond), C-CH3 ~1.52 Å (single bond), trans omega.
fn create_ace_residue(
    upstream_res: &Residue,
    upstream_res_id: i32,
    _chain: &str,
    starting_serial: i32,
) -> Result<Residue, CapError> {
    let c_atom = upstream_res
        .atoms
        .iter()
        .find(|a| a.name == "C")
        .ok_or(CapError::MissingBackboneAtom("C".to_string()))?
        .clone();

    let o_atom = upstream_res
        .atoms
        .iter()
        .find(|a| a.name == "O")
        .ok_or(CapError::MissingBackboneAtom("O".to_string()))?
        .clone();

    let ca_atom = upstream_res
        .atoms
        .iter()
        .find(|a| a.name == "CA")
        .ok_or(CapError::MissingBackboneAtom("CA".to_string()))?
        .clone();

    // ACE cap: place CY (carbonyl C), OY (oxygen), and CAY (methyl) from upstream backbone
    // CY is bonded to the upstream C-CA-O backbone
    // Use ideal bond lengths: C-C ~1.52 Å (C to CY), C=O ~1.25 Å (CY to OY), C-C ~1.52 Å (CY to CAY)
    // Angles: ~120° sp2/sp3 geometry

    // Place CY extending from upstream C, using N-CA-C angle reference
    let cy_coords = Nerf::place_atom(
        &[o_atom.coords, c_atom.coords, ca_atom.coords],
        1.52, // C-C single bond from upstream C
        120.0, // sp2-like angle
        180.0, // trans to existing O
    );

    // Place OY (oxygen) doubly bonded to CY
    let oy_coords = Nerf::place_atom(
        &[ca_atom.coords, c_atom.coords, cy_coords],
        1.25, // C=O double bond
        123.0, // sp2 angle
        180.0, // trans to CA
    );

    // Place CAY (methyl carbon) bonded to CY
    let cay_coords = Nerf::place_atom(
        &[oy_coords, cy_coords, c_atom.coords],
        1.52, // C-C single bond
        120.0, // sp3-like angle
        0.0, // positioned in remaining space
    );

    let mut atoms = vec![
        Atom {
            name: "CY".to_string(),
            element: "C".to_string(),
            coords: cy_coords,
            alt_loc: ' ',
            serial: starting_serial,
            b_factor: 0.0,
            occupancy: 1.0,
            is_hetatm: false,
        },
        Atom {
            name: "OY".to_string(),
            element: "O".to_string(),
            coords: oy_coords,
            alt_loc: ' ',
            serial: starting_serial + 1,
            b_factor: 0.0,
            occupancy: 1.0,
            is_hetatm: false,
        },
        Atom {
            name: "CAY".to_string(),
            element: "C".to_string(),
            coords: cay_coords,
            alt_loc: ' ',
            serial: starting_serial + 2,
            b_factor: 0.0,
            occupancy: 1.0,
            is_hetatm: false,
        },
    ];

    // Ensure all atoms have finite coordinates
    for atom in &mut atoms {
        if !atom.coords[0].is_finite() || !atom.coords[1].is_finite() || !atom.coords[2].is_finite() {
            atom.coords = c_atom.coords; // fallback to upstream C position
        }
    }

    Ok(Residue {
        name: "ACE".to_string(),
        res_id: upstream_res_id,
        insertion_code: 'A',
        atoms,
    })
}

/// Create an NME (N-methyl amide) residue capping the N-terminal end of the downstream residue.
/// NME has two atoms: N (NT) and CH3 (CAT).
/// Geometry: N-C ~1.33 Å (peptide-like), ideal angle ~120°.
fn create_nme_residue(
    downstream_res: &Residue,
    downstream_res_id: i32,
    _chain: &str,
    starting_serial: i32,
) -> Result<Residue, CapError> {
    let n_atom = downstream_res
        .atoms
        .iter()
        .find(|a| a.name == "N")
        .ok_or(CapError::MissingBackboneAtom("N".to_string()))?
        .clone();

    let ca_atom = downstream_res
        .atoms
        .iter()
        .find(|a| a.name == "CA")
        .ok_or(CapError::MissingBackboneAtom("CA".to_string()))?
        .clone();

    let c_atom = downstream_res
        .atoms
        .iter()
        .find(|a| a.name == "C")
        .ok_or(CapError::MissingBackboneAtom("C".to_string()))?
        .clone();

    // NME cap: place N and CH3 from downstream backbone
    // N is already there, so we place CH3 bonded to N
    // N-C bond ~1.33 Å (sp2), angle ~120°, trans to CA

    let cat_coords = Nerf::place_atom(
        &[c_atom.coords, ca_atom.coords, n_atom.coords],
        1.33, // N-C single bond (peptide-like)
        120.0, // C-N-C angle
        180.0, // trans to CA
    );

    let mut atoms = vec![
        Atom {
            name: "NT".to_string(),
            element: "N".to_string(),
            coords: n_atom.coords,
            alt_loc: ' ',
            serial: starting_serial,
            b_factor: 0.0,
            occupancy: 1.0,
            is_hetatm: false,
        },
        Atom {
            name: "CAT".to_string(),
            element: "C".to_string(),
            coords: cat_coords,
            alt_loc: ' ',
            serial: starting_serial + 1,
            b_factor: 0.0,
            occupancy: 1.0,
            is_hetatm: false,
        },
    ];

    // Ensure finite coordinates
    for atom in &mut atoms {
        if !atom.coords[0].is_finite() || !atom.coords[1].is_finite() || !atom.coords[2].is_finite() {
            atom.coords = n_atom.coords; // fallback
        }
    }

    Ok(Residue {
        name: "NME".to_string(),
        res_id: downstream_res_id,
        insertion_code: 'B',
        atoms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Chain;
    use crate::sanitizers::capping::CappingSanitizer;


    /// Build a minimal backbone residue (N, CA, C, O)
    fn build_backbone_residue(name: &str, res_id: i32, serial_base: i32) -> Residue {
        Residue {
            name: name.to_string(),
            res_id,
            insertion_code: ' ',
            atoms: vec![
                Atom {
                    name: "N".to_string(),
                    element: "N".to_string(),
                    coords: [0.0, 0.0, 0.0],
                    alt_loc: ' ',
                    serial: serial_base,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "CA".to_string(),
                    element: "C".to_string(),
                    coords: [1.458, 0.0, 0.0],
                    alt_loc: ' ',
                    serial: serial_base + 1,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "C".to_string(),
                    element: "C".to_string(),
                    coords: [2.009, 1.391, 0.0],
                    alt_loc: ' ',
                    serial: serial_base + 2,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "O".to_string(),
                    element: "O".to_string(),
                    coords: [1.221, 2.338, 0.0],
                    alt_loc: ' ',
                    serial: serial_base + 3,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
            ],
        }
    }

    #[test]
    fn test_cap_internal_break_creates_ace_and_nme() {
        let res1 = build_backbone_residue("ALA", 1, 1);
        let mut res2 = build_backbone_residue("GLY", 2, 5);
        // Place res2 far away to create a break
        res2.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [15.0, 0.0, 0.0];
        for atom in &mut res2.atoms {
            if atom.name != "N" {
                atom.coords[0] += 15.0 - 1.458;
            }
        }

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2],
            }],
        };

        let capping_sanitizer = CappingSanitizer::new(&mut topology);
        let breaks = capping_sanitizer.detect_breaks();
        assert_eq!(breaks.len(), 1);

        let mut cap_sanitizer = CapSanitizer::new(&mut topology);
        let applied_caps = cap_sanitizer.run(&breaks).unwrap();

        // Should have inserted 2 caps (ACE and NME)
        assert_eq!(applied_caps.len(), 2);
        assert_eq!(applied_caps[0].kind, CapKind::Ace);
        assert_eq!(applied_caps[1].kind, CapKind::Nme);

        // Chain should now have 4 residues (ALA, ACE, GLY, NME)
        assert_eq!(topology.chains[0].residues.len(), 4);

        // Find ACE residue
        let ace_res = topology.chains[0]
            .residues
            .iter()
            .find(|r| r.name == "ACE")
            .unwrap();
        assert_eq!(ace_res.atoms.len(), 3, "ACE should have 3 atoms");
        let ace_atom_names: Vec<_> = ace_res.atoms.iter().map(|a| a.name.clone()).collect();
        assert!(ace_atom_names.contains(&"CY".to_string()));
        assert!(ace_atom_names.contains(&"OY".to_string()));
        assert!(ace_atom_names.contains(&"CAY".to_string()));

        // Find NME residue
        let nme_res = topology.chains[0]
            .residues
            .iter()
            .find(|r| r.name == "NME")
            .unwrap();
        assert_eq!(nme_res.atoms.len(), 2, "NME should have 2 atoms");
        let nme_atom_names: Vec<_> = nme_res.atoms.iter().map(|a| a.name.clone()).collect();
        assert!(nme_atom_names.contains(&"NT".to_string()));
        assert!(nme_atom_names.contains(&"CAT".to_string()));

        // All atoms should have finite coordinates
        for res in &topology.chains[0].residues {
            for atom in &res.atoms {
                assert!(atom.coords[0].is_finite());
                assert!(atom.coords[1].is_finite());
                assert!(atom.coords[2].is_finite());
            }
        }
    }

    #[test]
    fn test_cap_residues_sorted_by_res_id() {
        let res1 = build_backbone_residue("ALA", 1, 1);
        let mut res2 = build_backbone_residue("GLY", 2, 5);
        res2.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [15.0, 0.0, 0.0];
        for atom in &mut res2.atoms {
            if atom.name != "N" {
                atom.coords[0] += 15.0 - 1.458;
            }
        }

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2],
            }],
        };

        let capping_sanitizer = CappingSanitizer::new(&mut topology);
        let breaks = capping_sanitizer.detect_breaks();

        let mut cap_sanitizer = CapSanitizer::new(&mut topology);
        cap_sanitizer.run(&breaks).unwrap();

        // Caps must sit between the flanking residues, in order
        let names: Vec<_> = topology.chains[0].residues.iter().map(|r| r.name.clone()).collect();
        assert_eq!(names, vec!["ALA", "ACE", "NME", "GLY"], "caps must be ordered between flanking residues");
        let keys: Vec<_> = topology.chains[0].residues.iter().map(|r| (r.res_id, r.insertion_code)).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "residues sorted by (res_id, insertion_code)");
    }

    #[test]
    fn test_ace_atoms_have_finite_coords() {
        let res1 = build_backbone_residue("ALA", 1, 1);
        let mut res2 = build_backbone_residue("GLY", 2, 5);
        res2.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [15.0, 0.0, 0.0];
        for atom in &mut res2.atoms {
            if atom.name != "N" {
                atom.coords[0] += 15.0 - 1.458;
            }
        }

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2],
            }],
        };

        let capping_sanitizer = CappingSanitizer::new(&mut topology);
        let breaks = capping_sanitizer.detect_breaks();

        let mut cap_sanitizer = CapSanitizer::new(&mut topology);
        cap_sanitizer.run(&breaks).unwrap();

        // Check ACE atoms
        let ace_res = topology.chains[0]
            .residues
            .iter()
            .find(|r| r.name == "ACE")
            .unwrap();
        for atom in &ace_res.atoms {
            assert!(atom.coords[0].is_finite(), "ACE {} x not finite", atom.name);
            assert!(atom.coords[1].is_finite(), "ACE {} y not finite", atom.name);
            assert!(atom.coords[2].is_finite(), "ACE {} z not finite", atom.name);
            assert_ne!(atom.coords, [0.0, 0.0, 0.0], "ACE {} has placeholder coords", atom.name);
        }
    }

    #[test]
    fn test_nme_atoms_have_finite_coords() {
        let res1 = build_backbone_residue("ALA", 1, 1);
        let mut res2 = build_backbone_residue("GLY", 2, 5);
        res2.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [15.0, 0.0, 0.0];
        for atom in &mut res2.atoms {
            if atom.name != "N" {
                atom.coords[0] += 15.0 - 1.458;
            }
        }

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2],
            }],
        };

        let capping_sanitizer = CappingSanitizer::new(&mut topology);
        let breaks = capping_sanitizer.detect_breaks();

        let mut cap_sanitizer = CapSanitizer::new(&mut topology);
        cap_sanitizer.run(&breaks).unwrap();

        // Check NME atoms
        let nme_res = topology.chains[0]
            .residues
            .iter()
            .find(|r| r.name == "NME")
            .unwrap();
        for atom in &nme_res.atoms {
            assert!(atom.coords[0].is_finite(), "NME {} x not finite", atom.name);
            assert!(atom.coords[1].is_finite(), "NME {} y not finite", atom.name);
            assert!(atom.coords[2].is_finite(), "NME {} z not finite", atom.name);
        }
    }

    #[test]
    fn test_cap_atom_serials_unique() {
        let res1 = build_backbone_residue("ALA", 1, 1);
        let mut res2 = build_backbone_residue("GLY", 2, 5);
        res2.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [15.0, 0.0, 0.0];
        for atom in &mut res2.atoms {
            if atom.name != "N" {
                atom.coords[0] += 15.0 - 1.458;
            }
        }

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2],
            }],
        };

        let capping_sanitizer = CappingSanitizer::new(&mut topology);
        let breaks = capping_sanitizer.detect_breaks();

        let mut cap_sanitizer = CapSanitizer::new(&mut topology);
        cap_sanitizer.run(&breaks).unwrap();

        // Collect all serials
        let mut serials = Vec::new();
        for chain in &topology.chains {
            for res in &chain.residues {
                for atom in &res.atoms {
                    serials.push(atom.serial);
                }
            }
        }

        // Check uniqueness
        let mut sorted_serials = serials.clone();
        sorted_serials.sort();
        sorted_serials.dedup();
        assert_eq!(
            serials.len(),
            sorted_serials.len(),
            "Atom serials should be unique"
        );
    }

    #[test]
    fn test_nme_residue_id_non_consecutive_break() {
        // Regression test: NME should have res_id of downstream residue, not upstream
        // when break_site has non-consecutive res_ids (e.g. upstream=5, downstream=10)
        let res1 = build_backbone_residue("ALA", 5, 1);
        let mut res2 = build_backbone_residue("GLY", 10, 5);
        res2.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [15.0, 0.0, 0.0];
        for atom in &mut res2.atoms {
            if atom.name != "N" {
                atom.coords[0] += 15.0 - 1.458;
            }
        }

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2],
            }],
        };

        let capping_sanitizer = CappingSanitizer::new(&mut topology);
        let breaks = capping_sanitizer.detect_breaks();
        assert_eq!(breaks.len(), 1);

        let mut cap_sanitizer = CapSanitizer::new(&mut topology);
        let applied_caps = cap_sanitizer.run(&breaks).unwrap();

        // ACE should have res_id 5 (upstream)
        let ace_cap = applied_caps.iter().find(|c| c.kind == CapKind::Ace).unwrap();
        assert_eq!(ace_cap.res_id, 5, "ACE should have upstream res_id");

        // NME should have res_id 10 (downstream), not 5
        // THIS SHOULD FAIL WITH CURRENT CODE because NME gets res_id_upstream instead of res_id_downstream
        let nme_cap = applied_caps.iter().find(|c| c.kind == CapKind::Nme).unwrap();
        assert_eq!(nme_cap.res_id, 10, "NME should have downstream res_id (currently fails because code uses upstream)");

        // After sort, residues should be: ALA(5), ACE(5,'A'), NME(10,'B'), GLY(10)
        let ace_res = topology.chains[0]
            .residues
            .iter()
            .find(|r| r.name == "ACE")
            .unwrap();
        assert_eq!(ace_res.res_id, 5, "ACE residue should have res_id 5");

        let nme_res = topology.chains[0]
            .residues
            .iter()
            .find(|r| r.name == "NME")
            .unwrap();
        assert_eq!(nme_res.res_id, 10, "NME residue should have res_id 10 (currently fails because code uses upstream)");
    }

    #[test]
    fn test_ace_cy_atom_not_at_backbone_c() {
        // Regression test: ACE's CY atom should NOT be co-located with upstream backbone C.
        // CY should be placed using NeRF to extend the chain geometry properly.
        let res1 = build_backbone_residue("ALA", 1, 1);
        let mut res2 = build_backbone_residue("GLY", 2, 5);
        res2.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [15.0, 0.0, 0.0];
        for atom in &mut res2.atoms {
            if atom.name != "N" {
                atom.coords[0] += 15.0 - 1.458;
            }
        }

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2],
            }],
        };

        let capping_sanitizer = CappingSanitizer::new(&mut topology);
        let breaks = capping_sanitizer.detect_breaks();

        let mut cap_sanitizer = CapSanitizer::new(&mut topology);
        cap_sanitizer.run(&breaks).unwrap();

        // Find ACE and check CY position
        let ace_res = topology.chains[0]
            .residues
            .iter()
            .find(|r| r.name == "ACE")
            .unwrap();

        let cy_atom = ace_res.atoms.iter().find(|a| a.name == "CY").unwrap();
        let upstream_c_coords = [2.009, 1.391, 0.0]; // From build_backbone_residue

        // CY should NOT be at the same position as upstream C
        assert_ne!(cy_atom.coords, upstream_c_coords, "CY should not be co-located with upstream backbone C");

        // CY should have finite coordinates (not NaN/Inf)
        assert!(cy_atom.coords[0].is_finite(), "CY x coordinate should be finite");
        assert!(cy_atom.coords[1].is_finite(), "CY y coordinate should be finite");
        assert!(cy_atom.coords[2].is_finite(), "CY z coordinate should be finite");

        // CY should be close to upstream C but not identical (reasonable bond distance ~1.5 Å)
        let dx = cy_atom.coords[0] - upstream_c_coords[0];
        let dy = cy_atom.coords[1] - upstream_c_coords[1];
        let dz = cy_atom.coords[2] - upstream_c_coords[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Bond distance should be between 1 Å and 2 Å (reasonable for C-C)
        assert!(distance > 0.8 && distance < 2.0, "CY should be bonded at reasonable distance, got {}", distance);
    }
}
