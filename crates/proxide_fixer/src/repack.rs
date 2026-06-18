use crate::models::{Atom, Residue, Topology, Chain};
use log::warn;
use proxide_rotlib::RotamerLibrary;
use std::collections::HashMap;
use thiserror::Error;

/// Error type for sidechain repacking operations.
#[derive(Error, Debug)]
pub enum RepackError {
    #[error("rotamer library error: {0}")]
    RotlibError(#[from] proxide_rotlib::error::RotlibError),

    #[error("invalid backbone: {0}")]
    InvalidBackbone(String),

    #[error("missing sidechain atoms for {aa} at {chain}:{res_id}")]
    MissingSidechain { chain: String, res_id: i32, aa: String },
}

/// Enum for selecting which residues to repack.
#[derive(Debug, Clone, Copy)]
pub enum RepackMode {
    /// Only fill in missing sidechain atoms (incomplete residues).
    CompleteMissingOnly,
    /// Repack all non-GLY/ALA residues.
    RepackAll,
}

/// Report of sidechain repacking results.
#[derive(Debug, Clone)]
pub struct RepackReport {
    /// Number of residues processed.
    pub residues_processed: usize,
    /// Number of residues successfully repacked.
    pub residues_completed: usize,
    /// Per-residue details: (chain_id, res_id, aa_name, chosen_rotamer_index, clash_penalty).
    pub residue_details: Vec<(String, i32, String, usize, f64)>,
}

/// Greedy SCWRL-lite sidechain repacker over Dunbrack BBDEP2010 rotamer library.
pub struct SidechainRepacker<'a> {
    topology: &'a mut Topology,
    lib: &'a RotamerLibrary,
}

/// Van der Waals radii for common protein atoms (in Ångströms).
/// Reference: Bondi, A. (1964). J. Phys. Chem. 68(3), 441-451.
fn get_element_radius(element: &str) -> f32 {
    match element {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "F" => 1.47,
        "Cl" => 1.75,
        "Br" => 1.85,
        "I" => 1.98,
        _ => 1.70, // Default
    }
}

/// Calculate the clash penalty for a rotamer candidate against already-placed atoms.
/// Uses soft repulsion: penalty accumulates for any pair where distance < r_i + r_j.
fn calculate_clash_penalty(
    candidate_atoms: &[(String, [f32; 3])],
    placed_atoms: &[Atom],
) -> f64 {
    let mut penalty: f64 = 0.0;

    for (cand_name, cand_coords) in candidate_atoms {
        let cand_element = extract_element(cand_name);
        let cand_radius = get_element_radius(&cand_element) as f64;

        for placed_atom in placed_atoms {
            let placed_element = &placed_atom.element;
            let placed_radius = get_element_radius(placed_element) as f64;
            let min_dist = cand_radius + placed_radius;

            let dx = (cand_coords[0] - placed_atom.coords[0]) as f64;
            let dy = (cand_coords[1] - placed_atom.coords[1]) as f64;
            let dz = (cand_coords[2] - placed_atom.coords[2]) as f64;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let dist = dist_sq.sqrt();

            if dist < min_dist {
                let overlap = min_dist - dist;
                // Quadratic repulsion penalty
                penalty += overlap * overlap;
            }
        }
    }

    penalty
}

/// Extract element symbol from atom name (first letter, or first two letters for multi-char).
fn extract_element(atom_name: &str) -> String {
    let name = atom_name.trim();
    if name.is_empty() {
        return "C".to_string();
    }

    // Remove leading digits and positions (common in PDB atom names)
    let trimmed = name.trim_start_matches(char::is_numeric);
    if trimmed.is_empty() {
        return "C".to_string();
    }

    // Take first character, uppercase
    let first_char = trimmed.chars().next().unwrap();
    if first_char.is_alphabetic() {
        let upper = first_char.to_uppercase().to_string();
        // Check if second character suggests a two-letter element
        if let Some(second_char) = trimmed.chars().nth(1) {
            if second_char.is_lowercase() && second_char.is_alphabetic() {
                return format!("{}{}", upper, second_char);
            }
        }
        return upper;
    }

    "C".to_string() // Fallback
}

/// Derive φ (phi) and ψ (psi) angles from backbone N, CA, C atoms of a residue and its neighbors.
/// Returns (phi, psi) where 9999.0 indicates missing angle (e.g., at termini).
fn derive_backbone_angles(
    prev_residue: Option<&Residue>,
    current_residue: &Residue,
    next_residue: Option<&Residue>,
) -> (f64, f64) {
    // phi = angle(prev_C, N, CA, C)
    let phi = if let Some(prev) = prev_residue {
        if let (Some(prev_c), Some(curr_n), Some(curr_ca), Some(curr_c)) = (
            find_atom(prev, "C"),
            find_atom(current_residue, "N"),
            find_atom(current_residue, "CA"),
            find_atom(current_residue, "C"),
        ) {
            dihedral_angle(prev_c, curr_n, curr_ca, curr_c)
        } else {
            9999.0
        }
    } else {
        9999.0 // N-terminus
    };

    // psi = angle(N, CA, C, next_N)
    let psi = if let Some(next) = next_residue {
        if let (Some(curr_n), Some(curr_ca), Some(curr_c), Some(next_n)) = (
            find_atom(current_residue, "N"),
            find_atom(current_residue, "CA"),
            find_atom(current_residue, "C"),
            find_atom(next, "N"),
        ) {
            dihedral_angle(curr_n, curr_ca, curr_c, next_n)
        } else {
            9999.0
        }
    } else {
        9999.0 // C-terminus
    };

    (phi, psi)
}

/// Find an atom by name in a residue.
fn find_atom(residue: &Residue, name: &str) -> Option<[f32; 3]> {
    residue
        .atoms
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.coords)
}

/// Calculate a dihedral angle from four points (in degrees).
fn dihedral_angle(p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], p4: [f32; 3]) -> f64 {
    // Vector from p2 to p1
    let v1 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
    // Vector from p2 to p3
    let v2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
    // Vector from p3 to p4
    let v3 = [p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]];

    // Normals to planes
    let n1 = cross_product(v1, v2);
    let n2 = cross_product(v2, v3);

    // Dihedral angle
    let dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
    let len_n1 = (n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]).sqrt();
    let len_n2 = (n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]).sqrt();

    if len_n1 < 1e-6 || len_n2 < 1e-6 {
        return 9999.0;
    }

    let cos_angle = (dot / (len_n1 * len_n2)).max(-1.0).min(1.0);
    let angle_rad = (cos_angle as f64).acos();
    let pi = std::f32::consts::PI as f64;
    let mut angle_deg = angle_rad * 180.0 / pi;

    // Adjust sign using cross product
    let cross = cross_product(n1, n2);
    let dot_v2 = cross[0] * v2[0] + cross[1] * v2[1] + cross[2] * v2[2];
    if dot_v2 < 0.0 {
        angle_deg = -angle_deg;
    }

    angle_deg as f64
}

/// Compute cross product of two 3D vectors.
fn cross_product(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Check if a residue's sidechain is incomplete (missing some library-defined heavy atoms).
fn is_incomplete(residue: &Residue, lib: &RotamerLibrary) -> bool {
    let aa = &residue.name;

    // GLY and ALA never need sidechain repacking
    if aa == "GLY" || aa == "ALA" {
        return false;
    }

    let lib_names = match lib.sidechain_atom_names(aa) {
        Ok(names) => names,
        Err(_) => return false, // Unknown AA, skip
    };

    let present_names: std::collections::HashSet<_> = residue
        .atoms
        .iter()
        .map(|a| a.name.as_str())
        .collect();

    // Incomplete if any library atom is missing
    lib_names.iter().any(|name| !present_names.contains(name.as_str()))
}

/// Calculate approximate burial order: residues with more neighbors within ~8 Å are more buried.
fn calculate_burial_order(chains: &[Chain]) -> Vec<(usize, usize, usize)> {
    // Collect all residue indices and their neighbor counts
    let mut residues_with_neighbors: Vec<((usize, usize, usize), usize)> = Vec::new();

    for (chain_idx, chain) in chains.iter().enumerate() {
        for (res_idx, residue) in chain.residues.iter().enumerate() {
            let mut neighbor_count = 0;

            for other_chain in chains {
                for other_res in &other_chain.residues {
                    if std::ptr::eq(residue, other_res) {
                        continue;
                    }

                    // Check if any atom pair is within 8 Å
                    for atom1 in &residue.atoms {
                        for atom2 in &other_res.atoms {
                            let dx = atom1.coords[0] - atom2.coords[0];
                            let dy = atom1.coords[1] - atom2.coords[1];
                            let dz = atom1.coords[2] - atom2.coords[2];
                            let dist_sq = dx * dx + dy * dy + dz * dz;
                            if dist_sq < 64.0 {
                                // 8 Å squared
                                neighbor_count += 1;
                                break; // One pair is enough
                            }
                        }
                        if neighbor_count > 0 {
                            break;
                        }
                    }
                }
            }

            residues_with_neighbors.push(((chain_idx, res_idx, 0), neighbor_count));
        }
    }

    // Sort by descending neighbor count
    residues_with_neighbors.sort_by(|a, b| b.1.cmp(&a.1));
    residues_with_neighbors.into_iter().map(|(idx, _)| idx).collect()
}

impl<'a> SidechainRepacker<'a> {
    /// Create a new repacker.
    pub fn new(topology: &'a mut Topology, lib: &'a RotamerLibrary) -> Self {
        Self { topology, lib }
    }

    /// Run the repacking algorithm.
    pub fn complete_and_repack(&mut self, mode: RepackMode) -> Result<RepackReport, RepackError> {
        let mut report = RepackReport {
            residues_processed: 0,
            residues_completed: 0,
            residue_details: Vec::new(),
        };

        // Calculate burial order
        let burial_order = calculate_burial_order(&self.topology.chains);

        // Pre-compute angles for all residues to avoid borrowing issues
        let mut angles_map: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
        for (chain_idx, res_idx, _) in &burial_order {
            let chain = &self.topology.chains[*chain_idx];
            let prev_res = if *res_idx > 0 {
                Some(&chain.residues[res_idx - 1])
            } else {
                None
            };
            let next_res = if res_idx + 1 < chain.residues.len() {
                Some(&chain.residues[res_idx + 1])
            } else {
                None
            };
            let angles = derive_backbone_angles(prev_res, &chain.residues[*res_idx], next_res);
            angles_map.insert((*chain_idx, *res_idx), angles);
        }

        // Pre-collect all atom positions to avoid borrowing issues
        let mut all_atoms_map: HashMap<(usize, usize), Vec<Atom>> = HashMap::new();
        for (chain_idx, chain) in self.topology.chains.iter().enumerate() {
            for (res_idx, residue) in chain.residues.iter().enumerate() {
                all_atoms_map.insert((chain_idx, res_idx), residue.atoms.clone());
            }
        }

        // Process residues in burial order
        for (chain_idx, res_idx, _) in burial_order {
            // Get residue info without mutable borrow first
            let (aa, res_id, chain_id) = {
                let chain = &self.topology.chains[chain_idx];
                let residue = &chain.residues[res_idx];
                (
                    residue.name.clone(),
                    residue.res_id,
                    chain.id.clone(),
                )
            };

            // Skip GLY and ALA unless in RepackAll mode
            if (aa == "GLY" || aa == "ALA") && !matches!(mode, RepackMode::RepackAll) {
                continue;
            }

            // Check if should process
            let should_process = {
                let chain = &self.topology.chains[chain_idx];
                let residue = &chain.residues[res_idx];
                match mode {
                    RepackMode::CompleteMissingOnly => is_incomplete(residue, self.lib),
                    RepackMode::RepackAll => aa != "GLY" && aa != "ALA",
                }
            };

            if !should_process {
                continue;
            }

            report.residues_processed += 1;

            // Get backbone atoms
            let (n_atom, ca_atom, c_atom) = {
                let chain = &self.topology.chains[chain_idx];
                let residue = &chain.residues[res_idx];
                (find_atom(residue, "N"), find_atom(residue, "CA"), find_atom(residue, "C"))
            };

            if n_atom.is_none() || ca_atom.is_none() || c_atom.is_none() {
                warn!(
                    "Skipping {chain_id}:{res_id} {aa}: missing backbone atoms",
                    chain_id = chain_id,
                    res_id = res_id,
                    aa = aa
                );
                continue;
            }

            let n = n_atom.unwrap();
            let ca = ca_atom.unwrap();
            let c = c_atom.unwrap();

            // Get pre-computed phi/psi
            let (phi, psi) = angles_map
                .get(&(chain_idx, res_idx))
                .copied()
                .unwrap_or((9999.0, 9999.0));

            // Get number of rotamers in this bin
            let nr = self.lib.num_rotamers(&aa, phi, psi, false)?;
            if nr == 0 {
                warn!(
                    "No rotamers available for {chain_id}:{res_id} {aa} at ({phi:.1}, {psi:.1})",
                    chain_id = chain_id,
                    res_id = res_id,
                    aa = aa,
                    phi = phi,
                    psi = psi
                );
                continue;
            }

            // Collect all already-placed atoms
            let mut placed_atoms = Vec::new();
            for (other_chain_idx, chain) in self.topology.chains.iter().enumerate() {
                for (other_res_idx, residue) in chain.residues.iter().enumerate() {
                    if (other_chain_idx, other_res_idx) != (chain_idx, res_idx) {
                        for atom in &residue.atoms {
                            placed_atoms.push(atom.clone());
                        }
                    }
                }
            }

            // Try rotamers in descending probability order
            let mut best_rot = 0;
            let mut best_score = f64::NEG_INFINITY;

            for rot_idx in 0..nr {
                // Get rotamer probability
                let prob = self.lib.rotamer_probability(&aa, rot_idx, phi, psi)?;

                // Place the rotamer
                let placed_rot = self.lib.place_rotamer(
                    &aa,
                    phi,
                    psi,
                    rot_idx,
                    false,
                    [n[0] as f64, n[1] as f64, n[2] as f64],
                    [ca[0] as f64, ca[1] as f64, ca[2] as f64],
                    [c[0] as f64, c[1] as f64, c[2] as f64],
                )?;

                // Convert placed atoms to (name, coords) pairs
                let candidate_atoms: Vec<(String, [f32; 3])> = placed_rot
                    .atoms
                    .iter()
                    .map(|a| (a.name.clone(), [a.xyz[0] as f32, a.xyz[1] as f32, a.xyz[2] as f32]))
                    .collect();

                // Calculate clash penalty
                let clash = calculate_clash_penalty(&candidate_atoms, &placed_atoms);

                // Score = probability - lambda * clash
                let lambda = 1.0; // Weight of clash penalty
                let score = prob - lambda * clash;

                if score > best_score {
                    best_score = score;
                    best_rot = rot_idx;
                }
            }

            // Place the best rotamer
            let best_placed = self.lib.place_rotamer(
                &aa,
                phi,
                psi,
                best_rot,
                false,
                [n[0] as f64, n[1] as f64, n[2] as f64],
                [ca[0] as f64, ca[1] as f64, ca[2] as f64],
                [c[0] as f64, c[1] as f64, c[2] as f64],
            )?;

            // Calculate candidate atoms and clash before updating the residue
            let candidate_atoms: Vec<(String, [f32; 3])> = best_placed
                .atoms
                .iter()
                .map(|a| (a.name.clone(), [a.xyz[0] as f32, a.xyz[1] as f32, a.xyz[2] as f32]))
                .collect();
            let final_clash = calculate_clash_penalty(&candidate_atoms, &placed_atoms);

            // Now get mutable residue and update atoms with the placed sidechain
            let chain = &mut self.topology.chains[chain_idx];
            let residue = &mut chain.residues[res_idx];
            for placed_atom in &best_placed.atoms {
                // Check if this atom already exists in the residue
                if let Some(pos) = residue.atoms.iter().position(|a| a.name == placed_atom.name) {
                    // Update coordinates
                    residue.atoms[pos].coords = [
                        placed_atom.xyz[0] as f32,
                        placed_atom.xyz[1] as f32,
                        placed_atom.xyz[2] as f32,
                    ];
                } else {
                    // Add new atom
                    let element = extract_element(&placed_atom.name);
                    residue.atoms.push(Atom {
                        name: placed_atom.name.clone(),
                        element,
                        coords: [
                            placed_atom.xyz[0] as f32,
                            placed_atom.xyz[1] as f32,
                            placed_atom.xyz[2] as f32,
                        ],
                        alt_loc: ' ',
                        serial: residue.atoms.len() as i32,
                        b_factor: 0.0,
                        occupancy: 1.0,
                        is_hetatm: false,
                    });
                }
            }

            report.residues_completed += 1;
            report.residue_details.push((chain_id, res_id, aa, best_rot, final_clash));
        }

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proxide_io::formats::pdb;
    use std::path::Path;

    fn load_test_topology(fixture_name: &str) -> Option<Topology> {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let fixture_path = Path::new(manifest_dir)
            .join("tests")
            .join("data")
            .join(fixture_name);

        if !fixture_path.exists() {
            return None;
        }

        match pdb::parse_pdb_file(&fixture_path) {
            Ok((raw_data, _model_ids)) => Some(Topology::from_raw_atom_data(&raw_data)),
            Err(_) => None,
        }
    }

    fn load_rotlib() -> Option<RotamerLibrary> {
        // Try to load from the project's rotlib path
        let rotlib_paths = [
            "/home/marielle/projects/proxide/testfiles/rotlib.pb.zst",
            "testfiles/rotlib.pb.zst",
        ];

        for path_str in &rotlib_paths {
            let path = Path::new(path_str);
            if path.exists() {
                if let Ok(lib) = RotamerLibrary::load_pb(path) {
                    return Some(lib);
                }
            }
        }

        None
    }

    #[test]
    fn test_truncated_lys_completion() {
        let lib = match load_rotlib() {
            Some(l) => l,
            None => {
                eprintln!("Skipping test: rotamer library not found");
                return;
            }
        };

        let mut topology = match load_test_topology("truncated_sidechain.pdb") {
            Some(t) => t,
            None => {
                eprintln!("Skipping test: truncated_sidechain.pdb not found");
                return;
            }
        };

        // Verify that LYS only has N/CA/C/O/CB (5 atoms)
        assert_eq!(topology.chains.len(), 1);
        let residue = &topology.chains[0].residues[0];
        assert_eq!(residue.name, "LYS");
        let atom_names: Vec<_> = residue.atoms.iter().map(|a| a.name.as_str()).collect();
        assert!(atom_names.contains(&"CB"), "CB should be present");

        let missing_before = vec!["CG", "CD", "CE", "NZ"];
        for atom_name in &missing_before {
            assert!(
                !atom_names.contains(atom_name),
                "{} should not be present before repacking",
                atom_name
            );
        }

        // Run repacker
        let mut repacker = SidechainRepacker::new(&mut topology, &lib);
        let report = repacker
            .complete_and_repack(RepackMode::CompleteMissingOnly)
            .expect("repacking should succeed");

        // Verify repacking occurred
        assert!(
            report.residues_completed > 0,
            "At least one residue should be repacked"
        );

        // Verify sidechain is now complete
        let residue = &topology.chains[0].residues[0];
        let atom_names: Vec<_> = residue.atoms.iter().map(|a| a.name.as_str()).collect();

        // Check that CG is now present (should have been added)
        assert!(
            atom_names.contains(&"CG"),
            "CG should be added after repacking"
        );

        // Verify CB-CG distance is reasonable (~1.52 Ångströms for LYS)
        let cb_atom = residue.atoms.iter().find(|a| a.name == "CB").unwrap();
        let cg_atom = residue.atoms.iter().find(|a| a.name == "CG");
        if let Some(cg) = cg_atom {
            let dx = cb_atom.coords[0] - cg.coords[0];
            let dy = cb_atom.coords[1] - cg.coords[1];
            let dz = cb_atom.coords[2] - cg.coords[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            // LYS CB-CG bond distance should be approximately 1.52 Ångströms ± 0.3
            assert!(
                distance > 1.2 && distance < 1.8,
                "CB-CG distance should be ~1.52 Å, got {:.2} Å",
                distance
            );
        }
    }

    #[test]
    fn test_clash_penalty_calculation() {
        // Test the clash penalty function directly
        let candidate = vec![("CG".to_string(), [0.0, 0.0, 0.0])];
        let placed = vec![
            Atom {
                name: "N".to_string(),
                element: "N".to_string(),
                coords: [0.1, 0.0, 0.0], // Very close, should cause clash
                alt_loc: ' ',
                serial: 1,
                b_factor: 0.0,
                occupancy: 1.0,
                is_hetatm: false,
            },
        ];

        let penalty = calculate_clash_penalty(&candidate, &placed);
        assert!(penalty > 0.0, "Penalty should be positive for a clash");
    }

    #[test]
    fn test_element_extraction() {
        assert_eq!(extract_element("CA"), "C");
        assert_eq!(extract_element("CB"), "C");
        assert_eq!(extract_element("N"), "N");
        assert_eq!(extract_element("OG"), "O");
        assert_eq!(extract_element("2HD"), "H");
        assert_eq!(extract_element("SG"), "S");
    }

    #[test]
    fn test_is_incomplete() {
        let lib = match load_rotlib() {
            Some(l) => l,
            None => {
                eprintln!("Skipping test: rotamer library not found");
                return;
            }
        };

        // LYS with only backbone atoms should be incomplete
        let residue_incomplete = Residue {
            name: "LYS".to_string(),
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
                    coords: [1.0, 0.0, 0.0],
                    alt_loc: ' ',
                    serial: 2,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
            ],
        };

        assert!(
            is_incomplete(&residue_incomplete, &lib),
            "LYS with only N and CA should be incomplete"
        );

        // GLY should never be incomplete
        let residue_gly = Residue {
            name: "GLY".to_string(),
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
            ],
        };

        assert!(
            !is_incomplete(&residue_gly, &lib),
            "GLY should never be incomplete"
        );
    }
}
