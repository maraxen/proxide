use crate::geometry::cell_list::find_neighbors_within_cutoff_fast;
use crate::geometry::distances::euclidean_distance_squared;
use proxide_core::forcefield::topology::{Bond, Topology};
use proxide_core::forcefield::ResidueTemplate;
use proxide_core::processing::ProcessedStructure;
use std::collections::{HashMap, HashSet};

/// Assign template hydrogens to unmatched H atoms based on connectivity
pub fn assign_template_hydrogens(
    processed: &ProcessedStructure,
    template: &ResidueTemplate,
    local_to_global: &HashMap<&str, usize>,
    claimed_template_atoms: &mut HashSet<String>,
    unmatched_h_indices: &[usize],
    ff: &proxide_core::forcefield::ForceField,
) -> HashMap<usize, String> {
    let mut mapping = HashMap::new();
    let res_start = if !unmatched_h_indices.is_empty() {
        processed.raw_atoms.res_ids[unmatched_h_indices[0]]
    } else {
        0
    };

    // Build template bond adjacency: heavy_atom_name -> [H_atom_names bonded to it]
    let mut template_h_by_parent: HashMap<&str, Vec<&str>> = HashMap::new();
    for (name1, name2) in &template.bonds {
        let a1_is_h = template.atoms.iter()
            .find(|a| &a.name == name1)
            .and_then(|ta| ff.get_atom_type(&ta.atom_type))
            .map(|at| at.element.eq_ignore_ascii_case("H"))
            .unwrap_or(name1.starts_with('H') && name1 != "HG");
        
        let a2_is_h = template.atoms.iter()
            .find(|a| &a.name == name2)
            .and_then(|ta| ff.get_atom_type(&ta.atom_type))
            .map(|at| at.element.eq_ignore_ascii_case("H"))
            .unwrap_or(name2.starts_with('H'));

        if a1_is_h && !a2_is_h {
            template_h_by_parent.entry(name2.as_str()).or_default().push(name1.as_str());
        } else if a2_is_h && !a1_is_h {
            template_h_by_parent.entry(name1.as_str()).or_default().push(name2.as_str());
        }
    }

    // Collect matched heavy atoms in this residue: (global_idx, template_name)
    // We need to find which residue these hydrogens belong to.
    // For simplicity, we assume all unmatched_h_indices are in the same residue passed in 'template'.
    let res_info = processed.residue_info.iter().find(|r| r.res_name == template.name && r.res_id == res_start);
    
    if let Some(res_info) = res_info {
        let res_heavy_atoms: Vec<(usize, &str)> = local_to_global
            .iter()
            .filter(|(_, &gidx)| {
                gidx >= res_info.start_atom
                    && gidx < res_info.start_atom + res_info.num_atoms
                    && !processed.raw_atoms.elements[gidx].eq_ignore_ascii_case("H")
            })
            .map(|(&tname, &gidx)| (gidx, tname))
            .collect();

        for &h_idx in unmatched_h_indices {
            let hx = processed.raw_atoms.coords[h_idx * 3];
            let hy = processed.raw_atoms.coords[h_idx * 3 + 1];
            let hz = processed.raw_atoms.coords[h_idx * 3 + 2];

            let mut best_heavy: Option<&str> = None;
            let mut best_dist = f32::MAX;

            for &(heavy_idx, tname) in &res_heavy_atoms {
                let dx = hx - processed.raw_atoms.coords[heavy_idx * 3];
                let dy = hy - processed.raw_atoms.coords[heavy_idx * 3 + 1];
                let dz = hz - processed.raw_atoms.coords[heavy_idx * 3 + 2];
                let dist = dx * dx + dy * dy + dz * dz;
                if dist < best_dist {
                    best_dist = dist;
                    best_heavy = Some(tname);
                }
            }

            // Sanity: typical C-H bond ~1.09A, N-H ~1.01A; reject if > 1.5A
            if best_dist > 2.25 {
                continue;
            }

            if let Some(parent_name) = best_heavy {
                if let Some(h_candidates) = template_h_by_parent.get(parent_name) {
                    for &h_template_name in h_candidates {
                        if !claimed_template_atoms.contains(h_template_name) {
                            mapping.insert(h_idx, h_template_name.to_string());
                            claimed_template_atoms.insert(h_template_name.to_string());
                            break;
                        }
                    }
                }
            }
        }
    }

    mapping
}
/// Values from Cambridge Structural Database / inner logic of many tools
pub fn get_covalent_radius(element: &str) -> f32 {
    match element.to_uppercase().as_str() {
        "H" => 0.31,
        "C" => 0.76,
        "N" => 0.71,
        "O" => 0.66,
        "S" => 1.05,
        "P" => 1.07,
        "F" => 0.57,
        "CL" => 1.02,
        "BR" => 1.20,
        "I" => 1.39,
        "FE" => 1.32,
        "ZN" => 1.22,
        "MG" => 1.41,
        "CA" => 1.76,
        _ => 1.50, // Generic fallback
    }
}

/// Infer bonds based on distance criteria
///
/// Use covalent radii + tolerance to detect bonds.
/// Tolerance factor typically 1.3 to include slightly stretched bonds but exclude non-bonded.
pub fn infer_bonds(coords: &[[f32; 3]], elements: &[String], tolerance: f32) -> Vec<[usize; 2]> {
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }

    // Maximum possible bond length to check
    // Max covalent radius ~1.8 (S/Metals) * 2 * tolerance
    let max_cutoff = 4.0;

    // Find candidate pairs efficiently using cell list
    let neighbors = find_neighbors_within_cutoff_fast(coords, max_cutoff);

    let mut bonds = Vec::new();

    for (i, neighbor_indices) in neighbors.iter().enumerate() {
        let r1 = get_covalent_radius(&elements[i]);

        for &j in neighbor_indices {
            // Check only i < j to avoid duplicates
            if i >= j {
                continue;
            }

            let r2 = get_covalent_radius(&elements[j]);
            let threshold = (r1 + r2) * tolerance;

            // Re-calculate squared distance (neighbor search did it but didn't return it)
            // Could optimize neighbor search to return distances, but cheap enough here
            let dist_sq = euclidean_distance_squared(&coords[i], &coords[j]);

            // Check lower bound too? Bonds shouldn't be too short (< 0.4A)
            if dist_sq <= threshold * threshold && dist_sq > 0.16 {
                bonds.push([i, j]);
            }
        }
    }

    bonds
}

/// Build a complete molecular topology from coordinates and elements
pub fn generate_topology(coords: &[[f32; 3]], elements: &[String], tolerance: f32) -> Topology {
    let bond_pairs = infer_bonds(coords, elements, tolerance);
    let bonds: Vec<Bond> = bond_pairs.iter().map(|&[i, j]| Bond::new(i, j)).collect();
    Topology::new(bonds, elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_bonds_methane() {
        // Methane CH4
        // C at origin
        // H at ~1.09 A
        let coords = [
            [0.0, 0.0, 0.0],       // C
            [1.09, 0.0, 0.0],      // H1
            [-0.36, 1.03, 0.0],    // H2 (approx)
            [-0.36, -0.51, 0.89],  // H3
            [-0.36, -0.51, -0.89], // H4
        ];

        let elements = vec![
            "C".to_string(),
            "H".to_string(),
            "H".to_string(),
            "H".to_string(),
            "H".to_string(),
        ];

        let bonds = infer_bonds(&coords, &elements, 1.3);

        // Should have 4 bonds (C-H)
        assert_eq!(bonds.len(), 4);

        // Check connectivity
        for bond in bonds {
            assert!(
                bond[0] == 0 || bond[1] == 0,
                "All bonds must connect to C (index 0)"
            );
        }
    }

    #[test]
    fn test_infer_bonds_water() {
        let coords = [
            [0.0, 0.0, 0.0],    // O
            [0.96, 0.0, 0.0],   // H1
            [-0.24, 0.93, 0.0], // H2
        ];

        let elements = vec!["O".to_string(), "H".to_string(), "H".to_string()];

        let bonds = infer_bonds(&coords, &elements, 1.3);

        // 2 bonds
        assert_eq!(bonds.len(), 2);
    }
}
