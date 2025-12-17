//! Topology and bond inference
//!
//! Provides methods to infer chemical bonds and topology from atomic coordinates.

use crate::geometry::cell_list::find_neighbors_within_cutoff_fast;
use crate::geometry::distances::euclidean_distance_squared;

/// Get covalent radius for an element symbol (in Angstroms)
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
