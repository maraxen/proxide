//! Atomic mass assignment utilities
//!
//! Provides functions to assign atomic masses based on element type.
//! Used for MD simulations.

/// Default mass for unknown elements
pub const DEFAULT_MASS: f32 = 12.0;

/// Get atomic mass for an element symbol
pub fn get_mass(element: &str) -> f32 {
    match element {
        "H" => 1.008,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "S" => 32.06,
        "P" => 30.97,
        "F" => 18.998,
        "Cl" => 35.45,
        "Br" => 79.904,
        "I" => 126.90,
        "Na" => 22.990,
        "K" => 39.098,
        "Ca" => 40.078,
        "Mg" => 24.305,
        "Zn" => 65.38,
        "Fe" => 55.845,
        "Cu" => 63.546,
        "Mn" => 54.938,
        "Se" => 78.971,
        _ => DEFAULT_MASS,
    }
}

/// Assign atomic masses based on atom names
///
/// Infers the element from the first character(s) of the atom name
/// and looks up the mass from the standard atomic masses table.
///
/// # Arguments
/// * `atom_names` - List of atom names (e.g., ["N", "CA", "C", "O", "H"])
///
/// # Returns
/// * List of masses in amu
pub fn assign_masses(atom_names: &[String]) -> Vec<f32> {
    atom_names
        .iter()
        .map(|name| {
            let element = infer_element(name);
            get_mass(element)
        })
        .collect()
}

/// Infer element from atom name
///
/// Follows PDB conventions where element is typically the first 1-2 characters
fn infer_element(atom_name: &str) -> &str {
    let name = atom_name.trim();
    if name.is_empty() {
        return "C";
    }

    // Check for common two-character elements first
    if name.len() >= 2 {
        let two_char = &name[..2];
        match two_char {
            "Cl" | "CL" => return "Cl",
            "Br" | "BR" => return "Br",
            "Na" | "NA" => return "Na",
            "Mg" | "MG" => return "Mg",
            "Zn" | "ZN" => return "Zn",
            "Fe" | "FE" => return "Fe",
            "Cu" | "CU" => return "Cu",
            "Mn" | "MN" => return "Mn",
            "Se" | "SE" => return "Se",
            _ => {}
        }
    }

    // Default: use first character
    let first = name.chars().next().unwrap();
    match first {
        'H' | 'h' => "H",
        'C' | 'c' => "C",
        'N' | 'n' => "N",
        'O' | 'o' => "O",
        'S' | 's' => "S",
        'P' | 'p' => "P",
        'F' | 'f' => "F",
        'I' | 'i' => "I",
        'K' | 'k' => "K",
        _ => "C", // Default to carbon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_masses() {
        let names = vec![
            "N".to_string(),
            "CA".to_string(),
            "C".to_string(),
            "O".to_string(),
            "H".to_string(),
        ];
        let masses = assign_masses(&names);

        assert!((masses[0] - 14.007).abs() < 0.001); // N
        assert!((masses[1] - 12.011).abs() < 0.001); // CA -> C
        assert!((masses[2] - 12.011).abs() < 0.001); // C
        assert!((masses[3] - 15.999).abs() < 0.001); // O
        assert!((masses[4] - 1.008).abs() < 0.001); // H
    }

    #[test]
    fn test_infer_element() {
        assert_eq!(infer_element("CA"), "C"); // Alpha carbon
        assert_eq!(infer_element("N"), "N");
        assert_eq!(infer_element("CL"), "Cl");
        assert_eq!(infer_element("Na"), "Na");
        assert_eq!(infer_element("FE"), "Fe");
    }
}
