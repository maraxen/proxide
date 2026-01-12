//! Water model definitions and parameterization.
//!
//! Provides standard water model parameters (TIP3P, SPC/E, TIP4P-Ew)
//! for explicit solvent simulations.

use std::collections::HashMap;

/// Parameters for a specific water model.
#[derive(Debug, Clone)]
pub struct WaterModel {
    pub name: String,
    pub atoms: Vec<String>,
    pub charges: HashMap<String, f32>,
    pub sigmas: HashMap<String, f32>,   // Angstroms
    pub epsilons: HashMap<String, f32>, // kcal/mol
    /// Bonds: (atom1_name, atom2_name, equilibrium_length, force_constant)
    pub bonds: Vec<(String, String, f32, f32)>,
    /// Angles: (a1, a2, a3, theta_radians, force_constant)
    pub angles: Vec<(String, String, String, f32, f32)>,
    /// Constraints: (atom1_name, atom2_name, distance)
    pub constraints: Vec<(String, String, f32)>,
    /// Whether this model has virtual sites (e.g., TIP4P)
    pub has_virtual_sites: bool,
}

// Conversion factor
const DEG2RAD: f32 = std::f32::consts::PI / 180.0;

/// Get parameters for TIP3P water model.
fn tip3p() -> WaterModel {
    let mut charges = HashMap::new();
    charges.insert("O".to_string(), -0.834);
    charges.insert("H1".to_string(), 0.417);
    charges.insert("H2".to_string(), 0.417);

    let mut sigmas = HashMap::new();
    sigmas.insert("O".to_string(), 3.15061);
    sigmas.insert("H1".to_string(), 0.0001);
    sigmas.insert("H2".to_string(), 0.0001);

    let mut epsilons = HashMap::new();
    epsilons.insert("O".to_string(), 0.1521);
    epsilons.insert("H1".to_string(), 0.0);
    epsilons.insert("H2".to_string(), 0.0);

    // r(OH) = 0.9572 Å, theta(HOH) = 104.52°
    // HH dist = 2 * 0.9572 * sin(104.52/2) ≈ 1.5139
    WaterModel {
        name: "TIP3P".to_string(),
        atoms: vec!["O".to_string(), "H1".to_string(), "H2".to_string()],
        charges,
        sigmas,
        epsilons,
        bonds: vec![
            ("O".to_string(), "H1".to_string(), 0.9572, 450.0),
            ("O".to_string(), "H2".to_string(), 0.9572, 450.0),
        ],
        angles: vec![(
            "H1".to_string(),
            "O".to_string(),
            "H2".to_string(),
            104.52 * DEG2RAD,
            100.0,
        )],
        constraints: vec![
            ("O".to_string(), "H1".to_string(), 0.9572),
            ("O".to_string(), "H2".to_string(), 0.9572),
            ("H1".to_string(), "H2".to_string(), 1.513_900_6),
        ],
        has_virtual_sites: false,
    }
}

/// Get parameters for SPC/E water model.
fn spce() -> WaterModel {
    let mut charges = HashMap::new();
    charges.insert("O".to_string(), -0.8476);
    charges.insert("H1".to_string(), 0.4238);
    charges.insert("H2".to_string(), 0.4238);

    let mut sigmas = HashMap::new();
    sigmas.insert("O".to_string(), 3.166);
    sigmas.insert("H1".to_string(), 0.0001);
    sigmas.insert("H2".to_string(), 0.0001);

    let mut epsilons = HashMap::new();
    epsilons.insert("O".to_string(), 0.1553);
    epsilons.insert("H1".to_string(), 0.0);
    epsilons.insert("H2".to_string(), 0.0);

    // r(OH) = 1.0 Å, theta(HOH) = 109.47°
    // HH dist = 2 * 1.0 * sin(109.47/2) ≈ 1.6330
    WaterModel {
        name: "SPCE".to_string(),
        atoms: vec!["O".to_string(), "H1".to_string(), "H2".to_string()],
        charges,
        sigmas,
        epsilons,
        bonds: vec![
            ("O".to_string(), "H1".to_string(), 1.0, 450.0),
            ("O".to_string(), "H2".to_string(), 1.0, 450.0),
        ],
        angles: vec![(
            "H1".to_string(),
            "O".to_string(),
            "H2".to_string(),
            109.47 * DEG2RAD,
            100.0,
        )],
        constraints: vec![
            ("O".to_string(), "H1".to_string(), 1.0),
            ("O".to_string(), "H2".to_string(), 1.0),
            ("H1".to_string(), "H2".to_string(), 1.632_980_8),
        ],
        has_virtual_sites: false,
    }
}

/// Get parameters for TIP4P-Ew water model.
fn tip4pew() -> WaterModel {
    let mut charges = HashMap::new();
    charges.insert("O".to_string(), 0.0);
    charges.insert("H1".to_string(), 0.52422);
    charges.insert("H2".to_string(), 0.52422);
    charges.insert("M".to_string(), -1.04844);

    let mut sigmas = HashMap::new();
    sigmas.insert("O".to_string(), 3.16435);
    sigmas.insert("H1".to_string(), 0.0001);
    sigmas.insert("H2".to_string(), 0.0001);
    sigmas.insert("M".to_string(), 0.0001);

    let mut epsilons = HashMap::new();
    epsilons.insert("O".to_string(), 0.16275);
    epsilons.insert("H1".to_string(), 0.0);
    epsilons.insert("H2".to_string(), 0.0);
    epsilons.insert("M".to_string(), 0.0);

    WaterModel {
        name: "TIP4PEW".to_string(),
        atoms: vec![
            "O".to_string(),
            "H1".to_string(),
            "H2".to_string(),
            "M".to_string(),
        ],
        charges,
        sigmas,
        epsilons,
        bonds: vec![
            ("O".to_string(), "H1".to_string(), 0.9572, 450.0),
            ("O".to_string(), "H2".to_string(), 0.9572, 450.0),
        ],
        angles: vec![(
            "H1".to_string(),
            "O".to_string(),
            "H2".to_string(),
            104.52 * DEG2RAD,
            100.0,
        )],
        constraints: vec![
            ("O".to_string(), "H1".to_string(), 0.9572),
            ("O".to_string(), "H2".to_string(), 0.9572),
            ("H1".to_string(), "H2".to_string(), 1.513_900_6),
        ],
        has_virtual_sites: true,
    }
}

/// Get water model parameters by name.
///
/// Supported models: TIP3P, SPCE (SPC/E), TIP4PEW (TIP4P-Ew)
///
/// If `rigid` is true, bond and angle force constants are set to 0.0,
/// matching OpenMM's rigidWater=True behavior.
pub fn get_water_model(name: &str, rigid: bool) -> Result<WaterModel, String> {
    let upper = name.to_uppercase();
    let mut model = match upper.as_str() {
        "TIP3P" => tip3p(),
        "SPCE" | "SPC/E" => spce(),
        "TIP4PEW" | "TIP4P-EW" => tip4pew(),
        "HOH" | "WAT" | "SOL" => tip3p(), // Default fallback
        _ => return Err(format!("Unknown water model: {}", name)),
    };

    if rigid {
        // Zero out force constants, keep equilibrium values for constraints
        model.bonds = model
            .bonds
            .into_iter()
            .map(|(a1, a2, length, _)| (a1, a2, length, 0.0))
            .collect();
        model.angles = model
            .angles
            .into_iter()
            .map(|(a1, a2, a3, theta, _)| (a1, a2, a3, theta, 0.0))
            .collect();
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tip3p() {
        let model = get_water_model("TIP3P", false).unwrap();
        assert_eq!(model.name, "TIP3P");
        assert_eq!(model.atoms.len(), 3);
        assert!((model.charges["O"] - (-0.834)).abs() < 0.001);
        assert!(!model.has_virtual_sites);
    }

    #[test]
    fn test_tip4pew() {
        let model = get_water_model("TIP4PEW", false).unwrap();
        assert_eq!(model.atoms.len(), 4);
        assert!(model.has_virtual_sites);
        assert!((model.charges["M"] - (-1.04844)).abs() < 0.001);
    }

    #[test]
    fn test_rigid_water() {
        let model = get_water_model("TIP3P", true).unwrap();
        // Force constants should be 0
        for (_, _, _, k) in &model.bonds {
            assert_eq!(*k, 0.0);
        }
        for (_, _, _, _, k) in &model.angles {
            assert_eq!(*k, 0.0);
        }
        // But equilibrium values preserved
        assert!((model.bonds[0].2 - 0.9572).abs() < 0.001);
    }
}
