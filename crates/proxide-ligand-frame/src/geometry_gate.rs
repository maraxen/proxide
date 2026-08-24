use crate::errors::LigandFrameError;

/// Cordero et al. (2008) single-bond covalent radii, Angstrom, for the
/// elements this crate's error surface treats as supported. An element
/// outside this table is `UnsupportedElement`, never silently defaulted.
fn covalent_radius(element: &str) -> Option<f64> {
    match element {
        "H" => Some(0.31),
        "C" => Some(0.76),
        "N" => Some(0.71),
        "O" => Some(0.66),
        "F" => Some(0.57),
        "P" => Some(1.07),
        "S" => Some(1.05),
        "Cl" => Some(1.02),
        "Br" => Some(1.20),
        "I" => Some(1.39),
        _ => None,
    }
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Reference-geometry validity gate (spec §3, closes Finding 11): must
/// pass before Espaloma charge inference runs.
pub fn validate_reference_geometry(
    elements: &[String],
    bonds: &[(usize, usize, u8)],
    positions: &[[f64; 3]],
) -> Result<(), LigandFrameError> {
    for (idx, p) in positions.iter().enumerate() {
        if !p.iter().all(|c| c.is_finite()) {
            return Err(LigandFrameError::InvalidReferenceGeometry {
                reason: format!("atom {idx} has non-finite coordinates"),
            });
        }
    }

    let mut radii = Vec::with_capacity(elements.len());
    for e in elements {
        radii.push(
            covalent_radius(e)
                .ok_or_else(|| LigandFrameError::UnsupportedElement { element: e.clone() })?,
        );
    }

    for &(i, j, _order) in bonds {
        let ref_len = radii[i] + radii[j];
        let dist = distance(&positions[i], &positions[j]);
        if dist < 0.5 * ref_len || dist > 2.5 * ref_len {
            return Err(LigandFrameError::InvalidReferenceGeometry {
                reason: format!(
                    "bond ({i}, {j}) length {dist:.3} A outside [0.5, 2.5] x reference {ref_len:.3} A"
                ),
            });
        }
    }

    let bonded: std::collections::HashSet<(usize, usize)> = bonds
        .iter()
        .map(|&(i, j, _)| (i.min(j), i.max(j)))
        .collect();
    let n = positions.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if bonded.contains(&(i, j)) {
                continue;
            }
            let dist = distance(&positions[i], &positions[j]);
            let clash_threshold = 0.7 * (radii[i] + radii[j]);
            if dist < clash_threshold {
                return Err(LigandFrameError::InvalidReferenceGeometry {
                    reason: format!(
                        "non-bonded atoms {i} and {j} are {dist:.3} A apart, closer than clash threshold {clash_threshold:.3} A"
                    ),
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch4_elements() -> Vec<String> {
        vec!["C", "H", "H", "H", "H"].into_iter().map(String::from).collect()
    }

    #[test]
    fn valid_methane_geometry_passes() {
        // Tetrahedral-ish, bond length ~1.09 A, well-separated H's.
        let positions = [
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ];
        let bonds = [(0, 1, 1u8), (0, 2, 1), (0, 3, 1), (0, 4, 1)];
        assert!(validate_reference_geometry(&ch4_elements(), &bonds, &positions).is_ok());
    }

    #[test]
    fn non_finite_coordinate_rejected() {
        let positions = [[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let elements = vec!["C".to_string(), "H".to_string()];
        let err = validate_reference_geometry(&elements, &[(0, 1, 1)], &positions).unwrap_err();
        assert!(matches!(err, LigandFrameError::InvalidReferenceGeometry { .. }));
    }

    #[test]
    fn bond_declared_but_atoms_far_apart_rejected() {
        let positions = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let elements = vec!["C".to_string(), "C".to_string()];
        let err = validate_reference_geometry(&elements, &[(0, 1, 1)], &positions).unwrap_err();
        assert!(matches!(err, LigandFrameError::InvalidReferenceGeometry { .. }));
    }

    #[test]
    fn clashing_nonbonded_atoms_rejected() {
        // Two non-bonded carbons 0.5 A apart -- well under 0.7*(0.76+0.76).
        let positions = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let elements = vec!["C".to_string(), "C".to_string()];
        let err = validate_reference_geometry(&elements, &[], &positions).unwrap_err();
        assert!(matches!(err, LigandFrameError::InvalidReferenceGeometry { .. }));
    }

    #[test]
    fn unsupported_element_rejected() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let elements = vec!["C".to_string(), "Xx".to_string()];
        let err = validate_reference_geometry(&elements, &[(0, 1, 1)], &positions).unwrap_err();
        assert!(matches!(err, LigandFrameError::UnsupportedElement { .. }));
    }
}
