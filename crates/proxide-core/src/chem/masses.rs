//! Atomic mass assignment utilities
//!
//! Provides functions to assign atomic masses based on element type.
//! Used for MD simulations.

/// Get atomic mass for an element symbol (in amu/Da)
///
/// Returns `None` when the element has no tabulated mass. A mass is never
/// guessed or defaulted to carbon (see project hard rule on unknown values,
/// ledger A1).
pub fn get_mass(element: &str) -> Option<f32> {
    match element {
        "H" => Some(1.008),
        "C" => Some(12.011),
        "N" => Some(14.007),
        "O" => Some(15.999),
        "S" => Some(32.06),
        "P" => Some(30.97),
        "F" => Some(18.998),
        "Cl" => Some(35.45),
        "Br" => Some(79.904),
        "I" => Some(126.90),
        "Na" => Some(22.990),
        "K" => Some(39.098),
        "Ca" => Some(40.078),
        "Mg" => Some(24.305),
        "Zn" => Some(65.38),
        "Fe" => Some(55.845),
        "Cu" => Some(63.546),
        "Mn" => Some(54.938),
        "Se" => Some(78.971),
        // IUPAC 2021 conventional standard atomic weight.
        "Si" => Some(28.085),
        // Deuterium (2H), IUPAC: 2.014 amu.
        "D" => Some(2.014),
        _ => None,
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
///
/// # Errors
/// Returns an error if an element cannot be inferred from an atom name,
/// or if an inferred element has no tabulated atomic mass. A mass is
/// never guessed or defaulted (see the project hard rule on unknown values,
/// ledger A1).
pub fn assign_masses(atom_names: &[String]) -> Result<Vec<f32>, String> {
    atom_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let elem = infer_element(name).ok_or_else(|| {
                format!("cannot infer an element from atom name {name:?} at index {i}")
            })?;
            get_mass(elem).ok_or_else(|| {
                format!(
                    "no tabulated mass for element {elem:?} from atom name {name:?} at index {i}"
                )
            })
        })
        .collect()
}

/// Infer element from atom name
///
/// Follows PDB conventions where element is typically the first 1-2 characters.
/// Public so format parsers (e.g. `proxide-io`'s PDB reader) can reuse this
/// two-letter-aware logic instead of reimplementing a naive first-character-only
/// fallback -- see backlog #5052 (prolix), where a duplicated, wrong version of
/// this exact inference in `proxide-io/src/formats/pdb.rs` mis-elementized a
/// chloride ion ("Cl") as carbon ("C").
///
/// Strips a leading digit before inference: PDB hydrogen-naming conventions
/// commonly prefix a remoteness/branch digit onto the element letter(s), e.g.
/// "1HB", "2HG1", "1CL" -- without stripping it, `"1HB".chars().next()` is
/// `'1'`, which matches no element and silently falls through to the "C"
/// default (sprint 24, decision g; task 260922_autonomous-loop). Digits
/// elsewhere in the name (e.g. the trailing branch index in "HG1") are left
/// alone -- only a *leading* digit is a PDB naming-convention prefix, not
/// part of the element.
///
/// Returns `None` when the name does not determine an element: empty names,
/// names of only digits, and any other leading letter. This used to return
/// `"C"`, which made "carbon" and "unknown" the same value (ledger A1). Callers
/// must decide what unknown means for them; parsers raise.
pub fn infer_element(atom_name: &str) -> Option<&'static str> {
    let name = atom_name.trim();
    let name = name.trim_start_matches(|c: char| c.is_ascii_digit());
    if name.is_empty() {
        return None;
    }

    // Check for common two-character elements first
    if name.len() >= 2 {
        let two_char = &name[..2];
        match two_char {
            "Cl" | "CL" => return Some("Cl"),
            "Br" | "BR" => return Some("Br"),
            "Na" | "NA" => return Some("Na"),
            "Mg" | "MG" => return Some("Mg"),
            "Zn" | "ZN" => return Some("Zn"),
            "Fe" | "FE" => return Some("Fe"),
            "Cu" | "CU" => return Some("Cu"),
            "Mn" | "MN" => return Some("Mn"),
            "Se" | "SE" => return Some("Se"),
            "Si" | "SI" => return Some("Si"),
            _ => {}
        }
    }

    // Default: use first character
    let first = name.chars().next()?;
    match first {
        'H' | 'h' => Some("H"),
        'C' | 'c' => Some("C"),
        'N' | 'n' => Some("N"),
        'O' | 'o' => Some("O"),
        'S' | 's' => Some("S"),
        'P' | 'p' => Some("P"),
        'F' | 'f' => Some("F"),
        'I' | 'i' => Some("I"),
        'K' | 'k' => Some("K"),
        _ => None,
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
        let masses = assign_masses(&names).unwrap();

        assert!((masses[0] - 14.007).abs() < 0.001); // N
        assert!((masses[1] - 12.011).abs() < 0.001); // CA -> C
        assert!((masses[2] - 12.011).abs() < 0.001); // C
        assert!((masses[3] - 15.999).abs() < 0.001); // O
        assert!((masses[4] - 1.008).abs() < 0.001); // H
    }

    #[test]
    fn test_infer_element() {
        assert_eq!(infer_element("CA"), Some("C")); // Alpha carbon
        assert_eq!(infer_element("N"), Some("N"));
        assert_eq!(infer_element("CL"), Some("Cl"));
        assert_eq!(infer_element("Na"), Some("Na"));
        assert_eq!(infer_element("FE"), Some("Fe"));
    }

    #[test]
    fn test_infer_element_comprehensive_corpus() {
        // Comprehensive conformance test for all supported atoms, both ALL-CAPS (PDB format)
        // and title-case (canonical). This is the guard against silent element-inference bugs.
        // See tests/test_element_inference_conformance.py for the structural anti-pattern scan.

        // Single-letter elements (PDB form: all-caps, canonical: uppercase)
        assert_eq!(infer_element("H"), Some("H"));
        assert_eq!(infer_element("h"), Some("H"));
        assert_eq!(infer_element("C"), Some("C"));
        assert_eq!(infer_element("c"), Some("C"));
        assert_eq!(infer_element("N"), Some("N"));
        assert_eq!(infer_element("n"), Some("N"));
        assert_eq!(infer_element("O"), Some("O"));
        assert_eq!(infer_element("o"), Some("O"));
        assert_eq!(infer_element("S"), Some("S"));
        assert_eq!(infer_element("s"), Some("S"));
        assert_eq!(infer_element("P"), Some("P"));
        assert_eq!(infer_element("p"), Some("P"));
        assert_eq!(infer_element("F"), Some("F"));
        assert_eq!(infer_element("f"), Some("F"));
        assert_eq!(infer_element("I"), Some("I"));
        assert_eq!(infer_element("i"), Some("I"));
        assert_eq!(infer_element("K"), Some("K"));
        assert_eq!(infer_element("k"), Some("K"));

        // Two-letter elements: PDB names are ALL-CAPS, must map to title-case canonical form
        // (implemented for PDB format + title-case, not bare lowercase, per real-world PDB usage)
        assert_eq!(infer_element("CL"), Some("Cl"));
        assert_eq!(infer_element("Cl"), Some("Cl"));
        assert_eq!(infer_element("BR"), Some("Br"));
        assert_eq!(infer_element("Br"), Some("Br"));
        assert_eq!(infer_element("NA"), Some("Na"));
        assert_eq!(infer_element("Na"), Some("Na"));
        assert_eq!(infer_element("MG"), Some("Mg"));
        assert_eq!(infer_element("Mg"), Some("Mg"));
        assert_eq!(infer_element("ZN"), Some("Zn"));
        assert_eq!(infer_element("Zn"), Some("Zn"));
        assert_eq!(infer_element("FE"), Some("Fe"));
        assert_eq!(infer_element("Fe"), Some("Fe"));
        assert_eq!(infer_element("CU"), Some("Cu"));
        assert_eq!(infer_element("Cu"), Some("Cu"));
        assert_eq!(infer_element("MN"), Some("Mn"));
        assert_eq!(infer_element("Mn"), Some("Mn"));
        assert_eq!(infer_element("SE"), Some("Se"));
        assert_eq!(infer_element("Se"), Some("Se"));

        // Critical edge case: CA (alpha carbon, protein backbone) must stay "C", not "Ca"
        assert_eq!(infer_element("CA"), Some("C"));
        assert_eq!(infer_element("Ca"), Some("C")); // title-case variant must also be "C"
        assert_eq!(infer_element("ca"), Some("C"));
    }

    #[test]
    fn test_infer_element_leading_digit_stripped() {
        // Sprint 24 decision g (task 260922_autonomous-loop): PDB hydrogen
        // naming conventions prefix a remoteness/branch digit onto the
        // element letter(s) (e.g. "1HB", "2HG1"). Without stripping the
        // leading digit, `infer_element` silently fell through to the "C"
        // default -- a latent bug at proxide-io/src/formats/pdb.rs:68 (blank
        // element column + digit-prefixed hydrogen name) predating this fix.
        assert_eq!(infer_element("1HB"), Some("H"));
        assert_eq!(infer_element("2HG1"), Some("H"));
        assert_eq!(infer_element("1CL"), Some("Cl"));
        assert_eq!(infer_element("2Cl"), Some("Cl"));
        assert_eq!(infer_element("3HD1"), Some("H"));

        // Not digit-prefixed: unaffected, including the critical CA case.
        assert_eq!(infer_element("CA"), Some("C"));
        assert_eq!(infer_element("HB"), Some("H"));

        // A name that is nothing but digits is unknown, like an empty name.
        assert_eq!(infer_element("1"), None);
    }

    #[test]
    fn unknown_names_are_not_carbon() {
        for name in ["", "   ", "1", "QZ", "XX", "D1", "EPW", "AU", "LI"] {
            assert_eq!(infer_element(name), None, "{name:?} must be unknown");
        }
    }

    #[test]
    fn assign_masses_refuses_unknown_names_and_names_them() {
        let err = assign_masses(&["CA".to_string(), "QZ".to_string()]).unwrap_err();
        assert!(err.contains("QZ") && err.contains("index 1"), "{err}");
    }

    #[test]
    fn get_mass_deuterium_and_title_case_symbols() {
        assert!((get_mass("Zn").unwrap() - 65.38).abs() < 1e-3);
        assert!((get_mass("Cl").unwrap() - 35.45).abs() < 1e-3);
        // Deuterium: 2.014 amu, not the carbon fallback.
        assert!((get_mass("D").unwrap() - 2.014).abs() < 1e-3);
    }

    #[test]
    fn test_get_mass_untabulated_returns_none() {
        // Untabulated elements must return None, never default mass 12.0.
        for elem in [
            "", "X", "AU", "Au", "LI", "Li", "U", "He", "Ne", "Ar", "Kr", "Xe", "Rn", "foo",
        ] {
            assert_eq!(get_mass(elem), None, "{elem:?} must return None");
        }
    }
}
