//! PDB format parser bridging proxide-io into WASM-compatible types.
//!
//! This module provides zero-copy PDB parsing for WASM by leveraging
//! proxide-io's PDB parser and converting to WasmMol.

use crate::{WasmAtom, WasmMol};
use proxide_io::formats::pdb::parse_pdb_from_reader;

/// Parse a PDB string and return a WasmMol.
///
/// # Arguments
/// * `text` - PDB file contents as a string slice
///
/// # Returns
/// * `Ok(WasmMol)` on successful parse
/// * `Err(String)` on parse error or other issues
///
/// # Notes
/// - PDB files typically don't include bond connectivity (CONECT records),
///   so the returned WasmMol will have an empty bonds vec.
/// - Residue names are preserved for each atom (parallel to atoms vec).
pub fn parse_pdb(text: &str) -> Result<WasmMol, String> {
    // Parse using proxide-io's PDB parser
    let (raw_data, _model_ids) = parse_pdb_from_reader(text.as_bytes())
        .map_err(|e| format!("PDB parse error: {}", e))?;

    // Convert atoms from RawAtomData to WasmAtom
    let mut atoms = Vec::with_capacity(raw_data.num_atoms);
    for i in 0..raw_data.num_atoms {
        let element = &raw_data.elements[i];
        let atomic_num = element_to_z(element)
            .ok_or_else(|| format!("Unknown element: {}", element))?;

        atoms.push(WasmAtom {
            idx: i,
            atomic_num,
            formal_charge: 0,    // PDB doesn't provide formal charges
            implicit_h: 0,       // PDB doesn't distinguish implicit H; deferred to topology
            in_ring: false,      // Ring detection deferred to topology module
            atom_type: String::new(), // To be populated by assign_params()
        });
    }

    // Residue names (one per atom)
    let residue_names = raw_data.res_names.clone();

    // Bonds: PDB files typically don't include CONECT; empty for now
    // Bond inference from residue topology is deferred to a later task
    let bonds = Vec::new();

    Ok(WasmMol {
        atoms,
        bonds,
        residue_names,
    })
}

/// Convert element symbol to atomic number.
/// Handles common elements in PDB files.
fn element_to_z(element: &str) -> Option<u8> {
    match element.trim().to_uppercase().as_str() {
        "H" => Some(1),
        "C" => Some(6),
        "N" => Some(7),
        "O" => Some(8),
        "S" => Some(16),
        "P" => Some(15),
        "F" => Some(9),
        "CL" => Some(17),
        "BR" => Some(35),
        "I" => Some(53),
        "FE" => Some(26),
        "MG" => Some(12),
        "CA" => Some(20),
        "K" => Some(19),
        "NA" => Some(11),
        "ZN" => Some(30),
        "CU" => Some(29),
        "NI" => Some(28),
        "SE" => Some(34),
        "AS" => Some(33),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALANINE_PDB: &str = "\
ATOM      1  N   ALA A   1       1.458   0.000   0.000  1.00  0.00           N  \n\
ATOM      2  CA  ALA A   1       2.621   0.000   0.819  1.00  0.00           C  \n\
ATOM      3  C   ALA A   1       3.921   0.573   0.235  1.00  0.00           C  \n\
END\n\
";

    #[test]
    fn pdb_alanine_atom_count() {
        let mol = parse_pdb(ALANINE_PDB).unwrap();
        assert_eq!(mol.atoms.len(), 3);
    }

    #[test]
    fn pdb_residue_names_preserved() {
        let mol = parse_pdb(ALANINE_PDB).unwrap();
        assert!(
            mol.residue_names.iter().all(|r| r == "ALA"),
            "residue names should be ALA"
        );
    }

    #[test]
    fn pdb_atomic_numbers_from_element_column() {
        let mol = parse_pdb(ALANINE_PDB).unwrap();
        assert_eq!(mol.atoms[0].atomic_num, 7, "N should be 7");
        assert_eq!(mol.atoms[1].atomic_num, 6, "CA should be C=6");
        assert_eq!(mol.atoms[2].atomic_num, 6, "C should be C=6");
    }

    #[test]
    fn pdb_empty_bonds_for_proteins() {
        let mol = parse_pdb(ALANINE_PDB).unwrap();
        assert_eq!(
            mol.bonds.len(),
            0,
            "PDB files don't provide CONECT; bonds are empty for now"
        );
    }

    #[test]
    fn pdb_no_atoms_fails() {
        let empty_pdb = "END\n";
        let result = parse_pdb(empty_pdb);
        assert!(
            result.is_err(),
            "PDB with no atoms should fail to parse"
        );
    }

    #[test]
    fn pdb_atom_indices_match() {
        let mol = parse_pdb(ALANINE_PDB).unwrap();
        for (i, atom) in mol.atoms.iter().enumerate() {
            assert_eq!(atom.idx, i, "atom index should match position");
        }
    }

    #[test]
    fn element_to_z_common() {
        assert_eq!(element_to_z("H"), Some(1));
        assert_eq!(element_to_z("C"), Some(6));
        assert_eq!(element_to_z("N"), Some(7));
        assert_eq!(element_to_z("O"), Some(8));
        assert_eq!(element_to_z("S"), Some(16));
        assert_eq!(element_to_z("P"), Some(15));
    }

    #[test]
    fn element_to_z_case_insensitive() {
        assert_eq!(element_to_z("c"), Some(6));
        assert_eq!(element_to_z("n"), Some(7));
        assert_eq!(element_to_z("O"), Some(8));
    }

    #[test]
    fn element_to_z_unknown() {
        assert_eq!(element_to_z("XX"), None);
    }
}
