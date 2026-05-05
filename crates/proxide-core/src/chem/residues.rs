//! Chemistry constants for protein structures
//!
//! This module contains constants ported from proxide/chem/residues.py
//! for residue types, atom ordering, and residue-specific atom masks.
//!
//! Note: Some utilities are used internally and will be fully exposed later.

#![allow(dead_code)]

use std::collections::HashMap;

/// Standard 20 amino acids in alphabetical order (AlphaFold convention)
pub const RESTYPES: [&str; 20] = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y",
    "V",
];

/// 1-letter to 3-letter code mapping
pub const RESTYPE_1TO3: [(&str, &str); 20] = [
    ("A", "ALA"),
    ("R", "ARG"),
    ("N", "ASN"),
    ("D", "ASP"),
    ("C", "CYS"),
    ("Q", "GLN"),
    ("E", "GLU"),
    ("G", "GLY"),
    ("H", "HIS"),
    ("I", "ILE"),
    ("L", "LEU"),
    ("K", "LYS"),
    ("M", "MET"),
    ("F", "PHE"),
    ("P", "PRO"),
    ("S", "SER"),
    ("T", "THR"),
    ("W", "TRP"),
    ("Y", "TYR"),
    ("V", "VAL"),
];

/// Atom37 ordering - standard atom types
pub const ATOM_TYPES: [&str; 37] = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1",
    "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2",
    "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
];

/// Number of atom types in atom37 format
pub const ATOM_TYPE_NUM: usize = 37;

/// Number of standard residue types (20 amino acids)
pub const RESTYPE_NUM: usize = 20;

/// Index for unknown residue type
pub const UNK_RESTYPE_INDEX: usize = 20;

/// Build atom order mapping (atom name -> index)
pub fn build_atom_order() -> HashMap<String, usize> {
    ATOM_TYPES
        .iter()
        .enumerate()
        .map(|(i, name)| (name.to_string(), i))
        .collect()
}

/// Build 3-letter to index mapping
pub fn build_resname_to_idx() -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, (_, three)) in RESTYPE_1TO3.iter().enumerate() {
        map.insert(three.to_string(), i);
    }
    map.insert("UNK".to_string(), UNK_RESTYPE_INDEX);
    map
}

/// Build 1-letter to index mapping
pub fn build_restype_order() -> HashMap<String, usize> {
    RESTYPES
        .iter()
        .enumerate()
        .map(|(i, letter)| (letter.to_string(), i))
        .collect()
}

/// Atoms present in each residue type (excluding hydrogens)
/// Returns a 21x37 mask array where mask[restype][atom_type] = 1 if present
pub fn build_standard_atom_mask() -> Vec<Vec<u8>> {
    let atom_order = build_atom_order();
    let mut mask = vec![vec![0u8; ATOM_TYPE_NUM]; RESTYPE_NUM + 1];

    // Define atoms for each residue type
    let residue_atoms = get_residue_atoms();

    for (i, (_, three_letter)) in RESTYPE_1TO3.iter().enumerate() {
        if let Some(atoms) = residue_atoms.get(*three_letter) {
            for atom_name in atoms {
                if let Some(&atom_idx) = atom_order.get(*atom_name) {
                    mask[i][atom_idx] = 1;
                }
            }
        }
    }

    // Index 20 is for unknown residues - all zeros
    mask
}

/// Get atoms for each residue type
fn get_residue_atoms() -> HashMap<&'static str, Vec<&'static str>> {
    let mut map = HashMap::new();

    map.insert("ALA", vec!["C", "CA", "CB", "N", "O"]);
    map.insert(
        "ARG",
        vec![
            "C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2",
        ],
    );
    map.insert("ASP", vec!["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"]);
    map.insert("ASN", vec!["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"]);
    map.insert("CYS", vec!["C", "CA", "CB", "N", "O", "SG"]);
    map.insert(
        "GLU",
        vec!["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    );
    map.insert(
        "GLN",
        vec!["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    );
    map.insert("GLY", vec!["C", "CA", "N", "O"]);
    map.insert(
        "HIS",
        vec!["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    );
    map.insert("ILE", vec!["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"]);
    map.insert("LEU", vec!["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"]);
    map.insert(
        "LYS",
        vec!["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    );
    map.insert("MET", vec!["C", "CA", "CB", "CG", "CE", "N", "O", "SD"]);
    map.insert(
        "PHE",
        vec![
            "C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O",
        ],
    );
    map.insert("PRO", vec!["C", "CA", "CB", "CG", "CD", "N", "O"]);
    map.insert("SER", vec!["C", "CA", "CB", "N", "O", "OG"]);
    map.insert("THR", vec!["C", "CA", "CB", "CG2", "N", "O", "OG1"]);
    map.insert(
        "TRP",
        vec![
            "C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "N", "NE1", "O",
        ],
    );
    map.insert(
        "TYR",
        vec![
            "C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH",
        ],
    );
    map.insert("VAL", vec!["C", "CA", "CB", "CG1", "CG2", "N", "O"]);

    map
}

/// Atom14 reduced representation - defines which atoms to include for each residue
pub fn build_restype_atom14_names() -> HashMap<&'static str, Vec<&'static str>> {
    let mut map = HashMap::new();

    map.insert(
        "ALA",
        vec![
            "N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "ARG",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", "",
        ],
    );
    map.insert(
        "ASN",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "ASP",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "CYS",
        vec![
            "N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "GLN",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", "",
        ],
    );
    map.insert(
        "GLU",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", "",
        ],
    );
    map.insert(
        "GLY",
        vec!["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    );
    map.insert(
        "HIS",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", "",
        ],
    );
    map.insert(
        "ILE",
        vec![
            "N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "LEU",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "LYS",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", "",
        ],
    );
    map.insert(
        "MET",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "PHE",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", "",
        ],
    );
    map.insert(
        "PRO",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "SER",
        vec![
            "N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "THR",
        vec![
            "N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "TRP",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
        ],
    );
    map.insert(
        "TYR",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", "",
        ],
    );
    map.insert(
        "VAL",
        vec![
            "N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", "",
        ],
    );
    map.insert(
        "UNK",
        vec!["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    );

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_order() {
        let atom_order = build_atom_order();
        assert_eq!(atom_order.len(), 37);
        assert_eq!(atom_order["N"], 0);
        assert_eq!(atom_order["CA"], 1);
        assert_eq!(atom_order["C"], 2);
        assert_eq!(atom_order["CB"], 3);
        assert_eq!(atom_order["O"], 4);
    }

    #[test]
    fn test_resname_to_idx() {
        let resname_idx = build_resname_to_idx();
        assert_eq!(resname_idx["ALA"], 0);
        assert_eq!(resname_idx["ARG"], 1);
        assert_eq!(resname_idx["UNK"], 20);
    }

    #[test]
    fn test_standard_atom_mask() {
        let mask = build_standard_atom_mask();
        assert_eq!(mask.len(), 21); // 20 + unknown
        assert_eq!(mask[0].len(), 37); // ALA

        // ALA should have N, CA, C, CB, O
        let atom_order = build_atom_order();
        assert_eq!(mask[0][atom_order["N"]], 1);
        assert_eq!(mask[0][atom_order["CA"]], 1);
        assert_eq!(mask[0][atom_order["CB"]], 1);

        // ALA should not have CG
        assert_eq!(mask[0][atom_order["CG"]], 0);
    }

    #[test]
    fn test_restype_constants() {
        assert_eq!(RESTYPES.len(), 20);
        assert_eq!(RESTYPE_1TO3.len(), 20);
        assert_eq!(ATOM_TYPE_NUM, 37);
        assert_eq!(RESTYPE_NUM, 20);
    }
}
