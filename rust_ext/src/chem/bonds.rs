//! Bond order definitions and lookup utilities
//!
//! Provides bond order inference for standard residues.
//! Default bond order is 1 (Single).
//!
//! Bond orders:
//! 1 = Single
//! 2 = Double
//! 3 = Triple
//! 4 = Aromatic (PyMOL/ChemFiles convention often uses 4, or treated as 1.5)

/// Get bond order for a bond between two atoms in a residue.
///
/// # Arguments
/// * `res_name` - Residue name (e.g. "ALA")
/// * `atom1` - Atom 1 name
/// * `atom2` - Atom 2 name
///
/// # Returns
/// Bond order (1, 2, 3, or 4). Default is 1.
pub fn get_bond_order(res_name: &str, atom1: &str, atom2: &str) -> u8 {
    // Normalize names (trim whitespace if needed, though they usually come clean)
    // Order doesn't matter, check both ways by sorting
    let (a, b) = if atom1 < atom2 {
        (atom1, atom2)
    } else {
        (atom2, atom1)
    };

    // 1. Backbone Carbonyl: (C, O) is always Double (2) in standard amino acids
    if a == "C" && b == "O" {
        return 2;
    }

    // 2. Sidechain Rules by Residue
    match res_name {
        "ASP" => {
            if a == "CG" && (b == "OD1" || b == "OD2") {
                return 2;
            }
        }
        "GLU" => {
            if a == "CD" && (b == "OE1" || b == "OE2") {
                return 2;
            }
        }
        "ASN" => {
            // CG-OD1 (Double), CG-ND2 (Single)
            if a == "CG" && b == "OD1" {
                return 2;
            }
        }
        "GLN" => {
            // CD-OE1 (Double), CD-NE2 (Single)
            if a == "CD" && b == "OE1" {
                return 2;
            }
        }
        "ARG" => {
            return 4; // Treat as aromatic/resonant within the group
        }
        "HIS" => {
            // Imidazole ring: Aromatic
            if is_ring_atom_his(a) && is_ring_atom_his(b) {
                return 4;
            }
        }
        "PHE" | "TYR" => {
            // Phenyl ring
            if is_ring_atom_phe_tyr(a) && is_ring_atom_phe_tyr(b) {
                return 4;
            }
        }
        "TRP" => {
            // Indole ring
            if is_ring_atom_trp(a) && is_ring_atom_trp(b) {
                return 4;
            }
        }
        _ => {}
    }

    // Default Single
    1
}

fn is_ring_atom_his(name: &str) -> bool {
    matches!(name, "CG" | "ND1" | "CE1" | "NE2" | "CD2")
}

fn is_ring_atom_phe_tyr(name: &str) -> bool {
    matches!(name, "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ")
}

fn is_ring_atom_trp(name: &str) -> bool {
    matches!(
        name,
        "CG" | "CD1" | "CD2" | "NE1" | "CE2" | "CE3" | "CZ2" | "CZ3" | "CH2"
    )
}
