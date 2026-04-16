//! Solvent removal utilities
//!
//! Provides functions to identify and filter solvent molecules (water, ions)
//! from protein structures.

#![allow(dead_code)]

use crate::processing::ProcessedStructure;
use crate::structure::RawAtomData;

/// Common solvent residue names
pub const SOLVENT_RESIDUES: &[&str] = &[
    "HOH", "WAT", "TIP3", "TIP4", "TIP5", "SOL", "DOD", "SPC", "SPCE",
];

/// Common ion residue names
pub const ION_RESIDUES: &[&str] = &[
    "NA", "CL", "K", "MG", "CA", "ZN", "FE", "CU", "MN", "CO", "NI", "NA+", "CL-", "K+", "MG2+",
    "CA2+", "ZN2+", "FE2+", "FE3+",
];

/// Check if a residue name is a solvent
pub fn is_solvent(res_name: &str) -> bool {
    SOLVENT_RESIDUES.contains(&res_name.trim().to_uppercase().as_str())
}

/// Check if a residue name is an ion
pub fn is_ion(res_name: &str) -> bool {
    let name = res_name.trim().to_uppercase();
    // Check both short (element) and ionized forms
    ION_RESIDUES
        .iter()
        .any(|&ion| ion.to_uppercase() == name || name.len() <= 2 && ion.starts_with(&name))
}

/// Remove solvent atoms from RawAtomData
/// Returns a new RawAtomData with solvent atoms removed.
pub fn remove_solvent(raw: &RawAtomData) -> RawAtomData {
    let mut filtered = RawAtomData::new();

    for i in 0..raw.num_atoms {
        if !is_solvent(&raw.res_names[i]) {
            // Copy atom data
            filtered.coords.push(raw.coords[i * 3]);
            filtered.coords.push(raw.coords[i * 3 + 1]);
            filtered.coords.push(raw.coords[i * 3 + 2]);
            filtered.atom_names.push(raw.atom_names[i].clone());
            filtered.elements.push(raw.elements[i].clone());
            filtered.serial_numbers.push(raw.serial_numbers[i]);
            filtered.alt_locs.push(raw.alt_locs[i]);
            filtered.res_names.push(raw.res_names[i].clone());
            filtered.res_ids.push(raw.res_ids[i]);
            filtered.insertion_codes.push(raw.insertion_codes[i]);
            filtered.chain_ids.push(raw.chain_ids[i].clone());
            filtered.b_factors.push(raw.b_factors[i]);
            filtered.occupancy.push(raw.occupancy[i]);
            filtered.is_hetatm.push(raw.is_hetatm[i]);

            // Copy optional fields if present
            if let Some(ref charges) = raw.charges {
                if i < charges.len() {
                    filtered
                        .charges
                        .get_or_insert_with(Vec::new)
                        .push(charges[i]);
                }
            }
            if let Some(ref radii) = raw.radii {
                if i < radii.len() {
                    filtered.radii.get_or_insert_with(Vec::new).push(radii[i]);
                }
            }
            if let Some(ref sigmas) = raw.sigmas {
                if i < sigmas.len() {
                    filtered.sigmas.get_or_insert_with(Vec::new).push(sigmas[i]);
                }
            }
            if let Some(ref epsilons) = raw.epsilons {
                if i < epsilons.len() {
                    filtered
                        .epsilons
                        .get_or_insert_with(Vec::new)
                        .push(epsilons[i]);
                }
            }

            filtered.num_atoms += 1;
        }
    }

    filtered
}

/// Remove both solvent and ion atoms from RawAtomData
pub fn remove_solvent_and_ions(raw: &RawAtomData) -> RawAtomData {
    let mut filtered = RawAtomData::new();

    for i in 0..raw.num_atoms {
        let res_name = &raw.res_names[i];
        if !is_solvent(res_name) && !is_ion(res_name) {
            // Copy atom data (same as above)
            filtered.coords.push(raw.coords[i * 3]);
            filtered.coords.push(raw.coords[i * 3 + 1]);
            filtered.coords.push(raw.coords[i * 3 + 2]);
            filtered.atom_names.push(raw.atom_names[i].clone());
            filtered.elements.push(raw.elements[i].clone());
            filtered.serial_numbers.push(raw.serial_numbers[i]);
            filtered.alt_locs.push(raw.alt_locs[i]);
            filtered.res_names.push(raw.res_names[i].clone());
            filtered.res_ids.push(raw.res_ids[i]);
            filtered.insertion_codes.push(raw.insertion_codes[i]);
            filtered.chain_ids.push(raw.chain_ids[i].clone());
            filtered.b_factors.push(raw.b_factors[i]);
            filtered.occupancy.push(raw.occupancy[i]);
            filtered.is_hetatm.push(raw.is_hetatm[i]);

            filtered.num_atoms += 1;
        }
    }

    filtered
}

/// Get solvent atom indices from a ProcessedStructure
pub fn get_solvent_indices(structure: &ProcessedStructure) -> &[usize] {
    &structure.solvent_atoms
}

/// Get ion atom indices from a ProcessedStructure
pub fn get_ion_indices(structure: &ProcessedStructure) -> &[usize] {
    &structure.ion_atoms
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::AtomRecord;

    #[test]
    fn test_is_solvent() {
        assert!(is_solvent("HOH"));
        assert!(is_solvent("WAT"));
        assert!(is_solvent("TIP3"));
        assert!(is_solvent("hoh")); // Case insensitive
        assert!(!is_solvent("ALA"));
        assert!(!is_solvent("GLY"));
    }

    #[test]
    fn test_is_ion() {
        assert!(is_ion("NA"));
        assert!(is_ion("CL"));
        assert!(is_ion("ZN"));
        assert!(is_ion("na")); // Case insensitive
        assert!(!is_ion("ALA"));
    }

    #[test]
    fn test_remove_solvent() {
        let mut raw = RawAtomData::new();

        // Add protein atom
        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        // Add water atom
        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "O".to_string(),
            alt_loc: ' ',
            res_name: "HOH".to_string(),
            chain_id: "A".to_string(),
            res_seq: 100,
            i_code: ' ',
            x: 10.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 30.0,
            element: "O".to_string(),
            charge: None,
            radius: None,
            is_hetatm: true,
        });

        let filtered = remove_solvent(&raw);

        assert_eq!(filtered.num_atoms, 1);
        assert_eq!(filtered.res_names[0], "ALA");
    }
}
