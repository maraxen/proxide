//! Chemistry constants for protein structures
//!
//! This module contains constants ported from proxide/chem/residues.py
//! for residue types, atom ordering, and residue-specific atom masks.
//!
//! Note: Some utilities are used internally and will be fully exposed later.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

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
pub(crate) fn get_residue_atoms() -> HashMap<&'static str, Vec<&'static str>> {
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

/// How to treat residue names matching the CHARMM/AMBER protonation-variant
/// candidate table (see [`build_variant_alias_table`]), which are also,
/// unfortunately, real (but unrelated) CCD codes for other molecules in every
/// case checked so far -- e.g. CCD `HSE` is L-homoserine, CCD `CYM` is
/// S-methylcysteine, CCD `HIP` is ND1-phosphonohistidine. Blindly renaming by
/// string match alone would silently misidentify genuine depositions of those
/// molecules as histidine/cysteine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidueNamingConvention {
    /// Only apply an alias when the residue's observed heavy-atom composition
    /// matches the aliased target's expected template (see
    /// [`AliasMatchConfig`]). Safe default for files of unknown/mixed
    /// provenance -- correctly rejects e.g. a genuine CCD `CYM`
    /// (S-methylcysteine, has an extra methyl carbon) while still resolving a
    /// true CHARMM/AMBER `CYM`.
    #[default]
    Standard,
    /// Apply the alias table unconditionally, with NO atom-composition check.
    /// Only use this when the caller has out-of-band knowledge that the input
    /// is genuinely force-field-prepped output (e.g. a CHARMM/AMBER QM/MM
    /// simulation reference structure) -- in that context there is no real
    /// ambiguity, since such files never contain the coincidentally-named real
    /// ligands/amino-acids the candidate table collides with.
    ForceFieldPrepped,
}

/// Tunable strictness for the atom-composition check used by `Standard` mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AliasMatchConfig {
    /// Max number of OBSERVED heavy atoms allowed that are NOT in the aliased
    /// target's expected template. Default 0: a strict subset check (every
    /// observed heavy atom must be expected; missing/disordered atoms are
    /// still fine, since this only counts atoms present that shouldn't be).
    /// This is what distinguishes true CHARMM/AMBER `HIP` (His heavy atoms
    /// only) from real CCD `HIP`/phosphohistidine (extra phosphate atoms) --
    /// raising this above 0 starts blurring that distinction, so it's opt-in.
    pub max_unexpected_atoms: usize,
}

/// Candidate residue-name aliases for known CHARMM/AMBER protonation-state and
/// disulfide-bonding conventions. Deliberately excludes codes that are
/// unambiguously resolvable via the real CCD (e.g. `MSE`, selenomethionine --
/// see `chem::ccd`) -- this table exists ONLY for names that are
/// simulation-tool-internal conventions and/or collide with a different, real
/// CCD entry (verified directly against the CCD; see
/// `crates/proxide-core/data/ccd_chem_comp_types.pb.zst` provenance).
pub fn build_variant_alias_table() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    for name in ["HSD", "HSE", "HSP", "HID", "HIE", "HIP"] {
        map.insert(name, "HIS");
    }
    map.insert("ASH", "ASP");
    map.insert("GLH", "GLU");
    map.insert("LYN", "LYS");
    map.insert("CYX", "CYS");
    map.insert("CYM", "CYS");
    map
}

/// Resolve `res_name` to its canonical parent residue if it matches a known
/// CHARMM/AMBER protonation-variant/disulfide-state alias, per `convention`.
///
/// `observed_heavy_atom_names` should contain the atom names actually present
/// on this residue instance, EXCLUDING hydrogens (filter on
/// `AtomRecord::element != "H"`, not name-pattern-guessing). Under `Standard`
/// convention, the alias is only accepted if the count of observed heavy atoms
/// NOT in the target's expected template is `<= match_config.max_unexpected_atoms`
/// -- this is what rejects a genuine, coincidentally-named CCD molecule (e.g.
/// real `CYM` is S-methylcysteine, which has an extra methyl carbon CYS's
/// template lacks).
pub fn resolve_variant_alias(
    res_name: &str,
    observed_heavy_atom_names: &HashSet<&str>,
    convention: ResidueNamingConvention,
    match_config: &AliasMatchConfig,
) -> Option<&'static str> {
    let alias_table = build_variant_alias_table();
    let target = *alias_table.get(res_name)?;

    match convention {
        ResidueNamingConvention::ForceFieldPrepped => Some(target),
        ResidueNamingConvention::Standard => {
            let residue_atoms = get_residue_atoms();
            let expected: HashSet<&str> = residue_atoms.get(target)?.iter().copied().collect();
            let n_unexpected = observed_heavy_atom_names
                .iter()
                .filter(|a| !expected.contains(*a))
                .count();
            if n_unexpected <= match_config.max_unexpected_atoms {
                Some(target)
            } else {
                None
            }
        }
    }
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

    #[test]
    fn test_variant_alias_table_contents() {
        let table = build_variant_alias_table();
        for name in ["HSD", "HSE", "HSP", "HID", "HIE", "HIP"] {
            assert_eq!(table.get(name), Some(&"HIS"), "{name} should alias to HIS");
        }
        assert_eq!(table.get("ASH"), Some(&"ASP"));
        assert_eq!(table.get("GLH"), Some(&"GLU"));
        assert_eq!(table.get("LYN"), Some(&"LYS"));
        assert_eq!(table.get("CYX"), Some(&"CYS"));
        assert_eq!(table.get("CYM"), Some(&"CYS"));
        // MSE is deliberately excluded: it's an unambiguous, genuine CCD entry
        // (selenomethionine) handled by the CCD lookup path, not this table.
        assert_eq!(table.get("MSE"), None);
    }

    fn his_heavy_atoms() -> HashSet<&'static str> {
        get_residue_atoms()["HIS"].iter().copied().collect()
    }

    #[test]
    fn test_resolve_variant_alias_standard_accepts_true_variant() {
        // Real CHARMM HSD: exactly His's heavy-atom set, no extras.
        let observed = his_heavy_atoms();
        let resolved = resolve_variant_alias(
            "HSD",
            &observed,
            ResidueNamingConvention::Standard,
            &AliasMatchConfig::default(),
        );
        assert_eq!(resolved, Some("HIS"));
    }

    #[test]
    fn test_resolve_variant_alias_standard_accepts_partial_atoms() {
        // Disordered/partially-resolved CHARMM HSD: missing atoms are fine,
        // only EXTRA atoms should be rejected.
        let mut observed = his_heavy_atoms();
        observed.remove("CE1");
        let resolved = resolve_variant_alias(
            "HSD",
            &observed,
            ResidueNamingConvention::Standard,
            &AliasMatchConfig::default(),
        );
        assert_eq!(resolved, Some("HIS"));
    }

    #[test]
    fn test_resolve_variant_alias_standard_rejects_real_ccd_collision() {
        // Real CCD CYM is S-methylcysteine: CYS's heavy atoms plus an extra
        // methyl carbon ("CM") that CYS's template does not have. This must
        // NOT be aliased to CYS under the default (Standard, strict) config.
        let mut observed: HashSet<&str> = get_residue_atoms()["CYS"].iter().copied().collect();
        observed.insert("CM");
        let resolved = resolve_variant_alias(
            "CYM",
            &observed,
            ResidueNamingConvention::Standard,
            &AliasMatchConfig::default(),
        );
        assert_eq!(
            resolved, None,
            "a genuine CCD collision (extra atom not in CYS's template) must not be aliased"
        );
    }

    #[test]
    fn test_resolve_variant_alias_force_field_prepped_trusts_unconditionally() {
        // Same mismatched (S-methylcysteine-shaped) atom set as above, but the
        // caller asserts the file is known force-field-prepped output -- the
        // alias should apply unconditionally, with no atom check at all.
        let mut observed: HashSet<&str> = get_residue_atoms()["CYS"].iter().copied().collect();
        observed.insert("CM");
        let resolved = resolve_variant_alias(
            "CYM",
            &observed,
            ResidueNamingConvention::ForceFieldPrepped,
            &AliasMatchConfig::default(),
        );
        assert_eq!(resolved, Some("CYS"));
    }

    #[test]
    fn test_resolve_variant_alias_max_unexpected_atoms_is_tunable() {
        let mut observed: HashSet<&str> = get_residue_atoms()["CYS"].iter().copied().collect();
        observed.insert("CM");

        // Default (0) rejects the single extra atom.
        assert_eq!(
            resolve_variant_alias(
                "CYM",
                &observed,
                ResidueNamingConvention::Standard,
                &AliasMatchConfig::default(),
            ),
            None
        );

        // Loosening to 1 admits exactly one extra atom.
        assert_eq!(
            resolve_variant_alias(
                "CYM",
                &observed,
                ResidueNamingConvention::Standard,
                &AliasMatchConfig {
                    max_unexpected_atoms: 1
                },
            ),
            Some("CYS")
        );
    }

    #[test]
    fn test_resolve_variant_alias_unknown_name_returns_none() {
        let observed = his_heavy_atoms();
        assert_eq!(
            resolve_variant_alias(
                "ALA",
                &observed,
                ResidueNamingConvention::Standard,
                &AliasMatchConfig::default(),
            ),
            None,
            "ALA is not a candidate alias name at all"
        );
    }
}
