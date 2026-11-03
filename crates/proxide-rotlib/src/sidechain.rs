/// True for peptide backbone heavy atoms and hydrogens.
pub fn is_backbone_or_hydrogen(name: &str) -> bool {
    matches!(name, "N" | "CA" | "C" | "O" | "H") || name.starts_with('H')
}

/// Determines whether an atom contributes to sidechain-sidechain contact calculations.
///
/// In standard residue topologies, Cβ is bonded to Cα and shares the rigid residue
/// backbone frame. For Alanine, Cβ is the sole sidechain heavy atom and is retained;
/// for all other amino acids, Cβ is excluded to prevent rigid backbone-level contacts
/// from artificially inflating rotamer contact degree.
pub fn counts_as_sidechain(atom_name: &str, aa: &str) -> bool {
    if is_backbone_or_hydrogen(atom_name) {
        return false;
    }
    if atom_name == "CB" && aa != "ALA" {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_backbone_or_hydrogen_backbone() {
        assert!(is_backbone_or_hydrogen("N"));
        assert!(is_backbone_or_hydrogen("CA"));
        assert!(is_backbone_or_hydrogen("C"));
        assert!(is_backbone_or_hydrogen("O"));
        assert!(is_backbone_or_hydrogen("H"));
    }

    #[test]
    fn test_is_backbone_or_hydrogen_hydrogens() {
        assert!(is_backbone_or_hydrogen("H1"));
        assert!(is_backbone_or_hydrogen("H2"));
        assert!(is_backbone_or_hydrogen("HB"));
        assert!(is_backbone_or_hydrogen("HG1"));
    }

    #[test]
    fn test_is_backbone_or_hydrogen_non_hydrogen() {
        assert!(!is_backbone_or_hydrogen("CB"));
        assert!(!is_backbone_or_hydrogen("CG"));
        assert!(!is_backbone_or_hydrogen("CG1"));
        assert!(!is_backbone_or_hydrogen("OG"));
    }

    #[test]
    fn test_counts_as_sidechain_backbone_atoms() {
        assert!(!counts_as_sidechain("N", "ALA"));
        assert!(!counts_as_sidechain("CA", "VAL"));
        assert!(!counts_as_sidechain("C", "LEU"));
        assert!(!counts_as_sidechain("O", "ILE"));
    }

    #[test]
    fn test_counts_as_sidechain_hydrogens() {
        assert!(!counts_as_sidechain("H", "ALA"));
        assert!(!counts_as_sidechain("HB", "LEU"));
        assert!(!counts_as_sidechain("H1", "GLY"));
    }

    #[test]
    fn test_counts_as_sidechain_cb_ala() {
        assert!(counts_as_sidechain("CB", "ALA"));
    }

    #[test]
    fn test_counts_as_sidechain_cb_other() {
        assert!(!counts_as_sidechain("CB", "VAL"));
        assert!(!counts_as_sidechain("CB", "LEU"));
        assert!(!counts_as_sidechain("CB", "GLY"));
    }

    #[test]
    fn test_counts_as_sidechain_other_atoms() {
        assert!(counts_as_sidechain("CG", "LEU"));
        assert!(counts_as_sidechain("OG", "SER"));
        assert!(counts_as_sidechain("CD", "LYS"));
    }
}
