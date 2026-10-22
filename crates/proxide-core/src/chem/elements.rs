//! Canonical element symbols (debt #2366).
//!
//! All 118 IUPAC element symbols in title case, plus `D` (deuterium), `T`
//! (tritium) and `X` (unknown, as wwPDB uses it). The PDB and mmCIF readers use
//! this table to validate and canonicalise element columns.
//!
//! This is deliberately separate from proxide-io's Amber `parm7` table, which
//! covers only atomic numbers 1-86 and uses upper case. That table is not
//! changed by this module.

/// Canonical (title-case) element symbols, plus `D`, `T` and `X`.
pub const ELEMENT_SYMBOLS: &[&str] = &[
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og", "D", "T", "X",
];

/// Canonical symbol for a raw element-column value, or `None` if it is empty or
/// not in [`ELEMENT_SYMBOLS`].
///
/// The input is trimmed and matched ASCII case-insensitively, so `" CL "`,
/// `"cl"` and `"Cl"` all give `Some("Cl")`.
pub fn canonical_element_symbol(raw: &str) -> Option<&'static str> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    ELEMENT_SYMBOLS
        .iter()
        .copied()
        .find(|symbol| symbol.eq_ignore_ascii_case(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_118_iupac_symbols_plus_d_t_x() {
        assert_eq!(ELEMENT_SYMBOLS.len(), 121);
        let mut upper: Vec<String> = ELEMENT_SYMBOLS
            .iter()
            .map(|s| s.to_ascii_uppercase())
            .collect();
        upper.sort();
        upper.dedup();
        assert_eq!(upper.len(), 121, "duplicate symbol in table");
    }

    #[test]
    fn accepts_real_symbols_case_insensitively() {
        assert_eq!(canonical_element_symbol("CL"), Some("Cl"));
        assert_eq!(canonical_element_symbol(" zn "), Some("Zn"));
        assert_eq!(canonical_element_symbol("Cl"), Some("Cl"));
        assert_eq!(canonical_element_symbol("U"), Some("U"));
        assert_eq!(canonical_element_symbol("TH"), Some("Th"));
        assert_eq!(canonical_element_symbol("Og"), Some("Og"));
        assert_eq!(canonical_element_symbol("D"), Some("D"));
        assert_eq!(canonical_element_symbol("T"), Some("T"));
        assert_eq!(canonical_element_symbol("X"), Some("X"));
    }

    #[test]
    fn rejects_unknown_and_empty_values() {
        assert_eq!(canonical_element_symbol("XX"), None);
        assert_eq!(canonical_element_symbol("C1"), None);
        assert_eq!(canonical_element_symbol("Q"), None);
        assert_eq!(canonical_element_symbol(""), None);
        assert_eq!(canonical_element_symbol("   "), None);
    }
}
