//! Three-letter <-> one-letter amino acid code lookup.
//!
//! Built on the canonical [`proxide_core::chem::residues::RESTYPE_1TO3`]
//! table rather than a new residue-code table — that table already holds
//! every `(one_letter, three_letter)` pair this crate needs; TM-align's
//! sequence-identity fields (`seqID1`/`seqID2`/`seqID_ali`) just need the
//! reverse (three -> one) direction, which this module derives.

use proxide_core::chem::residues::RESTYPE_1TO3;

/// One-letter code for a three-letter residue name (e.g. `"ALA"` -> `'A'`).
///
/// Returns `'X'` for unrecognized residues, matching TM-align's own
/// convention for non-standard residues (`basic_fun.h`'s `aa3to1`).
pub fn three_to_one(res_name: &str) -> char {
    RESTYPE_1TO3
        .iter()
        .find(|&&(_, three)| three.eq_ignore_ascii_case(res_name))
        .and_then(|&(one, _)| one.chars().next())
        .unwrap_or('X')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_residues_map_correctly() {
        assert_eq!(three_to_one("ALA"), 'A');
        assert_eq!(three_to_one("GLY"), 'G');
        assert_eq!(three_to_one("TRP"), 'W');
        assert_eq!(three_to_one("ala"), 'A'); // case-insensitive
    }

    #[test]
    fn unknown_residue_maps_to_x() {
        assert_eq!(three_to_one("XYZ"), 'X');
        assert_eq!(three_to_one("HOH"), 'X'); // water, not an amino acid
    }

    #[test]
    fn every_restype_1to3_entry_round_trips() {
        for &(one, three) in RESTYPE_1TO3.iter() {
            assert_eq!(three_to_one(three), one.chars().next().unwrap());
        }
    }
}
