//! Canonical topology and per-frame geometric feature extraction for
//! ligand reference frames.
//!
//! Sits beside `proxide-gaff2` rather than inside it: that crate's own
//! module docs scope it to atom typing only
//! (`crates/proxide-gaff2/src/lib.rs:6-13`), so canonicalization,
//! torsion/pucker definition, and charge wiring live here as a *caller* of
//! `proxide_gaff2::assign_gaff2_atom_types`, not an extension of it.

pub mod canon;
pub mod charges;
pub mod connectivity;
pub mod errors;
pub mod geometry_gate;
pub mod pucker;
pub mod torsions;
pub mod typing;

pub use errors::LigandFrameError;
pub use pucker::RingPucker;
pub use typing::{canonicalize_ligand_topology, LigandTopology};

#[cfg(test)]
mod smoke_tests {
    #[test]
    fn crate_links_against_gaff2_and_core() {
        use proxide_gaff2::mol::{Bond, BondOrder, MolGraph};

        let mol = MolGraph::new(
            vec!["C".to_string(), "H".to_string()],
            vec![Bond {
                i: 0,
                j: 1,
                order: BondOrder::Single,
                aromatic: false,
            }],
            None,
            None,
            None,
        )
        .expect("MolGraph::new should accept a minimal well-formed molecule");
        assert_eq!(mol.elements.len(), 2);

        let weights = proxide_core::chem::inference::EspalomaWeights::from_bytes(
            proxide_core::chem::inference::EMBEDDED_WEIGHTS,
        )
        .expect("embedded Espaloma weights should parse");
        assert_eq!(
            weights.sages.len(),
            proxide_core::chem::inference::N_SAGE_LAYERS
        );
    }
}
