mod common;

use common::{
    make_synthetic_backbone, make_synthetic_backbone_no_atoms,
    make_synthetic_backbone_partial_atoms,
};
use proxide_confind::cache::build_bb_atoms;
use proxide_confind::coords::ResidueIndex;

#[test]
fn bb_atoms_count_full_residue() {
    // Single residue with all 4 atoms (N, CA, C, O all Some)
    let bb = make_synthetic_backbone(1, 3.8);
    let (atoms, atom_res) = build_bb_atoms(&bb);

    assert_eq!(atoms.len(), 4, "Full residue should emit 4 atoms");
    assert_eq!(atom_res.len(), 4, "atom_res should match atoms length");
    for ri in &atom_res {
        assert_eq!(
            *ri,
            ResidueIndex(0),
            "All atoms should belong to ResidueIndex(0)"
        );
    }
}

#[test]
fn bb_atoms_count_partial_residue() {
    // Residue with n=Some, ca=Some, c=None, o=None
    let bb = make_synthetic_backbone_partial_atoms(1, 3.8, &[0]);
    let (atoms, atom_res) = build_bb_atoms(&bb);

    assert_eq!(atoms.len(), 2, "Partial residue should emit 2 atoms");
    assert_eq!(atom_res.len(), 2, "atom_res should match atoms length");
}

#[test]
fn bb_atoms_two_residues_ordering() {
    // Two full residues (4 atoms each)
    let bb = make_synthetic_backbone(2, 3.8);
    let (atoms, atom_res) = build_bb_atoms(&bb);

    assert_eq!(
        atoms.len(),
        8,
        "Two full residues should emit 8 atoms total"
    );
    assert_eq!(atom_res.len(), 8, "atom_res should match atoms length");

    // First 4 atoms belong to ResidueIndex(0)
    for i in 0..4 {
        assert_eq!(atom_res[i], ResidueIndex(0));
    }
    // Next 4 atoms belong to ResidueIndex(1)
    for i in 4..8 {
        assert_eq!(atom_res[i], ResidueIndex(1));
    }
}

#[test]
fn bb_atoms_empty_backbone() {
    // Zero residues → both vecs empty
    let bb = make_synthetic_backbone(0, 3.8);
    let (atoms, atom_res) = build_bb_atoms(&bb);

    assert_eq!(atoms.len(), 0, "Empty backbone should emit no atoms");
    assert_eq!(atom_res.len(), 0, "Empty backbone should emit no atom_res");
}

#[test]
fn bb_atoms_all_none_atoms() {
    // Residue with n=None, ca=None, c=None, o=None → zero atoms emitted
    let bb = make_synthetic_backbone_no_atoms(1);
    let (atoms, atom_res) = build_bb_atoms(&bb);

    assert_eq!(
        atoms.len(),
        0,
        "Residue with all None atoms should emit no atoms"
    );
    assert_eq!(atom_res.len(), 0, "atom_res should be empty");
}
