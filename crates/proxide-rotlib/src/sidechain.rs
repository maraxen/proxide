/// True iff `atom_name` counts as a sidechain atom in ConFind-style clash detection.
/// CB is excluded for all AAs except ALA (doNotCountCB=true semantics).
pub fn counts_as_sidechain(_atom_name: &str, _aa: &str) -> bool {
    todo!("implement in Phase 6")
}

/// True for backbone atoms (N, CA, C, O, H) and hydrogens (name starts with 'H').
pub fn is_backbone_or_hydrogen(_name: &str) -> bool {
    todo!("implement in Phase 6")
}
