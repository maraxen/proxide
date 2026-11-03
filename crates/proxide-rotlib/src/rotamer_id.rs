/// Identifies a specific rotamer: amino acid, phi/psi bin, and rotamer index within that bin.
/// `Ord` is lexicographic (aa, bin_index, rot_index) — used for stable accumulation order.
#[derive(Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct RotamerId {
    pub aa: String,
    pub bin_index: u32,
    pub rot_index: u32,
}

/// A rotamer placed into lab-frame coordinates.
#[derive(Clone, Debug)]
pub struct PlacedRotamer {
    pub id: RotamerId,
    /// Whether `default_bin` was used because `phi` or `psi` was undefined (`None`, Debt #1897).
    pub used_default_bin: bool,
    /// All sidechain heavy atoms in library atom order. Includes CB.
    /// Does NOT include backbone atoms N, CA, C, O.
    pub atoms: Vec<PlacedAtom>,
}

/// A single placed atom with its lab-frame position.
#[derive(Clone, Debug)]
pub struct PlacedAtom {
    pub name: String,
    pub xyz: [f64; 3],
}
