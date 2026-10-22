//! Per-format residue alias policies.
//!
//! Two decisions are kept apart:
//!
//! * Residue TYPING (MSE -> MET, the model-facing `aatype`) happens once, in
//!   `proxide-core`, for every array format. It is not a per-format choice.
//! * Atom PLACEMENT (which layout residue's atom table lays a residue out, and
//!   which structure atom names are renamed to that table's names) is per array
//!   format. Each format declares its own policy and its formatter consults only
//!   that policy. A future format (e.g. atom72) can declare differently.

/// One alias: an as-read residue name, its layout residue and atom renames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutAlias {
    /// As-read residue name (`res_name`) this alias applies to.
    pub res_name: &'static str,
    /// Residue whose atom table lays this residue out. Atom14 uses it to pick its
    /// slot layout. Atom37 places atoms by name and does not use it; it applies
    /// only `renames`.
    pub layout_residue: &'static str,
    /// `(structure atom name, layout atom name)` pairs.
    pub renames: &'static [(&'static str, &'static str)],
}

/// A format's alias policy.
pub type AliasPolicy = &'static [LayoutAlias];

/// MSE is laid out as MET with its selenium SE in the SD slot (AlphaFold convention).
const MSE_AS_MET: LayoutAlias = LayoutAlias {
    res_name: "MSE",
    layout_residue: "MET",
    renames: &[("SE", "SD")],
};

/// Alias policy of the Atom14 format.
pub const ATOM14_ALIASES: AliasPolicy = &[MSE_AS_MET];

/// Alias policy of the Atom37 format (renames only; see `LayoutAlias::layout_residue`).
pub const ATOM37_ALIASES: AliasPolicy = &[MSE_AS_MET];

/// Residues whose writers emit HETATM records (their own name, not a standard
/// residue record).
pub const HETATM_RESIDUES: &[&str] = &["MSE"];

/// The alias `policy` declares for `res_name`, if any.
pub fn lookup_alias(policy: AliasPolicy, res_name: &str) -> Option<&'static LayoutAlias> {
    policy.iter().find(|alias| alias.res_name == res_name)
}

/// Structure atom name as it appears in the layout after the alias renames.
pub fn rename_layout_atom(name: &str, renames: &[(&str, &str)]) -> String {
    renames
        .iter()
        .find(|(from, _)| *from == name)
        .map_or_else(|| name.to_string(), |(_, to)| to.to_string())
}
