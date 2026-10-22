//! Per-format residue alias policies.
//!
//! An array format decides how a residue whose as-read name has a different
//! layout residue is laid out: which residue's atom table it uses and which
//! structure atom names are renamed to that table's names. Each format declares
//! its own policy and its formatter consults only that policy. Nothing here is
//! applied globally, so a future format (e.g. atom72) can declare differently.

/// One alias: an as-read residue name, its layout residue and atom renames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutAlias {
    /// As-read residue name (`res_name`) this alias applies to.
    pub res_name: &'static str,
    /// Residue whose atom table lays this residue out.
    pub layout_residue: &'static str,
    /// `(name in the structure, name in the layout)` pairs.
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

/// Alias policy of the Atom37 format.
pub const ATOM37_ALIASES: AliasPolicy = &[MSE_AS_MET];

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
