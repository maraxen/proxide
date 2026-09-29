//! Generalized Born Surface Area (GBSA) support for implicit solvent.
//!
//! Assigns intrinsic Born radii (mbondi2) and OBC2 screening factors.
//!
//! # Two things here are load-bearing
//!
//! **The numbers are not in this file.** They live in `data/mbondi2.xml` with a
//! provenance block naming the reference implementation and pinned revision they
//! were transcribed from. See [`crate::physics::gb_params`].
//!
//! **Dispatch is on the resolved element, never on a prefix of the atom name.**
//! The previous implementation matched `name.chars().next()`, which handed
//! `"SE"` sulfur's radius, `"NA"` nitrogen's, `"CU"` carbon's and `"FE"`
//! fluorine's -- each a confident, plausible, wrong number with no trace that a
//! guess had occurred (CLAUDE.md ledger A2). Element resolution is now delegated
//! to the single canonical `infer_element`.
//!
//! # Unknown is a state, not a number
//!
//! mbondi2 does not define radii for Se, Na, Cu, Fe, Zn, Mn, Br, I, K or Mg.
//! The reference implementation substitutes a documented catch-all for them and
//! so do we -- but every atom that receives one is tagged
//! [`ParameterSource::Fallback`] in a parallel provenance array, so a consumer
//! can tell a tabulated measurement from a stand-in. The plain `assign_*`
//! functions keep their original signatures and drop that channel; prefer the
//! `*_with_provenance` variants anywhere the distinction matters.

use std::collections::HashMap;

use proxide_core::chem::masses::infer_element;

use crate::physics::gb_params::{table, ParameterSource};

/// Radii plus the provenance of each one.
///
/// The two vectors are the same length as the input and index-aligned with it.
/// They are returned together, as one value, on purpose: a caller cannot obtain
/// the numbers while accidentally dropping the record of which were guessed.
#[derive(Debug, Clone, PartialEq)]
pub struct AssignedParameters {
    /// The assigned values, index-aligned with the input atom names.
    pub values: Vec<f32>,
    /// Per-atom [`ParameterSource`] discriminants, index-aligned with `values`.
    pub sources: Vec<u8>,
}

impl AssignedParameters {
    /// Indices of atoms whose value is a stand-in rather than a real parameter.
    ///
    /// An empty slice means every atom got a licensed mbondi2 value. A non-empty
    /// one means those atoms carry the catch-all and their true parameter is
    /// unknown -- see `OutputSpec.strict_parameterization` to make that a hard
    /// error instead of a value to check.
    pub fn unlicensed_atoms(&self) -> Vec<usize> {
        self.sources
            .iter()
            .enumerate()
            .filter(|(_, &code)| !source_is_licensed(code))
            .map(|(i, _)| i)
            .collect()
    }

    /// True when every atom received a real mbondi2 parameter.
    pub fn all_licensed(&self) -> bool {
        self.sources.iter().all(|&c| source_is_licensed(c))
    }
}

/// Whether a raw `u8` provenance code denotes a licensed parameter.
///
/// An unrecognised code is treated as **unlicensed**. That direction is
/// deliberate: a consumer reading provenance written by a newer version must
/// fail towards "I cannot vouch for this", never towards silent acceptance.
fn source_is_licensed(code: u8) -> bool {
    match code {
        c if c == ParameterSource::Tabulated.as_u8() => true,
        c if c == ParameterSource::HydrogenBondedToNitrogen.as_u8() => true,
        c if c == ParameterSource::HydrogenDefault.as_u8() => true,
        _ => false,
    }
}

/// Build an adjacency list from an undirected bond list.
///
/// Neighbour order follows bond order, matching the reference implementation's
/// `_get_bonded_atom_list`, because the mbondi2 hydrogen rule inspects the
/// *first* bonded neighbour and is therefore order-sensitive.
fn adjacency(n_atoms: usize, bonds: &[[usize; 2]]) -> HashMap<usize, Vec<usize>> {
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::with_capacity(n_atoms);
    for i in 0..n_atoms {
        adj.insert(i, Vec::new());
    }
    for bond in bonds {
        // Out-of-range indices are ignored rather than panicking: a bad topology
        // must not take down radius assignment. The affected hydrogen simply
        // fails the bonded-to-nitrogen test and is tagged accordingly.
        if let Some(neighbors) = adj.get_mut(&bond[0]) {
            neighbors.push(bond[1]);
        }
        if let Some(neighbors) = adj.get_mut(&bond[1]) {
            neighbors.push(bond[0]);
        }
    }
    adj
}

/// Assign intrinsic radii using the mbondi2 scheme, recording where each came
/// from.
///
/// See [`crate::physics::gb_params`] for the table and its provenance. Elements
/// outside mbondi2 receive the documented catch-all and are tagged
/// [`ParameterSource::Fallback`].
pub fn assign_mbondi2_radii_with_provenance(
    atom_names: &[String],
    bonds: &[[usize; 2]],
) -> AssignedParameters {
    let params = table();
    let n_atoms = atom_names.len();
    let adj = adjacency(n_atoms, bonds);

    let mut values = vec![0.0f32; n_atoms];
    let mut sources = vec![0u8; n_atoms];

    for (i, name) in atom_names.iter().enumerate() {
        let element = infer_element(name);

        if element == "H" {
            // mbondi2's hydrogen radius depends on the element it is bonded to.
            // The reference implementation inspects only the FIRST bonded
            // neighbour, so we do too -- a hydrogen with more than one bond is
            // unphysical, and quietly disagreeing with the reference on it would
            // be an undocumented divergence.
            let first_neighbor = adj.get(&i).and_then(|neighbors| neighbors.first().copied());

            match first_neighbor {
                Some(neighbor) => {
                    let bonded_to_nitrogen = atom_names
                        .get(neighbor)
                        .map(|n| infer_element(n) == "N")
                        .unwrap_or(false);
                    values[i] = params.hydrogen_radius(bonded_to_nitrogen);
                    sources[i] = if bonded_to_nitrogen {
                        ParameterSource::HydrogenBondedToNitrogen.as_u8()
                    } else {
                        ParameterSource::HydrogenDefault.as_u8()
                    };
                }
                None => {
                    // No bonds at all: the rule cannot be evaluated. Use the
                    // unbonded branch's value but record that it was not derived
                    // from chemistry, because this nearly always means the
                    // topology was never built.
                    values[i] = params.hydrogen_radius(false);
                    sources[i] = ParameterSource::HydrogenUnbonded.as_u8();
                }
            }
            continue;
        }

        match params.radius(element) {
            Some(r) => {
                values[i] = r;
                sources[i] = ParameterSource::Tabulated.as_u8();
            }
            None => {
                // mbondi2 does not define this element. Substituting the
                // catch-all is an unlicensed inference, so it is recorded.
                values[i] = params.fallback_radius();
                sources[i] = ParameterSource::Fallback.as_u8();
            }
        }
    }

    AssignedParameters { values, sources }
}

/// Assign scaling factors for the OBC2 GBSA model, recording provenance.
pub fn assign_obc2_scaling_factors_with_provenance(atom_names: &[String]) -> AssignedParameters {
    let params = table();
    let mut values = Vec::with_capacity(atom_names.len());
    let mut sources = Vec::with_capacity(atom_names.len());

    for name in atom_names {
        match params.screen(infer_element(name)) {
            Some(s) => {
                values.push(s);
                sources.push(ParameterSource::Tabulated.as_u8());
            }
            None => {
                values.push(params.fallback_screen());
                sources.push(ParameterSource::Fallback.as_u8());
            }
        }
    }

    AssignedParameters { values, sources }
}

/// Assign intrinsic radii using the mbondi2 scheme.
///
/// Values only. The provenance channel is discarded, so a caller cannot tell a
/// tabulated radius from the catch-all substituted for an element mbondi2 does
/// not define -- prefer [`assign_mbondi2_radii_with_provenance`] wherever that
/// distinction matters. This signature is retained for existing callers.
pub fn assign_mbondi2_radii(atom_names: &[String], bonds: &[[usize; 2]]) -> Vec<f32> {
    assign_mbondi2_radii_with_provenance(atom_names, bonds).values
}

/// Assign scaling factors for the OBC2 GBSA model.
///
/// Values only; see [`assign_mbondi2_radii`] on the discarded provenance.
pub fn assign_obc2_scaling_factors(atom_names: &[String]) -> Vec<f32> {
    assign_obc2_scaling_factors_with_provenance(atom_names).values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_mbondi2_radii_basic() {
        let atom_names = names(&["N", "CA", "C", "O"]);
        let bonds = vec![[0, 1], [1, 2], [2, 3]];
        let radii = assign_mbondi2_radii(&atom_names, &bonds);

        assert!((radii[0] - 1.55).abs() < 0.01); // N
        assert!((radii[1] - 1.70).abs() < 0.01); // CA is the alpha carbon
        assert!((radii[2] - 1.70).abs() < 0.01); // C
        assert!((radii[3] - 1.50).abs() < 0.01); // O
    }

    #[test]
    fn test_obc2_scaling() {
        let atom_names = names(&["N", "CA", "O", "S", "H"]);
        let f = assign_obc2_scaling_factors(&atom_names);
        assert!((f[0] - 0.79).abs() < 0.001);
        assert!((f[1] - 0.72).abs() < 0.001);
        assert!((f[2] - 0.85).abs() < 0.001);
        assert!((f[3] - 0.96).abs() < 0.001);
        assert!((f[4] - 0.85).abs() < 0.001);
    }

    #[test]
    fn hydrogen_rule_follows_first_bonded_neighbour() {
        // H bonded to N gets 1.30; H bonded to C gets 1.20.
        let atom_names = names(&["N", "H", "CA", "HA"]);
        let bonds = vec![[0, 1], [2, 3]];
        let p = assign_mbondi2_radii_with_provenance(&atom_names, &bonds);

        assert!((p.values[1] - 1.30).abs() < 0.01);
        assert_eq!(
            p.sources[1],
            ParameterSource::HydrogenBondedToNitrogen.as_u8()
        );

        assert!((p.values[3] - 1.20).abs() < 0.01);
        assert_eq!(p.sources[3], ParameterSource::HydrogenDefault.as_u8());
        assert!(p.all_licensed());
    }

    #[test]
    fn unbonded_hydrogen_is_recorded_not_silently_defaulted() {
        let atom_names = names(&["H"]);
        let p = assign_mbondi2_radii_with_provenance(&atom_names, &[]);

        assert!((p.values[0] - 1.20).abs() < 0.01);
        assert_eq!(p.sources[0], ParameterSource::HydrogenUnbonded.as_u8());
        assert!(!p.all_licensed());
        assert_eq!(p.unlicensed_atoms(), vec![0]);
    }

    /// The regression this module exists for. Each of these atom names used to
    /// be dispatched on its first character and silently handed a DIFFERENT
    /// element's tabulated parameter.
    #[test]
    fn two_letter_elements_no_longer_borrow_another_elements_parameters() {
        let atom_names = names(&["SE", "NA", "CU", "FE", "CL"]);
        let p = assign_mbondi2_radii_with_provenance(&atom_names, &[]);

        // Se must NOT get sulfur's 1.80; Na must NOT get nitrogen's 1.55;
        // Cu must NOT get carbon's 1.70. mbondi2 defines none of them.
        for (i, element) in ["Se", "Na", "Cu", "Fe"].iter().enumerate() {
            assert_eq!(
                p.sources[i],
                ParameterSource::Fallback.as_u8(),
                "{element} is absent from mbondi2 and must be tagged as a fallback"
            );
            assert!((p.values[i] - 1.50).abs() < 0.01);
        }

        // Chloride IS in mbondi2 -- it must now resolve as chlorine rather than
        // reaching carbon's arm by way of its first character.
        assert_eq!(p.sources[4], ParameterSource::Tabulated.as_u8());
        assert!((p.values[4] - 1.70).abs() < 0.01);

        assert_eq!(p.unlicensed_atoms(), vec![0, 1, 2, 3]);
    }

    /// The screening table is a separate borrow path with the same defect:
    /// `"CL"` was taking carbon's 0.72 and `"SE"` sulfur's 0.96.
    #[test]
    fn two_letter_elements_get_the_fallback_screening_factor() {
        let atom_names = names(&["CL", "SE", "NA", "CU", "FE"]);
        let p = assign_obc2_scaling_factors_with_provenance(&atom_names);

        for (i, element) in ["Cl", "Se", "Na", "Cu", "Fe"].iter().enumerate() {
            assert_eq!(
                p.sources[i],
                ParameterSource::Fallback.as_u8(),
                "{element} has no OBC2 screening factor of its own"
            );
            assert!((p.values[i] - 0.80).abs() < 0.001);
        }
    }

    /// Silicon was reaching sulfur's arm through its first character.
    #[test]
    fn silicon_resolves_to_its_own_radius() {
        // `infer_element` must resolve "SI" before this test means anything; if
        // it cannot, the value falls back rather than silently becoming sulfur's.
        let p = assign_mbondi2_radii_with_provenance(&names(&["SI"]), &[]);
        assert_ne!(
            p.values[0], 1.80,
            "silicon must never receive sulfur's tabulated radius"
        );
    }

    #[test]
    fn provenance_is_index_aligned_with_values() {
        let atom_names = names(&["N", "CA", "SE", "O", "H"]);
        let bonds = vec![[0, 4]];
        let p = assign_mbondi2_radii_with_provenance(&atom_names, &bonds);
        assert_eq!(p.values.len(), atom_names.len());
        assert_eq!(p.sources.len(), atom_names.len());
    }

    #[test]
    fn plain_api_returns_the_same_values_as_the_provenance_api() {
        let atom_names = names(&["N", "CA", "C", "O", "SE", "CL", "H"]);
        let bonds = vec![[0, 6]];
        assert_eq!(
            assign_mbondi2_radii(&atom_names, &bonds),
            assign_mbondi2_radii_with_provenance(&atom_names, &bonds).values
        );
        assert_eq!(
            assign_obc2_scaling_factors(&atom_names),
            assign_obc2_scaling_factors_with_provenance(&atom_names).values
        );
    }

    /// An out-of-range bond index must not panic radius assignment.
    #[test]
    fn out_of_range_bond_indices_do_not_panic() {
        let atom_names = names(&["N", "H"]);
        let bonds = vec![[0, 99], [42, 1]];
        let p = assign_mbondi2_radii_with_provenance(&atom_names, &bonds);
        assert_eq!(p.values.len(), 2);
    }

    /// An unrecognised provenance code must read as unlicensed, never as fine.
    #[test]
    fn unknown_source_codes_are_treated_as_unlicensed() {
        let p = AssignedParameters {
            values: vec![1.5],
            sources: vec![250],
        };
        assert!(!p.all_licensed());
        assert_eq!(p.unlicensed_atoms(), vec![0]);
    }
}
