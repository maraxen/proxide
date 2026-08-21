//! Owned molecular graph representation for GAFF2 atom typing.
//!
//! Defines the input type for GAFF2 atom type assignment. This is an owned structure
//! (not dependent on external topology types) to maintain clean parity boundaries and
//! enable standalone wasm compilation.

/// Bond order in a Kekulized molecule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
}

/// A single bond between atoms i and j.
#[derive(Debug, Clone)]
pub struct Bond {
    /// Atom index (0-indexed)
    pub i: usize,
    /// Atom index (0-indexed)
    pub j: usize,
    /// Kekulized bond order
    pub order: BondOrder,
    /// Whether the bond connects aromatic atoms (RDKit GetIsAromatic preserved by Kekulize)
    pub aromatic: bool,
}

/// Owned molecular graph for GAFF2 atom typing.
///
/// Encodes connectivity, bond orders, formal charges, and optional ring perception.
/// All bonds must be in Kekulized form (no Aromatic bond type) — aromaticity is tracked
/// per-atom via `aromatic_atoms`, preserving RDKit GetIsAromatic through Kekulize(clearAromaticFlags=False).
///
/// Bond order is **strictly preserved from input** to ensure `atadjust()` coloring
/// parity: that algorithm sweeps bonds in molecule file order and depends on
/// first-match-wins semantics per pass. HashMap-based adjacency or random bond ordering
/// produces different (unstable) results.
#[derive(Debug)]
pub struct MolGraph {
    /// Element symbols (e.g., "C", "H", "N", "O", "S")
    pub elements: Vec<String>,
    /// Formal charges per atom
    pub formal_charges: Vec<i8>,
    /// Bonds in molecule order — **order is load-bearing for atadjust() parity**
    pub bonds: Vec<Bond>,
    /// Optional SSSR from ring perception (e.g., RDKit AtomRings()).
    /// Phase 1: caller-supplied from RDKit.
    /// Phase 2: native implementation (gated separately for independent validation).
    pub rings: Option<Vec<Vec<usize>>>,
}

impl MolGraph {
    /// Create a new molecular graph.
    ///
    /// # Arguments
    /// * `elements` - Element symbols for each atom
    /// * `bonds` - Bonds in molecule order (order is load-bearing)
    /// * `formal_charges` - Formal charge per atom; `None` defaults to all-zero
    ///   (matches RDKit's `GetFormalCharge()` default for an atom with no
    ///   explicit charge set)
    /// * `aromatic_atoms` - Optional per-atom aromatic flags (preserved from
    ///   RDKit's `GetIsAromatic()`), validated for length if present but
    ///   **not stored**: nothing in this crate reads a per-atom aromaticity
    ///   array -- `AtomBondFacts`/f8 aromaticity-class matching derive
    ///   aromaticity entirely from `Bond::aromatic` and ring-based
    ///   classification (see `atom_bond_facts::classify_ring_aromaticity`),
    ///   never from a per-atom flag. Accepted (rather than dropped from the
    ///   signature) so a caller that only has RDKit's per-atom
    ///   `GetIsAromatic()` handy can still pass it for a cheap length/sanity
    ///   check without pre-deriving per-bond flags itself.
    /// * `rings` - Optional SSSR from ring perception (e.g. RDKit
    ///   `GetRingInfo().AtomRings()`); every ring's atom indices are
    ///   validated against `elements`' bounds
    ///
    /// # Returns
    /// A new molecular graph, or an error describing which structural
    /// invariant was violated.
    pub fn new(
        elements: Vec<String>,
        bonds: Vec<Bond>,
        formal_charges: Option<Vec<i8>>,
        aromatic_atoms: Option<Vec<bool>>,
        rings: Option<Vec<Vec<usize>>>,
    ) -> Result<Self, String> {
        let n = elements.len();

        if let Some(fc) = &formal_charges {
            if fc.len() != n {
                return Err(format!(
                    "formal_charges has {} entries, expected {n} (one per atom)",
                    fc.len()
                ));
            }
        }
        if let Some(aa) = &aromatic_atoms {
            if aa.len() != n {
                return Err(format!(
                    "aromatic_atoms has {} entries, expected {n} (one per atom)",
                    aa.len()
                ));
            }
        }
        for (bond_idx, bond) in bonds.iter().enumerate() {
            if bond.i >= n || bond.j >= n {
                return Err(format!(
                    "bond {bond_idx} references atom index out of range \
                     (i={}, j={}, n_atoms={n})",
                    bond.i, bond.j
                ));
            }
            if bond.i == bond.j {
                return Err(format!(
                    "bond {bond_idx} is a self-loop (i == j == {})",
                    bond.i
                ));
            }
        }
        if let Some(rings) = &rings {
            for (ring_idx, ring) in rings.iter().enumerate() {
                for &atom_idx in ring {
                    if atom_idx >= n {
                        return Err(format!(
                            "ring {ring_idx} references atom index {atom_idx} \
                             out of range (n_atoms={n})"
                        ));
                    }
                }
            }
        }

        let formal_charges = formal_charges.unwrap_or_else(|| vec![0; n]);

        Ok(MolGraph {
            elements,
            formal_charges,
            bonds,
            rings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_a_well_formed_molecule() {
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
        .unwrap();
        assert_eq!(mol.elements, vec!["C", "H"]);
        assert_eq!(mol.formal_charges, vec![0, 0]);
        assert_eq!(mol.bonds.len(), 1);
        assert!(mol.rings.is_none());
    }

    #[test]
    fn new_rejects_out_of_range_bond_index() {
        let err = MolGraph::new(
            vec!["C".to_string()],
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
        .unwrap_err();
        assert!(err.contains("out of range"), "unexpected error: {err}");
    }

    #[test]
    fn new_rejects_self_loop_bond() {
        let err = MolGraph::new(
            vec!["C".to_string(), "C".to_string()],
            vec![Bond {
                i: 0,
                j: 0,
                order: BondOrder::Single,
                aromatic: false,
            }],
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(err.contains("self-loop"), "unexpected error: {err}");
    }

    #[test]
    fn new_rejects_mismatched_formal_charges_length() {
        let err = MolGraph::new(
            vec!["C".to_string(), "H".to_string()],
            vec![],
            Some(vec![0]),
            None,
            None,
        )
        .unwrap_err();
        assert!(err.contains("formal_charges"), "unexpected error: {err}");
    }

    #[test]
    fn new_rejects_ring_index_out_of_range() {
        let err = MolGraph::new(
            vec!["C".to_string(), "C".to_string()],
            vec![],
            None,
            None,
            Some(vec![vec![0, 1, 2]]),
        )
        .unwrap_err();
        assert!(err.contains("ring"), "unexpected error: {err}");
    }

    #[test]
    fn new_preserves_bond_order_and_formal_charges_when_provided() {
        let mol = MolGraph::new(
            vec!["C".to_string(), "O".to_string()],
            vec![Bond {
                i: 0,
                j: 1,
                order: BondOrder::Double,
                aromatic: false,
            }],
            Some(vec![0, -1]),
            None,
            Some(vec![]),
        )
        .unwrap();
        assert_eq!(mol.formal_charges, vec![0, -1]);
        assert!(matches!(mol.bonds[0].order, BondOrder::Double));
    }
}
