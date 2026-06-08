use crate::coords::ResidueIndex;

/// Threshold above which two residue positions are considered "poised to interact" —
/// i.e., the amino-acid identity at one position influences the identity at the other.
/// Used in dTERMen TERM neighbourhood construction.
///
/// Residue pairs with `cd > CONTACT_THRESHOLD` are included in TERM
/// neighbourhoods for dTERMen sequence optimisation. This is the canonical
/// published value from Zheng & Grigoryan (2017) PLoS ONE 12(5): e0178272, eq. 9.
///
/// # Design decision
///
/// `CONTACT_THRESHOLD` is applied once at output filtering and has no
/// coupling to the residue cache (unlike `CLASH_DIST`/`CONT_DIST` which
/// invalidate cached sidechain grids if changed). Downstream consumers
/// with different TERM-density requirements (or non-dTERMen uses of
/// contact degree) may pass a custom threshold to `ContactList::filter()`.
/// For canonical dTERMen behaviour, pass `CONTACT_THRESHOLD`.
///
/// See ADR: `.praxia/docs/decisions/260602_contact-threshold-adr.md`
///
/// Source: Zheng & Grigoryan (2017) PLoS ONE 12(5): e0178272, eq. 9.
pub const CONTACT_THRESHOLD: f64 = 0.02;

/// Output of a [`crate::ConFind::contacts`] query.
///
/// Stores (residue_a, residue_b) contact pairs and their associated
/// contact-degree values.  Pairs are stored in insertion order; use
/// [`ordered_pairs`](ContactList::ordered_pairs) for a stable sorted view.
#[derive(Debug, Default)]
pub struct ContactList {
    /// Unordered contact pairs.  Each element corresponds to `degrees[i]`.
    pub pairs: Vec<(ResidueIndex, ResidueIndex)>,
    /// Contact-degree value for each pair (same index as `pairs`).
    pub degrees: Vec<f64>,
}

impl ContactList {
    /// Return the contact degree for the pair `(a, b)`, or `None` if not found.
    ///
    /// Accepts either ordering of the residue indices.
    pub fn degree(&self, a: ResidueIndex, b: ResidueIndex) -> Option<f64> {
        self.pairs
            .iter()
            .position(|&(x, y)| (x == a && y == b) || (x == b && y == a))
            .map(|i| self.degrees[i])
    }

    /// Pairs sorted ascending by (res_a, res_b).
    pub fn ordered_pairs(&self) -> Vec<(ResidueIndex, ResidueIndex)> {
        let mut v = self.pairs.clone();
        v.sort_unstable();
        v
    }

    /// Filter to pairs with `cd > threshold`.
    ///
    /// Pass [`CONTACT_THRESHOLD`] for canonical dTERMen behaviour.
    /// Downstream consumers with different TERM-density requirements may
    /// pass a custom threshold without re-running ConFind.
    pub fn filter(&self, threshold: f64) -> ContactList {
        let mut out = ContactList::default();
        for (i, &pair) in self.pairs.iter().enumerate() {
            if self.degrees[i] > threshold {
                out.pairs.push(pair);
                out.degrees.push(self.degrees[i]);
            }
        }
        out
    }
}
