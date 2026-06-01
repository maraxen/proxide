use crate::coords::ResidueIndex;

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
}
