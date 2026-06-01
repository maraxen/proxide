use crate::coords::ResidueIndex;

#[derive(Debug, Default)]
pub struct ContactList {
    pub pairs: Vec<(ResidueIndex, ResidueIndex)>,
    pub degrees: Vec<f64>,
}

impl ContactList {
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
