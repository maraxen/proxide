use crate::coords::ResidueIndex;

/// Contact-degree threshold defining "poised to interact" residue pairs.
///
/// Residue pairs with `cd > CONTACT_THRESHOLD` are included in TERM
/// neighbourhoods for dTERMen sequence optimisation. This is the canonical
/// published value from Zheng & Grigoryan (2017) *PLoS ONE* 12(5): e0178272, eq. 9.
///
/// Downstream consumers with different TERM-density requirements may pass
/// a custom threshold; pass `CONTACT_THRESHOLD` for canonical dTERMen behaviour.
///
/// See ADR: `.praxia/docs/decisions/260602_contact-threshold-adr.md`
pub const CONTACT_THRESHOLD: f64 = 0.02;

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
