use crate::errors::LigandFrameError;

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

fn count_components(n_atoms: usize, bonds: &[(usize, usize)]) -> usize {
    let mut uf = UnionFind::new(n_atoms);
    for &(i, j) in bonds {
        uf.union(i, j);
    }
    let mut roots: Vec<usize> = (0..n_atoms).map(|i| uf.find(i)).collect();
    roots.sort_unstable();
    roots.dedup();
    roots.len()
}

/// Connectivity validation (spec §4, closes Finding 9): errors on any
/// multi-fragment input rather than silently canonicalizing a disconnected
/// graph.
pub fn validate_connected(n_atoms: usize, bonds: &[(usize, usize)]) -> Result<(), LigandFrameError> {
    let component_count = count_components(n_atoms, bonds);
    if component_count > 1 {
        return Err(LigandFrameError::DisconnectedGraph { component_count });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_fragment_passes() {
        // 0-1-2 chain.
        assert!(validate_connected(3, &[(0, 1), (1, 2)]).is_ok());
    }

    #[test]
    fn two_fragments_reported_with_correct_count() {
        // 0-1 and 2-3, two disjoint fragments.
        let err = validate_connected(4, &[(0, 1), (2, 3)]).unwrap_err();
        assert_eq!(err, LigandFrameError::DisconnectedGraph { component_count: 2 });
    }
}
