//! Canonical atom ranking: graph-invariant refinement (O'Boyle & Sayle
//! style) with the two-layer tie-break from spec §1. Never references 3D
//! coordinates -- canonicalization is a graph-labeling problem.

fn atomic_number(element: &str) -> u8 {
    match element {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "Br" => 35,
        "I" => 53,
        _ => 0,
    }
}

fn dense_ranks<T: Ord + Clone>(keys: &[T]) -> Vec<u32> {
    let mut sorted: Vec<T> = keys.to_vec();
    sorted.sort();
    sorted.dedup();
    keys.iter()
        .map(|k| sorted.binary_search(k).unwrap() as u32)
        .collect()
}

/// Canonical-rank atoms via iterative neighbor-rank refinement to a fixed
/// point, then the spec §1 two-layer tie-break. Returns `canonical_order`
/// where `canonical_order[input_idx]` is the atom's 0-based canonical
/// index -- i.e. the permutation `LigandTopology.canonical_order` carries.
pub fn canonical_order(
    elements: &[String],
    bonds: &[(usize, usize)],
    formal_charges: &[i8],
    atom_aromatic: &[bool],
    atom_in_ring: &[bool],
) -> Vec<usize> {
    let n = elements.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j) in bonds {
        adjacency[i].push(j);
        adjacency[j].push(i);
    }

    // Seed invariant: atomic number, degree, formal charge, aromaticity,
    // ring membership (spec §1).
    let initial_keys: Vec<(u8, usize, i8, bool, bool)> = (0..n)
        .map(|i| {
            (
                atomic_number(&elements[i]),
                adjacency[i].len(),
                formal_charges[i],
                atom_aromatic[i],
                atom_in_ring[i],
            )
        })
        .collect();
    let mut ranks = dense_ranks(&initial_keys);

    // Neighbor-rank-multiset refinement to a fixed point.
    loop {
        let refined_keys: Vec<(u32, Vec<u32>)> = (0..n)
            .map(|i| {
                let mut nb_ranks: Vec<u32> = adjacency[i].iter().map(|&j| ranks[j]).collect();
                nb_ranks.sort_unstable();
                (ranks[i], nb_ranks)
            })
            .collect();
        let new_ranks = dense_ranks(&refined_keys);
        if new_ranks == ranks {
            break;
        }
        ranks = new_ranks;
    }

    // Layer 1: lexicographically-smallest canonical-rank-sorted neighbor
    // sequence. Layer 2: ascending original input index (final fallback
    // for genuine graph automorphisms -- spec §1's documented, accepted
    // limitation).
    let mut input_order: Vec<usize> = (0..n).collect();
    input_order.sort_by(|&a, &b| {
        ranks[a].cmp(&ranks[b]).then_with(|| {
            let mut na: Vec<u32> = adjacency[a].iter().map(|&j| ranks[j]).collect();
            let mut nb: Vec<u32> = adjacency[b].iter().map(|&j| ranks[j]).collect();
            na.sort_unstable();
            nb.sort_unstable();
            na.cmp(&nb).then_with(|| a.cmp(&b))
        })
    });

    let mut canonical_order = vec![0usize; n];
    for (canon_idx, &input_idx) in input_order.iter().enumerate() {
        canonical_order[input_idx] = canon_idx;
    }
    canonical_order
}

#[cfg(test)]
mod tests {
    use super::canonical_order;

    /// The load-bearing property from spec §1: two different input
    /// orderings of the identical (non-automorphic) molecule collapse to
    /// the same canonical *sequence* of elements.
    #[test]
    fn same_molecule_different_input_order_yields_same_canonical_element_sequence() {
        // Variant A: C(0)-F(1),Cl(2),Br(3),H(4)
        let elements_a = vec!["C", "F", "Cl", "Br", "H"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let bonds_a = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let charges = vec![0i8; 5];
        let aromatic = vec![false; 5];
        let in_ring = vec![false; 5];

        // Variant B: same molecule, substituents relabeled in reverse order.
        let elements_b = vec!["C", "H", "Br", "Cl", "F"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let bonds_b = vec![(0, 1), (0, 2), (0, 3), (0, 4)];

        let order_a = canonical_order(&elements_a, &bonds_a, &charges, &aromatic, &in_ring);
        let order_b = canonical_order(&elements_b, &bonds_b, &charges, &aromatic, &in_ring);

        let canonical_elements_a = canonical_sequence(&elements_a, &order_a);
        let canonical_elements_b = canonical_sequence(&elements_b, &order_b);
        assert_eq!(canonical_elements_a, canonical_elements_b);
    }

    /// Layer 2 tie-break (spec §1): the 4 equivalent H atoms of methane are
    /// a genuine graph automorphism -- broken by ascending original input
    /// index, deterministically.
    #[test]
    fn automorphic_atoms_are_tie_broken_by_ascending_input_index() {
        let elements = vec!["C", "H", "H", "H", "H"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let bonds = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let charges = vec![0i8; 5];
        let aromatic = vec![false; 5];
        let in_ring = vec![false; 5];

        let order = canonical_order(&elements, &bonds, &charges, &aromatic, &in_ring);
        // H atoms (input indices 1..4) keep ascending order among
        // themselves; carbon (index 0) is the sole atom in its own class.
        assert!(order[1] < order[2]);
        assert!(order[2] < order[3]);
        assert!(order[3] < order[4]);
    }

    fn canonical_sequence(elements: &[String], order: &[usize]) -> Vec<String> {
        let mut seq = vec![String::new(); elements.len()];
        for (input_idx, &canon_idx) in order.iter().enumerate() {
            seq[canon_idx] = elements[input_idx].clone();
        }
        seq
    }
}
