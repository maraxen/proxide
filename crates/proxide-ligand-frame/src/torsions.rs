//! Restricted-rotation detection (bounded v1 pattern list: amide,
//! ester/thioester, sp2-sp2 twist bond) and deterministic torsion
//! definitions (spec §2). Operates in canonical index space -- "highest
//! canonical rank" reduces to "largest canonical index."

fn heavy_degree_excluding(elements: &[String], adjacency: &[Vec<usize>], atom: usize, exclude: usize) -> usize {
    adjacency[atom]
        .iter()
        .filter(|&&nb| nb != exclude && elements[nb] != "H")
        .count()
}

/// `bonds`: canonical-index `(i, j, order, is_aromatic)`. Returns one flag
/// per bond, aligned by position (index-alignment contract, spec §4).
pub fn detect_restricted_rotation(elements: &[String], bonds: &[(usize, usize, u8, bool)]) -> Vec<bool> {
    let n = elements.len();
    let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n]; // (neighbor, bond_idx)
    for (b_idx, &(i, j, _, _)) in bonds.iter().enumerate() {
        adjacency[i].push((j, b_idx));
        adjacency[j].push((i, b_idx));
    }

    let is_carbonyl_carbon = |atom: usize| {
        elements[atom] == "C"
            && adjacency[atom]
                .iter()
                .any(|&(nb, b_idx)| elements[nb] == "O" && bonds[b_idx].2 == 2)
    };
    let is_sp2 = |atom: usize| {
        adjacency[atom]
            .iter()
            .any(|&(_, b_idx)| bonds[b_idx].2 == 2 || bonds[b_idx].3)
    };

    bonds
        .iter()
        .map(|&(i, j, order, aromatic)| {
            if aromatic {
                return false; // aromatic ring bonds are handled by pucker
            }
            if order != 1 {
                // Only a formally-single bond adjoining a carbonyl (the
                // amide C-N / ester C-O(S) bond) exhibits restricted
                // rotation from conjugation; the C=O double bond itself is
                // excluded from torsion selection elsewhere (it's a
                // terminal-oxygen bond) and must not self-match here --
                // is_carbonyl_carbon(j) is trivially true for the carbonyl
                // carbon's own C=O bond, which would otherwise flag that
                // bond as "ester" against its own double-bonded oxygen.
                return false;
            }
            let amide = (is_carbonyl_carbon(i) && elements[j] == "N")
                || (is_carbonyl_carbon(j) && elements[i] == "N");
            let ester = (is_carbonyl_carbon(i) && (elements[j] == "O" || elements[j] == "S"))
                || (is_carbonyl_carbon(j) && (elements[i] == "O" || elements[i] == "S"));
            let twist = order == 1 && is_sp2(i) && is_sp2(j);
            amide || ester || twist
        })
        .collect()
}

/// `bond_in_ring[k]` is true when both endpoints of `bonds[k]` belong to a
/// common SSSR ring -- ring-internal bonds are handled by pucker (spec
/// §2a), never as independent torsions, even when not aromatic.
pub fn torsion_definitions(
    elements: &[String],
    bonds: &[(usize, usize, u8, bool, bool)], // i, j, order, is_aromatic, restricted_rotation
    bond_in_ring: &[bool],
) -> Vec<[usize; 4]> {
    let n = elements.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j, ..) in bonds {
        adjacency[i].push(j);
        adjacency[j].push(i);
    }

    let branch_atom = |center: usize, partner: usize| -> Option<usize> {
        adjacency[center]
            .iter()
            .copied()
            .filter(|&nb| nb != partner && elements[nb] != "H")
            .max() // canonical index space: highest canonical rank == largest index
    };

    let mut defs = Vec::new();
    for (b_idx, &(i, j, _order, aromatic, restricted)) in bonds.iter().enumerate() {
        if aromatic || restricted || bond_in_ring[b_idx] {
            continue;
        }
        if heavy_degree_excluding(elements, &adjacency, i, j) < 1
            || heavy_degree_excluding(elements, &adjacency, j, i) < 1
        {
            continue; // terminal atom or freely-rotating methyl exclusion
        }
        if let (Some(a), Some(d)) = (branch_atom(i, j), branch_atom(j, i)) {
            defs.push([a, i, j, d]);
        }
    }
    defs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amide_bond_is_flagged_restricted_plain_single_bond_is_not() {
        // O=C(0)-N(2), C(0)-C(1) also present (plain single bond).
        let elements = vec!["O", "C", "N", "C"].into_iter().map(String::from).collect::<Vec<_>>();
        let bonds = vec![(0, 1, 2u8, false), (1, 2, 1, false), (1, 3, 1, false)];
        let flags = detect_restricted_rotation(&elements, &bonds);
        assert_eq!(flags, vec![false, true, false]);
    }

    #[test]
    fn terminal_methyl_bond_excluded_from_torsions() {
        // C(0)-C(1), C(1)'s only heavy neighbor besides C(0) is none (methyl).
        let elements = vec!["C", "C", "H", "H", "H"].into_iter().map(String::from).collect::<Vec<_>>();
        let bonds = vec![
            (0, 1, 1u8, false, false),
            (1, 2, 1, false, false),
            (1, 3, 1, false, false),
            (1, 4, 1, false, false),
        ];
        let defs = torsion_definitions(&elements, &bonds, &[false; 4]);
        assert!(defs.is_empty());
    }

    #[test]
    fn torsion_definition_uses_highest_canonical_index_branch_atoms() {
        // Central bond 0-1; atom 0 also bonded to heavy atoms 2 and 5
        // (5 > 2, so 5 is the deterministic branch atom); atom 1 bonded to
        // heavy atom 3.
        let elements = vec!["C"; 6].into_iter().map(String::from).collect::<Vec<_>>();
        let bonds = vec![
            (0, 1, 1u8, false, false),
            (0, 2, 1, false, false),
            (0, 5, 1, false, false),
            (1, 3, 1, false, false),
        ];
        let defs = torsion_definitions(&elements, &bonds, &[false; 4]);
        assert_eq!(defs, vec![[5, 0, 1, 3]]);
    }

    #[test]
    fn ring_bond_excluded_even_when_not_aromatic() {
        // Cyclohexane-like: 6 carbons in a ring, all single, non-aromatic.
        let elements = vec!["C"; 6].into_iter().map(String::from).collect::<Vec<_>>();
        let bonds: Vec<(usize, usize, u8, bool, bool)> = (0..6)
            .map(|i| (i, (i + 1) % 6, 1u8, false, false))
            .collect();
        let defs = torsion_definitions(&elements, &bonds, &[true; 6]);
        assert!(defs.is_empty());
    }
}
