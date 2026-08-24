use crate::canon::canonical_order;
use crate::connectivity::validate_connected;
use crate::errors::LigandFrameError;
use crate::pucker::RingPucker;

#[derive(Debug, Clone)]
pub struct LigandTopology {
    pub ligand_id: String,
    pub canonical_order: Vec<usize>,
    pub elements: Vec<String>,
    pub atom_names: Vec<String>,
    pub gaff2_types: Vec<String>,
    pub formal_charges: Vec<i8>,
    pub partial_charges: Vec<f64>,
    pub aromaticity: Vec<bool>,
    pub ring_membership: Vec<Vec<usize>>,
    pub bonds: Vec<(usize, usize, u8, bool, bool)>,
    pub torsion_definitions: Vec<[usize; 4]>,
    pub pucker_definitions: Vec<RingPucker>,
    pub unrepresented_ring_dof: Vec<Vec<usize>>,
}

/// Builds the frame-invariant topology: canonical ordering, gaff2 typing
/// (calls into `proxide_gaff2`). Rings/aromaticity are caller-supplied
/// (RDKit-derived in the Python layer, spec §1) rather than computed here.
/// `partial_charges` is populated in Task 7; this task leaves it
/// `vec![0.0; n]`. `torsion_definitions`/`pucker_definitions`/
/// `unrepresented_ring_dof` are populated in Tasks 5-6.
#[allow(clippy::too_many_arguments)] // matches proxide-confind/-tmalign/-rotlib's existing convention
pub fn canonicalize_ligand_topology(
    ligand_id: &str,
    elements: &[String],
    atom_names: &[String],
    bonds_in: &[(usize, usize, u8)],
    bond_is_aromatic: &[bool],
    rings: &[Vec<usize>],
    formal_charges: Option<&[i8]>,
    ref_positions: &[[f64; 3]],
) -> Result<LigandTopology, LigandFrameError> {
    let n = elements.len();
    let pairs: Vec<(usize, usize)> = bonds_in.iter().map(|&(i, j, _)| (i, j)).collect();
    validate_connected(n, &pairs)?;
    crate::geometry_gate::validate_reference_geometry(elements, bonds_in, ref_positions)?;

    let formal_charges_vec: Vec<i8> = formal_charges.map(|f| f.to_vec()).unwrap_or_else(|| vec![0; n]);

    // Per-atom aromaticity/ring-membership derived from the caller-supplied
    // per-bond/per-ring inputs (spec §1).
    let mut atom_aromatic = vec![false; n];
    for (b_idx, &(i, j, _)) in bonds_in.iter().enumerate() {
        if bond_is_aromatic[b_idx] {
            atom_aromatic[i] = true;
            atom_aromatic[j] = true;
        }
    }
    let mut atom_in_ring = vec![false; n];
    for ring in rings {
        for &a in ring {
            atom_in_ring[a] = true;
        }
    }

    let canon = canonical_order(elements, &pairs, &formal_charges_vec, &atom_aromatic, &atom_in_ring);

    // gaff2 typing runs in INPUT index space (matches assign_gaff_atom_types's
    // own contract), then results are reordered into canonical space.
    let gaff2_bonds: Vec<proxide_gaff2::mol::Bond> = bonds_in
        .iter()
        .zip(bond_is_aromatic.iter())
        .map(|(&(i, j, order), &aromatic)| {
            let order = match order {
                1 => proxide_gaff2::mol::BondOrder::Single,
                2 => proxide_gaff2::mol::BondOrder::Double,
                3 => proxide_gaff2::mol::BondOrder::Triple,
                other => panic!("invalid bond order {other}: expected 1, 2, or 3"),
            };
            proxide_gaff2::mol::Bond { i, j, order, aromatic }
        })
        .collect();
    let mol = proxide_gaff2::mol::MolGraph::new(
        elements.to_vec(),
        gaff2_bonds,
        Some(formal_charges_vec.clone()),
        None,
        if rings.is_empty() { None } else { Some(rings.to_vec()) },
    )
    .map_err(|reason| LigandFrameError::SssrInputInvalid { reason })?;
    let gaff2_types_input_order =
        proxide_gaff2::assign_gaff2_atom_types(&mol).map_err(|reason| LigandFrameError::InvalidValence {
            atom_index: reason.len(), // best-effort: orchestrate returns a String, not an index
        })?;

    // Reorder every per-atom/per-bond field into canonical index space.
    let mut canon_elements = vec![String::new(); n];
    let mut canon_atom_names = vec![String::new(); n];
    let mut canon_gaff2_types = vec![String::new(); n];
    let mut canon_formal_charges = vec![0i8; n];
    let mut canon_aromaticity = vec![false; n];
    for input_idx in 0..n {
        let c = canon[input_idx];
        canon_elements[c] = elements[input_idx].clone();
        canon_atom_names[c] = atom_names[input_idx].clone();
        canon_gaff2_types[c] = gaff2_types_input_order[input_idx].clone();
        canon_formal_charges[c] = formal_charges_vec[input_idx];
        canon_aromaticity[c] = atom_aromatic[input_idx];
    }
    let canon_ring_membership: Vec<Vec<usize>> = rings
        .iter()
        .map(|ring| {
            let mut r: Vec<usize> = ring.iter().map(|&a| canon[a]).collect();
            r.sort_unstable();
            r
        })
        .collect();
    let canon_bonds: Vec<(usize, usize, u8, bool, bool)> = bonds_in
        .iter()
        .zip(bond_is_aromatic.iter())
        .map(|(&(i, j, order), &aromatic)| {
            let (ci, cj) = (canon[i], canon[j]);
            (ci.min(cj), ci.max(cj), order, aromatic, false) // restricted_rotation: Task 5
        })
        .collect();

    Ok(LigandTopology {
        ligand_id: ligand_id.to_string(),
        canonical_order: canon,
        elements: canon_elements,
        atom_names: canon_atom_names,
        gaff2_types: canon_gaff2_types,
        formal_charges: canon_formal_charges,
        partial_charges: vec![0.0; n],
        aromaticity: canon_aromaticity,
        ring_membership: canon_ring_membership,
        bonds: canon_bonds,
        torsion_definitions: Vec::new(),
        pucker_definitions: Vec::new(),
        unrepresented_ring_dof: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Methanol: C-O bond exists, but each endpoint's heavy-degree
    /// excluding its partner is 0 (methyl's only heavy neighbor is O; O's
    /// only heavy neighbor is C) -- correctly zero torsion_definitions at
    /// this checkpoint (Task 5 adds the selection logic; this molecule has
    /// none to select).
    #[test]
    fn canonicalizes_methanol_with_correct_types_and_canonical_bond_reordering() {
        let elements = vec!["C", "O", "H", "H", "H", "H"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let atom_names = vec!["C1", "O1", "H1", "H2", "H3", "H4"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let bonds_in = vec![(0, 1, 1u8), (0, 2, 1), (0, 3, 1), (0, 4, 1), (1, 5, 1)];
        let bond_is_aromatic = vec![false; 5];
        let ref_positions = vec![
            [0.0, 0.0, 0.0],
            [1.43, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
            [1.94, 0.75, 0.0],
        ];

        let topology = canonicalize_ligand_topology(
            "methanol",
            &elements,
            &atom_names,
            &bonds_in,
            &bond_is_aromatic,
            &[],
            None,
            &ref_positions,
        )
        .expect("well-formed methanol should canonicalize");

        assert_eq!(topology.ligand_id, "methanol");
        assert_eq!(topology.canonical_order.len(), 6);
        assert_eq!(topology.elements.len(), 6);
        assert_eq!(topology.gaff2_types.len(), 6);
        assert_eq!(topology.bonds.len(), 5);
        // No selectable torsion yet (Task 5 adds the machinery).
        assert!(topology.torsion_definitions.is_empty());
        assert_eq!(topology.partial_charges, vec![0.0; 6]);

        // Content assertions on the actual index-remap direction (not just
        // lengths): the three methyl H's (input 2,3,4) and hydroxyl H
        // (input 5) are automorphic-tie-broken to canonical 0-3, C (input 0)
        // to canonical 4, O (input 1) to canonical 5.
        assert_eq!(topology.canonical_order, vec![4, 5, 0, 1, 2, 3]);
        assert_eq!(
            topology.elements,
            vec!["H", "H", "H", "H", "C", "O"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            topology.atom_names,
            vec!["H1", "H2", "H3", "H4", "C1", "O1"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );

        // Canonical-space bonds, order-independent (i < j within each tuple
        // by construction; compare as a sorted set since the emission order
        // follows input-bond order, not canonical order).
        let mut expected_bonds: Vec<(usize, usize, u8, bool, bool)> = vec![
            (4, 5, 1, false, false),
            (0, 4, 1, false, false),
            (1, 4, 1, false, false),
            (2, 4, 1, false, false),
            (3, 5, 1, false, false),
        ];
        expected_bonds.sort();
        let mut actual_bonds = topology.bonds.clone();
        actual_bonds.sort();
        assert_eq!(actual_bonds, expected_bonds);
    }

    #[test]
    fn disconnected_input_is_rejected() {
        let elements = vec!["C", "H", "C", "H"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let atom_names = elements.clone();
        let bonds_in = vec![(0, 1, 1u8), (2, 3, 1)];
        let bond_is_aromatic = vec![false; 2];
        let ref_positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]];

        let err = canonicalize_ligand_topology(
            "two-fragments",
            &elements,
            &atom_names,
            &bonds_in,
            &bond_is_aromatic,
            &[],
            None,
            &ref_positions,
        )
        .unwrap_err();
        assert_eq!(err, LigandFrameError::DisconnectedGraph { component_count: 2 });
    }
}
