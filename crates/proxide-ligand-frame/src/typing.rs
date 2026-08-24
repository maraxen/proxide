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
/// `partial_charges` is populated via Espaloma charge inference (Task 7,
/// `crate::charges`). `torsion_definitions` is populated (Task 5, via
/// `crate::torsions`); `pucker_definitions`/`unrepresented_ring_dof` are
/// populated in Task 6.
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
    espaloma_features: &[[f32; proxide_core::chem::inference::FEATURE_UNITS]],
    espaloma_senders: &[u32],
    espaloma_receivers: &[u32],
    espaloma_total_charge: f32,
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
                other => return Err(LigandFrameError::InvalidBondOrder { order: other }),
            };
            Ok(proxide_gaff2::mol::Bond { i, j, order, aromatic })
        })
        .collect::<Result<Vec<_>, LigandFrameError>>()?;
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
            (ci.min(cj), ci.max(cj), order, aromatic, false) // restricted_rotation: filled in below
        })
        .collect();

    let bonds_no_restriction: Vec<(usize, usize, u8, bool)> = canon_bonds
        .iter()
        .map(|&(i, j, order, aromatic, _)| (i, j, order, aromatic))
        .collect();
    let restricted = crate::torsions::detect_restricted_rotation(&canon_elements, &bonds_no_restriction);
    let canon_bonds: Vec<(usize, usize, u8, bool, bool)> = canon_bonds
        .into_iter()
        .zip(restricted.iter())
        .map(|((i, j, order, aromatic, _), &r)| (i, j, order, aromatic, r))
        .collect();

    // A bond is ring-internal only if both endpoints belong to a *common*
    // SSSR ring (matches torsions.rs's own doc comment on
    // `torsion_definitions`'s `bond_in_ring` parameter) -- not merely "each
    // endpoint is in *some* ring", which would wrongly mark the acyclic
    // bond joining two separate rings (e.g. bicyclohexyl's connecting C-C
    // bond) as ring-internal and silently drop a real rotatable-torsion DOF.
    let bond_in_ring: Vec<bool> = canon_bonds
        .iter()
        .map(|&(i, j, ..)| canon_ring_membership.iter().any(|ring| ring.contains(&i) && ring.contains(&j)))
        .collect();
    let torsion_definitions =
        crate::torsions::torsion_definitions(&canon_elements, &canon_bonds, &bond_in_ring);

    let mut canon_adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j, ..) in &canon_bonds {
        canon_adjacency[i].push(j);
        canon_adjacency[j].push(i);
    }
    let (pucker_definitions, unrepresented_ring_dof) =
        crate::pucker::build_ring_puckers(&canon_ring_membership, &canon_adjacency);

    // Reference-geometry gate already ran above (Task 4); charge inference
    // runs on ref_positions in INPUT atom order (features arrive
    // pre-ordered to match), independent of canonical reordering.
    let partial_charges = crate::charges::infer_partial_charges(
        espaloma_features,
        espaloma_senders,
        espaloma_receivers,
        espaloma_total_charge,
    )?;
    let mut canon_partial_charges = vec![0.0f64; n];
    for input_idx in 0..n {
        canon_partial_charges[canon[input_idx]] = partial_charges[input_idx];
    }

    Ok(LigandTopology {
        ligand_id: ligand_id.to_string(),
        canonical_order: canon,
        elements: canon_elements,
        atom_names: canon_atom_names,
        gaff2_types: canon_gaff2_types,
        formal_charges: canon_formal_charges,
        partial_charges: canon_partial_charges,
        aromaticity: canon_aromaticity,
        ring_membership: canon_ring_membership,
        bonds: canon_bonds,
        torsion_definitions,
        pucker_definitions,
        unrepresented_ring_dof,
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

        let features = vec![[0.0f32; proxide_core::chem::inference::FEATURE_UNITS]; 6];
        let topology = canonicalize_ligand_topology(
            "methanol",
            &elements,
            &atom_names,
            &bonds_in,
            &bond_is_aromatic,
            &[],
            None,
            &ref_positions,
            &features,
            &[],
            &[],
            0.0f32,
        )
        .expect("well-formed methanol should canonicalize");

        assert_eq!(topology.ligand_id, "methanol");
        assert_eq!(topology.canonical_order.len(), 6);
        assert_eq!(topology.elements.len(), 6);
        assert_eq!(topology.gaff2_types.len(), 6);
        assert_eq!(topology.bonds.len(), 5);
        // No selectable torsion yet (Task 5 adds the machinery).
        assert!(topology.torsion_definitions.is_empty());
        // Zero-filled features + no message-passing edges + total_charge=0.0
        // -> deterministic near-zero charges (f32-precision noise, not
        // exact 0.0; see charges::infer_partial_charges).
        for &c in &topology.partial_charges {
            assert!(c.abs() < 1e-6, "expected near-zero partial charge, got {c}");
        }

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

        let features = vec![[0.0f32; proxide_core::chem::inference::FEATURE_UNITS]; 4];
        let err = canonicalize_ligand_topology(
            "two-fragments",
            &elements,
            &atom_names,
            &bonds_in,
            &bond_is_aromatic,
            &[],
            None,
            &ref_positions,
            &features,
            &[],
            &[],
            0.0f32,
        )
        .unwrap_err();
        assert_eq!(err, LigandFrameError::DisconnectedGraph { component_count: 2 });
    }

    #[test]
    fn molecule_with_real_rotatable_bond_gets_a_torsion_definition() {
        // 2,3-dimethylbutane skeleton: (CH3)2CH-CH(CH3)2 central bond is
        // rotatable (each central carbon has 2 heavy substituents besides
        // its partner).
        let elements = vec!["C", "C", "C", "C", "C", "C"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let atom_names = elements.clone();
        let bonds_in = vec![(0, 1, 1u8), (0, 2, 1), (0, 3, 1), (1, 4, 1), (1, 5, 1)];
        let bond_is_aromatic = vec![false; 5];
        let ref_positions = vec![
            [0.0, 0.0, 0.0],
            [1.53, 0.0, 0.0],
            [-0.5, 1.4, 0.0],
            [-0.5, -1.4, 0.0],
            [2.03, 1.4, 0.0],
            [2.03, -1.4, 0.0],
        ];
        let features = vec![[0.0f32; proxide_core::chem::inference::FEATURE_UNITS]; 6];
        let topology = canonicalize_ligand_topology(
            "dimethylbutane",
            &elements,
            &atom_names,
            &bonds_in,
            &bond_is_aromatic,
            &[],
            None,
            &ref_positions,
            &features,
            &[],
            &[],
            0.0f32,
        )
        .expect("should canonicalize");
        assert_eq!(topology.torsion_definitions.len(), 1);
    }

    #[test]
    fn aromatic_ring_gets_a_pucker_definition_and_no_torsions() {
        let elements = vec!["C"; 6].into_iter().map(String::from).collect::<Vec<_>>();
        let atom_names = elements.clone();
        let bonds_in: Vec<(usize, usize, u8)> = (0..6).map(|i| (i, (i + 1) % 6, 1u8)).collect();
        let bond_is_aromatic = vec![true; 6];
        let ring = vec![vec![0, 1, 2, 3, 4, 5]];
        let radius = 1.4;
        let ref_positions: Vec<[f64; 3]> = (0..6)
            .map(|j| {
                let angle = 2.0 * std::f64::consts::PI * j as f64 / 6.0;
                [radius * angle.cos(), radius * angle.sin(), 0.0]
            })
            .collect();
        let features = vec![[0.0f32; proxide_core::chem::inference::FEATURE_UNITS]; 6];
        let topology = canonicalize_ligand_topology(
            "benzene", &elements, &atom_names, &bonds_in, &bond_is_aromatic, &ring, None, &ref_positions,
            &features, &[], &[], 0.0f32,
        )
        .expect("benzene should canonicalize");
        assert_eq!(topology.pucker_definitions.len(), 1);
        assert_eq!(topology.pucker_definitions[0].ring_size, 6);
        assert!(topology.torsion_definitions.is_empty());
        assert!(topology.unrepresented_ring_dof.is_empty());
    }

    /// Finding 3 (260824 final-review fix round): a bond order outside
    /// {1,2,3} -- e.g. TRIPOS "nc" (not-connected), which
    /// `_parse_mol2_bond_type` in `molecule.py` maps to `0` -- must return a
    /// clean `Err`, not `panic!` across the FFI boundary (a panic there
    /// crashes the whole Python process, not just this call).
    #[test]
    fn invalid_bond_order_is_rejected_not_panicked() {
        let elements = vec!["C".to_string(), "C".to_string()];
        let atom_names = elements.clone();
        let bonds_in = vec![(0, 1, 0u8)]; // order 0: outside {1, 2, 3}
        let bond_is_aromatic = vec![false];
        let ref_positions = vec![[0.0, 0.0, 0.0], [1.53, 0.0, 0.0]];
        let features = vec![[0.0f32; proxide_core::chem::inference::FEATURE_UNITS]; 2];

        let err = canonicalize_ligand_topology(
            "bad-bond-order", &elements, &atom_names, &bonds_in, &bond_is_aromatic, &[], None,
            &ref_positions, &features, &[], &[], 0.0,
        )
        .unwrap_err();
        assert_eq!(err, LigandFrameError::InvalidBondOrder { order: 0 });
    }

    /// Finding 2 (260824 final-review fix round): the acyclic C-C bond
    /// joining two separate 6-membered rings (a bicyclohexyl-like
    /// structure: two cyclohexanes, each atom-disjoint, connected by one
    /// single bond) is a genuine rotatable torsion DOF -- neither endpoint
    /// is ring-internal, since no *single* SSSR ring contains both. The
    /// pre-fix implementation flattened all rings into one `HashSet` and
    /// asked "is each endpoint in *some* ring", which wrongly said yes for
    /// this bond (atom 0 is in ring A, atom 6 is in ring B) and silently
    /// dropped the real DOF (`torsion_definitions.len() == 0` instead of 1).
    #[test]
    fn interring_bond_between_two_disjoint_rings_gets_a_torsion_not_dropped_as_ring_internal() {
        let elements = vec!["C"; 12].into_iter().map(String::from).collect::<Vec<_>>();
        let atom_names = elements.clone();

        // Ring A: 0-1-2-3-4-5-0. Ring B: 6-7-8-9-10-11-6. Connecting bond: 0-6.
        let mut bonds_in: Vec<(usize, usize, u8)> = (0..6).map(|i| (i, (i + 1) % 6, 1u8)).collect();
        bonds_in.extend((0..6).map(|i| (6 + i, 6 + (i + 1) % 6, 1u8)));
        bonds_in.push((0, 6, 1u8));
        let bond_is_aromatic = vec![false; bonds_in.len()];
        let rings = vec![vec![0, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]];

        // Two regular hexagons (side length == circumradius R=1.51, a
        // typical C-C ring bond length) sharing an axis, with ring A's
        // atom 0 and ring B's atom 6 -- the connecting bond's endpoints --
        // facing each other exactly `1.53` A apart (a typical C-C bond
        // length), and the ring centers far enough apart that no other
        // interring atom pair comes anywhere near the geometry gate's
        // clash threshold (0.7 * 1.52 A =~ 1.06 A for C-C).
        let r = 1.51;
        let bond_len = 1.53;
        let center_b_x = 2.0 * r + bond_len;
        let mut ref_positions = Vec::with_capacity(12);
        for i in 0..6 {
            let angle = std::f64::consts::PI * (i as f64) / 3.0; // i * 60 deg, ring A base 0 deg
            ref_positions.push([r * angle.cos(), r * angle.sin(), 0.0]);
        }
        for i in 0..6 {
            let angle = std::f64::consts::PI + std::f64::consts::PI * (i as f64) / 3.0; // ring B base 180 deg
            ref_positions.push([center_b_x + r * angle.cos(), r * angle.sin(), 0.0]);
        }

        let features = vec![[0.0f32; proxide_core::chem::inference::FEATURE_UNITS]; 12];
        let topology = canonicalize_ligand_topology(
            "bicyclohexyl", &elements, &atom_names, &bonds_in, &bond_is_aromatic, &rings, None,
            &ref_positions, &features, &[], &[], 0.0f32,
        )
        .expect("bicyclohexyl-shaped topology should canonicalize");
        assert_eq!(topology.torsion_definitions.len(), 1);
    }

    #[test]
    fn invalid_reference_geometry_blocks_charge_inference() {
        let elements = vec!["C".to_string(), "C".to_string()];
        let atom_names = elements.clone();
        let bonds_in = vec![(0, 1, 1u8)];
        let bond_is_aromatic = vec![false];
        // Declared bond, but positions are 10 A apart -- fails the
        // reference-geometry gate before charge inference ever runs.
        let ref_positions = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let features = vec![[0.0f32; proxide_core::chem::inference::FEATURE_UNITS]; 2];

        let err = canonicalize_ligand_topology(
            "bad-geom", &elements, &atom_names, &bonds_in, &bond_is_aromatic, &[], None,
            &ref_positions, &features, &[], &[], 0.0,
        )
        .unwrap_err();
        assert!(matches!(err, LigandFrameError::InvalidReferenceGeometry { .. }));
    }
}
