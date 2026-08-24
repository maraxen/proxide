use proxide_geometry::geometry::angles::{bond_angle_f64, dihedral_angle_f64};

use crate::errors::LigandFrameError;
use crate::pucker::cremer_pople_phase;
use crate::typing::LigandTopology;

#[derive(Debug, Clone)]
pub struct LigandFrameCoordinates {
    pub positions: Vec<Vec<[f64; 3]>>,
    pub torsions: Vec<Vec<f64>>,
    pub feature_mask: Vec<bool>,
    pub frame_validity: Vec<bool>,
    pub pucker_phase: Vec<Vec<f64>>,
    pub bond_lengths: Vec<Vec<f64>>,
    pub bond_angles: Vec<Vec<f64>>,
}

const COLLINEAR_EPSILON_RAD: f64 = 1e-3;

fn bond_angle_triples(bonds: &[(usize, usize, u8, bool, bool)]) -> Vec<[usize; 3]> {
    let n = bonds
        .iter()
        .map(|&(i, j, ..)| i.max(j))
        .max()
        .map_or(0, |m| m + 1);
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j, ..) in bonds {
        adjacency[i].push(j);
        adjacency[j].push(i);
    }
    let mut triples = Vec::new();
    for (center, neighbors) in adjacency.iter().enumerate() {
        for x in 0..neighbors.len() {
            for y in (x + 1)..neighbors.len() {
                triples.push([neighbors[x], center, neighbors[y]]);
            }
        }
    }
    triples
}

/// Per-frame extraction, indexed by an already-built `LigandTopology`.
/// Validates the topology<->positions atom-identity contract via
/// `input_elements` before processing (spec §4, closes Finding 8).
pub fn extract_ligand_frame_coordinates(
    topology: &LigandTopology,
    positions: &[Vec<[f64; 3]>],
    input_elements: &[String],
) -> Result<LigandFrameCoordinates, LigandFrameError> {
    let n_atoms = topology.canonical_order.len();

    if input_elements.len() != n_atoms {
        return Err(LigandFrameError::TopologyPositionMismatch {
            expected_atoms: n_atoms,
            got_atoms: input_elements.len(),
        });
    }
    let mut canonical_input_elements = vec![String::new(); n_atoms];
    for (input_idx, &canon_idx) in topology.canonical_order.iter().enumerate() {
        canonical_input_elements[canon_idx] = input_elements[input_idx].clone();
    }
    if canonical_input_elements != topology.elements {
        return Err(LigandFrameError::TopologyPositionMismatch {
            expected_atoms: n_atoms,
            got_atoms: input_elements.len(),
        });
    }

    let n_frames = positions.len();
    let mut canonical_positions: Vec<Vec<[f64; 3]>> = Vec::with_capacity(n_frames);
    for frame in positions {
        if frame.len() != n_atoms {
            return Err(LigandFrameError::TopologyPositionMismatch {
                expected_atoms: n_atoms,
                got_atoms: frame.len(),
            });
        }
        let mut reordered = vec![[0.0; 3]; n_atoms];
        for (input_idx, &canon_idx) in topology.canonical_order.iter().enumerate() {
            reordered[canon_idx] = frame[input_idx];
        }
        canonical_positions.push(reordered);
    }

    let mut frame_validity = vec![true; n_frames];
    for (f, frame) in canonical_positions.iter().enumerate() {
        if frame.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
            frame_validity[f] = false;
        }
    }

    let n_torsions = topology.torsion_definitions.len();
    let mut torsions = vec![vec![0.0f64; n_frames]; n_torsions];
    for (t_idx, quad) in topology.torsion_definitions.iter().enumerate() {
        let [a, b, c, d] = *quad;
        for (f, frame) in canonical_positions.iter().enumerate() {
            if !frame_validity[f] {
                torsions[t_idx][f] = f64::NAN;
                continue;
            }
            let abc = bond_angle_f64(&frame[a], &frame[b], &frame[c]);
            let bcd = bond_angle_f64(&frame[b], &frame[c], &frame[d]);
            let near_collinear =
                |theta: f64| theta.min(std::f64::consts::PI - theta) < COLLINEAR_EPSILON_RAD;
            if near_collinear(abc) || near_collinear(bcd) {
                frame_validity[f] = false;
                torsions[t_idx][f] = f64::NAN;
                continue;
            }
            torsions[t_idx][f] = dihedral_angle_f64(&frame[a], &frame[b], &frame[c], &frame[d]);
        }
    }
    let feature_mask = vec![true; n_torsions];

    let n_rings = topology.pucker_definitions.len();
    let mut pucker_phase = vec![vec![0.0f64; n_frames]; n_rings];
    for (r_idx, ring) in topology.pucker_definitions.iter().enumerate() {
        for (f, frame) in canonical_positions.iter().enumerate() {
            if !frame_validity[f] {
                pucker_phase[r_idx][f] = f64::NAN;
                continue;
            }
            let ring_positions: Vec<[f64; 3]> = ring.ring_atoms.iter().map(|&a| frame[a]).collect();
            pucker_phase[r_idx][f] = cremer_pople_phase(&ring_positions);
        }
    }

    let n_bonds = topology.bonds.len();
    let mut bond_lengths = vec![vec![0.0f64; n_frames]; n_bonds];
    for (b_idx, &(i, j, ..)) in topology.bonds.iter().enumerate() {
        for (f, frame) in canonical_positions.iter().enumerate() {
            let d = [
                frame[i][0] - frame[j][0],
                frame[i][1] - frame[j][1],
                frame[i][2] - frame[j][2],
            ];
            bond_lengths[b_idx][f] = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        }
    }

    let angle_triples = bond_angle_triples(&topology.bonds);
    let mut bond_angles = vec![vec![0.0f64; n_frames]; angle_triples.len()];
    for (a_idx, &[a, b, c]) in angle_triples.iter().enumerate() {
        for (f, frame) in canonical_positions.iter().enumerate() {
            bond_angles[a_idx][f] = bond_angle_f64(&frame[a], &frame[b], &frame[c]);
        }
    }

    Ok(LigandFrameCoordinates {
        positions: canonical_positions,
        torsions,
        feature_mask,
        frame_validity,
        pucker_phase,
        bond_lengths,
        bond_angles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pucker::RingPucker;
    use std::f64::consts::PI;

    fn two_atom_topology() -> LigandTopology {
        LigandTopology {
            ligand_id: "test".to_string(),
            canonical_order: vec![1, 0], // input 0 -> canonical 1, input 1 -> canonical 0
            elements: vec!["H".to_string(), "C".to_string()],
            atom_names: vec!["H1".to_string(), "C1".to_string()],
            gaff2_types: vec!["hc".to_string(), "c3".to_string()],
            formal_charges: vec![0, 0],
            partial_charges: vec![0.0, 0.0],
            aromaticity: vec![false, false],
            ring_membership: vec![],
            bonds: vec![(0, 1, 1, false, false)],
            torsion_definitions: vec![],
            pucker_definitions: vec![],
            unrepresented_ring_dof: vec![],
        }
    }

    #[test]
    fn atom_count_mismatch_rejected() {
        let topology = two_atom_topology();
        let positions = vec![vec![[0.0, 0.0, 0.0]]]; // only 1 atom, topology expects 2
        let input_elements = vec!["C".to_string()];
        let err =
            extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap_err();
        assert!(matches!(
            err,
            LigandFrameError::TopologyPositionMismatch { .. }
        ));
    }

    #[test]
    fn element_order_inconsistent_with_canonical_order_rejected() {
        let topology = two_atom_topology();
        let positions = vec![vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]];
        // Wrong: claims input order is H, C. The fixture's canonical_order
        // ([1, 0]) + elements ([H, C]) were built from input order C, H (as
        // `positions_reordered_into_canonical_atom_order` below confirms is
        // the correct/accepted order), so H, C is inconsistent and must be
        // rejected.
        let input_elements = vec!["H".to_string(), "C".to_string()];
        let err =
            extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap_err();
        assert!(matches!(
            err,
            LigandFrameError::TopologyPositionMismatch { .. }
        ));
    }

    #[test]
    fn positions_reordered_into_canonical_atom_order() {
        let topology = two_atom_topology();
        // Input order: atom0=C at (1,0,0), atom1=H at (0,0,0).
        let positions = vec![vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]];
        let input_elements = vec!["C".to_string(), "H".to_string()];
        let result =
            extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
        // canonical index 0 = H (input 1) = (0,0,0); canonical index 1 = C (input 0) = (1,0,0)
        assert_eq!(result.positions[0][0], [0.0, 0.0, 0.0]);
        assert_eq!(result.positions[0][1], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn frame_validity_false_for_nan_input_positions() {
        let topology = two_atom_topology();
        let positions = vec![vec![[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]]];
        let input_elements = vec!["C".to_string(), "H".to_string()];
        let result =
            extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
        assert_eq!(result.frame_validity, vec![false]);
    }

    #[test]
    fn torsion_extraction_matches_known_perpendicular_dihedral() {
        let mut topology = two_atom_topology();
        topology.canonical_order = vec![0, 1, 2, 3];
        topology.elements = vec!["C".to_string(); 4];
        topology.atom_names = topology.elements.clone();
        topology.gaff2_types = vec!["c3".to_string(); 4];
        topology.formal_charges = vec![0; 4];
        topology.partial_charges = vec![0.0; 4];
        topology.aromaticity = vec![false; 4];
        topology.bonds = vec![
            (0, 1, 1, false, false),
            (1, 2, 1, false, false),
            (2, 3, 1, false, false),
        ];
        topology.torsion_definitions = vec![[0, 1, 2, 3]];

        // Same 90-degree dihedral fixture as angles.rs's own
        // test_dihedral_angle_perpendicular.
        let frame = vec![
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ];
        let positions = vec![frame];
        let input_elements = topology.elements.clone();

        let result =
            extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
        assert!((result.torsions[0][0].abs() - std::f64::consts::FRAC_PI_2).abs() < 0.1);
    }

    #[test]
    fn frame_validity_false_for_near_collinear_torsion_triple() {
        // Same 4-atom torsion topology as
        // torsion_extraction_matches_known_perpendicular_dihedral, but this
        // time atoms 0,1,2 are placed almost perfectly collinear so the
        // A-B-C bond angle is within COLLINEAR_EPSILON_RAD of 0 (or PI),
        // making the dihedral about the B-C bond undefined.
        let mut topology = two_atom_topology();
        topology.canonical_order = vec![0, 1, 2, 3];
        topology.elements = vec!["C".to_string(); 4];
        topology.atom_names = topology.elements.clone();
        topology.gaff2_types = vec!["c3".to_string(); 4];
        topology.formal_charges = vec![0; 4];
        topology.partial_charges = vec![0.0; 4];
        topology.aromaticity = vec![false; 4];
        topology.bonds = vec![
            (0, 1, 1, false, false),
            (1, 2, 1, false, false),
            (2, 3, 1, false, false),
        ];
        topology.torsion_definitions = vec![[0, 1, 2, 3]];

        // A=(0,0,0), B=(1,0,0), C=(2,0,0) are exactly collinear (angle
        // A-B-C = PI, well inside the epsilon band); D is off-axis so only
        // the A-B-C leg triggers the guard.
        let frame = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],
        ];
        let positions = vec![frame];
        let input_elements = topology.elements.clone();

        let result =
            extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
        assert_eq!(result.frame_validity, vec![false]);
        assert!(result.torsions[0][0].is_nan());
    }

    #[test]
    fn bond_lengths_are_index_aligned_with_topology_bonds() {
        // Global Constraints (plan): bonds<->bond_lengths must be
        // index-aligned by construction and covered by a test asserting
        // that alignment, not just lengths matching. A length-only check
        // (`bond_lengths.len() == bonds.len()`) cannot distinguish correct
        // alignment from a transposition bug (e.g. swapping bond_lengths[0]
        // and bond_lengths[1]) -- so this fixture uses a 4-atom chain with
        // THREE bonds of distinct, hand-computed lengths (1.0, 1.5, 2.0 A),
        // placed on the x-axis so each bond's Euclidean length is an exact,
        // easily-verified value:
        //   A=(0,0,0), B=(1.0,0,0)  -> |A-B| = 1.0  (bond 0)
        //   C=(2.5,0,0)             -> |B-C| = 1.5  (bond 1)
        //   D=(4.5,0,0)             -> |C-D| = 2.0  (bond 2)
        // A swap of any two of bond_lengths[0..3] produces a mismatch
        // against a *specific* bond's known length, which this test would
        // catch (verified via RED/GREEN in the task-8 fix-round report).
        let mut topology = two_atom_topology();
        topology.canonical_order = vec![0, 1, 2, 3];
        topology.elements = vec!["C".to_string(); 4];
        topology.atom_names = topology.elements.clone();
        topology.gaff2_types = vec!["c3".to_string(); 4];
        topology.formal_charges = vec![0; 4];
        topology.partial_charges = vec![0.0; 4];
        topology.aromaticity = vec![false; 4];
        topology.bonds = vec![
            (0, 1, 1, false, false),
            (1, 2, 1, false, false),
            (2, 3, 1, false, false),
        ];

        let frame = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ];
        let positions = vec![frame];
        let input_elements = topology.elements.clone();

        let result =
            extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
        assert_eq!(result.bond_lengths.len(), 3);
        assert!(
            (result.bond_lengths[0][0] - 1.0).abs() < 1e-9,
            "bond 0 (A-B) expected length 1.0, got {}",
            result.bond_lengths[0][0]
        );
        assert!(
            (result.bond_lengths[1][0] - 1.5).abs() < 1e-9,
            "bond 1 (B-C) expected length 1.5, got {}",
            result.bond_lengths[1][0]
        );
        assert!(
            (result.bond_lengths[2][0] - 2.0).abs() < 1e-9,
            "bond 2 (C-D) expected length 2.0, got {}",
            result.bond_lengths[2][0]
        );
    }

    #[test]
    fn pucker_phase_is_index_aligned_with_topology_pucker_definitions() {
        // Global Constraints (plan): pucker_definitions<->pucker_phase must
        // be index-aligned by construction and covered by a test asserting
        // that alignment. A single ring can't prove alignment (there's
        // nothing to transpose against with n_rings=1), so this fixture
        // uses TWO independent 6-membered rings (atoms 0-5 and 6-11) with
        // deliberately different, hand-derived Cremer-Pople phases.
        //
        // Derivation: for a planar regular hexagon (atoms at
        // theta_j = 2*pi*j/6, radius R, in the xy-plane) displaced
        // out-of-plane by a pure m=2 Fourier mode
        //   zeta_j = A*cos(2*theta_j) + B*sin(2*theta_j),
        // the mode is orthogonal (over the 6-point discrete Fourier basis)
        // to the m=1 harmonics that would otherwise tilt the fitted mean
        // plane, so the Cremer-Pople normal vector comes out exactly along
        // the z-axis (verified by hand: r_prime_z = r_double_prime_z = 0
        // for any A, B). That makes the per-atom CP height
        // z[j] = -zeta_j exactly, and the dominant-mode coefficients:
        //   a = sum_j z[j]*cos(2*theta_j) = -3A
        //   b = -sum_j z[j]*sin(2*theta_j) = 3B
        // (using sum_j cos^2(2*theta_j) = sum_j sin^2(2*theta_j) = 3 and
        // sum_j cos(2*theta_j)*sin(2*theta_j) = 0 for n=6), giving
        // phase = atan2(b, a) = atan2(B, -A).
        //
        // Ring 0: A=-S, B=S  -> atan2(S, S)   = pi/4
        // Ring 1: A= S, B=S  -> atan2(S, -S)  = 3*pi/4
        // for any scale S != 0 (S=0.3 here) -- two well-separated,
        // hand-computable phases far from the atan2 branch cut, so a
        // transposition of pucker_phase[0] and pucker_phase[1] fails this
        // test by ~pi/2, far outside the 1e-6 tolerance.
        let mut topology = two_atom_topology();
        topology.canonical_order = (0..12).collect();
        topology.elements = vec!["C".to_string(); 12];
        topology.atom_names = topology.elements.clone();
        topology.gaff2_types = vec!["c3".to_string(); 12];
        topology.formal_charges = vec![0; 12];
        topology.partial_charges = vec![0.0; 12];
        topology.aromaticity = vec![false; 12];
        topology.bonds = vec![];
        topology.pucker_definitions = vec![
            RingPucker {
                ring_atoms: (0..6).collect(),
                ring_size: 6,
            },
            RingPucker {
                ring_atoms: (6..12).collect(),
                ring_size: 6,
            },
        ];

        let radius = 1.4;
        let scale = 0.3;
        let ring0: Vec<[f64; 3]> = (0..6)
            .map(|j| {
                let theta = 2.0 * PI * j as f64 / 6.0;
                let zeta = scale * (-(2.0 * theta).cos() + (2.0 * theta).sin());
                [radius * theta.cos(), radius * theta.sin(), zeta]
            })
            .collect();
        let ring1: Vec<[f64; 3]> = (0..6)
            .map(|j| {
                let theta = 2.0 * PI * j as f64 / 6.0;
                let zeta = scale * ((2.0 * theta).cos() + (2.0 * theta).sin());
                [radius * theta.cos(), radius * theta.sin(), zeta]
            })
            .collect();
        let mut frame = ring0;
        frame.extend(ring1);
        let positions = vec![frame];
        let input_elements = topology.elements.clone();

        let result =
            extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
        assert_eq!(result.pucker_phase.len(), 2);
        assert!(
            (result.pucker_phase[0][0] - PI / 4.0).abs() < 1e-6,
            "ring 0 expected phase pi/4, got {}",
            result.pucker_phase[0][0]
        );
        assert!(
            (result.pucker_phase[1][0] - 3.0 * PI / 4.0).abs() < 1e-6,
            "ring 1 expected phase 3*pi/4, got {}",
            result.pucker_phase[1][0]
        );
    }
}
