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
    let n = bonds.iter().map(|&(i, j, ..)| i.max(j)).max().map_or(0, |m| m + 1);
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j, ..) in bonds {
        adjacency[i].push(j);
        adjacency[j].push(i);
    }
    let mut triples = Vec::new();
    for center in 0..n {
        let neighbors = &adjacency[center];
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
            let near_collinear = |theta: f64| theta.min(std::f64::consts::PI - theta) < COLLINEAR_EPSILON_RAD;
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
        let err = extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap_err();
        assert!(matches!(err, LigandFrameError::TopologyPositionMismatch { .. }));
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
        let err = extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap_err();
        assert!(matches!(err, LigandFrameError::TopologyPositionMismatch { .. }));
    }

    #[test]
    fn positions_reordered_into_canonical_atom_order() {
        let topology = two_atom_topology();
        // Input order: atom0=C at (1,0,0), atom1=H at (0,0,0).
        let positions = vec![vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]];
        let input_elements = vec!["C".to_string(), "H".to_string()];
        let result = extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
        // canonical index 0 = H (input 1) = (0,0,0); canonical index 1 = C (input 0) = (1,0,0)
        assert_eq!(result.positions[0][0], [0.0, 0.0, 0.0]);
        assert_eq!(result.positions[0][1], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn frame_validity_false_for_nan_input_positions() {
        let topology = two_atom_topology();
        let positions = vec![vec![[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]]];
        let input_elements = vec!["C".to_string(), "H".to_string()];
        let result = extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
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
        topology.bonds = vec![(0, 1, 1, false, false), (1, 2, 1, false, false), (2, 3, 1, false, false)];
        topology.torsion_definitions = vec![[0, 1, 2, 3]];

        // Same 90-degree dihedral fixture as angles.rs's own
        // test_dihedral_angle_perpendicular.
        let frame = vec![[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]];
        let positions = vec![frame];
        let input_elements = topology.elements.clone();

        let result = extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
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
        topology.bonds = vec![(0, 1, 1, false, false), (1, 2, 1, false, false), (2, 3, 1, false, false)];
        topology.torsion_definitions = vec![[0, 1, 2, 3]];

        // A=(0,0,0), B=(1,0,0), C=(2,0,0) are exactly collinear (angle
        // A-B-C = PI, well inside the epsilon band); D is off-axis so only
        // the A-B-C leg triggers the guard.
        let frame = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 1.0]];
        let positions = vec![frame];
        let input_elements = topology.elements.clone();

        let result = extract_ligand_frame_coordinates(&topology, &positions, &input_elements).unwrap();
        assert_eq!(result.frame_validity, vec![false]);
        assert!(result.torsions[0][0].is_nan());
    }
}
