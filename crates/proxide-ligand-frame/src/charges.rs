//! Espaloma charge inference wiring and the 3-part cache/fingerprint key
//! (spec §3, closes Finding 5 and Finding 15).

use nalgebra::DMatrix;
use proxide_core::chem::inference::{infer_charges, EspalomaWeights, EMBEDDED_WEIGHTS, FEATURE_UNITS};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::errors::LigandFrameError;

/// `(graph_fingerprint, ref_frame_geometry_hash, espaloma_weights_version)`
/// -- the full cache key from spec §3. NOT graph identity alone: two
/// callers with the same graph but different reference frames must NOT
/// collide.
pub type ChargeCacheKey = (u64, u64, u64);

/// Content hash over canonical-order-invariant graph facts: elements,
/// gaff2 types, and canonical-index bond tuples.
pub fn compute_graph_fingerprint(
    elements: &[String],
    gaff2_types: &[String],
    bonds: &[(usize, usize, u8, bool, bool)],
) -> u64 {
    let mut hasher = DefaultHasher::new();
    elements.hash(&mut hasher);
    gaff2_types.hash(&mut hasher);
    for b in bonds {
        b.hash(&mut hasher);
    }
    hasher.finish()
}

/// Hash of `ref_positions`, rounded to 3 decimal Angstrom to tolerate
/// float noise (spec §3).
pub fn compute_ref_frame_geometry_hash(positions: &[[f64; 3]]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for p in positions {
        for &c in p {
            let rounded = (c * 1000.0).round() as i64;
            rounded.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Version token for the embedded Espaloma weights blob -- invalidates
/// every cache entry if the weights are ever updated (closes Finding 15).
pub fn espaloma_weights_version() -> u64 {
    let mut hasher = DefaultHasher::new();
    EMBEDDED_WEIGHTS.hash(&mut hasher);
    hasher.finish()
}

pub fn charge_cache_key(
    elements: &[String],
    gaff2_types: &[String],
    bonds: &[(usize, usize, u8, bool, bool)],
    ref_positions: &[[f64; 3]],
) -> ChargeCacheKey {
    (
        compute_graph_fingerprint(elements, gaff2_types, bonds),
        compute_ref_frame_geometry_hash(ref_positions),
        espaloma_weights_version(),
    )
}

/// Runs Espaloma inference from precomputed graph features via the
/// existing native message-passing path (the same body
/// `assign_espaloma_charges` in `proxide_py::py_chemistry` wraps).
pub(crate) fn infer_partial_charges(
    espaloma_features: &[[f32; FEATURE_UNITS]],
    espaloma_senders: &[u32],
    espaloma_receivers: &[u32],
    espaloma_total_charge: f32,
) -> Result<Vec<f64>, LigandFrameError> {
    let weights = EspalomaWeights::from_bytes(EMBEDDED_WEIGHTS)
        .map_err(|reason| LigandFrameError::ChargeInferenceFailure { reason })?;

    let n_atoms = espaloma_features.len();
    let mut x = DMatrix::zeros(n_atoms, FEATURE_UNITS);
    for (i, row) in espaloma_features.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            x[(i, j)] = v;
        }
    }
    let segment_ids = vec![0u32; n_atoms];
    let charges = infer_charges(
        &weights,
        &x,
        espaloma_senders,
        espaloma_receivers,
        &segment_ids,
        1,
        &[espaloma_total_charge],
    );
    Ok(charges.into_iter().map(|c| c as f64).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_fingerprint_stable_for_identical_inputs() {
        let elements = vec!["C".to_string(), "H".to_string()];
        let types = vec!["c3".to_string(), "hc".to_string()];
        let bonds = vec![(0, 1, 1u8, false, false)];
        assert_eq!(
            compute_graph_fingerprint(&elements, &types, &bonds),
            compute_graph_fingerprint(&elements, &types, &bonds)
        );
    }

    #[test]
    fn ref_frame_geometry_hash_tolerates_sub_millidegree_noise_but_not_real_moves() {
        let a = vec![[0.0, 0.0, 0.0], [1.089, 0.0, 0.0]];
        let a_plus_noise = vec![[0.0, 0.0, 0.0], [1.089_000_000_1, 0.0, 0.0]];
        let b_moved = vec![[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]];
        assert_eq!(compute_ref_frame_geometry_hash(&a), compute_ref_frame_geometry_hash(&a_plus_noise));
        assert_ne!(compute_ref_frame_geometry_hash(&a), compute_ref_frame_geometry_hash(&b_moved));
    }

    /// Load-bearing regression test for Finding 5: same graph, different
    /// reference frame -> different cache key, so two campaigns never
    /// silently collide on stale charges.
    #[test]
    fn charge_cache_key_differs_when_reference_geometry_differs() {
        let elements = vec!["C".to_string(), "H".to_string()];
        let types = vec!["c3".to_string(), "hc".to_string()];
        let bonds = vec![(0, 1, 1u8, false, false)];
        let pos_a = vec![[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]];
        let pos_b = vec![[0.0, 0.0, 0.0], [1.50, 0.0, 0.0]];
        let key_a = charge_cache_key(&elements, &types, &bonds, &pos_a);
        let key_b = charge_cache_key(&elements, &types, &bonds, &pos_b);
        assert_ne!(key_a, key_b);
        assert_eq!(key_a.0, key_b.0); // same graph fingerprint
        assert_ne!(key_a.1, key_b.1); // different geometry hash
    }

    #[test]
    fn infer_partial_charges_returns_one_charge_per_atom_conserving_total() {
        let features = vec![[0.0f32; FEATURE_UNITS]; 2];
        let senders = vec![0u32, 1];
        let receivers = vec![1u32, 0];
        let charges = infer_partial_charges(&features, &senders, &receivers, 0.0).unwrap();
        assert_eq!(charges.len(), 2);
        assert!((charges[0] + charges[1]).abs() < 1e-5);
    }
}
